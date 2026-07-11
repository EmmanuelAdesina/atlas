import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from atlas.config.settings import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_second: int):
        self.calls_per_second = calls_per_second
        self._semaphore = asyncio.Semaphore(calls_per_second)
        self._interval = 1.0 / calls_per_second
    
    async def acquire(self):
        await self._semaphore.acquire()
        asyncio.create_task(self._release())
    
    async def _release(self):
        await asyncio.sleep(self._interval)
        self._semaphore.release()

class AtlasHTTPClient:
    """Single HTTP client for all collectors."""
    
    def __init__(self):
        self.settings = settings
        self.rate_limiter = RateLimiter(self.settings.rate_limit_per_second)
        self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout),
                headers={"User-Agent": self.settings.user_agent},
                follow_redirects=True,
            )
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with rate limiting and retries."""
        
        # Rate limit
        await self.rate_limiter.acquire()
        
        # Build request
        client = await self._get_client()
        req_headers = {"User-Agent": self.settings.user_agent}
        if headers:
            req_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            logger.debug(f"Request: {method} {url}")
            response = await client.request(
                method=method,
                url=url,
                params=params,
                headers=req_headers,
                json=json,
                **kwargs
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Response: {response.status_code} in {duration:.2f}s")
            
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("POST", url, **kwargs)
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

# Singleton instance
default_client = AtlasHTTPClient()
