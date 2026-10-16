"""Read-only Polymarket Gamma discovery collector.

Uses the public Gamma API (https://gamma-api.polymarket.com) — no API key,
no wallet, no trading. Endpoints verified against official docs 2026-10-12:
  GET /markets?closed=<bool>&limit=<n>          list markets (simple)
  GET /markets/keyset?...&after_cursor=<cursor>  keyset pagination
  GET /markets/<id> | /markets/slug/<slug>       fetch one market
  GET /events/keyset, /events/<id>, /public-search?q=

Every fetch persists an immutable raw snapshot BEFORE mapping, and returns a
CollectionResult (brief section 9 + existing atlas/schemas/collection.py).
"""

from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from atlas.config.settings import settings
from atlas.http.client import default_client
from atlas.schemas.collection import CollectionResult, CollectionStatus
from atlas.services.snapshots import save_raw_snapshot

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
SOURCE = "polymarket_gamma"


def _classify_error(e: Exception) -> CollectionStatus:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code in (401, 403):
            return CollectionStatus.AUTH_FAILED
        if code == 429:
            return CollectionStatus.RATE_LIMITED
        return CollectionStatus.FAILED
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
        return CollectionStatus.NETWORK_ERROR
    return CollectionStatus.FAILED


class PolymarketGammaCollector:
    """Discover Polymarket markets/events via the public read-only Gamma API."""

    def __init__(self, client=None, raw_data_path: Optional[str] = None):
        self.client = client or default_client
        self.raw_data_path = raw_data_path or settings.raw_data_path

    async def _fetch(self, path: str, params: dict[str, Any]) -> CollectionResult:
        collected_at = datetime.now(timezone.utc)
        query = urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{GAMMA_BASE_URL}{path}"
        source_url = f"{url}?{query}" if query else url
        try:
            response = await self.client.get(url, params=params)
            payload = response.json()
        except Exception as e:  # network / HTTP / decode failures
            return CollectionResult(
                source=SOURCE,
                collected_at=collected_at,
                status=_classify_error(e),
                raw_data=None,
                metadata={"source_url": source_url},
                error=f"{type(e).__name__}: {e}",
            )
        snapshot_path = save_raw_snapshot(
            source=SOURCE,
            endpoint=source_url,
            payload=payload,
            raw_data_path=self.raw_data_path,
            collected_at=collected_at,
        )
        count = len(payload) if isinstance(payload, list) else 1
        return CollectionResult(
            source=SOURCE,
            collected_at=collected_at,
            status=CollectionStatus.SUCCESS if count else CollectionStatus.EMPTY,
            raw_data=payload,
            metadata={
                "source_url": source_url,
                "snapshot_path": str(snapshot_path),
                "count": count,
            },
        )

    async def discover_markets(
        self,
        limit: int = 20,
        closed: bool = False,
        tag_id: Optional[int] = None,
    ) -> CollectionResult:
        """List markets. `tag_id` filters by category (see Gamma /tags)."""
        params: dict[str, Any] = {"closed": str(closed).lower(), "limit": limit}
        if tag_id is not None:
            params["tag_id"] = tag_id
        return await self._fetch("/markets", params)

    async def fetch_market(self, market_id: str) -> CollectionResult:
        """Fetch one market by numeric id."""
        return await self._fetch(f"/markets/{market_id}", {})

    async def fetch_market_by_slug(self, slug: str) -> CollectionResult:
        """Fetch one market by slug."""
        return await self._fetch(f"/markets/slug/{slug}", {})

    async def discover_events(self, limit: int = 20, closed: bool = False) -> CollectionResult:
        """List events via keyset feed."""
        return await self._fetch(
            "/events/keyset", {"closed": str(closed).lower(), "limit": limit}
        )
