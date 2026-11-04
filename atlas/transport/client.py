"""TransportClient: synchronous httpx with a pre/unknown/accepted discipline.

Retry policy (the whole point of this class):

  * GETs are idempotent — at most 2 retries, and only when the failure is
    provably pre-send (DNS/connect/TLS) or a 5xx delivered before acceptance.
  * POSTs are NEVER retried. One attempt; the caller reconciles.
  * Read timeouts after send raise OutcomeUnknownError, never RejectedError
    and never a silent retry.
  * 401/403 (or auth-shaped bizCode surfaced by sources) raise
    SessionExpiredError — the runner must fail the cycle closed.
  * Responses are delivered as raw bytes + status. Parsing, schema validation
    and archiving happen in sources — this client has no opinions on content.

No connection pooling surprises: explicit limits, explicit timeouts, one
client per process, closed on shutdown via ``close()``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from atlas.transport.errors import (
    OutcomeUnknownError,
    PreSendError,
    RejectedError,
    SessionExpiredError,
)

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
DEFAULT_LIMITS = httpx.Limits(max_connections=4, max_keepalive_connections=2)

HOST = "https://www.sportybet.com"


@dataclass
class RawResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes

    def json_text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class TransportClient:
    def __init__(self, session, timeout: httpx.Timeout = DEFAULT_TIMEOUT,
                 limits: httpx.Limits = DEFAULT_LIMITS,
                 transport: Optional[httpx.BaseTransport] = None):
        # `session` is atlas.transport.session.Session — duck-typed to keep
        # this module import-light.
        self._session = session
        self._client = httpx.Client(
            base_url=HOST,
            timeout=timeout,
            limits=limits,
            follow_redirects=False,  # a redirect on an API call is a signal, not a path
            transport=transport,
            http2=False,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TransportClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def get(self, path: str, *, params: Optional[dict] = None,
            referer: Optional[str] = None, retries: int = 2) -> RawResponse:
        return self._request(
            "GET", path, params=params, referer=referer, max_attempts=1 + retries,
        )

    def post(self, path: str, *, body: Optional[bytes | str | dict] = None,
             referer: Optional[str] = None) -> RawResponse:
        # POSTs: exactly one attempt, ever.
        return self._request("POST", path, body=body, referer=referer, max_attempts=1)

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, *, params=None, body=None,
                 referer=None, max_attempts: int) -> RawResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._client.request(
                    method,
                    path,
                    params=params,
                    content=self._encode_body(body),
                    headers=self._session.request_headers(referer),
                )
            except (httpx.ConnectError, httpx.ConnectTimeout,
                    httpx.WriteError, httpx.WriteTimeout) as exc:
                # Nothing reached the server (or the write never completed).
                if attempt < max_attempts:
                    self._backoff(attempt)
                    continue
                raise PreSendError(f"{method} {path}: {type(exc).__name__}") from exc
            except (httpx.ReadTimeout, httpx.ReadError, httpx.PoolTimeout,
                    httpx.RemoteProtocolError) as exc:
                # Request went out; we cannot know the outcome.
                raise OutcomeUnknownError(
                    f"{method} {path}: {type(exc).__name__} after send"
                ) from exc
            except httpx.UnsupportedProtocol as exc:
                raise PreSendError(f"{method} {path}: {exc}") from exc

            if resp.status_code in (401, 403):
                raise SessionExpiredError(f"{method} {path}: HTTP {resp.status_code}")
            if resp.status_code == 429 or resp.status_code >= 500:
                if method == "GET" and attempt < max_attempts:
                    self._backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                    continue
                if method == "GET":
                    raise RejectedError(
                        f"{method} {path}: HTTP {resp.status_code} after retries"
                    )
                # POST: 5xx after send is ambiguous, not rejection.
                raise OutcomeUnknownError(f"{method} {path}: HTTP {resp.status_code}")
            if 400 <= resp.status_code < 500:
                raise RejectedError(f"{method} {path}: HTTP {resp.status_code}")
            return RawResponse(
                status_code=resp.status_code,
                headers={k.lower(): v for k, v in resp.headers.items()},
                body=resp.content,
            )

    @staticmethod
    def _encode_body(body):
        if body is None:
            return b""
        if isinstance(body, (bytes, str)):
            return body
        # dict → urlencoded, matching the captured content-type profile
        from urllib.parse import urlencode
        return urlencode(body).encode("utf-8")

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str] = None) -> None:
        try:
            if retry_after:
                delay = min(float(retry_after), 5.0)
            else:
                delay = min(0.5 * (2 ** (attempt - 1)), 4.0)
        except ValueError:
            delay = 1.0
        time.sleep(delay * (0.5 + random.random()))
