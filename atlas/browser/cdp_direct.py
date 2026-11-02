"""Direct DevTools transport — no daemon, no extension, no middleman.

Everything page-local the live loop does (evaluate, clicks, tab
find/create) is served straight from chrome's DevTools endpoint.

  GET  /json/list            enumerate page targets
  PUT  /json/new?url=...     create a target AT the url (probe-proven on
                             this chrome: the URL is carried; an
                             about:blank landing is detected and a
                             Page.navigate fallback runs over the socket)
  GET  /json/close/<id>      close a target
  WS   Runtime.evaluate / Page.navigate / Input.dispatchMouseEvent

Why this exists: the kimi daemon's call plane (chrome.debugger relayed
through the extension) answered 502 for every evaluate for a full day
(2026-11-02, survived a v2.0.30->v2.0.32 upgrade and a restart) while
the SAME page answered Runtime.evaluate on 9222 the whole time (settle
scrape 16:02 UTC count 264; board innerText 0->436->942). One
transport, first principles, inspectable code.

Thread model: every call takes a FRESH websocket — no shared socket
state, so the capture main thread and the placement worker thread can
use the transport concurrently without locks. ~30-80ms extra per call,
vastly more robust.

Error discipline (money-critical, mirrors the daemon timeout semantics):

  CDPPreError   nothing was sent — transport down before execution.
                Safe to fail over to another channel.
  CDPPostError  the request was sent and the outcome is unknown or the
                socket died mid-call. NEVER replayed on a second
                channel: a mutating script that half-ran must not be
                blind-retried. Fail-closed; callers record unknown.
  WebBridgeError(non-transient)
                the page itself raised (exceptionDetails) — a
                deterministic answer, not a transport fault.

Envelope parity: evaluate() returns the same {'type','value'} dict
`WebBridgeAdapter._cdp_evaluate` returns, so `_unwrap` and every JS
caller keep working unchanged.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from collections import deque
from typing import Any, Optional

from atlas.browser.webbridge import WebBridgeError

CDP_DEBUG_HOST = "http://127.0.0.1:9222"

# Reads must fit inside the capture round's budget (read actions at the
# daemon keep 15s; the fallback chain shares it). Override via env.
DEFAULT_EVAL_TIMEOUT_S: float = 12.0
_HTTP_TIMEOUT_S: float = 5.0
_OPEN_TIMEOUT_S: float = 8.0
_WS_CONNECT_TIMEOUT_S: float = 10.0

# A wedged chrome must not collect probe tabs: creating a target is
# capped. Beyond the cap the call fails transient and the cycle retries
# later instead of stacking about:blank SPA targets on a drowning box.
MAX_CREATES_PER_MINUTE = 3


class CDPPreError(WebBridgeError):
    """Nothing was sent — failing over to another channel is safe."""

    def __init__(self, message: str) -> None:
        super().__init__(message, transient=True)


class CDPPostError(WebBridgeError):
    """Request sent, outcome unknown (or socket died mid-call).

    Never fail over to a second channel after this: a mutating script
    that actually executed would be replayed. The caller records the
    unknown outcome and the verify layer (book truth) resolves it.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, transient=True)


_created_at: deque[float] = deque()


def eval_timeout() -> float:
    import os

    raw = os.environ.get("ATLAS_CDP_EVAL_TIMEOUT", "")
    try:
        return max(2.0, float(raw)) if raw else DEFAULT_EVAL_TIMEOUT_S
    except ValueError:
        return DEFAULT_EVAL_TIMEOUT_S


def debug_host() -> str:
    import os

    return os.environ.get("ATLAS_CDP_HOST", CDP_DEBUG_HOST).rstrip("/")


def _http_json(url: str, method: str = "GET", timeout: float = _HTTP_TIMEOUT_S) -> dict:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    req = Request(url, method=method)  # noqa: S310 — local debug endpoint only
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
    except (HTTPError, URLError, OSError, ValueError) as exc:
        # 2026-11-02: Chrome 156 launched WITHOUT --remote-debugging-port
        # still holds the 9222 socket but 404s every /json endpoint. That
        # escaped here as a raw HTTPError, which is not failoverable, so
        # CDPFirstAdapter.evaluate re-raised it and EVERY page-local read
        # in the loop died as CaptureError -> rows "0" — the loop then
        # reported "on the live board (0 rows)" against a board it could
        # never actually read, and never reached the healthy daemon
        # failback. Nothing was sent to any page, so this is exactly the
        # definition of a failover-safe pre-error.
        raise CDPPreError(
            f"cdp: debug endpoint unreachable: {type(exc).__name__}: {str(exc)[:120]}"
        ) from exc


def _ws_create(url: str, timeout: float):
    """Seam for tests; the real path needs websocket-client in the venv."""
    import websocket  # VPS venv: pip install --no-deps websocket-client

    return websocket.create_connection(url, timeout=timeout)


def _timeout_excs() -> tuple[type[BaseException], ...]:
    """Exception classes that mean 'peer busy, keep draining'.

    Mirrors `_CDPTab.call` in bet_history_scrape: no DATA inside one recv
    window is not a dead socket; the wall-clock deadline rules the wait
    out. Resolved lazily so tests can run without websocket-client.
    """
    try:
        import websocket

        return (websocket.WebSocketTimeoutException,)
    except ImportError:
        return ()


def list_pages(host: Optional[str] = None) -> list[dict]:
    """All DevTools targets (callers filter on type=='page')."""
    return _http_json((host or debug_host()) + "/json/list") or []


def _is_page(t: dict) -> bool:
    return t.get("type") == "page"


def find_page(prefixes: tuple[str, ...], host_hint: str = "", host: Optional[str] = None) -> Optional[dict]:
    """The page target to ride, PREFIX-major so a betslip context wins
    over a stray board tab when both exist:

      tier 1: any caller prefix (match-panel markers first, then the
              board list url) checked against every page — in order
      tier 2: the host hint (any sportybet page)
    """
    pages = [t for t in list_pages(host) if _is_page(t)]
    for tier in (tuple(prefixes), (host_hint,)):
        for pref in tier:
            if not pref:
                continue
            for t in pages:
                if pref in (t.get("url") or ""):
                    return t
    return None


def _create_budget_ok() -> bool:
    now = time.monotonic()
    while _created_at and now - _created_at[0] > 60.0:
        _created_at.popleft()
    return len(_created_at) < MAX_CREATES_PER_MINUTE


def open_page(url: str, host: Optional[str] = None) -> dict:
    """Create a target AT `url`, verifying the landing.

    Chrome variants disagree on whether `PUT /json/new?url=` carries the
    URL (this box's build does — probe-proven; docs say Chrome ignores it
    since ~156 and starts about:blank). Both are handled: an about:blank
    landing gets a deadline-protected Page.navigate over the target's own
    socket; the CALLER's readiness polling proves the final arrival.
    """
    if not _create_budget_ok():
        raise CDPPreError(
            "cdp: target-creation budget exhausted "
            f"(max {MAX_CREATES_PER_MINUTE}/min)"
        )
    quoted = urllib.parse.quote(url, safe="/:?&=%.[]~")
    try:
        target = _http_json(
            (host or debug_host()) + "/json/new?" + quoted, method="PUT",
            timeout=_OPEN_TIMEOUT_S,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CDPPreError(f"cdp: /json/new unreachable: {str(exc)[:120]}") from exc
    tid = target.get("id") or ""
    if not tid:
        raise CDPPreError("cdp: /json/new returned no target id")
    _created_at.append(time.monotonic())
    landed = (target.get("url") or "")[:2048]
    if landed in ("", "about:blank"):
        try:
            ws = _ws_create(target["webSocketDebuggerUrl"], _WS_CONNECT_TIMEOUT_S)
        except Exception as exc:
            raise CDPPreError(
                f"cdp: target landed about:blank and socket refused: {str(exc)[:120]}"
            ) from exc
        try:
            call(ws, "Page.navigate", {"url": url}, timeout=15.0)
        finally:
            try:
                ws.close()
            except Exception:
                pass
    return target


def ensure_page(
    prefixes: tuple[str, ...], create_url: Optional[str] = None,
    host_hint: str = "", host: Optional[str] = None,
) -> tuple[dict, bool]:
    """find-or-create. Returns (target, created). Called by evaluate().

    A missing page is NORMAL here (chrome recycles tabs on its wedges);
    the create is the self-heal the daemon path never had.
    """
    page = find_page(prefixes, host_hint=host_hint, host=host)
    if page is not None:
        return page, False
    if not create_url:
        raise CDPPreError(
            "cdp: no page target matches "
            f"{[p[:60] for p in prefixes if p]} and no create_url given"
        )
    return open_page(create_url, host=host), True


def connect(target: dict) -> Any:
    """One fresh socket for one call (thread-safe by construction)."""
    try:
        return _ws_create(target["webSocketDebuggerUrl"], _WS_CONNECT_TIMEOUT_S)
    except Exception as exc:  # chrome down / port closed / refused
        raise CDPPreError(
            f"cdp: cannot connect {str(target.get('url') or '')[:60]}: "
            f"{str(exc)[:120]}"
        ) from exc


def call(ws: Any, method: str, params: Optional[dict] = None, timeout: float = 15.0) -> dict:
    """One CDP round-trip, deadline-protected. Semantics copied from the
    production-proven `_CDPTab.call` (bet_history_scrape): recv-window
    timeouts drain toward the wall deadline; every socket-layer failure
    after the send is CDPPostError, never a raw crash.
    """
    deadline = time.monotonic() + timeout
    drain_excs = _timeout_excs()
    try:
        ws.send(json.dumps({
            "id": 1, "method": method, "params": params or {},
        }))
    except Exception as exc:
        # Nothing reached the browser: pre-execution by definition.
        raise CDPPreError(
            f"cdp {method}: send failed ({str(exc)[:120]})"
        ) from exc
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CDPPostError(
                f"cdp {method}: stalled >{timeout:.0f}s (no matching response)"
            )
        try:
            ws.settimeout(min(15.0, max(1.0, remaining)))
        except Exception as exc:
            raise CDPPostError(f"cdp {method}: socket state lost ({str(exc)[:90]})") from exc
        try:
            msg = json.loads(ws.recv())
        except drain_excs:
            continue  # busy peer; the deadline rules the wait out
        except Exception as exc:
            raise CDPPostError(
                f"cdp {method}: transport gone mid-call ({str(exc)[:120]})"
            ) from exc
        if not isinstance(msg, dict):
            continue
        if msg.get("id") != 1:
            continue  # events / other traffic: keep draining
        if "error" in msg:
            raise CDPPostError(f"cdp {method}: {msg['error']}")
        return msg.get("result") or {}


def evaluate_on(target: dict, code: str, timeout_s: Optional[float] = None) -> dict:
    """Runtime.evaluate on one target -> the {'type','value'} envelope."""
    ws = connect(target)
    try:
        res = call(
            ws, "Runtime.evaluate",
            {
                "expression": code,
                "returnByValue": True,
                "awaitPromise": True,
                "userGesture": True,
            },
            timeout=timeout_s or eval_timeout(),
        )
    finally:
        try:
            ws.close()
        except Exception:
            pass
    if res.get("exceptionDetails"):
        detail = res["exceptionDetails"]
        raise WebBridgeError(
            "CDP evaluate threw: "
            f"{detail.get('text', '')} {detail.get('exception', {})}"
        )
    inner = res.get("result") or {}
    return {"type": inner.get("type"), "value": inner.get("value")}


def evaluate(
    code: str,
    url_prefixes: tuple[str, ...] = (),
    create_url: Optional[str] = None,
    host_hint: str = "",
    timeout_s: Optional[float] = None,
) -> dict:
    """find-or-create a page target, then Runtime.evaluate on it."""
    page, _created = ensure_page(url_prefixes, create_url=create_url, host_hint=host_hint)
    return evaluate_on(page, code, timeout_s=timeout_s)


def dispatch_click(
    selector: str,
    url_prefixes: tuple[str, ...],
    create_url: Optional[str] = None,
    host_hint: str = "",
    timeout_s: Optional[float] = None,
) -> dict:
    """One TRUSTED click on the marked control, same fidelity the daemon
    click served: chrome.debugger == this Input domain.

    Read the control's viewport rect via Runtime.evaluate (scroll it into
    view first), then Input.dispatchMouseEvent press/release at its
    center. A rect of None is pre-execution (nothing was pressed) and
    raises CDPPreError, so the daemon fallback and the betslip's retry
    paths both stay available.
    """
    page, _created = ensure_page(url_prefixes, create_url=create_url, host_hint=host_hint)
    ws = connect(page)
    try:
        probe = call(
            ws, "Runtime.evaluate",
            {
                "returnByValue": True,
                "expression": (
                    "(() => {"
                    f"const el = document.querySelector({json.dumps(selector)});"
                    "if (!el) return 'ATLAS-NX';"
                    "el.scrollIntoView({block:'center', inline:'center'});"
                    "const r = el.getBoundingClientRect();"
                    "return JSON.stringify({"
                    "x: r.x + r.width/2, y: r.y + r.height/2,"
                    "w: r.width, h: r.height});"
                    "})()"
                ),
            },
            timeout=timeout_s or eval_timeout(),
        )
        value = (probe.get("result") or {}).get("value")
        if value == "ATLAS-NX":
            raise CDPPreError(f"cdp click: no element for selector {selector[:80]}")
        rect = json.loads(value) if isinstance(value, str) else None
        if not rect or not rect.get("w"):
            raise CDPPreError(
                f"cdp click: selector present but zero-size rect ({selector[:80]})"
            )
        x, y = float(rect["x"]), float(rect["y"])
        event = {"x": x, "y": y, "button": "left", "clickCount": 1, "pointerType": "mouse"}
        for state in ("mousePressed", "mouseReleased"):
            call(ws, "Input.dispatchMouseEvent", {**event, "type": state}, timeout=10.0)
            time.sleep(0.03)
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return {"clicked": True, "selector": selector[:120]}
