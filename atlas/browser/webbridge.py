"""Minimal WebBridge browser adapter (read-path first).

Talks to the Kimi WebBridge local daemon over HTTP using only the actions
documented in the kimi-webbridge skill: navigate, find_tab, snapshot,
click, fill, evaluate, screenshot, network, list_tabs, close_tab,
close_session. No invented endpoints, no coordinate-based automation:
element interaction uses snapshot @e refs / CSS selectors.

Bet placement is deliberately NOT exposed here. This adapter only reads
and navigates market sections. Execution (if ever enabled) lives behind
a separate, explicitly authorized path with pre-submission verification.

Separation: this file knows about browser mechanics only. Probability
estimation and strategy live in atlas.live.* and never import this module.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

# The daemon is normally local (127.0.0.1:10086). A split-plane deployment
# (brain on an always-on box, eyes in a browser elsewhere — e.g. over a
# Tailscale mesh) points ATLAS_WEBBRIDGE_DAEMON_URL at the remote daemon;
# wide-target like the split-brain deployment.
DAEMON_URL_ENV = "ATLAS_WEBBRIDGE_DAEMON_URL"
DEFAULT_DAEMON_URL = os.environ.get(
    DAEMON_URL_ENV, "http://127.0.0.1:10086/command"
)

# Transient daemon answers. The extension wedges on the heavy live pages and
# the daemon then answers 502/503/504; httpx.TransportError appears while the
# daemon itself is restarting. A single 502 used to abort a whole cycle and
# was reported as "the board is empty" — indistinguishable from a genuinely
# empty board (2026-10-29). Retry these, bounded.
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

# How long one daemon call may take. The live board is a heavy SPA: a cold
# navigation measures ~25s and a market read ~1s warm, so the old 20s client
# timeout fired on healthy work. A timeout does NOT cancel the command — the
# extension still executes it — so each timed-out call that is retried adds
# another queued command behind the first. On 2026-10-29 that feedback loop
# grew the extension's queue into minutes of latency and no cycle could
# finish. One generous timeout, no retry on timeout.
DEFAULT_TIMEOUT_S = 45.0

# NEVER retried: a request that timed out may still have been executed, and a
# second click can place a second bet or toggle a control twice. Money moves
# through click(), so ambiguity is resolved by the caller, never here.
NON_IDEMPOTENT_ACTIONS = frozenset({
    "click", "fill", "upload", "save_as_pdf", "close_tab", "close_session",
})

# Actions taken straight from the skill reference. Anything else is rejected
# client-side so a typo can't become an undocumented call.
ALLOWED_ACTIONS = frozenset({
    "navigate", "find_tab", "snapshot", "click", "fill", "evaluate",
    "screenshot", "network", "upload", "save_as_pdf", "list_tabs",
    "close_tab", "close_session", "cdp",
})

# Read paths get a SHORTER ceiling than the 45s default: they are the calls
# a WEDGED page answers never, and each one waiting out its full 45s (+ the
# CDP fallback's own 45s) is how a cycle silently stretched past five
# minutes (2026-10-31, Tottenham panel). Warm reads measure ~1s, so 15s
# still tolerates a heavy page while failing 3× faster. Money-path actions
# (click/fill/upload) and cold navigations keep the generous default.
READ_ACTIONS = frozenset({"evaluate", "snapshot", "list_tabs", "network"})
READ_TIMEOUT_S = 15.0


class WebBridgeError(RuntimeError):
    """Daemon unreachable, non-envelope reply, or ok:false from the daemon.

    `transient` marks failures that say nothing about the page: the daemon
    was restarting, the extension was mid-recovery, or a proxy answered
    502. Callers may retry those without concluding the session is dead.
    """

    def __init__(self, message: str, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


@dataclass
class WebBridgeAdapter:
    """Sync adapter bound to one task session (= one browser tab group)."""

    session: str
    daemon_url: str = DEFAULT_DAEMON_URL
    client: Optional[httpx.Client] = None
    # Bounded retry for transient daemon failures. Tests set this to 0.
    attempts: int = 3
    retry_backoff_s: float = 0.5
    # One daemon call's ceiling. See DEFAULT_TIMEOUT_S for why it is large.
    timeout_s: float = DEFAULT_TIMEOUT_S

    def _http(self) -> httpx.Client:
        """The shared client. One per adapter, not one per call.

        A fresh `httpx.Client` per request leaked a connection per call and
        left the daemon holding sockets from abandoned clients.
        """
        if self.client is None:
            self.client = httpx.Client(timeout=self.timeout_s)
        return self.client

    def _post(
        self,
        action: str,
        args: dict[str, Any],
        retry_safe: Optional[bool] = None,
    ) -> Any:
        if action not in ALLOWED_ACTIONS:
            raise WebBridgeError(f"refusing undocumented action {action!r}")
        if retry_safe is None:
            retry_safe = action not in NON_IDEMPOTENT_ACTIONS
        attempts = max(1, self.attempts) if retry_safe else 1
        body = {"action": action, "args": args, "session": self.session}
        client = self._http()
        # Reads fail fast (see READ_TIMEOUT_S); everything else keeps the
        # generous default. Override per request, not per client: the same
        # client serves both in one cycle.
        call_timeout = READ_TIMEOUT_S if action in READ_ACTIONS else self.timeout_s
        last: WebBridgeError = WebBridgeError(
            f"daemon call {action} failed without an attempt"
        )
        for attempt in range(1, attempts + 1):
            try:
                resp = client.post(self.daemon_url, json=body, timeout=call_timeout)
                if resp.status_code in TRANSIENT_STATUS:
                    last = WebBridgeError(
                        f"daemon call {action} returned HTTP "
                        f"{resp.status_code} (transient)",
                        transient=True,
                    )
                else:
                    resp.raise_for_status()
                    envelope = resp.json()
                    if not isinstance(envelope, dict) or "ok" not in envelope:
                        raise WebBridgeError(
                            f"daemon call {action}: not an ok-envelope"
                        )
                    if not envelope["ok"]:
                        err = envelope.get("error", {})
                        raise WebBridgeError(
                            f"daemon call {action} returned "
                            f"{err.get('code')}: {err.get('message')}"
                        )
                    return envelope["data"]
            except WebBridgeError as e:
                if not e.transient:
                    raise
                last = e
            except httpx.TimeoutException as e:
                # The command may still run in the extension; retrying it
                # would queue a duplicate and deepen the backlog that caused
                # the timeout. Fail this call and let the cycle retry later.
                raise WebBridgeError(
                    f"daemon call {action} timed out after "
                    f"{self.timeout_s:.0f}s: {type(e).__name__}"
                ) from e
            except (httpx.TransportError, ValueError) as e:
                # ValueError: a proxy/daemon error page instead of JSON.
                last = WebBridgeError(
                    f"daemon call {action} failed: {e}", transient=True
                )
            if attempt < attempts:
                time.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
        raise last

    # --- Read / navigate primitives ---

    def list_tabs(self) -> Any:
        return self._post("list_tabs", {})

    def navigate(self, url: str, new_tab: bool = False,
                 group_title: Optional[str] = None) -> Any:
        args: dict[str, Any] = {"url": url}
        if new_tab:
            args["newTab"] = True
        if group_title:
            args["group_title"] = group_title
        return self._post("navigate", args)

    def find_tab(self, url: str, active: bool = False) -> Any:
        args: dict[str, Any] = {"url": url}
        if active:
            args["active"] = True
        return self._post("find_tab", args)

    def snapshot(self) -> Any:
        """Accessibility tree of the current tab (text + @e refs)."""
        return self._post("snapshot", {})

    def click(self, selector: str) -> Any:
        """Click a tab/section control. Never used on odds buttons by policy."""
        return self._post("click", {"selector": selector})

    def evaluate(self, code: str) -> Any:
        """Run JS in the page; callers must keep payloads compact JSON."""
        return self._post("evaluate", {"code": code})

    def evaluate_safe(self, code: str) -> Any:
        """Evaluate for READ-ONLY scans, with a CDP fallback.

        On the heavy live list the renderer can die to
        chrome-error://chromewebdata/ or stall mid-navigation; the
        extension's evaluate then times out while CDP (handled by the
        browser process) still answers. The capture path uses this; the
        betslip's mutating JS uses evaluate() and is never replayed through
        a second channel.
        """
        try:
            return self._post("evaluate", {"code": code}, retry_safe=False)
        except WebBridgeError as e:
            if not e.transient:
                raise
            return self._cdp_evaluate(code, cause=e)

    def _cdp_evaluate(self, code: str, cause: Optional[Exception] = None) -> Any:
        try:
            data = self._post(
                "cdp",
                {
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": code,
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                },
                retry_safe=False,
            )
        except WebBridgeError as e:
            raise WebBridgeError(
                f"evaluate and CDP fallback both failed: {e}; "
                f"first failure: {cause}",
                transient=True,
            ) from e
        result = data.get("result", {}) if isinstance(data, dict) else {}
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"]
            raise WebBridgeError(
                "CDP evaluate threw: "
                f"{detail.get('text', '')} {detail.get('exception', {})}"
            )
        inner = result.get("result", {})
        return {"type": inner.get("type"), "value": inner.get("value")}

    def screenshot(self, format: str = "jpeg", quality: int = 60) -> Any:
        return self._post("screenshot", {"format": format, "quality": quality})

    def cdp(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Raw CDP passthrough — the recovery path for a wedged page.

        CDP calls are handled by the browser process, so Page.navigate still
        works when the page's own JS thread is stuck and evaluate() times
        out. Never retried: a CDP method can do anything.
        """
        args: dict[str, Any] = {"method": method}
        if params:
            args["params"] = params
        return self._post("cdp", args, retry_safe=False)

    # --- Network capture (per tab, start before reading, detail before stop) ---

    def network(self, cmd: str, filter: Optional[str] = None,
                request_id: Optional[str] = None) -> Any:
        args: dict[str, Any] = {"cmd": cmd}
        if filter is not None:
            args["filter"] = filter
        if request_id is not None:
            args["requestId"] = request_id
        return self._post("network", args)

    def close_session(self) -> Any:
        """Close the whole tab group. Only on explicit user request."""
        return self._post("close_session", {})
