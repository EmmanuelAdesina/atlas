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

from dataclasses import dataclass
from typing import Any, Optional

import httpx

DEFAULT_DAEMON_URL = "http://127.0.0.1:10086/command"

# Actions taken straight from the skill reference. Anything else is rejected
# client-side so a typo can't become an undocumented call.
ALLOWED_ACTIONS = frozenset({
    "navigate", "find_tab", "snapshot", "click", "fill", "evaluate",
    "screenshot", "network", "upload", "save_as_pdf", "list_tabs",
    "close_tab", "close_session",
})


class WebBridgeError(RuntimeError):
    """Daemon unreachable, non-envelope reply, or ok:false from the daemon."""


@dataclass
class WebBridgeAdapter:
    """Sync adapter bound to one task session (= one browser tab group)."""

    session: str
    daemon_url: str = DEFAULT_DAEMON_URL
    client: Optional[httpx.Client] = None

    def _post(self, action: str, args: dict[str, Any]) -> Any:
        if action not in ALLOWED_ACTIONS:
            raise WebBridgeError(f"refusing undocumented action {action!r}")
        body = {"action": action, "args": args, "session": self.session}
        client = self.client or httpx.Client(timeout=60.0)
        try:
            resp = client.post(self.daemon_url, json=body)
            resp.raise_for_status()
            envelope = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise WebBridgeError(f"daemon call {action} failed: {e}") from e
        if not isinstance(envelope, dict) or "ok" not in envelope:
            raise WebBridgeError(f"daemon call {action}: not an ok-envelope")
        if not envelope["ok"]:
            err = envelope.get("error", {})
            raise WebBridgeError(
                f"daemon call {action} returned "
                f"{err.get('code')}: {err.get('message')}"
            )
        return envelope["data"]

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

    def screenshot(self, format: str = "jpeg", quality: int = 60) -> Any:
        return self._post("screenshot", {"format": format, "quality": quality})

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
