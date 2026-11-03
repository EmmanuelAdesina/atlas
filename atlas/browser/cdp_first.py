"""CDPFirstAdapter — the one-transport adapter.

Operator decision (2026-11-02, week of zero placements on a sick
extension relay): the kimi daemon's evaluate/click plane is unfit — it
502'd every page call for a full day and survived an upgrade and a
restart, while raw DevTools on 9222 answered the whole time. This
subclass promotes the raw transport to FIRST choice for everything
page-local and demotes the daemon to TWO roles:

  * fallback for calls that were PROVEN not to execute (CDPPreError:
    nothing left this process toward chrome), and
  * the still-working paths it actually serves (find_tab, list_tabs,
    navigate metadata, snapshot a11y reads).

Precedence per call:

  evaluate / evaluate_safe -> CDP Runtime.evaluate, then daemon evaluate
  click                    -> CDP Input.dispatchMouseEvent (TRUSTED click,
                              same fidelity the daemon click served via
                              chrome.debugger), then daemon click
  everything else          -> inherited daemon behavior

Money discipline (unchanged walls, new socket):

  * walls live in control.py/decisions.py — they decide BEFORE any
    driver JS runs; a transport swap cannot bypass them.
  * a CDPPostError (request sent, outcome unknown) is NEVER replayed on
    the daemon: half-executed mutating JS must not get a blind second
    attempt. The outcome stays unknown; the ledger/verify layer resolves
    it against the book's own bet history — same protocol as the old
    daemon timeout semantics.
  * allow_submit/allow_execute arming in main.py is untouched.
"""

from __future__ import annotations

from typing import Any, Optional

from atlas.browser import cdp_direct
from atlas.browser.webbridge import DEFAULT_DAEMON_URL, WebBridgeAdapter, WebBridgeError
from atlas.live.capture import LIVE_BOARD_URL


class CDPFirstAdapter(WebBridgeAdapter):
    """WebBridgeAdapter whose page-local plane rides raw DevTools."""

    transport_name = "cdp-first"

    def __init__(
        self,
        session: str,
        daemon_url: str = DEFAULT_DAEMON_URL,
        # Prefix-major, match-panel first: the betslip lives on the match
        # page; a plain board tab must never hijack a placement context
        # when one exists. Markers mirror capture.MATCH_MARKERS + the
        # board list, then the host hint catches any sportybet page.
        url_prefixes: tuple[str, ...] = (
            "/sr:match:", "/football/live/", LIVE_BOARD_URL,
        ),
        create_url: str = LIVE_BOARD_URL,
        host_hint: str = "sportybet.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(session=session, daemon_url=daemon_url, **kwargs)
        self.cdp_url_prefixes = tuple(url_prefixes)
        self.cdp_create_url = create_url
        self.cdp_host_hint = host_hint

    # --- the prefixes passed to the direct transport ---------------------

    def _direct_kwargs(self) -> dict:
        return {
            "url_prefixes": self.cdp_url_prefixes,
            "create_url": self.cdp_create_url,
            "host_hint": self.cdp_host_hint,
        }

    @staticmethod
    def _failoverable(exc: WebBridgeError) -> bool:
        """Only a PROVEN no-op may be retried on the daemon channel."""
        return isinstance(exc, cdp_direct.CDPPreError)

    # --- page-local plane -------------------------------------------------

    def evaluate(self, code: str) -> Any:
        try:
            return cdp_direct.evaluate(code, **self._direct_kwargs())
        except WebBridgeError as exc:
            if self._failoverable(exc):
                return super().evaluate(code)
            raise

    def evaluate_safe(self, code: str) -> Any:
        # Reads rotate across answering targets on proven stalls: a
        # prefix-preferred tab with a wedged renderer must not capture
        # the whole scan plane while another tab answers. Mutating
        # evaluate() above never rotates (wrong-tab betslip risk).
        try:
            primary = cdp_direct.find_page(
                self.cdp_url_prefixes, host_hint=self.cdp_host_hint)
            if primary is None:
                raise cdp_direct.CDPPreError("cdp evaluate_safe: no matching target")
            extras = self._extra_targets(primary)
        except WebBridgeError as pre:
            if pre.transient and not isinstance(pre, cdp_direct.CDPPostError):
                return super().evaluate_safe(code)
            raise
        try:
            return cdp_direct.evaluate_read(
                code, **self._direct_kwargs(), extras=extras, primary=primary)
        except WebBridgeError as exc:
            if self._failoverable(exc):
                return super().evaluate_safe(code)
            raise

    def _extra_targets(self, primary: dict) -> list[dict]:
        try:
            pages = cdp_direct.list_pages()
        except WebBridgeError:
            return []
        out = []
        for t in pages:
            url = t.get("url") or ""
            if t.get("type") != "page" or t.get("id") == primary.get("id"):
                continue
            if any(p in url for p in self.cdp_url_prefixes) or (
                self.cdp_host_hint and self.cdp_host_hint in url
            ):
                out.append(t)
        return out

    def click(self, selector: str) -> Any:
        try:
            return cdp_direct.dispatch_click(selector, **self._direct_kwargs())
        except WebBridgeError as exc:
            if self._failoverable(exc):
                return super().click(selector)
            raise

    def navigate(self, url: str, new_tab: bool = False,
                 group_title: Optional[str] = None) -> Any:
        """Navigate over CDP: Page.navigate on the reserved target, or a
        URL-carrying target creation when none matches (probe-proven to
        land).

        Page.navigate answers fast by design; ARRIVAL is proven by the
        caller's readiness checks (capture.load_board verifies
        location.href) — the daemon navigate this replaces timed out on
        the same waits anyway. group_title is a daemon-only concept and
        is accepted for call-compatibility, stored nowhere.
        """
        if new_tab or group_title:
            try:
                page = cdp_direct.open_page(url)
                ws = cdp_direct.connect(page)
                try:
                    return cdp_direct.call(
                        ws, "Page.navigate", {"url": url}, timeout=20.0)
                finally:
                    try:
                        ws.close()
                    except Exception:
                        pass
            except WebBridgeError as exc:
                if self._failoverable(exc):
                    return super().navigate(
                        url, new_tab=new_tab, group_title=group_title)
                raise
        try:
            page, _created = cdp_direct.ensure_page(
                self.cdp_url_prefixes, create_url=url,
                host_hint=self.cdp_host_hint)
            ws = cdp_direct.connect(page)
            try:
                return cdp_direct.call(
                    ws, "Page.navigate", {"url": url}, timeout=20.0)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except WebBridgeError as exc:
            if self._failoverable(exc):
                return super().navigate(url)
            raise

    def cdp(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Served locally for the one method the recovery paths need.
        Everything else fails FAST (transient-unknown, nothing mailed to
        any page) instead of daemon-502-stalling the caller."""
        if method == "Page.navigate":
            url = (params or {}).get("url")
            if url:
                return self.navigate(url)
        raise WebBridgeError(
            f"cdp-first transport does not serve {method!r} — "
            "nothing was executed",
            transient=True,
        )
