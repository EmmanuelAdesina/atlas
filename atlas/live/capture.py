"""Live capture: read the SportyBet live board through the browser.

Everything here was learned by driving the real board, and the workarounds
are not optional — they are the only way the page behaves:

  - The live board NEVER fires 'load' (it holds a websocket open), so
    `navigate()` to it always times out and then unregisters the tab,
    which makes every later daemon call fail. The board is therefore
    reached by loading a light page and navigating client-side.
  - The board renders progressively. A snapshot taken too early carries the
    header but not the score, the clock or the market table, so every
    capture polls against a readiness gate.
  - The accessibility 'snapshot' action intermittently fails on the heavy
    live page, so it is retried and then re-read with a lighter path.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from atlas.live.extract import (
    extract_live_match_state,
    extract_markets,
    iter_texts,
    panel_is_ready,
)
from atlas.live.observe import LiveMatchState

LIVE_BOARD_URL = "https://www.sportybet.com/ng/sport/football/live_list/"
LIGHT_URL = "https://sportybet.com/ng/"


@dataclass
class CaptureResult:
    """One capture attempt."""

    ready: bool
    reason: str
    markets: list[dict] = field(default_factory=list)
    state: Optional[LiveMatchState] = None
    observed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    match_id: str = "live"


@dataclass
class BoardCandidate:
    """A live match on the board, for the Boss to choose between."""

    match_id: str
    home_team: str
    away_team: str
    score: str
    minute: Optional[float]
    period: str
    num_markets: int
    index: int
    # False for a scheduled/pre-match row: the list also carries upcoming
    # kick-offs, and opening one trades live pricing for a pre-match panel.
    live: bool = True


class CaptureError(RuntimeError):
    """The board could not be read at all."""


class LiveCapture:
    """Reads the live board through a WebBridge adapter."""

    def __init__(
        self,
        adapter: Any,
        url: str = LIVE_BOARD_URL,
        poll_s: float = 3.0,
        max_wait_s: float = 90.0,
        min_markets: int = 10,
        match_id: str = "live",
        pause_scale: float = 1.0,
    ) -> None:
        self.adapter = adapter
        self.url = url
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self.min_markets = min_markets
        self.match_id = match_id
        # Scales the waits in the recovery path so tests do not pay for them.
        self.pause_scale = max(0.0, pause_scale)
        self._loaded = False
        # Where the tab actually is (last successful URL read) and the last
        # browser-level failure. The heartbeat publishes both: "the board is
        # empty" is unactionable when the real fault is a daemon 502 or a tab
        # parked on another page.
        self.last_href: str = ""
        self.last_transport_error: str = ""
        # Serialises PAGE MUTATIONS (navigate / reload / tab / panel clicks)
        # against the placement worker's betslip flow. LiveRunner replaces
        # this with its own re-entrant lock so the whole process agrees on
        # one page owner. READ paths deliberately do not take it: the cycle
        # must keep pricing while a placement is in flight.
        self.page_lock = threading.RLock()
        # Consecutive failed probes: after a few, the session is treated as
        # dead rather than flaky, so the loop fails fast instead of waiting
        # out the daemon's timeout on every call.
        self._consecutive_failures = 0
        # True when the last session check could not read the page at all.
        # 'expired: False' then means UNKNOWN, not 'logged in'; the heartbeat
        # publishes this so an unattached extension cannot hide behind a
        # healthy-looking session (2026-10-30).
        self.session_probe_failed = False

    # --- browser primitives ---------------------------------------------

    @staticmethod
    def _unwrap(raw: Any) -> Any:
        """Accept either the daemon envelope or the already-unwrapped data.

        `WebBridgeAdapter._post` returns `envelope['data']`, but a test
        double may hand back the whole envelope, so both are accepted.
        """
        if isinstance(raw, dict):
            if "tree" in raw or "url" in raw or "value" in raw:
                return raw
            if "data" in raw:
                return raw["data"]
        return raw

    def _evaluate(self, code: str, retries: int = 2) -> Any:
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                # Read-only scans use the CDP-capable path when the adapter
                # offers it: a dead renderer still answers through CDP.
                evaluate = (
                    getattr(self.adapter, "evaluate_safe", None)
                    or self.adapter.evaluate
                )
                data = self._unwrap(evaluate(code))
                self._consecutive_failures = 0
                self.last_transport_error = ""
                if isinstance(data, dict) and "value" in data:
                    return data["value"]
                return data
            except Exception as e:  # the extension intermittently fails
                last = e
                transient = bool(getattr(e, "transient", False))
                if transient or "daemon call" in str(e):
                    self.last_transport_error = str(e)
                if transient:
                    # The daemon said it is wedged/restarting (5xx or a
                    # transport error) after its own bounded retries. That is
                    # a BROWSER fault, not evidence that the page is empty,
                    # and another full round only burns the cycle's budget.
                    raise CaptureError(
                        "browser automation unreachable (the local daemon "
                        f"did not answer): {e}"
                    ) from e
                self._consecutive_failures += 1
                if self._consecutive_failures >= 2:
                    # The session is dead, not flaky. Every further probe
                    # would wait out the daemon's own timeout, so fail fast
                    # and let the supervisor recover instead.
                    #
                    # The usual cause is that the extension only drives a
                    # FOREGROUND tab: if the operator has moved to another
                    # tab or window, the commands are sent and never
                    # executed. That is worth saying, because it is the
                    # difference between "restart the browser" and "click
                    # the SportyBet tab".
                    raise CaptureError(
                        "the browser session is not responding — the tab may "
                        "need to be in the foreground (the extension only "
                        f"drives the visible tab): {e}"
                    ) from e
                time.sleep(1.5 * self.pause_scale)
        raise CaptureError(f"evaluate failed: {last}")

    def _snapshot_texts(self) -> list[str]:
        data = self._unwrap(self.adapter.snapshot())
        tree = data.get("tree") or [] if isinstance(data, dict) else []
        return iter_texts(tree)

    # The book's live URLs, as served on 2026-10-29:
    #   board: /ng/sport/football/live_list/   (Multi View)
    #   match: /ng/sport/football/live/<Country>/<Comp>/<Home>_vs_<Away>/sr:match:<id>
    # and the older /ng/sport/live/ still redirects into the first shape. The
    # gate must accept all three: matching only '/sport/live' made this
    # permanently False, so every capture() re-navigated the board to the
    # site root and threw away the match panel it had just opened (the loop
    # then read Multi View, found no market tables, and reported "the board
    # is empty" forever).
    LIVE_URL_MARKERS = ("/sport/live", "/live_list", "/sr:match:")
    BOARD_LIST_MARKERS = ("/live_list",)
    MATCH_MARKERS = ("/sr:match:", "/football/live/")

    def current_url(self) -> str:
        """The tab's REAL URL, cached for the heartbeat and the nav checks."""
        try:
            href = self._evaluate("location.href")
        except CaptureError:
            return ""
        if isinstance(href, str) and href:
            self.last_href = href.strip('"')
        return self.last_href

    def at_board_list(self) -> bool:
        return any(m in self.last_href for m in self.BOARD_LIST_MARKERS)

    def at_match(self) -> bool:
        return any(m in self.last_href for m in self.MATCH_MARKERS)

    def board_row_count(self) -> int:
        try:
            raw = self._evaluate(
                "document.querySelectorAll('.match-row').length"
            )
        except CaptureError:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def board_loaded(self) -> bool:
        """True when the tab is on the board LIST or on a match panel.

        The bare hub (/sport/live/) is deliberately NOT accepted: it matches
        the live-URL marker but hosts neither the list nor a panel, and the
        loop sat on it reporting "the board is empty" while the real list
        was one route away (2026-10-29). On the hub, rows decide.
        """
        href = self.current_url()
        if not href:
            return False
        if self.at_board_list() or self.at_match():
            return True
        if any(marker in href for marker in self.LIVE_URL_MARKERS):
            return self.board_row_count() > 0
        return False

    def _wait_until(self, predicate: Any, timeout_s: float) -> bool:
        """Poll a predicate, scaled by pause_scale so tests pay nothing."""
        deadline = time.monotonic() + timeout_s * self.pause_scale
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.05, 0.25 * self.pause_scale))

    def _on_board_list(self) -> bool:
        self.current_url()
        return self.at_board_list()

    def board_failure_reason(self, fallback: str) -> str:
        """Say WHAT failed: browser, route, or a genuinely empty feed.

        "the board is empty" was reported for a daemon 502, a tab parked on
        another route, and a real empty board alike — three different
        operator actions behind one message (2026-10-29).
        """
        where = self.last_href or "unknown page"
        if self.last_transport_error:
            return (
                "browser automation unreachable — no page could be read "
                f"({self.last_transport_error}); last known page {where}"
            )
        if not self.at_board_list() and not self.at_match():
            return (
                f"{fallback} — the tab is not on the live board list, so "
                f"navigation did not arrive (last page {where})"
            )
        return f"{fallback} (last page {where})"

    def load_board(self, force: bool = False) -> None:
        """Get the live board loaded, working around the never-firing 'load'.

        The direct navigation always times out, so the board is reached by
        loading a light page and then navigating client-side, which does not
        wait for a load event. The live URL is NEVER passed to navigate().

        In Multi View the whole board is shown; in Single View one match is.
        The board is left in Multi View because the match selection needs the
        list of live matches.

        Arrival is VERIFIED against location.href before the board is used.
        The old fixed sleeps declared success blindly: the loop could scan
        the previous document forever while reporting "board empty"
        (2026-10-29).
        """
        with self.page_lock:
            if self.board_loaded() and not force:
                self._loaded = True
                self.ensure_multi_view()
                return
            self._loaded = False
            # 1. A light page re-registers a tab the extension stopped
            #    answering on.
            try:
                self.adapter.navigate(LIGHT_URL)
            except Exception:
                pass
            time.sleep(0.5 * self.pause_scale)
            # 2. Client-side hop: no load event to wait for.
            try:
                self._evaluate(f"location.href={json.dumps(self.url)}; 1")
            except CaptureError:
                pass
            arrived = self._wait_until(self._on_board_list, 8.0)
            if not arrived:
                # A crashed renderer cannot run the client-side hop; CDP is
                # handled by the browser process and still can.
                cdp = getattr(self.adapter, "cdp", None)
                if callable(cdp):
                    try:
                        cdp("Page.navigate", {"url": LIGHT_URL})
                    except Exception:
                        pass
                    time.sleep(0.5 * self.pause_scale)
                    try:
                        self._evaluate(
                            f"location.href={json.dumps(self.url)}; 1"
                        )
                    except CaptureError:
                        pass
                    arrived = self._wait_until(self._on_board_list, 8.0)
            if not arrived:
                # The SPA router sometimes needs a reload to mount the list.
                try:
                    self._evaluate("location.reload(); 1")
                except CaptureError:
                    pass
                arrived = self._wait_until(self._on_board_list, 6.0)
            if arrived and self.board_row_count() == 0:
                # The list mounted but is still filling in: the board's own
                # data lags the route, and a reload populates it.
                try:
                    self._evaluate("location.reload(); 1")
                except CaptureError:
                    pass
                self._wait_until(lambda: self.board_row_count() > 0, 6.0)
            self._loaded = arrived
            self.ensure_multi_view()

    JS_MULTI_VIEW = r"""
    (() => {
      const text = document.body.innerText || '';
      if (!/Multi View/.test(text)) return JSON.stringify({ok: false, why: 'no multi view control'});
      const rows = document.querySelectorAll('.match-row').length;
      if (rows > 0) return JSON.stringify({ok: true, why: 'already multi', rows: rows});
      const el = Array.from(document.querySelectorAll('*')).find(e =>
        e.children.length === 0 && (e.textContent||'').trim() === 'Multi View');
      if (!el) return JSON.stringify({ok: false, why: 'no multi view control'});
      el.click();
      return JSON.stringify({ok: true, why: 'clicked'});
    })()
    """

    def ensure_multi_view(self) -> bool:
        """The match selection needs the LIST of live matches, which is the
        Multi View board. Single View shows one match at a time."""
        with self.page_lock:
            try:
                raw = self._evaluate(self.JS_MULTI_VIEW)
            except CaptureError:
                return False
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                return False
            if isinstance(data, dict) and data.get("ok"):
                time.sleep(1.5)
                return True
            return False

    # --- session health ---------------------------------------------------

    JS_SESSION = r"""
    (() => {
      const href = location.href || '';
      const text = document.body.innerText || '';
      const visible = (e) => e && e.offsetParent !== null &&
        getComputedStyle(e).display !== 'none' &&
        getComputedStyle(e).visibility !== 'hidden';
      // The login form stays in the DOM after a successful login, so only a
      // VISIBLE password field means the session has gone.
      const visiblePassword = Array.from(
        document.querySelectorAll('input[type=password]')).some(visible);
      const onLoginPage = /\/login|\/signin/i.test(href);
      const hasAccount = /My Account|Bet History/i.test(text);
      const depositPrompt = /Make a Deposit|Instant Registration/i.test(text);
      // Expired when the page asks to log in and there is no account header.
      const expired = visiblePassword || onLoginPage ||
        (depositPrompt && !hasAccount);
      return JSON.stringify({
        expired: expired,
        visiblePassword: visiblePassword,
        onLoginPage: onLoginPage,
        hasAccount: hasAccount,
        href: href
      });
    })()
    """

    def session_expired(self) -> dict:
        """Has the SportyBet session expired?

        The session is the operator's login. When it expires the page
        renders the login form, and a loop that does not notice just keeps
        planning against a board it can no longer bet on.
        """
        # An unreadable page is NOT "logged in". `expired` stays False so
        # callers that only branch on it keep working, but `probe_failed`
        # says the answer is unknown — the extension answering 502 with no
        # tab used to look exactly like a healthy session (2026-10-30).
        self.session_probe_failed = False
        try:
            raw = self._evaluate(self.JS_SESSION)
        except CaptureError as e:
            self.session_probe_failed = True
            return {"expired": False, "probe_failed": True, "error": str(e)}
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self.session_probe_failed = True
                return {
                    "expired": False,
                    "probe_failed": True,
                    "error": "session probe returned nothing",
                }
        else:
            data = raw if isinstance(raw, dict) else {}
        if not data:
            self.session_probe_failed = True
            return {
                "expired": False,
                "probe_failed": True,
                "error": "session probe returned nothing",
            }
        return data

    # --- capture ---------------------------------------------------------

    JS_HIDE_MEDIA = r"""
    (() => {
      // The 'SportyTV' media widget can expand over the scoreboard and hide
      // the score and clock. Its 'Hide' control puts the scoreboard back.
      const el = Array.from(document.querySelectorAll('*')).find(e =>
        e.children.length === 0 && /^Hide$/i.test((e.textContent||'').trim()));
      if (!el) return JSON.stringify({ok: false});
      el.click();
      return JSON.stringify({ok: true});
    })()
    """

    def hide_media_overlay(self) -> bool:
        """Collapse an expanded media widget that is covering the scoreboard."""
        with self.page_lock:
            try:
                raw = self._evaluate(self.JS_HIDE_MEDIA)
            except CaptureError:
                return False
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                return False
            return bool(isinstance(data, dict) and data.get("ok"))

    def read_scoreboard(self) -> dict:
        """Read the live scoreboard from the DOM.

        The scoreboard is not exposed in the accessibility tree, so the DOM is
        the only place the score, the clock and the period exist. This is
        the single most important read in the loop: without it the model
        abstains on every selection.
        """
        from atlas.live.extract import extract_scoreboard_js

        try:
            raw = self._evaluate(extract_scoreboard_js())
        except CaptureError:
            return {}
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}
        return raw if isinstance(raw, dict) else {}

    JS_OPEN_STATS = r"""
    (() => {
      const el = Array.from(document.querySelectorAll('*')).find(e =>
        e.children.length === 0 && /^Statistics$/i.test((e.textContent||'').trim()));
      if (!el) return JSON.stringify({ok: false, error: 'no statistics control'});
      el.click();
      return JSON.stringify({ok: true});
    })()
    """

    JS_CLOSE_STATS = r"""
    (() => {
      // The Statistics panel REPLACES the market panel (its tabs go with
      // it), so the market panel is restored with the 'Match Tracker'
      // control, not by clicking a market tab.
      const el = Array.from(document.querySelectorAll('*')).find(e =>
        e.children.length === 0 && /^Match Tracker$/i.test((e.textContent||'').trim()));
      if (!el) return JSON.stringify({ok: false});
      el.click();
      return JSON.stringify({ok: true});
    })()
    """

    def close_stats_panel(self) -> bool:
        """Return to the market panel after reading the statistics."""
        with self.page_lock:
            try:
                raw = self._evaluate(self.JS_CLOSE_STATS)
            except CaptureError:
                return False
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                return False
            return bool(isinstance(data, dict) and data.get("ok"))

    def read_stats(self) -> dict:
        """Open the Statistics panel and read the live statistics.

        The statistics are what the model's xG estimate is built from, and
        without them the model runs on priors alone — which the calibration
        then shrinks onto the book, leaving no edge. This read is the
        difference between a model that can compete and one that cannot.
        """
        from atlas.live.extract import extract_stats_js

        with self.page_lock:
            try:
                self._evaluate(self.JS_OPEN_STATS)
            except CaptureError:
                return {}
            # The panel loads asynchronously.
            for _ in range(6):
                time.sleep(0.8 * self.pause_scale)
                try:
                    raw = self._evaluate(extract_stats_js())
                except CaptureError:
                    continue
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if data:
                        return data
                elif isinstance(raw, dict) and raw:
                    return raw
            return {}

    @staticmethod
    def apply_stats(state: LiveMatchState, stats: dict) -> LiveMatchState:
        """Fold the live statistics into the observed state."""
        if not stats:
            return state

        def pair(name: str):
            v = stats.get(name)
            if not isinstance(v, dict):
                return (None, None)
            return (v.get("home"), v.get("away"))

        poss_h, poss_a = pair("possession_pct")
        sot_h, sot_a = pair("shots_on_target")
        corners_h, corners_a = pair("corners")
        att_h, att_a = pair("attempts")
        saves_h, saves_a = pair("saves")
        fouls_h, fouls_a = pair("fouls")
        off_h, off_a = pair("offsides")
        yc_h, yc_a = pair("yellow_cards")
        rc_h, rc_a = pair("red_cards")

        for team, values in (
            (state.home, (poss_h, sot_h, corners_h, att_h, saves_h, fouls_h, off_h, yc_h, rc_h)),
            (state.away, (poss_a, sot_a, corners_a, att_a, saves_a, fouls_a, off_a, yc_a, rc_a)),
        ):
            (
                poss, sot, corners, att, saves, fouls, offsides, yc, rc
            ) = values
            if poss is not None:
                team.possession_pct = poss
            if sot is not None:
                team.shots_on_target = sot
            if corners is not None:
                team.corners = corners
            if att is not None:
                team.shots_total = att
            if saves is not None:
                team.saves = saves
            if fouls is not None:
                team.fouls = fouls
            if offsides is not None:
                team.offsides = offsides
            if yc is not None:
                team.yellow_cards = yc
            if rc is not None:
                team.red_cards = rc
        return state

    JS_OPEN_MARKETS_TAB = r"""
    (() => {
      // The scoreboard lives in the market panel, so the panel must be the
      // active tab. Opening the Statistics panel replaces it.
      for (const name of ['Main', 'All']) {
        const el = Array.from(document.querySelectorAll('*')).find(e =>
          e.children.length === 0 && (e.textContent||'').trim() === name);
        if (el) { el.click(); return JSON.stringify({ok: true, tab: name}); }
      }
      return JSON.stringify({ok: false});
    })()
    """

    JS_FINISHED = r"""
    (() => {
      const t = document.body.innerText || '';
      if (/Match Ended/i.test(t)) return JSON.stringify({ended: true});
      const el = Array.from(document.querySelectorAll('*')).find(e =>
        e.children.length === 0 && /^ENDED$/i.test((e.textContent||'').trim()));
      return JSON.stringify({ended: !!el});
    })()
    """

    def _panel_is_finished(self) -> bool:
        """Has the focused match ended?

        A finished match shows 'ENDED' / 'Match Ended' instead of a
        scoreboard, so the score and clock cannot be read. The final score
        is read separately from the result block.
        """
        try:
            raw = self._evaluate(self.JS_FINISHED)
        except CaptureError:
            return False
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(isinstance(data, dict) and data.get("ended"))

    def open_markets_tab(self) -> bool:
        """Make sure the market panel (and its scoreboard) is the active tab."""
        with self.page_lock:
            try:
                raw = self._evaluate(self.JS_OPEN_MARKETS_TAB)
            except CaptureError:
                return False
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                return False
            return bool(isinstance(data, dict) and data.get("ok"))

    def recover(self) -> bool:
        """Recover a wedged tab, escalating only as far as needed.

        The extension intermittently stops answering on the heavy live page
        (the daemon returns 502). Escalation:
          1. navigate to a light page (re-registers a merely stuck tab),
          2. CDP Page.navigate (handled by the browser process, not the
             page's stuck JS thread),
          3. close the frozen TAB — never the whole session — and open a
             fresh one.

        Order matters: the old code closed the SESSION before proving it
        could reopen one, so a failed reopen left no tab at all and every
        later call answered "session not found" (2026-10-29).
        """
        with self.page_lock:
            href = self.current_url()
            crashed = href.startswith("chrome-error://") or href == "about:blank"
            alive = False
            if not crashed:
                # A live tab accepts a normal navigation; a crashed one
                # would only hang the extension for its full timeout.
                try:
                    self.adapter.navigate(LIGHT_URL)
                    alive = self._wait_until(self._usable, 6.0)
                except Exception:
                    pass
            if not alive:
                cdp = getattr(self.adapter, "cdp", None)
                if callable(cdp):
                    try:
                        cdp("Page.navigate", {"url": LIGHT_URL})
                    except Exception:
                        pass
                    alive = self._wait_until(self._usable, 6.0)
            if not alive:
                try:
                    self.adapter.close_tab()
                except Exception:
                    pass
                try:
                    self.adapter.navigate(LIGHT_URL)
                except Exception:
                    pass
                alive = self._wait_until(self._usable, 8.0)
            # Put the tab back on the board whether or not it answered: the
            # next capture must never read the light page.
            try:
                self._evaluate(f"location.href={json.dumps(self.url)}; 1")
            except CaptureError:
                pass
            self._wait_until(self._on_board_list, 6.0)
            if self.board_row_count() == 0:
                try:
                    self._evaluate("location.reload(); 1")
                except CaptureError:
                    pass
                self._wait_until(lambda: self.board_row_count() > 0, 4.0)
            self._loaded = True
            return alive

    def _responsive(self) -> bool:
        try:
            self._evaluate("1", retries=0)
            return True
        except CaptureError:
            return False

    def _usable(self) -> bool:
        """Responsive AND on a real page (not a crashed-tab error page)."""
        href = self.current_url()
        if not href:
            return False
        if href.startswith("chrome-error://") or href == "about:blank":
            return False
        return self._responsive()

    def open_match(self, url: str, wait_s: float = 25.0) -> bool:
        """Open a match by its DIRECT url and wait for its panel.

        The board's match LIST is the flakiest part of the site: it renders
        progressively, it redirects between sports, and it intermittently
        delivers nothing at all. A match url is stable, so once a match is
        known the loop can work it directly instead of re-deriving it from
        the list every cycle.
        """
        try:
            self._evaluate(f"location.href={json.dumps(url)}; 1")
        except CaptureError:
            return False
        deadline = time.monotonic() + wait_s
        with self.page_lock:
            while time.monotonic() < deadline:
                time.sleep(2.0)
                try:
                    if self.read_markets_dom():
                        return True
                except Exception:
                    continue
            return False

    def match_url(self) -> Optional[str]:
        """The url of the match currently open, if it is a match page."""
        try:
            href = self._evaluate("location.href")
        except CaptureError:
            return None
        if isinstance(href, str) and "/sr:match:" in href:
            return href
        return None

    def read_markets_dom(self) -> list[dict]:
        """Read the markets from the DOM.

        This is the PRIMARY path: the accessibility snapshot intermittently
        fails on the heavy live board (the daemon returns 502), so the
        markets are read with targeted JS and the snapshot is only a
        fallback.
        """
        from atlas.live.extract import extract_markets_js, markets_from_js_entries

        try:
            raw = self._evaluate(extract_markets_js())
        except CaptureError:
            return []
        if isinstance(raw, str):
            try:
                entries = json.loads(raw)
            except json.JSONDecodeError:
                return []
        else:
            entries = raw if isinstance(raw, list) else []
        if not entries:
            return []
        home, away = self._team_names()
        return markets_from_js_entries(entries, self.match_id, home, away)

    def _team_names(self) -> tuple[str, str]:
        """The viewed match's team names, from the panel header."""
        try:
            raw = self._evaluate(
                "(() => { let h = null; document.querySelectorAll('*').forEach(e => {"
                " if (!h && e.children.length === 0 && / vs /.test(e.textContent||'')"
                " && (e.textContent||'').length < 80) h = e; });"
                " return h ? (h.textContent||'').trim() : ''; })()"
            )
        except CaptureError:
            return ("", "")
        if isinstance(raw, str) and " vs " in raw:
            home, away = raw.split(" vs ", 1)
            return (home.strip(), away.strip())
        return ("", "")

    def read_markets_snapshot(self) -> list[dict]:
        """Read the markets from the accessibility tree (fallback path)."""
        try:
            texts = self._snapshot_texts()
        except Exception:
            return []
        return extract_markets(texts, self.match_id)

    def _state_from_dom(self) -> LiveMatchState:
        """Build the observed state from the DOM alone.

        The accessibility tree does not carry the scoreboard, so this is the
        normal path: the scoreboard comes from the DOM and the team names
        from the panel header.
        """
        from atlas.live.observe import TeamStats

        board = self.read_scoreboard()
        home, away = self._team_names()
        return LiveMatchState(
            match_id=self.match_id,
            home=TeamStats(
                name=home or "Home",
                goals=board.get("home_goals"),
            ),
            away=TeamStats(
                name=away or "Away",
                goals=board.get("away_goals"),
            ),
            period=board.get("period") or "unknown",
            clock_minute=board.get("clock_minute"),
            clock_label=board.get("clock_label"),
            observed_at=datetime.now(timezone.utc),
            source="sportybet_dom",
        )

    def capture(self, wait: bool = True) -> CaptureResult:
        """Read the focused match panel, waiting until it is ready."""
        deadline = time.monotonic() + (self.max_wait_s if wait else 0.0)
        reason = "not captured"
        media_hidden = False
        recovered = False
        empty_reads = 0
        # The board must be loaded before anything is read, and the live URL
        # is never passed to navigate().
        if not self.board_loaded():
            self.load_board()
        while True:
            # P0: a session that has expired is detected and reported, so
            # the loop can tell the operator instead of failing silently.
            session = self.session_expired()
            if session.get("expired"):
                return CaptureResult(
                    False,
                    "session expired (the page is showing the login form)",
                    observed_at=datetime.now(timezone.utc),
                    match_id=self.match_id,
                )
            # On the BOARD (not a match panel) there is nothing to wait for:
            # return at once so switch_match can open a live match. Waiting
            # out max_wait_s here burned ~90s per rotation and hammered the
            # heavy list with snapshots until its renderer crashed
            # (2026-10-29).
            if self.at_board_list():
                rows = self.board_row_count()
                if rows > 0:
                    return CaptureResult(
                        False,
                        f"on the live board ({rows} rows) — no match selected",
                        observed_at=datetime.now(timezone.utc),
                        match_id=self.match_id,
                    )
            # The DOM is the primary path; the snapshot is the fallback.
            markets = self.read_markets_dom()
            texts: list[str] = []
            if not markets:
                try:
                    texts = self._snapshot_texts()
                except Exception as e:
                    reason = f"snapshot failed: {e}"
                    if time.monotonic() >= deadline:
                        return CaptureResult(False, reason, match_id=self.match_id)
                    # The extension wedges on the heavy page and stops
                    # answering. Recovering is cheaper than failing the cycle.
                    if not recovered:
                        recovered = True
                        self.recover()
                    time.sleep(self.poll_s)
                    continue
                markets = extract_markets(texts, self.match_id)
                if not markets:
                    # The board rendered but carries no markets: the live feed
                    # is not delivering. A reload is the only thing that
                    # fixes it, so do it after a couple of empty reads.
                    empty_reads += 1
                    if empty_reads >= 2 and not recovered:
                        recovered = True
                        self.recover()
                    time.sleep(self.poll_s)
                    if time.monotonic() >= deadline:
                        return CaptureResult(
                            False,
                            self.board_failure_reason(
                                "the board is empty (the live feed is not "
                                "delivering)"
                            ),
                            markets=markets,
                            observed_at=datetime.now(timezone.utc),
                            match_id=self.match_id,
                        )
                    continue
            empty_reads = 0
            state = extract_live_match_state(texts, self.match_id) if texts else None
            if state is None:
                state = self._state_from_dom()
            # The DOM is authoritative for the scoreboard: the accessibility
            # tree does not expose it.
            board = self.read_scoreboard()
            if not board.get("period") and self._panel_is_finished():
                board = {"period": "ft", "home_goals": board.get("home_goals"),
                         "away_goals": board.get("away_goals")}
            if not board.get("period"):
                # The Statistics panel replaces the market panel (and its
                # scoreboard), so put the market tab back and read again.
                if self.open_markets_tab():
                    time.sleep(1.0)
                    board = self.read_scoreboard()
            if board:
                if board.get("home_goals") is not None:
                    state.home.goals = board["home_goals"]
                if board.get("away_goals") is not None:
                    state.away.goals = board["away_goals"]
                if board.get("clock_minute") is not None:
                    state.clock_minute = board["clock_minute"]
                if board.get("clock_label"):
                    state.clock_label = board["clock_label"]
                if board.get("period"):
                    state.period = board["period"]
            raw_markets = [
                {
                    "status": "open"
                    if len(m.get("selections", [])) >= 2
                    else "suspended"
                }
                for m in markets
            ]
            ready, why = panel_is_ready(
                state, raw_markets, min_markets=self.min_markets
            )
            if ready:
                # The statistics are read LAST: opening the panel REPLACES
                # the market panel, so the market list must be captured
                # first and the panel restored afterwards.
                stats = self.read_stats()
                if stats:
                    self.apply_stats(state, stats)
                    self.close_stats_panel()
                return CaptureResult(
                    True,
                    "ready",
                    markets=markets,
                    state=state,
                    observed_at=datetime.now(timezone.utc),
                    match_id=self.match_id,
                )
            reason = why
            # If the scoreboard is missing but the media widget is expanded,
            # collapse it once and try again. The period case is included:
            # "period not live (got 'unknown')" comes from exactly the same
            # hidden-scoreboard condition, and the widget sits over the only
            # place the clock and period exist.
            if (
                not media_hidden
                and (
                    "score unknown" in why
                    or "clock unknown" in why
                    or "period not live" in why
                )
                and time.monotonic() < deadline
            ):
                media_hidden = self.hide_media_overlay()
                if media_hidden:
                    time.sleep(1.0)
                    continue
            if time.monotonic() >= deadline:
                return CaptureResult(
                    False,
                    reason,
                    markets=markets,
                    state=state,
                    observed_at=datetime.now(timezone.utc),
                    match_id=self.match_id,
                )
            time.sleep(self.poll_s)

    # --- the board's match list -----------------------------------------

    JS_LIST_FALLBACK = r"""
    (() => {
      // Class-independent live-match rows, v2 — CLOCK-ANCHORED.
      // Ground truth from the live board (2026-10-28, 49 matches rendered,
      // 0 rows found by the old shape): each match row renders its teams in
      // SEPARATE spans — the visible 'vs' is styling, not text — so the
      // leaf 'Home v Away' shape never matches. The stable anchor is the
      // live CLOCK leaf ("80:10") beside a period chip ("H2"). From every
      // clock leaf we climb to the smallest ancestor that also carries
      // decimal odds — that container is the match row. Teams are read
      // from the row text: the alphabetic run between the clock/period
      // prefix and the score digits is the home team; the run between the
      // score and the first decimal price is the away team. The click that
      // opens a match is always a marked REAL click on the marked row.
      const rows = [];
      const seen = new Set();
      // scores are single-or-two digit tokens; '1923 Kragujevac' keeps its
      // year in the team name because 1923 cannot be a score
      const isScore = (s) => /^\d{1,2}$/.test(s);
      for (const e of document.querySelectorAll('*')) {
        if (e.children.length) continue;
        const txt = (e.textContent || '').trim();
        if (!/^\d{1,3}:\d{2}$/.test(txt)) continue;
        // Leftover BETSLIP selections render odds text too; a slip row
        // classified as a board row traps the picker back into the same
        // parked match.
        if (e.closest && e.closest('[class*=betslip],[class*=m-order],[class*=slip]')) continue;
        // Climb to the enclosing row: smallest container with the odds
        // cluster AND exactly ONE live clock — a container holding two
        // clocks has swallowed neighbouring matches and must not be a
        // click target.
        const clocksIn = (t) => (t.match(/\d{1,3}:\d{2}/g) || []).length;
        let row = e;
        let ok = false;
        for (let up = 0; up < 8; up++) {
          const t = row.innerText || '';
          const prices = (t.match(/\b\d{1,3}\.\d{2}\b/g) || []).length;
          if (prices >= 3 && clocksIn(t) <= 1) { ok = true; break; }
          if (!row.parentElement) break;
          row = row.parentElement;
        }
        if (!ok) continue;
        const text = (row.innerText || '').replace(/\s+/g, ' ').trim();
        const key = text.slice(0, 60);
        if (seen.has(key)) continue;
        seen.add(key);
        // Teams from text: everything from the CLOCK onwards, periods and
        // scores are digits so they cannot be confused with team words.
        const at = text.search(/\d{1,3}:\d{2}/);
        let seg = at >= 0 ? text.slice(at) : text;
        const pfx = seg.match(/^\d{1,3}:\d{2}\s*(?:H1|H2|HT|Penalty|AET|Pen)?\s*/);
        if (pfx) seg = seg.slice(pfx[0].length);
        const tokens = seg.split(' ');
        let i = 0;
        const home = [];
        const away = [];
        while (i < tokens.length && !isScore(tokens[i])) { home.push(tokens[i]); i++; }
        while (i < tokens.length && isScore(tokens[i])) i++;
        const priceRe = /^\d{1,3}\.\d{2}$/;
        while (
          i < tokens.length
          && !isScore(tokens[i])
          && !priceRe.test(tokens[i])
          && !/^[+]\d+$/.test(tokens[i])   // '+66' more-markets chips
        ) {
          away.push(tokens[i]); i++;
        }
        if (!home.length || !away.length) continue;
        const periodM = text.match(/\b(H1|H2|HT|Penalty|AET)\b/);
        const rowIdx = rows.length;
        if (!row.offsetParent) continue;
        row.setAttribute('data-atlas-live-row', String(rowIdx));
        rows.push({
          i: rowIdx,
          home: home.join(' ').slice(0, 45),
          away: away.join(' ').slice(0, 45),
          clock: txt,
          period: periodM ? periodM[1] : '',
          score: '',
          odds: ((text.match(/\b\d{1,3}\.\d{2}\b/g) || []).length),
          text: text.slice(0, 140),
        });
        if (rows.length >= 40) break;
      }
      return JSON.stringify({rows: rows, mode: 'clock'});
    })()
    """

    JS_ROWS_TEXT = r"""
    (() => {
      // v3 — structural .match-row scan (ground truth 2026-10-29).
      // The list keeps .match-row but the row build varies: classes get
      // renamed and the clock is split across child spans. Read the row
      // STRUCTURE first (.teams children 0/1 = home/away, .time child 0 =
      // clock), then leaf order. NEVER split the row text into teams:
      // 'Everton Newcastle 2 0' has no delimiter between the names.
      // marker: rows-text
      const rows = [];
      const seen = new Set();
      const periodRe = /\b(H1|H2|HT|Penalty|AET)\b/;
      const textOf = (el) => ((el && el.innerText) || '')
        .replace(/\s+/g, ' ').trim();
      document.querySelectorAll('.match-row').forEach((row, i) => {
        // getClientRects() (not offsetParent): a virtualised/transformed
        // list gives visible rows offsetParent === null, which would
        // discard every row.
        if (!row.getClientRects || row.getClientRects().length === 0) return;
        const text = textOf(row);
        if (!text) return;
        const cm = text.match(/\b\d{1,3}:\d{2}\b/);
        if (!cm) return;
        const key = text.slice(0, 60);
        if (seen.has(key)) return;
        let clock = cm[0];
        let home = '';
        let away = '';
        const teams = row.querySelector('.teams');
        if (teams && teams.children.length >= 2) {
          home = textOf(teams.children[0]);
          away = textOf(teams.children[1]);
        }
        if (!home || !away) {
          home = textOf(row.querySelector('.home-team'));
          away = textOf(row.querySelector('.away-team'));
        }
        const time = row.querySelector('.time');
        if (time) {
          const first = time.children[0];
          const tm = textOf(first || time).match(/\b\d{1,3}:\d{2}\b/);
          if (tm) clock = tm[0];
        }
        if (!home || !away) {
          // Last resort: the first two text leaves that are neither the
          // clock nor the match id (each team is one leaf on this build).
          const leaves = Array.from(row.querySelectorAll('*'))
            .filter((e) => !e.children.length)
            .map(textOf)
            .filter((t) => t && /[A-Za-z]/.test(t) && !/^ID:/.test(t)
                        && !/Multi View|Statistics|H2H/.test(t));
          if (leaves.length >= 2) {
            home = leaves[0];
            away = leaves[1];
          }
        }
        if (!home || !away) return;
        seen.add(key);
        const periodM = text.match(periodRe);
        // DOM index, not output index: select_match clicks rows[index] on
        // the raw .match-row list, so both scanners must speak that order.
        const idx = i;
        row.setAttribute('data-atlas-live-row', String(idx));
        rows.push({
          i: idx,
          home: home.slice(0, 45),
          away: away.slice(0, 45),
          clock: clock,
          period: periodM ? periodM[1] : '',
          score: '',
          odds: (text.match(/\b\d{1,3}\.\d{2}\b/g) || []).length,
          text: text.slice(0, 140),
        });
      });
      return JSON.stringify({rows: rows, mode: 'rows'});
    })()
    """

    JS_CARDS = r"""
    (() => {
      const out = [];
      // Multi View: one row per live match, with explicit classes.
      document.querySelectorAll('.match-row').forEach((row, i) => {
        const t = (el) => {
          const e = row.querySelector(el);
          return e ? (e.innerText || '').trim() : '';
        };
        const home = t('.home-team');
        const away = t('.away-team');
        if (!home || !away) return;
        const scores = Array.from(row.querySelectorAll('.score-item'))
          .map(e => (e.innerText || '').trim()).filter(v => /^\d{1,2}$/.test(v));
        const odds = Array.from(row.querySelectorAll('.m-outcome-odds')).length;
        out.push({
          i: i, home: home, away: away,
          clock: t('.clock-time'), period: t('.game-id'),
          score: scores.join(':'), odds: odds,
          text: (row.innerText || '').trim().replace(/\n/g, ' ').slice(0, 120)
        });
      });
      if (out.length) return JSON.stringify({rows: out, mode: 'multi'});

      // Single View: a card per match.
      document.querySelectorAll('.sr-lmt-plus-scb__wrapper').forEach((c, i) => {
        const teams = Array.from(c.querySelectorAll('.sr-lmt-plus-scb__team-title'))
          .map(e => (e.innerText || '').trim()).filter(Boolean);
        if (teams.length < 2) return;
        const period = ((c.querySelector('.sr-lmt-plus-scb__result-period') || {}).innerText || '').trim();
        const result = ((c.querySelector('.sr-lmt-plus-scb__result') || {}).innerText || '').trim();
        const text = (c.innerText || '').trim();
        let odds = 0;
        c.querySelectorAll('*').forEach(e => {
          if (e.children.length === 0 && /^[0-9]{1,3}\.[0-9]{2}$/.test((e.textContent||'').trim())) odds++;
        });
        out.push({i: i, home: teams[0], away: teams[1], clock: '', period: period,
                  score: result.replace(/\n/g, ' '), odds: odds,
                  text: text.replace(/\n/g, ' ').slice(0, 120)});
      });
      return JSON.stringify({rows: out, mode: 'single'});
    })()
    """

    @staticmethod
    def _finished_markers() -> tuple[str, ...]:
        return (
            "FT",
            "ENDED",
            "FINISHED",
            "POSTPONED",
            "CANCELED",
            "CANCELLED",
            "AWARDED",
            "ABANDONED",
            "SUSPENDED",
        )

    def _scan_rows(self, script: str) -> list[dict]:
        """Run one board scanner; failures are an empty scan, not a crash."""
        try:
            raw = self._evaluate(script)
        except CaptureError:
            return []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return []
        rows = (data or {}).get("rows", []) if isinstance(data, dict) else []
        return [r for r in rows if isinstance(r, dict)]

    def board_candidates(self) -> list[BoardCandidate]:
        """Every live match on the board, for the Boss to choose between.

        The two scanners see different ROW BUILDS, not different boards:
        on 2026-10-29 the class-shaped scan saw 12 of 24 rows (some rows
        omit .home-team) while the structural scan saw all 24. Run both and
        merge — a missing row is a match the loop can never trade.
        """
        rows = self._scan_rows(self.JS_CARDS)
        seen = {
            ((r.get("home") or "").upper(), (r.get("away") or "").upper())
            for r in rows
        }
        for r in self._scan_rows(self.JS_ROWS_TEXT):
            key = ((r.get("home") or "").upper(), (r.get("away") or "").upper())
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
        if not rows:
            # The book rebuilds its UI classes; the class-shaped scans above
            # can find nothing even while the list renders. Fall back to a
            # class-independent scan: any leaf text that looks like
            # 'Home v Away' inside a list region that also carries prices.
            rows = self._scan_rows(self.JS_LIST_FALLBACK)
        out: list[BoardCandidate] = []
        for e in rows:
            home = (e.get("home") or "").strip()
            away = (e.get("away") or "").strip()
            if not home or not away:
                continue
            period = (e.get("period") or "").replace("\u00a0", " ").strip()
            clock = (e.get("clock") or "").replace("\u00a0", " ").strip()
            minute = self._parse_minute(clock) if clock else None
            if minute is None:
                minute = self._parse_minute(period)
            out.append(
                BoardCandidate(
                    match_id=f"board-{e.get('i', len(out))}",
                    home_team=home,
                    away_team=away,
                    score=(e.get("score") or e.get("text") or "").replace("\n", " "),
                    minute=minute,
                    period=period or clock,
                    num_markets=int(e.get("odds") or 0),
                    index=int(e.get("i", 0)),
                    live=self._is_live_period(period),
                )
            )
        return out

    # Period labels that mean the match is being played. The list also
    # carries upcoming kick-offs ('14:30 ID: 15728'): they have no period,
    # and opening one trades live pricing for a pre-match panel.
    LIVE_PERIOD_MARKERS = (
        "H1", "H2", "1H", "2H", "HT", "1ST", "2ND", "HALF", "PEN", "AET",
    )

    @classmethod
    def _is_live_period(cls, period: str) -> bool:
        p = (period or "").upper()
        return any(marker in p for marker in cls.LIVE_PERIOD_MARKERS)

    @staticmethod
    def _parse_minute(period: str) -> Optional[float]:
        """Read the match clock from a card's period label.

        The label is not always a clock: on a finished match it carries the
        HALF-TIME score ('HT 1 : 0'), and on a scheduled match a kick-off
        time. Reading the '1' out of 'HT 1 : 0' as a minute would send the
        loop to a match that is long over.
        """
        text = (period or "").strip()
        if not text:
            return None
        # A clock: '67:12'
        m = re.match(r"^(\d{1,3}):(\d{2})$", text)
        if m:
            return int(m.group(1)) + int(m.group(2)) / 60.0
        # A quoted minute: "67'" or "67"
        m = re.search(r"(\d{1,3})\s*['\u2019]", text)
        if m:
            return float(m.group(1))

    # --- account balance (board header, no panel needed) -------------------

    JS_BALANCE = r"""
    (() => {
      // The header renders the balance beside the next control:
      // 'NGN 52.68Deposit' — innerText glues it on, so the terminator is
      // 'anything that cannot continue a number', not a newline.
      const t = document.body ? (document.body.innerText || '') : '';
      const m = t.match(/(?:^|[\n|])\s*(?:NGN|\u20a6)\s*([\d,]+\.\d{2})(?=[^\d,]|$)/m);
      return JSON.stringify({
        balance: m ? parseFloat(m[1].replace(/,/g, '')) : null
      });
    })()
    """

    def read_balance_ngn(self) -> Optional[float]:
        """The header account balance — readable on EVERY page.

        The match-panel extraction carries the balance only while a panel is
        open, so in board mode (or while the board is empty) the balance
        guard was blind all night: heartbeat account_balance_ngn stayed None
        while the header plainly showed NGN 52.68 (2026-10-28).
        """
        try:
            raw = self._evaluate(self.JS_BALANCE)
        except CaptureError:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        val = data.get("balance")
        if not isinstance(val, (int, float)):
            return None
        # Sanity: a real account balance is a small non-negative number.
        if not (0.0 <= float(val) <= 10_000_000.0):
            return None
        return float(val)
        # A spaced score ('1 : 0') is not a clock.
        if re.search(r"\d\s*:\s*\d", text):
            return None
        m = re.match(r"^(\d{1,3})\b", text)
        if m:
            return float(m.group(1))
        return None

    JS_SELECT = r"""
    (() => {
      const target = %(index)d;
      // Multi View: the row itself opens the match's market panel.
      const rows = document.querySelectorAll('.match-row');
      if (rows.length) {
        const row = rows[target];
        if (!row) return JSON.stringify({ok: false, error: 'no row'});
        const cell = row.querySelector('.home-team') || row;
        // NO synthetic click: this UI build ignores untrusted events (the
        // login form proved it). Mark the target for a REAL browser click.
        cell.setAttribute('data-atlas-open-match', '1');
        return JSON.stringify({ok: true, mode: 'multi', selector: '[data-atlas-open-match="1"]'});
      }
      // Single View: a card per match.
      const cards = Array.from(document.querySelectorAll('.sr-lmt-plus-scb__wrapper'));
      const card = cards[target];
      if (card) {
        const clickable = card.querySelector('.sr-lmt-plus-scb__team-title') || card;
        clickable.setAttribute('data-atlas-open-match', '1');
        return JSON.stringify({ok: true, mode: 'single', selector: '[data-atlas-open-match="1"]'});
      }
      // Class-independent fallback: the rows the list scan marked with
      // data-atlas-live-row (the reunited shape when the board re-skins).
      const fb = document.querySelector('[data-atlas-live-row="' + target + '"]');
      if (fb) {
        fb.setAttribute('data-atlas-open-match', '1');
        return JSON.stringify({ok: true, mode: 'fallback', selector: '[data-atlas-open-match="1"]'});
      }
      return JSON.stringify({ok: false, error: 'no row'});
    })()
    """

    def select_match(self, index: int) -> bool:
        """Open a specific match from the board list.

        The click must be a REAL browser click: the list's handlers ignore
        synthetic element.click() events on this UI build, so a marked
        element is clicked through the adapter.
        """
        try:
            with self.page_lock:
                raw = self._evaluate(self.JS_SELECT % {"index": index})
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, json.JSONDecodeError):
                    return False
                if not (
                    isinstance(data, dict)
                    and data.get("ok")
                    and data.get("selector")
                ):
                    return False
                # One re-mark retry: the list re-renders and can drop a marker.
                for _ in range(2):
                    try:
                        self.adapter.click(data.get("selector"))
                        return True
                    except Exception:
                        try:
                            self._evaluate(self.JS_SELECT % {"index": index})
                        except CaptureError:
                            return False
                return False
        except CaptureError:
            return False
