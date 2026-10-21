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

LIVE_BOARD_URL = "https://sportybet.com/ng/sport/live/"
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
        # Consecutive failed probes: after a few, the session is treated as
        # dead rather than flaky, so the loop fails fast instead of waiting
        # out the daemon's timeout on every call.
        self._consecutive_failures = 0

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
                data = self._unwrap(self.adapter.evaluate(code))
                self._consecutive_failures = 0
                if isinstance(data, dict) and "value" in data:
                    return data["value"]
                return data
            except Exception as e:  # the extension intermittently fails
                last = e
                self._consecutive_failures += 1
                if self._consecutive_failures >= 2:
                    # The session is dead, not flaky. Every further probe
                    # would wait out the daemon's own timeout, so fail fast
                    # and let the supervisor recover instead.
                    raise CaptureError(
                        f"the browser session is not responding ({e})"
                    ) from e
                time.sleep(1.5 * self.pause_scale)
        raise CaptureError(f"evaluate failed: {last}")

    def _snapshot_texts(self) -> list[str]:
        data = self._unwrap(self.adapter.snapshot())
        tree = data.get("tree") or [] if isinstance(data, dict) else []
        return iter_texts(tree)

    def board_loaded(self) -> bool:
        try:
            href = self._evaluate("location.href")
        except CaptureError:
            return False
        return isinstance(href, str) and "/sport/live" in href

    def load_board(self, force: bool = False) -> None:
        """Get the live board loaded, working around the never-firing 'load'.

        The direct navigation always times out, so the board is reached by
        loading a light page and then navigating client-side, which does not
        wait for a load event. The live URL is NEVER passed to navigate().

        In Multi View the whole board is shown; in Single View one match is.
        The board is left in Multi View because the match selection needs the
        list of live matches.
        """
        if self.board_loaded() and not force:
            self._loaded = True
            self.ensure_multi_view()
            return
        try:
            self.adapter.navigate(LIGHT_URL)
        except Exception:
            pass
        time.sleep(2.0)
        try:
            self._evaluate(f"location.href={json.dumps(self.url)}; 1")
        except CaptureError:
            pass
        # The board's own data needs a moment; a reload populates it.
        time.sleep(4.0)
        try:
            self._evaluate("location.reload(); 1")
        except CaptureError:
            pass
        time.sleep(3.0)
        self._loaded = True
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
        try:
            raw = self._evaluate(self.JS_SESSION)
        except CaptureError:
            return {"expired": False}
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {"expired": False}
        else:
            data = raw if isinstance(raw, dict) else {}
        return data or {"expired": False}

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
        """Recover a wedged browser tab.

        The extension intermittently stops answering on the heavy live page
        (the daemon returns 502). Navigating to a light page re-registers
        the tab, and the extension starts answering again — but only after a
        moment, so the follow-up is retried rather than assumed.
        """
        navigated = False
        try:
            # A FROZEN tab cannot be navigated: the navigation is queued to
            # the page, and the page's JS thread is what is stuck. Closing the
            # session discards the frozen tab, and a fresh one is healthy.
            try:
                self.adapter.close_session()
                time.sleep(1.5 * self.pause_scale)
            except Exception:
                pass
            self.adapter.navigate(LIGHT_URL)
            navigated = True
        except Exception:
            pass
        if navigated:
            time.sleep(2.5 * self.pause_scale)
            # The tab re-registers asynchronously; wait for it to answer.
            for _ in range(6):
                try:
                    self._evaluate("1")
                    break
                except CaptureError:
                    time.sleep(1.5 * self.pause_scale)
        try:
            self._evaluate(f"location.href={json.dumps(self.url)}; 1")
        except CaptureError:
            pass
        time.sleep(4.0 * self.pause_scale)
        try:
            self._evaluate("location.reload(); 1")
        except CaptureError:
            pass
        time.sleep(3.0 * self.pause_scale)
        self._loaded = True
        return True

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
                            "the board is empty (the live feed is not delivering)",
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
            # collapse it once and try again.
            if (
                not media_hidden
                and ("score unknown" in why or "clock unknown" in why)
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

    def board_candidates(self) -> list[BoardCandidate]:
        """Every live match on the board, for the Boss to choose between."""
        try:
            raw = self._evaluate(self.JS_CARDS)
        except CaptureError:
            return []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return []
        out: list[BoardCandidate] = []
        for e in (data or {}).get("rows", []):
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
                )
            )
        return out

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
        cell.click();
        return JSON.stringify({ok: true, mode: 'multi'});
      }
      // Single View: a card per match.
      const cards = Array.from(document.querySelectorAll('.sr-lmt-plus-scb__wrapper'));
      const card = cards[target];
      if (!card) return JSON.stringify({ok: false, error: 'no card'});
      const clickable = card.querySelector('.sr-lmt-plus-scb__team-title') || card;
      clickable.click();
      return JSON.stringify({ok: true, mode: 'single'});
    })()
    """

    def select_match(self, index: int) -> bool:
        """Open a specific match from the board list."""
        try:
            raw = self._evaluate(self.JS_SELECT % {"index": index})
        except CaptureError:
            return False
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(isinstance(data, dict) and data.get("ok"))
