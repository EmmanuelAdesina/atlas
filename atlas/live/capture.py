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
    ) -> None:
        self.adapter = adapter
        self.url = url
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self.min_markets = min_markets
        self.match_id = match_id
        self._loaded = False

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
                if isinstance(data, dict) and "value" in data:
                    return data["value"]
                return data
            except Exception as e:  # the extension intermittently fails
                last = e
                time.sleep(1.5)
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
        wait for a load event.
        """
        if self.board_loaded() and not force:
            self._loaded = True
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

    def capture(self, wait: bool = True) -> CaptureResult:
        """Read the focused match panel, waiting until it is ready."""
        deadline = time.monotonic() + (self.max_wait_s if wait else 0.0)
        reason = "not captured"
        media_hidden = False
        while True:
            try:
                texts = self._snapshot_texts()
            except Exception as e:
                reason = f"snapshot failed: {e}"
                if time.monotonic() >= deadline:
                    return CaptureResult(False, reason, match_id=self.match_id)
                time.sleep(self.poll_s)
                continue

            markets = extract_markets(texts, self.match_id)
            state = extract_live_match_state(texts, self.match_id)
            # The DOM is authoritative for the scoreboard: the accessibility
            # tree does not expose it.
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
      document.querySelectorAll('.sr-lmt-plus-scb__wrapper').forEach((c, i) => {
        const teams = Array.from(c.querySelectorAll('.sr-lmt-plus-scb__team-title'))
          .map(e => (e.innerText || '').trim()).filter(Boolean);
        const period = ((c.querySelector('.sr-lmt-plus-scb__result-period') || {}).innerText || '').trim();
        const result = ((c.querySelector('.sr-lmt-plus-scb__result') || {}).innerText || '').trim();
        // The card's mini-odds are a proxy for how deep the market list is.
        let odds = 0;
        c.querySelectorAll('*').forEach(e => {
          if (e.children.length === 0 && /^[0-9]{1,3}\.[0-9]{2}$/.test((e.textContent||'').trim())) odds++;
        });
        const text = (c.innerText || '').trim();
        if (teams.length >= 2) out.push({i, teams, period, result, odds, text});
      });
      return JSON.stringify(out);
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
            entries = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return []
        out: list[BoardCandidate] = []
        for e in entries or []:
            teams = e.get("teams") or []
            if len(teams) < 2:
                continue
            period = (e.get("period") or "").replace("\u00a0", " ").strip()
            result = (e.get("result") or "").replace("\u00a0", " ").strip()
            # The card's full text carries the status badge ('ENDED'), which
            # is a sibling of the result element, not inside it.
            card_text = (e.get("text") or "").replace("\u00a0", " ").strip()
            out.append(
                BoardCandidate(
                    match_id=f"board-{e.get('i', len(out))}",
                    home_team=teams[0],
                    away_team=teams[1],
                    score=(card_text or result).replace("\n", " "),
                    minute=self._parse_minute(period),
                    period=period,
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
      const cards = Array.from(document.querySelectorAll('.sr-lmt-plus-scb__wrapper'));
      const card = cards[target];
      if (!card) return JSON.stringify({error: 'no card'});
      const clickable = card.querySelector('.sr-lmt-plus-scb__team-title') || card;
      clickable.click();
      return JSON.stringify({ok: 1});
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
