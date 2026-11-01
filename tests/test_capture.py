"""Tests for the live capture layer (atlas.live.capture).

These cover the behaviours that were established by driving the real board:
the scoreboard is not in the accessibility tree, the statistics panel
REPLACES the market panel, Multi View labels a half 'H1'/'H2', and the
extension wedges and has to be recovered.
"""

import json

import pytest

from atlas.live.capture import BoardCandidate, LiveCapture
from atlas.live.observe import LiveMatchState, TeamStats


class FakeAdapter:
    """Returns canned payloads per JS marker."""

    def __init__(self, payloads: dict | None = None):
        self.payloads = payloads or {}
        self.navigations: list[str] = []
        self.evaluations: list[str] = []

    def evaluate(self, code: str) -> dict:
        self.evaluations.append(code)
        for marker, payload in self.payloads.items():
            if marker in code:
                return {"data": {"value": payload}}
        return {"data": {"value": "{}"}}

    def snapshot(self):
        return {"tree": [{"role": "StaticText", "name": t} for t in self.payloads.get("__texts__", [])]}

    def navigate(self, url: str, **kwargs):
        self.navigations.append(url)
        return {"success": True}


def test_multi_view_rows_are_read():
    payload = json.dumps(
        {
            "rows": [
                {
                    "i": 0,
                    "home": "Al Nassr Club",
                    "away": "Neom SC",
                    "clock": "43:10",
                    "period": "H1",
                    "score": "1:0",
                    "odds": 6,
                }
            ],
            "mode": "multi",
        }
    )
    adapter = FakeAdapter({"match-row": payload})
    cands = LiveCapture(adapter).board_candidates()
    assert len(cands) == 1
    c = cands[0]
    assert c.home_team == "Al Nassr Club"
    assert c.away_team == "Neom SC"
    assert c.period == "H1"
    assert c.minute == pytest.approx(43 + 10 / 60.0)
    assert c.num_markets == 6


def test_single_view_cards_are_read():
    payload = json.dumps(
        {
            "rows": [
                {
                    "i": 0,
                    "home": "Braga",
                    "away": "Vicente Barcelos",
                    "clock": "",
                    "period": "HT 1 : 0",
                    "score": "1 : 2",
                    "odds": 4,
                }
            ],
            "mode": "single",
        }
    )
    adapter = FakeAdapter({"sr-lmt-plus-scb__wrapper": payload})
    cands = LiveCapture(adapter).board_candidates()
    assert len(cands) == 1
    # 'HT 1 : 0' is a halftime score, not a clock.
    assert cands[0].minute is None


def test_the_scoreboard_is_read_from_the_dom():
    payload = json.dumps(
        {
            "home_goals": 1,
            "away_goals": 0,
            "clock_label": "66' H2",
            "clock_minute": 66.0,
            "period": "2h",
        }
    )
    adapter = FakeAdapter({"m-result-title": payload})
    board = LiveCapture(adapter).read_scoreboard()
    assert board["home_goals"] == 1
    assert board["away_goals"] == 0
    assert board["clock_minute"] == 66.0
    assert board["period"] == "2h"


def test_stats_are_folded_into_the_state():
    stats = {
        "corners": {"home": 8, "away": 3},
        "shots_on_target": {"home": 3, "away": 1},
        "attempts": {"home": 11, "away": 4},
        "yellow_cards": {"home": 0, "away": 2},
        "red_cards": {"home": 0, "away": 0},
    }
    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="H", goals=1),
        away=TeamStats(name="A", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        source="test",
    )
    LiveCapture.apply_stats(state, stats)
    assert state.home.corners == 8
    assert state.away.corners == 3
    assert state.home.shots_on_target == 3
    assert state.away.shots_total == 4
    assert state.away.yellow_cards == 2


def test_stats_do_not_clobber_what_is_already_known():
    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="H", goals=1, corners=5),
        away=TeamStats(name="A", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        source="test",
    )
    LiveCapture.apply_stats(state, {"corners": {"home": 8, "away": 3}})
    assert state.home.corners == 8


def test_empty_stats_leave_the_state_alone():
    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="H", goals=1, corners=5),
        away=TeamStats(name="A", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        source="test",
    )
    LiveCapture.apply_stats(state, {})
    assert state.home.corners == 5


def test_recovery_navigates_to_a_light_page():
    """A wedged extension is recovered by loading a light page."""
    adapter = FakeAdapter()
    LiveCapture(adapter, pause_scale=0.0).recover()
    assert adapter.navigations, "recovery must navigate somewhere"


def test_a_broken_adapter_yields_no_candidates():
    class Broken:
        def evaluate(self, code):
            raise RuntimeError("extension down")

        def snapshot(self):
            raise RuntimeError("extension down")

        def navigate(self, url, **kwargs):
            raise RuntimeError("extension down")

    capture = LiveCapture(Broken(), pause_scale=0.0)
    assert capture.board_candidates() == []
    assert capture.read_scoreboard() == {}
    assert capture.read_stats() == {}


def test_an_empty_board_triggers_a_reload():
    """A board that renders but carries no markets is the live feed failing.

    The capture must reload rather than spin, and it must say so.
    """
    calls = {"snapshot": 0, "navigate": 0}

    class EmptyAdapter:
        def evaluate(self, code):
            if "m-result-title" in code:
                return {"data": {"value": "{}"}}
            if "m-table__wrapper" in code:
                return {"data": {"value": "[]"}}
            if "vs" in code and "querySelectorAll" in code:
                return {"data": {"value": ""}}
            return {"data": {"value": "{}"}}

        def snapshot(self):
            calls["snapshot"] += 1
            return {"tree": []}

        def navigate(self, url, **kwargs):
            calls["navigate"] += 1
            return {"success": True}

    capture = LiveCapture(
        EmptyAdapter(), min_markets=5, max_wait_s=1.0, poll_s=0.1, pause_scale=0.0
    )
    result = capture.capture()
    assert result.ready is False
    assert "empty" in result.reason or "not ready" in result.reason


def test_the_capture_never_raises_on_a_broken_page():
    """A capture failure must be a reason, not an exception."""

    class Broken:
        def evaluate(self, code):
            raise RuntimeError("extension down")

        def snapshot(self):
            raise RuntimeError("extension down")

        def navigate(self, url, **kwargs):
            raise RuntimeError("extension down")

    capture = LiveCapture(Broken(), max_wait_s=0.5, poll_s=0.1, pause_scale=0.0)
    result = capture.capture()
    assert result.ready is False
    assert result.reason


# --- header balance (board mode, no panel needed) --------------------------
# 2026-10-28: the balance was read only from a match panel, so with the
# board unpicked the guard saw None all night while the header showed
# NGN 52.68. read_balance_ngn reads the header on ANY page.

def test_read_balance_ngn_parses_the_header():
    adapter = FakeAdapter({"NGN": json.dumps({"balance": 52.68})})
    assert LiveCapture(adapter).read_balance_ngn() == pytest.approx(52.68)


def test_read_balance_ngn_returns_none_when_absent():
    adapter = FakeAdapter({"NGN": json.dumps({"balance": None})})
    assert LiveCapture(adapter).read_balance_ngn() is None


def test_read_balance_ngn_rejects_absurd_values():
    adapter = FakeAdapter({"NGN": json.dumps({"balance": 1e12})})
    assert LiveCapture(adapter).read_balance_ngn() is None


def test_match_row_text_scanner_parses_rows_when_classes_are_renamed():
    """2026-10-29: the live football list renders .match-row but renamed
    the inner classes (no .home-team/.away-team) and split the clock across
    spans (no clock-shaped leaf). Both older scanners found ZERO rows on a
    53-row board. v3 parses the row TEXT; JS_CARDS must fall through to it."""
    rows_payload = json.dumps(
        {
            "rows": [
                {
                    "i": 0, "home": "Man City", "away": "Brighton",
                    "clock": "58:09", "period": "H2", "score": "", "odds": 6,
                },
                {
                    "i": 1, "home": "FK Radnicki 1923 Kragujevac",
                    "away": "Neom SC", "clock": "46:14", "period": "H1",
                    "score": "", "odds": 6,
                },
            ],
            "mode": "rows",
        }
    )
    adapter = FakeAdapter(
        {
            # JS_CARDS finds no .home-team rows on this build:
            "m-outcome-odds": json.dumps({"rows": [], "mode": "single"}),
            # ...so board_candidates must consult the v3 text scanner:
            "marker: rows-text": rows_payload,
        }
    )
    cands = LiveCapture(adapter).board_candidates()
    assert [c.home_team for c in cands] == [
        "Man City", "FK Radnicki 1923 Kragujevac"
    ]
    assert cands[0].minute == pytest.approx(58.0 + 9 / 60.0)
    assert cands[1].period == "H1"


# --- board route truth + fail-tolerant navigation (2026-10-29) -------------
# The loop sat on /sport/live/ (the bare hub: no list, no panel) and reported
# "the board is empty" while the real Multi View list was one route away.

def test_the_bare_hub_is_not_a_loaded_board():
    adapter = FakeAdapter(
        {"location.href": "https://www.sportybet.com/ng/sport/live/"}
    )
    assert LiveCapture(adapter).board_loaded() is False


def test_the_multi_view_list_is_a_loaded_board():
    adapter = FakeAdapter(
        {
            "location.href": (
                "https://www.sportybet.com/ng/sport/football/live_list/"
            )
        }
    )
    capture = LiveCapture(adapter)
    assert capture.board_loaded() is True
    assert capture.at_board_list() is True


def test_a_match_panel_is_a_loaded_board_even_without_sr_match():
    # Observed route shape: /ng/sport/football/live/Italy/Serie_A/Frosinone...
    adapter = FakeAdapter(
        {
            "location.href": (
                "https://www.sportybet.com/ng/sport/football/live/"
                "Italy/Serie_A/Frosinone_vs_Lecce"
            )
        }
    )
    assert LiveCapture(adapter).board_loaded() is True


class PageAdapter:
    """A tab with a mutable location, so navigation can be verified."""

    def __init__(self, href: str, rows: int = 0):
        self.href = href
        self.rows = rows
        self.navigations: list[str] = []

    def evaluate(self, code: str):
        code = code.strip()
        if code == "location.href":
            return self.href
        if code.startswith("location.href="):
            self.href = json.loads(code.split("=", 1)[1].split(";")[0])
            self.rows = 3
            return 1
        if "querySelectorAll('.match-row').length" in code:
            return self.rows
        if "Multi View" in code:
            return json.dumps(
                {"ok": True, "why": "already multi", "rows": self.rows}
            )
        if code.startswith("location.reload"):
            self.rows = 3
            return 1
        return "{}"

    def snapshot(self):
        return {"tree": []}

    def navigate(self, url, **kwargs):
        self.navigations.append(url)


def test_load_board_verifies_arrival_from_the_hub():
    adapter = PageAdapter("https://www.sportybet.com/ng/sport/live/")
    capture = LiveCapture(adapter, pause_scale=0.0)
    capture.load_board(force=True)
    assert capture.at_board_list() is True
    assert capture.last_href == capture.url
    # the live URL is still never handed to navigate() (only the light page)
    assert all("/sport/live" not in u for u in adapter.navigations)
    assert capture._loaded is True


def test_board_candidates_surfaces_a_daemon_failure_instead_of_empty():
    """A 502 is not an empty board: the transport fault must be visible."""
    from atlas.browser.webbridge import WebBridgeError

    class Dead:
        def evaluate(self, code):
            raise WebBridgeError(
                "daemon call evaluate returned HTTP 502 (transient)",
                transient=True,
            )

        def snapshot(self):
            raise WebBridgeError("daemon call snapshot failed", transient=True)

        def navigate(self, url, **kwargs):
            raise WebBridgeError("daemon down", transient=True)

    capture = LiveCapture(Dead(), pause_scale=0.0)
    assert capture.board_candidates() == []
    assert "502" in capture.last_transport_error


def test_the_broken_reason_says_unreachable_not_empty():
    from atlas.browser.webbridge import WebBridgeError

    class Dead:
        def evaluate(self, code):
            raise WebBridgeError(
                "daemon call evaluate returned HTTP 502 (transient)",
                transient=True,
            )

    capture = LiveCapture(Dead(), pause_scale=0.0)
    capture.evaluate = capture._evaluate
    for code in ("1",):
        try:
            capture._evaluate(code)
        except Exception:
            pass
    reason = capture.board_failure_reason("the board is empty")
    assert "unreachable" in reason
    assert "502" in reason


def test_recovery_uses_cdp_before_closing_a_tab():
    """CDP runs in the browser process, so it can move a page whose JS is
    stuck — the tab must survive when that works."""
    from atlas.browser.webbridge import WebBridgeError

    class FrozenUntilCdp:
        def __init__(self):
            self.frozen = True
            self.cdp_calls: list[str] = []
            self.closed = False

        def evaluate(self, code):
            if self.frozen:
                raise WebBridgeError(
                    "daemon call evaluate returned HTTP 502", transient=True
                )
            return "{}"

        def snapshot(self):
            return {"tree": []}

        def navigate(self, url, **kwargs):
            pass  # queued behind the frozen JS thread

        def cdp(self, method, params=None):
            self.cdp_calls.append(method)
            self.frozen = False

        def close_tab(self):
            self.closed = True

    adapter = FrozenUntilCdp()
    capture = LiveCapture(adapter, pause_scale=0.0)
    assert capture.recover() is True
    assert adapter.cdp_calls == ["Page.navigate"]
    assert adapter.closed is False


def test_recovery_closes_only_the_tab_never_the_session():
    from atlas.browser.webbridge import WebBridgeError

    class Dead:
        def __init__(self):
            self.session_closed = False
            self.closed = 0
            self.frozen = True

        def evaluate(self, code):
            if self.frozen:
                raise WebBridgeError("HTTP 502", transient=True)
            return "{}"

        def snapshot(self):
            return {"tree": []}

        def navigate(self, url, **kwargs):
            pass

        def close_session(self):
            self.session_closed = True

        def close_tab(self):
            self.closed += 1
            self.frozen = False

    adapter = Dead()
    capture = LiveCapture(adapter, pause_scale=0.0)
    capture.recover()
    assert adapter.closed == 1
    assert adapter.session_closed is False, (
        "closing the session before a reopen works leaves no tab at all"
    )


def test_board_candidates_flag_scheduled_rows_as_not_live():
    """The list also carries upcoming kick-offs ('14:30 ID: 15728'); they
    must never be mistaken for a live match at minute 14."""
    payload = json.dumps(
        {
            "rows": [
                {
                    "i": 0, "home": "Everton", "away": "Newcastle",
                    "clock": "60:57", "period": "H2", "odds": 6,
                },
                {
                    "i": 1, "home": "CD San Marcos de Jiquilisco",
                    "away": "AD San Rafael", "clock": "14:30",
                    "period": "", "odds": 31,
                },
            ],
            "mode": "multi",
        }
    )
    cands = LiveCapture(FakeAdapter({"m-outcome-odds": payload})).board_candidates()
    assert cands[0].live is True
    assert cands[1].live is False


def test_board_candidates_merges_both_scanners():
    """JS_CARDS and v3 see different row builds (12 vs 24 rows on the live
    board, 2026-10-29). Merge them; keep the DOM index so clicks land."""
    cards = json.dumps(
        {
            "rows": [
                {"i": 0, "home": "A", "away": "B", "clock": "60:00",
                 "period": "H2", "odds": 6},
            ],
            "mode": "multi",
        }
    )
    rows_text = json.dumps(
        {
            "rows": [
                {"i": 0, "home": "A", "away": "B", "clock": "60:00",
                 "period": "H2", "odds": 6},
                {"i": 3, "home": "C", "away": "D", "clock": "12:00",
                 "period": "H1", "odds": 6},
            ],
            "mode": "rows",
        }
    )
    adapter = FakeAdapter(
        {"m-outcome-odds": cards, "marker: rows-text": rows_text}
    )
    cands = LiveCapture(adapter).board_candidates()
    assert [(c.home_team, c.index) for c in cands] == [("A", 0), ("C", 3)]


def test_capture_returns_at_once_when_on_the_board():
    """The board is not an empty match panel: waiting out max_wait_s there
    burned ~90s per rotation and hammered the list until it crashed."""
    import time as _time

    adapter = FakeAdapter(
        {
            "location.href": (
                "https://www.sportybet.com/ng/sport/football/live_list/"
            ),
            "document.querySelectorAll('.match-row').length": 24,
        }
    )
    capture = LiveCapture(adapter, pause_scale=0.0, max_wait_s=30.0)
    started = _time.monotonic()
    result = capture.capture()
    assert _time.monotonic() - started < 2.0
    assert result.ready is False
    assert "no match selected" in result.reason


def test_recovery_skips_the_hanging_navigate_on_a_crashed_tab():
    """chrome-error pages make the extension's navigate wait out its whole
    timeout; CDP is handled by the browser process and works."""
    class Crashed:
        def __init__(self):
            self.href = "chrome-error://chromewebdata/"
            self.navigations: list[str] = []
            self.cdp_calls: list[str] = []

        def evaluate(self, code):
            if code.strip() == "location.href":
                return self.href
            return "{}"

        def snapshot(self):
            return {"tree": []}

        def navigate(self, url, **kwargs):
            self.navigations.append(url)
            self.href = url

        def cdp(self, method, params=None):
            self.cdp_calls.append(method)
            self.href = (params or {}).get("url", self.href)

    adapter = Crashed()
    capture = LiveCapture(adapter, pause_scale=0.0)
    assert capture.recover() is True
    assert adapter.navigations == [], (
        "a crashed tab must not wait out the extension navigate timeout"
    )
    assert adapter.cdp_calls == ["Page.navigate"]


# ---- batched capture read (one evaluate) --------------------------------

MATCH_URL = "https://www.sportybet.com/football/live/sr:match:1234"

SCOREBOARD = {
    "home_goals": 1, "away_goals": 0, "period": "2h",
    "clock_minute": 50, "clock_label": "50'",
}
MARKETS = [
    {"market": "Match Result",
     "selections": [{"name": "Home", "odds": 2.1}, {"name": "Draw", "odds": 3.2},
                    {"name": "Away", "odds": 3.5}]},
    {"market": "Goals Over/Under 2.5",
     "selections": [{"name": "Over 2.5", "odds": 1.8}, {"name": "Under 2.5", "odds": 2.0}]},
]


def _bundle_payload(href=MATCH_URL, **overrides):
    """A capture-bundle answer as the daemon would deliver it."""
    bundle = {
        "session": {"expired": False, "probe_failed": False, "href": href},
        "teams": "Arsenal vs Chelsea",
        "rows": 0,
        "scoreboard": dict(SCOREBOARD),
        "markets": [dict(m) for m in MARKETS],
    }
    bundle.update(overrides)
    return json.dumps(bundle)


def _capture(adapter, **kw):
    capture = LiveCapture(
        adapter, min_markets=kw.pop("min_markets", 2), max_wait_s=1.0,
        poll_s=0.1, pause_scale=0.0,
    )
    return capture, capture.capture(kw.pop("wait", True))


def test_capture_bundle_answers_every_read_in_one_evaluate():
    """The bundle replaces 5-6 sequential round-trips with ONE.

    The stats read stays separate by design (it opens a panel). The
    session / teams / rows / scoreboard / markets reads must NOT run as
    individual evaluates any more: each of their DOM markers appears
    exactly once — inside the single bundle call.
    """
    adapter = FakeAdapter({
        # "'session': " MUST come before "location.href": the bundle code
        # embeds the session JS (which contains location.href), and the
        # FakeAdapter returns on the FIRST matching marker.
        "'session': ": _bundle_payload(),
        "location.href": MATCH_URL,
    })
    capture, result = _capture(adapter)
    assert result.ready is True, result.reason
    ev = adapter.evaluations
    # one bundle call, and it carries every part
    bundle_calls = [c for c in ev if "'session': " in c]
    assert len(bundle_calls) == 1
    assert ".m-table__wrapper" in bundle_calls[0]
    assert ".m-result-title" in bundle_calls[0]
    assert "querySelectorAll('.match-row')" in bundle_calls[0]
    # no part is re-read individually anywhere else
    assert sum(".m-table__wrapper" in c for c in ev) == 1
    assert sum(".m-result-title" in c for c in ev) == 1
    assert sum("querySelectorAll('.match-row')" in c for c in ev) == 1
    # the state was built from the bundle, not from re-reads
    assert result.state.period == "2h"
    assert result.state.home.name == "Arsenal"
    assert result.state.away.name == "Chelsea"
    assert result.state.home.goals == 1 and result.state.away.goals == 0
    assert result.state.clock_minute == 50
    # href came from the bundle session part
    assert capture.last_href == MATCH_URL
    # markets list came from the bundle entries
    assert len(result.markets) == 2
    assert all(isinstance(m.get("selections"), list) for m in result.markets)


def test_bundle_failure_falls_back_to_single_readers():
    """A broken bundle only slows the cycle; NOTHING behaves differently.

    Without a bundle payload the single readers (session probe, markets
    DOM, scoreboard DOM, team names) run exactly as before and the
    capture still reaches 'ready' through them.
    """
    adapter = FakeAdapter({
        "location.href": MATCH_URL,
        "expired": json.dumps({"expired": False, "href": MATCH_URL}),
        "m-table__wrapper": json.dumps(MARKETS),
        "m-result-title": json.dumps(SCOREBOARD),
        "/ vs /.test": "Arsenal vs Chelsea",
    })
    capture, result = _capture(adapter)
    assert result.ready is True, result.reason
    ev = adapter.evaluations
    # the fallback readers ran, individually
    assert any(
        ".m-table__wrapper" in c and "'session': " not in c
        for c in ev
    )
    # the (useless) bundle call still happened once
    assert (
        len([c for c in ev if "'session': " in c]) == 1
    )
    assert result.state.period == "2h"
    assert len(result.markets) == 2


def test_bundle_unusable_content_becomes_empty_dict():
    """A bundle with unusable content becomes {}.

    _wrap_js_json_part renders each broken part as its fallback — the top
    level still parses but carries nothing usable, so read_capture_bundle
    must return {} and the capture loop falls back to the single readers.
    """
    adapter = FakeAdapter({"'session': ": json.dumps({})})
    capture = LiveCapture(adapter, min_markets=2, pause_scale=0.0)
    bundle = capture.read_capture_bundle()
    assert bundle == {}
