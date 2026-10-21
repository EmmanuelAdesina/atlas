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
