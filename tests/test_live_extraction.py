"""Regression tests for live-board extraction and 1UP/2UP pricing.

These bugs were found by running ATLAS against the real SportyBet live board:
every selection abstained (period label '2nd' was unreadable), the score was
read as the clock (73-2), and 1UP/2UP pricing ignored the current scoreline
deficit, manufacturing a +5.3 EV phantom edge at 47.00.
"""

import pytest
from datetime import datetime, timezone

from atlas.live.estimate import LiveStateSpaceModel
from atlas.live.extract import extract_clock_score, extract_score_from_dom
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.live.math_engine import price_from_joint_matrix, compute_matrix_marginals


# --- Score vs clock -------------------------------------------------------

def test_split_clock_is_not_read_as_a_score():
    """'73' ':' '52' is a clock, not a 73-2 score."""
    texts = [
        "Al-Raed Club vs AL Saqer FC",
        "Statistics",
        "2nd",
        "|",
        "73",
        ":",
        "52",
        "0",
        ":",
        "2",
    ]
    score = extract_score_from_dom(texts)
    assert (score["home_goals"], score["away_goals"]) == (0, 2)


def test_single_token_clock_is_not_read_as_a_score():
    texts = [
        "Al-Raed Club vs AL Saqer FC",
        "2nd",
        "|",
        "75:11",
        "0",
        ":",
        "2",
    ]
    score = extract_score_from_dom(texts)
    assert (score["home_goals"], score["away_goals"]) == (0, 2)


def test_split_clock_is_read_as_the_clock():
    texts = ["A vs B", "2nd", "|", "73", ":", "52", "0", ":", "2"]
    info = extract_clock_score(texts)
    assert info["clock_minute"] == pytest.approx(73 + 52 / 60.0)
    assert info["clock_label"] == "73:52"


def test_period_read_from_bare_second_half_label():
    texts = ["A vs B", "2nd", "|", "75:11", "0", ":", "2"]
    info = extract_clock_score(texts)
    assert info["period"] == "2h"


def test_bare_period_label_is_not_trusted_away_from_the_clock():
    """'2nd' elsewhere on the page (e.g. a table header) must not set period."""
    texts = ["A vs B", "2nd", "Some market", "0", ":", "2", "75:11"]
    info = extract_clock_score(texts)
    assert info["period"] is None


def test_board_card_clock_does_not_steal_the_panel_score():
    """The first live card's clock/score must not override the viewed match.

    Board cards render the two team names separately, so the panel is the
    first header containing ' vs '.
    """
    texts = [
        "Other FC",
        "45:00",
        "1",
        ":",
        "0",
        "Another FC",
        "Al-Raed Club vs AL Saqer FC",
        "2nd",
        "|",
        "81:45",
        "0",
        ":",
        "2",
    ]
    score = extract_score_from_dom(texts)
    assert (score["home_goals"], score["away_goals"]) == (0, 2)


# --- 1UP / 2UP pricing must respect the scoreline deficit -----------------

def _state(home_goals: int, away_goals: int, minute: float = 84.0):
    from datetime import datetime, timezone

    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=home_goals),
        away=TeamStats(name="Away", goals=away_goals),
        period="2h",
        clock_minute=minute,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


def _prob(mkey: str, selection: str, state) -> float:
    from datetime import datetime, timezone

    model = LiveStateSpaceModel(now=datetime.now(timezone.utc))
    est = model.estimate(state, mkey, selection)
    assert est.p is not None, f"model abstained on {mkey}/{selection}"
    return est.p


def test_1up_leading_side_cannot_be_a_big_edge():
    """Home 0-2 down at 84': leading by 1+ is nearly impossible.

    Regression: the 1UP probability ignored the deficit and summed the
    remaining-goal diff directly, giving ~13% at 47.00 (EV +5.3).
    """
    state = _state(0, 2)
    p = _prob("1x2_1up", "Home", state)
    assert p < 0.01, f"1UP Home probability {p:.4f} is implausible when 0-2 down"
    # A 47.00 shot must not look like value.
    assert p * 47.0 - 1.0 < 0.05


def test_1up_trail_side_probability_is_not_symmetrically_overstated():
    """A side that is behind by one should not read as ~50% to lead by 1."""
    state = _state(0, 1)
    p = _prob("1x2_1up", "Home", state)
    assert p < 0.20, f"1UP Home probability {p:.4f} too high when 0-1 down"


def test_2up_trail_side_cannot_be_a_big_edge():
    state = _state(0, 2)
    p = _prob("1x2_2up", "Home", state)
    assert p < 0.01, f"2UP Home probability {p:.4f} is implausible when 0-2 down"


def test_1up_leading_side_is_already_won():
    state = _state(2, 0)
    p = _prob("1x2_1up", "Home", state)
    assert p == pytest.approx(1.0)


# --- Readiness gate ------------------------------------------------------

def test_panel_is_not_ready_without_a_live_period():
    from atlas.live.extract import panel_is_ready

    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="A", goals=0),
        away=TeamStats(name="B", goals=0),
        period="unknown",
        clock_minute=70.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )
    ready, reason = panel_is_ready(state, [{"status": "open"}] * 10)
    assert ready is False
    assert "period" in reason


def test_panel_is_not_ready_without_a_score():
    from atlas.live.extract import panel_is_ready

    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="A"),
        away=TeamStats(name="B", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )
    ready, reason = panel_is_ready(state, [{"status": "open"}] * 10)
    assert ready is False
    assert "score" in reason


def test_panel_is_not_ready_without_enough_markets():
    from atlas.live.extract import panel_is_ready

    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="A", goals=1),
        away=TeamStats(name="B", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )
    ready, reason = panel_is_ready(state, [{"status": "open"}] * 3, min_markets=8)
    assert ready is False
    assert "market" in reason


def test_panel_is_ready_when_complete():
    from atlas.live.extract import panel_is_ready

    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="A", goals=1),
        away=TeamStats(name="B", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )
    ready, reason = panel_is_ready(state, [{"status": "open"}] * 20)
    assert ready is True
    assert reason == "ready"


# --- Calibration: Rest-of-Match is only the same event at a level score ---

def test_rest_of_match_is_not_used_to_calibrate_a_non_level_1x2():
    """'Rest of the Match (current score 1:0)' settles on the REMAINING time
    treated as a fresh match, so it is not the same event as the full-time
    1X2 unless the score is level.

    Calibrating across the two made the model believe a side had ~9% to win
    the full-time 1X2 at 65.00 — a +4.5 EV phantom edge.
    """
    from datetime import datetime, timezone

    from atlas.live.estimate import CalibratedModel, LiveStateSpaceModel
    from atlas.live.math_engine import devig_shin
    from atlas.live.observe import LiveMatchState, TeamStats

    # The live board's prices for a 1-0 match at 68'.
    full, _ = devig_shin({"Home": 1.07, "Draw": 6.5, "Away": 65.0})
    rest, _ = devig_shin({"Home": 2.6, "Draw": 1.7, "Away": 8.25})
    market_fair = {("1x2", k): v for k, v in full.items()}
    market_fair.update({("rest_of_match_1x2", k): v for k, v in rest.items()})

    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=1),
        away=TeamStats(name="Away", goals=0),
        period="2h",
        clock_minute=68.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )
    model = CalibratedModel(
        base_model=LiveStateSpaceModel(now=datetime.now(timezone.utc)),
        market_fair=market_fair,
    )
    est = model.estimate(state, "1x2", "Away")
    assert est.p is not None
    # The away side's chance of winning at 1-0 down is small, and the
    # calibrated probability must be in the same league as the book's own.
    assert est.p < 0.05, f"calibrated p {est.p:.4f} is far above the market's {full['Away']:.4f}"
    ev = est.p * 65.0 - 1.0
    assert ev < 0.05, f"phantom edge survived: EV {ev:+.3f} at 65.00"


def test_rest_of_match_is_used_when_the_score_is_level():
    """At 0-0 the remaining time IS the match, so the two are the same event."""
    from datetime import datetime, timezone

    from atlas.live.estimate import CalibratedModel, LiveStateSpaceModel
    from atlas.live.math_engine import devig_shin
    from atlas.live.observe import LiveMatchState, TeamStats

    full, _ = devig_shin({"Home": 2.2, "Draw": 3.4, "Away": 3.2})
    rest, _ = devig_shin({"Home": 2.0, "Draw": 3.3, "Away": 3.3})
    market_fair = {("1x2", k): v for k, v in full.items()}
    market_fair.update({("rest_of_match_1x2", k): v for k, v in rest.items()})
    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=0),
        away=TeamStats(name="Away", goals=0),
        period="2h",
        clock_minute=68.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )
    model = CalibratedModel(
        base_model=LiveStateSpaceModel(now=datetime.now(timezone.utc)),
        market_fair=market_fair,
    )
    est = model.estimate(state, "1x2", "Home")
    assert est.p is not None
    # Shrunk toward the rest-of-match prior, which is between the model's
    # own probability and the market's.
    assert est.p != full["Home"]
