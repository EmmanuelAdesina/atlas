"""Regression tests for live-board extraction and 1UP/2UP pricing.

These bugs were found by running ATLAS against the real SportyBet live board:
every selection abstained (period label '2nd' was unreadable), the score was
read as the clock (73-2), and 1UP/2UP pricing ignored the current scoreline
deficit, manufacturing a +5.3 EV phantom edge at 47.00.
"""

import pytest

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
