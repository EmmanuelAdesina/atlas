"""Regression: win-to-nil must be priced on BOTH sides.

`away_win_to_nil` carried a `sign` factor that inverted the final-score
comparison, so 'Yes' was always 0.0 and 'No' always 1.0 regardless of the
score. The loop therefore read p=1.0 on an away-win-to-nil 'No' at 0-0,
claimed a 24% edge against a 1.24 price, and placed real money on it
(2026-10-29, NK Brinje Grosuplje v Bravo Ljubljana).

These tests pin the value, not just "the pricer returned something":
the old bug produced a perfectly well-formed number.
"""

from datetime import datetime, timezone

import pytest

from atlas.live.math_engine import (
    build_remaining_score_matrix,
    compute_matrix_marginals,
    price_from_joint_matrix,
)
from atlas.live.observe import LiveMatchState, TeamStats


def _state(home_goals: int, away_goals: int, minute: float = 79.0) -> LiveMatchState:
    return LiveMatchState(
        match_id="test",
        home=TeamStats(name="Home", goals=home_goals),
        away=TeamStats(name="Away", goals=away_goals),
        period="2h",
        clock_minute=minute,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


@pytest.fixture(scope="module")
def pricing():
    mu_h, mu_a = 0.30, 0.30
    mat = build_remaining_score_matrix(mu_h, mu_a, max_goals=8)
    return mat, compute_matrix_marginals(mat), mu_h, mu_a


def _p(pricing, state, market_key, selection) -> float:
    mat, marg, mu_h, mu_a = pricing
    out = price_from_joint_matrix(
        mat, state, market_key, selection, mu_h, mu_a, marginals=marg
    )
    assert out is not None, f"{market_key}/{selection} was not priced"
    return out.p_win


def test_away_win_to_nil_is_not_degenerate_at_nil_nil(pricing):
    """At 0-0 the away side CAN still win to nil, so 'No' is not certain."""
    state = _state(0, 0)
    yes = _p(pricing, state, "away_win_to_nil", "Yes")
    no = _p(pricing, state, "away_win_to_nil", "No")
    assert yes > 0.05, f"away win to nil 'Yes' collapsed to {yes}"
    assert no < 0.95, f"away win to nil 'No' is a fake certainty at {no}"
    assert yes + no == pytest.approx(1.0)


def test_win_to_nil_sides_are_symmetric_at_nil_nil(pricing):
    """Equal intensities: both sides must price the same at 0-0."""
    state = _state(0, 0)
    assert _p(pricing, state, "home_win_to_nil", "Yes") == pytest.approx(
        _p(pricing, state, "away_win_to_nil", "Yes"), abs=1e-9
    )


def test_away_win_to_nil_is_impossible_once_the_home_side_scores(pricing):
    """Home already scored: away cannot win to nil, so 'No' is certain."""
    state = _state(1, 0)
    assert _p(pricing, state, "away_win_to_nil", "Yes") == pytest.approx(0.0)
    assert _p(pricing, state, "away_win_to_nil", "No") == pytest.approx(1.0)


def test_away_win_to_nil_when_away_leads_requires_a_home_shutout(pricing):
    """Away 1-0 up: 'away wins to nil' is exactly 'home does not score again'."""
    state = _state(0, 1)
    away_yes = _p(pricing, state, "away_win_to_nil", "Yes")
    assert 0.0 < away_yes < 1.0
    assert away_yes == pytest.approx(
        _p(pricing, state, "away_clean_sheet", "Yes"), abs=1e-9
    )
