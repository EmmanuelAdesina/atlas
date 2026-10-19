"""Tests for the remaining live-board families: GG/NG 2+, no-draw BTTS and
the period-scoped parity markets.
"""

from datetime import datetime, timezone

import pytest

from atlas.live.math_engine import (
    build_remaining_score_matrix,
    compute_matrix_marginals,
    price_from_joint_matrix,
)
from atlas.live.observe import LiveMatchState, TeamStats

MU_H, MU_A = 0.45, 0.35


@pytest.fixture(scope="module")
def env():
    mat = build_remaining_score_matrix(MU_H, MU_A, max_goals=8)
    return mat, compute_matrix_marginals(mat)


def state(home=0, away=0, ht_h=None, ht_a=None, period="2h"):
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=home),
        away=TeamStats(name="Away", goals=away),
        period=period,
        clock_minute=70.0,
        half_home_goals=ht_h,
        half_away_goals=ht_a,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


def price(env, st, key, sel):
    mat, marg = env
    out = price_from_joint_matrix(
        mat, st, key, sel, MU_H, MU_A, marginals=marg
    )
    return None if out is None else out.p_win


# --- GG/NG 2+ ------------------------------------------------------------

def test_btts_2plus_needs_two_each(env):
    """With one goal each, both sides need one more."""
    st = state(home=1, away=1)
    p = price(env, st, "btts_2plus", "Yes")
    assert p is not None
    assert 0.0 < p < 0.5


def test_btts_2plus_complements_itself(env):
    st = state(home=0, away=0)
    yes = price(env, st, "btts_2plus", "Yes")
    no = price(env, st, "btts_2plus", "No")
    assert yes + no == pytest.approx(1.0, abs=1e-12)


def test_btts_2plus_is_already_satisfied(env):
    st = state(home=2, away=2)
    assert price(env, st, "btts_2plus", "Yes") == pytest.approx(1.0, abs=1e-12)


# --- BTTS with no draw ---------------------------------------------------

def test_btts_no_draw_excludes_the_draw(env):
    st = state(home=0, away=0)
    p = price(env, st, "btts_no_draw", "Yes")
    assert p is not None
    assert 0.0 < p < 1.0


def test_btts_no_draw_is_zero_when_neither_team_has_scored(env):
    """With nobody scored, the 'both score' half of the market is already lost
    unless both score in the remaining time."""
    st = state(home=0, away=0)
    p = price(env, st, "btts_no_draw", "Yes")
    assert p is not None
    assert p < 0.5


def test_btts_no_draw_is_still_below_one_when_the_score_is_level(env):
    """1-1 with time left: both have scored, but the match can still end in a
    draw, so the 'no draw' condition can still fail."""
    st = state(home=1, away=1)
    p = price(env, st, "btts_no_draw", "Yes")
    assert p is not None
    assert 0.0 < p < 1.0
    # The complement sums to one.
    assert p + price(env, st, "btts_no_draw", "No") == pytest.approx(1.0, abs=1e-12)


# --- period-scoped parity ------------------------------------------------

def test_second_half_parity_uses_the_second_half_score(env):
    """At 1-1 with HT 0-1, one goal has been scored in the second half."""
    st = state(home=1, away=1, ht_h=0, ht_a=1)
    # The second half has 1 goal so far, which is odd.
    p_odd = price(env, st, "odd_even_2h", "Odd")
    p_even = price(env, st, "odd_even_2h", "Even")
    assert p_odd + p_even == pytest.approx(1.0, abs=1e-12)
    # One more goal makes it even, so Odd is not certain.
    assert p_odd < 1.0
    assert p_odd > p_even


def test_second_half_parity_needs_the_half_time_score(env):
    st = state(home=1, away=1, ht_h=None, ht_a=None)
    assert price(env, st, "odd_even_2h", "Odd") is None


def test_full_match_parity_differs_from_the_second_half(env):
    st = state(home=1, away=1, ht_h=0, ht_a=1)
    full = price(env, st, "odd_even", "Odd")
    second = price(env, st, "odd_even_2h", "Odd")
    assert full != pytest.approx(second, abs=1e-9)


def test_team_parity_is_scoped_to_that_team(env):
    st = state(home=1, away=1, ht_h=0, ht_a=1)
    home = price(env, st, "home_odd_even_2h", "Odd")
    away = price(env, st, "away_odd_even_2h", "Odd")
    assert home is not None and away is not None
    assert home != pytest.approx(away, abs=1e-9)
