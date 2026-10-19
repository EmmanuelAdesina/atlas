"""Tests for combination markets ('1X2 & Over/Under', 'O/U & GG/NG').

A combination settles on the JOINT outcome, so its probability is not the
product of the two marginals. These tests pin that down, because computing
it as a product is exactly the mistake that manufactures false edges.
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


def make_state(home_goals=0, away_goals=0, minute=70.0):
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=home_goals),
        away=TeamStats(name="Away", goals=away_goals),
        period="2h",
        clock_minute=minute,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


@pytest.fixture(scope="module")
def env():
    mat = build_remaining_score_matrix(MU_H, MU_A, max_goals=8)
    return mat, compute_matrix_marginals(mat)


def price(env, state, key, sel):
    mat, marg = env
    out = price_from_joint_matrix(
        mat, state, key, sel, MU_H, MU_A, marginals=marg
    )
    return None if out is None else out.p_win


def test_combination_is_not_the_product_of_marginals(env):
    """'Home & Over 2.5' must differ from P(home wins) * P(over 2.5).

    They differ because scoring more goals and winning are positively
    correlated. A product would overstate or understate the true joint
    probability depending on the direction of the correlation.
    """
    state = make_state(0, 0)
    joint = price(env, state, "combo_1x2__ou_2_5", "Home & Over")
    p_home = price(env, state, "1x2", "Home")
    p_over = price(env, state, "ou_2_5", "Over")
    assert joint is not None
    assert p_home is not None and p_over is not None
    assert joint != pytest.approx(p_home * p_over, abs=1e-6)
    # More goals is good for the home side, so the joint exceeds the product.
    assert joint > p_home * p_over


def test_combination_matches_the_joint_matrix(env):
    """The combination probability must be read straight off the matrix."""
    mat, _ = env
    state = make_state(0, 0)
    joint = price(env, state, "combo_1x2__ou_2_5", "Home & Over")
    expected = 0.0
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if i > j and (i + j) > 2.5:
                expected += mat[i][j]
    assert joint == pytest.approx(expected, abs=1e-12)


def test_already_settled_condition_collapses_to_the_other_component(env):
    """With the O/U already decided, the combination is the 1X2 alone."""
    state = make_state(1, 1)  # total already 2 > 1.5
    joint = price(env, state, "combo_1x2__ou_1_5", "Home & Over")
    p_home = price(env, state, "1x2", "Home")
    assert joint == pytest.approx(p_home, abs=1e-12)


def test_btts_already_satisfied_collapses_to_the_other_component(env):
    """Both teams already scored, so 'Yes' is certain."""
    state = make_state(1, 1)
    joint = price(env, state, "combo_1x2__btts", "Home & Yes")
    p_home = price(env, state, "1x2", "Home")
    assert joint == pytest.approx(p_home, abs=1e-12)


def test_mutually_exclusive_selections_partition_the_outcome(env):
    """Home/Draw/Away & Over must sum to P(Over)."""
    state = make_state(0, 0)
    p_over = price(env, state, "ou_2_5", "Over")
    total = sum(
        price(env, state, "combo_1x2__ou_2_5", f"{side} & Over")
        for side in ("Home", "Draw", "Away")
    )
    assert total == pytest.approx(p_over, abs=1e-12)


def test_unknown_selection_abstains_rather_than_mispricing(env):
    state = make_state(0, 0)
    assert price(env, state, "combo_1x2__ou_2_5", "Something Else & Over") is None
    assert price(env, state, "combo_1x2__ou_2_5", "Home") is None
    assert price(env, state, "combo_1x2__ou_2_5", "Home & Over & Draw") is None


def test_unknown_component_abstains(env):
    state = make_state(0, 0)
    assert price(env, state, "combo_1x2__mystery", "Home & Over") is None


def test_combination_probability_is_a_probability(env):
    state = make_state(0, 0)
    for sel in ("Home & Over", "Draw & Over", "Away & Under", "Home & Yes"):
        p = price(env, state, "combo_1x2__ou_2_5", sel)
        if p is not None:
            assert 0.0 <= p <= 1.0
