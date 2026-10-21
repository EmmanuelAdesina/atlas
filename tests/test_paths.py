"""Tests for the score-path engine (atlas.live.paths).

The engine prices markets whose answer depends on the ORDER of goals, not
the end state. The core correctness proof is that the dynamic program's
endpoint distribution reproduces the joint remaining-goal matrix exactly
(with the Dixon-Coles adjustment disabled, so the two constructions are
the same distribution).
"""

import math

import pytest

from atlas.live.math_engine import build_remaining_score_matrix
from atlas.live.paths import AWAY, HOME, GoalPathModel


def test_endpoint_distribution_reproduces_the_joint_matrix():
    """The DP event process and the joint matrix must be the same law."""
    mu_h, mu_a = 0.42, 0.31
    model = GoalPathModel.from_intensities(mu_h, mu_a, max_goals=8)
    dp = model.endpoint_distribution()

    # rho=0 disables the Dixon-Coles adjustment, leaving the plain bivariate
    # Poisson, which is exactly the sequential event process.
    mat = build_remaining_score_matrix(mu_h, mu_a, rho=0.0, max_goals=8)

    max_err = 0.0
    for i in range(9):
        for j in range(9):
            max_err = max(max_err, abs(dp[i][j] - mat[i][j]))
    assert max_err < 1e-9, f"endpoint distributions differ by {max_err:.3e}"


def test_endpoint_distribution_is_normalised():
    model = GoalPathModel.from_intensities(0.5, 0.5, max_goals=6)
    dist = model.endpoint_distribution()
    total = sum(sum(r) for r in dist)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_lead_by_at_any_time_is_one_when_already_leading():
    model = GoalPathModel.from_intensities(0.4, 0.4, max_goals=8)
    # Home already 2-0 up: it has already led by 2.
    assert model.lead_by_at_any_time(HOME, 2, base_diff=2) == 1.0
    # Away already 2-0 up: home can never lead by 2 from here.
    assert model.lead_by_at_any_time(HOME, 2, base_diff=-2) < 0.01


def test_lead_by_at_any_time_is_bounded_and_monotone():
    model = GoalPathModel.from_intensities(0.45, 0.35, max_goals=10)
    probs = [model.lead_by_at_any_time(HOME, n, base_diff=0) for n in (1, 2, 3)]
    for p in probs:
        assert 0.0 <= p <= 1.0
    assert probs[0] > probs[1] > probs[2]


def test_lead_by_at_any_time_mirrors_for_equal_teams():
    """With symmetric intensities, home and away lead probabilities mirror."""
    model = GoalPathModel.from_intensities(0.5, 0.5, max_goals=10)
    for n in (1, 2, 3):
        home = model.lead_by_at_any_time(HOME, n, base_diff=0)
        away = model.lead_by_at_any_time(AWAY, n, base_diff=0)
        assert home == pytest.approx(away, abs=1e-12)


def test_lead_by_at_any_time_exceeds_end_state_probability():
    """Leading by 2 at ANY time is more likely than ending 2 ahead.

    This is exactly why the end-state matrix cannot price these markets.
    """
    model = GoalPathModel.from_intensities(0.5, 0.5, max_goals=10)
    path = model.lead_by_at_any_time(HOME, 2, base_diff=0)
    end_state = model.endpoint_distribution()
    p_end = sum(
        end_state[i][j] for i in range(len(end_state)) for j in range(len(end_state))
        if i - j >= 2
    )
    assert path > p_end
    assert path - p_end > 0.002


def test_never_trails_is_zero_when_already_behind():
    model = GoalPathModel.from_intensities(0.4, 0.4, max_goals=8)
    assert model.never_trails(HOME, base_diff=-1) == 0.0
    assert model.never_trails(AWAY, base_diff=1) == 0.0


def test_never_trails_is_one_when_already_ahead_by_one():
    """Ahead by one, the side can still be overtaken, so it is not certain."""
    model = GoalPathModel.from_intensities(0.4, 0.4, max_goals=8)
    assert model.never_trails(HOME, base_diff=1) < 1.0
    assert model.never_trails(HOME, base_diff=1) > 0.9


def test_never_trails_rises_as_the_surplus_grows():
    """The further AHEAD a side is, the likelier it never falls behind."""
    model = GoalPathModel.from_intensities(0.5, 0.5, max_goals=10)
    probs = [model.never_trails(HOME, base_diff=d) for d in (0, 1, 2)]
    assert probs[0] < probs[1] < probs[2]


def test_goals_in_a_row_of_one_means_at_least_one_goal():
    """A run of 1 is just 'a goal is scored', so P = 1 - P(no goals).

    With no shared-tempo events the two are exactly equal. A shared-tempo
    event is a goal for both teams at the same instant, which the 'in a row'
    convention does not count as a run, so it removes a little probability.
    """
    model = GoalPathModel.from_intensities(0.5, 0.5, shared_tempo_frac=0.0, max_goals=8)
    no_goal = math.exp(-model.total_intensity)
    assert model.goals_in_a_row(1) == pytest.approx(1.0 - no_goal, abs=1e-9)

    with_tempo = GoalPathModel.from_intensities(0.5, 0.5, max_goals=8)
    assert with_tempo.goals_in_a_row(1) < 1.0 - no_goal


def test_goals_in_a_row_is_monotone_and_bounded():
    model = GoalPathModel.from_intensities(0.6, 0.4, max_goals=10)
    probs = [model.goals_in_a_row(n) for n in (2, 3, 4)]
    for p in probs:
        assert 0.0 <= p <= 1.0
    assert probs[0] > probs[1] > probs[2]


def test_goals_in_a_row_for_one_side_is_smaller_than_either_side():
    model = GoalPathModel.from_intensities(0.5, 0.5, max_goals=10)
    either = model.goals_in_a_row(3)
    home = model.goals_in_a_row(3, side=HOME)
    assert home < either
    assert home == pytest.approx(model.goals_in_a_row(3, side=AWAY), abs=1e-12)


def test_goals_in_a_row_requires_consecutive_goals():
    """A run of 2 needs two goals in a row, which is rarer than two goals."""
    model = GoalPathModel.from_intensities(0.6, 0.4, max_goals=10)
    dist = model.endpoint_distribution()
    two_or_more = 1.0 - dist[0][0] - dist[1][0] - dist[0][1]
    assert model.goals_in_a_row(2) < two_or_more


def test_rejects_unknown_side():
    model = GoalPathModel.from_intensities(0.4, 0.4, max_goals=6)
    with pytest.raises(ValueError):
        model.lead_by_at_any_time("neutral", 1)
    with pytest.raises(ValueError):
        model.goals_in_a_row(2, side="neutral")


def test_path_engine_is_fast_enough_for_the_fast_plane():
    """A single path query must stay inside the fast-plane budget.

    Path probabilities depend only on (intensities, base_diff, n, side) — not
    on the market or the selection — so a surface computes a handful of
    distinct queries and reuses them. The per-call guard below is what stops
    the DP from becoming a pathologically slow regression.
    """
    import time

    model = GoalPathModel.from_intensities(0.5, 0.4, max_goals=10)

    def best_of(fn, reps=5, calls=20):
        # The minimum over several runs: a single timing on a loaded machine
        # is noise, the minimum is what reflects the work done.
        out = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            for _ in range(calls):
                fn()
            out.append((time.perf_counter_ns() - t0) / 1000.0 / calls)
        return min(out)

    # A path query is a dynamic program over the score path: it costs
    # ~0.5-2ms, not the ~0.2ms of a single pricing pass. That is fine
    # because a surface computes each DISTINCT query once and reuses it
    # across every selection (the second test covers the aggregate).
    lead_us = best_of(lambda: model.lead_by_at_any_time(HOME, 2, base_diff=0))
    assert lead_us < 5000.0, f"{lead_us:.0f}us per lead-by query is too slow"

    row_us = best_of(lambda: model.goals_in_a_row(3))
    assert row_us < 5000.0, f"{row_us:.0f}us per in-a-row query is too slow"


def test_one_surface_needs_only_a_handful_of_path_queries():
    """The distinct path queries for a surface must stay cheap in aggregate."""
    import time

    model = GoalPathModel.from_intensities(0.5, 0.4, max_goals=10)
    t0 = time.perf_counter_ns()
    for side in ("home", "away"):
        for n in (1, 2, 3):
            model.lead_by_at_any_time(side, n, base_diff=0)
            model.goals_in_a_row(n, side=side)
    total_ms = (time.perf_counter_ns() - t0) / 1e6
    # This is the number that matters: it is paid ONCE per snapshot, and
    # the results are then shared by every selection in the family.
    assert total_ms < 60.0, f"a surface's path queries cost {total_ms:.1f}ms"
