"""The same-match portfolio math (atlas.ops.portfolio).

Ground rules established against real book behaviour:
  - covering every outcome of ONE partitioning market on ONE book loses the
    overround exactly (sum(1/odds) > 1 -> guaranteed negative P&L);
  - profit exists only when a cross-market cover beats the boot (sum < 1)
    -- those are the windows the live scanner hunts;
  - the stake split is closed-form, and the floor must be verified before
    anything moves.
"""

import pytest

from atlas.ops.portfolio import (
    Leg,
    best_equal_floor_split,
    cover_boot,
    equal_return_cover,
    evaluate_portfolio,
)


def test_one_market_covered_on_one_book_loses_the_margin():
    """1X2 at a realistic live book: 2.10 / 3.40 / 3.20 -- the overround."""
    boot = cover_boot([2.10, 3.40, 3.20])
    # Boot > 1 -> the equal cover is a guaranteed LOSS of ~7.6%, no slicing.
    assert boot > 1.0
    plan = equal_return_cover(["Home", "Draw", "Away"], [2.10, 3.40, 3.20], budget=300)
    assert plan.guaranteed_pnl == pytest.approx(-22.94, abs=0.1)


def test_cross_market_lag_arb_is_detected_and_sized():
    """When different market families disagree (price lag), sum < 1:
    2.20 beats 2.10 fair-probability home, 3.50 draw vs 3.40, 3.30 away...
    here a genuine arb: 0.4545 + 0.2857 + 0.3030 = 1.0432 -> still positive.
    Use a REAL arb shape instead: Home 2.30, Draw 3.80, Away 3.90
    = 0.4348 + 0.2632 + 0.2564 = 0.9544 -> ROI ~ +4.8%."""
    odds = [2.30, 3.80, 3.90]
    boot = cover_boot(odds)
    assert boot < 1.0
    plan = equal_return_cover(["Home", "Draw", "Away"], odds, budget=1000)
    assert plan.guaranteed_pnl > 0
    assert plan.roi == pytest.approx(1.0 / boot - 1.0, abs=1e-9)
    # every stake >= 0 and sums to the budget
    assert sum(plan.stakes) == pytest.approx(1000.0)
    assert all(s >= 0 for s in plan.stakes)


def test_outcome_matrix_quantifies_a_genuine_middle():
    """Over 1.5 @ 1.45 vs Under 3.5 @ 1.30 on total goals {0,1,2,3,4+}:
    - goals 0/1: Over loses, Under wins
    - goals 2:   BOTH win (the middle!) -> the tail jackpot
    - goals 3/4+: Over only.
    Both legs on one book pricing 1.45 & 1.30 -> boot = 1.689 > 1, so with
    any stake vector the AVERAGE outcome is negative; verify the matrix
    reports exactly which scores pay what for a split that maximises the
    middle.""
    """
    outcomes = ["0-1", "2", "3", "4+"]
    legs = [
        Leg(label="Over 1.5", odds=1.45, covers=("2", "3", "4+")),
        Leg(label="Under 3.5", odds=1.30, covers=("0-1", "2", "3")),
    ]
    ev = evaluate_portfolio(legs, [50.0, 50.0], outcomes)
    assert ev.total_stake == pytest.approx(100.0)
    # 0-1 goals: only Under 3.5 wins
    assert ev.pnl_by_outcome["0-1"] == pytest.approx(50 * 1.30 - 100)
    # 2 goals: BOTH win -- the middle jackpot
    assert ev.pnl_by_outcome["2"] == pytest.approx(50 * 1.45 + 50 * 1.30 - 100)
    # 3 goals: also BOTH win (Under 3.5 pays at 0..3 goals) -- wider middle
    assert ev.pnl_by_outcome["3"] == pytest.approx(50 * 1.45 + 50 * 1.30 - 100)
    # 4+ goals: only Over 1.5 wins
    assert ev.pnl_by_outcome["4+"] == pytest.approx(50 * 1.45 - 100)
    # the both-legs-win tail is the best the portfolio can ever do
    assert ev.best_pnl == ev.pnl_by_outcome["2"]


def test_best_equal_floor_split_only_solves_partitions():
    """The middle pair does not partition -> the LP is not faked."""
    legs = [
        Leg(label="Over 1.5", odds=1.45, covers=("2", "3", "4+")),
        Leg(label="Under 3.5", odds=1.30, covers=("0-1", "2", "3")),
    ]
    assert best_equal_floor_split(legs, ["0-1", "2", "3", "4+"], 100) is None
    # A strict partition solves to the exact equal return
    p = [
        Leg(label="Home", odds=2.30, covers=("H",)),
        Leg(label="Draw", odds=3.80, covers=("D",)),
        Leg(label="Away", odds=3.90, covers=("A",)),
    ]
    sol = best_equal_floor_split(p, ["H", "D", "A"], 1000)
    assert sol is not None
    assert all(v == pytest.approx(1000 / cover_boot([2.30, 3.80, 3.90]) - 1000) for v in sol.pnl_by_outcome.values())


def test_equal_return_cover_rejects_impossible_and_empty_inputs():
    with pytest.raises(ValueError):
        cover_boot([])
    with pytest.raises(ValueError):
        cover_boot([1.0, 2.0])  # decimal 1.00 is a not-a-price leg
    with pytest.raises(ValueError):
        equal_return_cover(["A"], [2.0], budget=0)
