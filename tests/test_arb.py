"""Tests for the sure-thing arbitrage scanner (atlas.live.arb)."""

import pytest

from atlas.live.arb import (
    Selection,
    best_arbs,
    find_cross_family_arbs,
    find_surething_arbs,
)


def sel(market_key, selection, odds, model_p):
    return Selection(market_key, selection, odds, model_p)


def test_finds_an_arb_when_inverse_prices_sum_below_one():
    sels = [
        sel("1x2", "Home", 2.20, 0.45),
        sel("1x2", "Draw", 3.80, 0.28),
        sel("1x2", "Away", 3.80, 0.27),
    ]
    arbs = find_surething_arbs(sels)
    assert len(arbs) == 1
    arb = arbs[0]
    assert arb.implied_total < 1.0
    assert arb.profit_pct > 0.0
    # 1/2.20 + 1/3.80 + 1/3.80 = 0.9809 -> a 1.9% sure-thing
    assert arb.implied_total == pytest.approx(1 / 2.20 + 1 / 3.80 + 1 / 3.80)
    assert arb.profit_pct == pytest.approx(1 - arb.implied_total)


def test_no_arb_when_the_book_has_a_margin():
    sels = [
        sel("1x2", "Home", 12.50, 0.08),
        sel("1x2", "Draw", 1.14, 0.80),
        sel("1x2", "Away", 9.00, 0.12),
    ]
    assert find_surething_arbs(sels) == []


def test_arb_stakes_guarantee_the_return():
    sels = [
        sel("1x2", "Home", 2.20, 0.45),
        sel("1x2", "Draw", 3.80, 0.28),
        sel("1x2", "Away", 3.80, 0.27),
    ]
    arb = find_surething_arbs(sels)[0]
    total_stake = sum(s for _, s, _ in arb.stakes())
    payouts = {round(p, 10) for _, _, p in arb.stakes()}
    # Whatever the outcome, one leg pays out and it always covers the stake.
    assert len(payouts) == 1
    payout = payouts.pop()
    assert payout == pytest.approx(1.0, abs=1e-9)
    assert total_stake < payout


def test_model_must_believe_the_family_is_exhaustive():
    """A family the model does not believe is exhaustive cannot carry an arb."""
    sels = [
        sel("1x2", "Home", 2.10, 0.10),
        sel("1x2", "Draw", 3.60, 0.10),
        sel("1x2", "Away", 3.60, 0.10),
    ]
    assert find_surething_arbs(sels) == []


def test_non_exhaustive_families_are_ignored():
    # A single Over/Under selection is not a covering set.
    sels = [sel("ou_2_5", "Over 2.5", 1.20, 0.55)]
    assert find_surething_arbs(sels) == []


def test_cross_family_arb_needs_complementary_legs():
    """1X2 Home and 1X2 Away are not complementary; Home and Draw are not
    either. Two legs cover the space only when their model probabilities
    sum to one."""
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
        sel("double_chance", "Home or Draw", 1.30, 0.73),
        sel("double_chance", "Draw or Away", 1.30, 0.55),
        sel("double_chance", "Home or Away", 6.00, 0.72),
    ]
    arbs = find_cross_family_arbs(sels)
    for arb in arbs:
        # Every reported cross-family arb must be genuinely complementary.
        assert arb.model_sum == pytest.approx(1.0, abs=0.02)
        assert arb.implied_total < 1.0


def test_cross_family_arb_is_found_when_families_disagree():
    """A 1X2 and a Double Chance priced inconsistently leak a sure-thing."""
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
        # 'Draw or Away' complements 'Home': 0.55 + 0.45 = 1.0
        sel("double_chance", "Draw or Away", 1.95, 0.55),
        sel("double_chance", "Home or Draw", 1.30, 0.73),
        sel("double_chance", "Home or Away", 6.00, 0.72),
    ]
    arbs = find_cross_family_arbs(sels)
    assert arbs, "expected a cross-family arb from inconsistent pricing"
    best = arbs[0]
    assert best.implied_total < 1.0
    assert best.profit_pct > 0.0


def test_best_arbs_is_sorted_by_profit():
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
    ]
    arbs = best_arbs(sels)
    profits = [a.profit_pct for a in arbs]
    assert profits == sorted(profits, reverse=True)


def test_min_edge_filters_out_marginal_arbs():
    sels = [
        sel("1x2", "Home", 3.00, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
    ]
    # 1/3 + 1/3.6 + 1/3.6 = 0.8889 -> a large arb, definitely reported.
    assert find_surething_arbs(sels, min_edge=0.05)
