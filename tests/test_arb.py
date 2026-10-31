"""Tests for the sure-thing arbitrage scanner (atlas.live.arb).

The scanner is allowed to miss arbs; it is NOT allowed to invent them. A
false "sure thing" invites the operator to size it like a hedge, so the
tests that matter most are the ones that pin the refusals:

  - a model probability sum of 1.0 does not make two events complementary
    (2026-10-28: btts No + rest_of_match_1x2 Home, "implied 0.8667");
  - an implied total far below one is a specification error, not a leak;
  - a leg's identity is (market_key, selection), never the selection text.
"""

import pytest

from atlas.live.arb import (
    MIN_PLAUSIBLE_IMPLIED,
    Selection,
    are_complementary,
    best_arbs,
    find_cross_family_arbs,
    find_surething_arbs,
    normalized_identity_set,
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


# --- structural complementarity -------------------------------------------


def test_one_x_two_side_is_complemented_by_the_covering_double_chance():
    assert are_complementary("1x2", "Home", "double_chance", "Draw or Away")
    assert are_complementary("1x2", "Away", "double_chance", "Home or Draw")
    assert are_complementary("1x2", "Draw", "double_chance", "Home or Away")
    # Scope is part of the relation: a first-half side is not covered by a
    # full-match double chance, and a rest-of-match side is not either.
    assert not are_complementary("1x2_1h", "Home", "double_chance", "Draw or Away")
    assert not are_complementary(
        "rest_of_match_1x2", "Home", "double_chance", "Draw or Away"
    )


def test_overlapping_selections_are_not_complementary():
    # 'Home' and 'Home or Draw' both win on a home win: overlap, not a
    # partition.
    assert not are_complementary("1x2", "Home", "double_chance", "Home or Draw")
    # Two sides of the same 1X2 are not a covering set either.
    assert not are_complementary("1x2", "Home", "1x2", "Away")


def test_btts_and_rest_of_match_are_not_complementary():
    """The 2026-10-28 false positive, pinned exactly.

    'btts No' (0.9308) + 'rest_of_match_1x2 Home' (0.0693) summed to
    1.0001, so the old probability-sum test admitted them. The events
    overlap (home scores, away does not: both win) and leave a gap (away
    scores, home does not: both lose).
    """
    assert not are_complementary("btts", "No", "rest_of_match_1x2", "Home")

    sels = [
        sel("btts", "No", 1.25, 0.9308),
        sel("btts", "Yes", 3.55, 0.0692),
        sel("rest_of_match_1x2", "Home", 15.00, 0.0693),
        sel("rest_of_match_1x2", "Draw", 1.30, 0.8700),
        sel("rest_of_match_1x2", "Away", 3.70, 0.0607),
    ]
    # The pair's implied total is still the number that used to fire...
    assert 1 / 1.25 + 1 / 15.00 == pytest.approx(0.8667, abs=1e-4)
    # ...and the scanner now refuses it.
    assert find_cross_family_arbs(sels) == []
    assert best_arbs(sels) == []


def test_cross_family_arb_needs_complementary_legs():
    """Only structurally complementary pairs may be reported, and only when
    the model agrees they cover the space."""
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
        sel("double_chance", "Home or Draw", 1.30, 0.73),
        sel("double_chance", "Draw or Away", 1.30, 0.55),
        sel("double_chance", "Home or Away", 6.00, 0.72),
    ]
    for arb in find_cross_family_arbs(sels):
        assert arb.model_sum == pytest.approx(1.0, abs=0.02)
        assert arb.implied_total < 1.0
        assert arb.implied_total >= MIN_PLAUSIBLE_IMPLIED


def test_cross_family_arb_is_found_when_families_disagree():
    """A 1X2 and a Double Chance priced inconsistently leak a sure-thing."""
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
        # 'Draw or Away' complements 'Home': 0.55 + 0.45 = 1.0
        # 1/2.10 + 1/1.95 = 0.9890 -> a plausible 1.1% leak.
        sel("double_chance", "Draw or Away", 1.95, 0.55),
        sel("double_chance", "Home or Draw", 1.30, 0.73),
        sel("double_chance", "Home or Away", 6.00, 0.72),
    ]
    arbs = find_cross_family_arbs(sels)
    assert arbs, "expected a cross-family arb from inconsistent pricing"
    best = arbs[0]
    assert best.implied_total < 1.0
    assert best.profit_pct > 0.0


# --- plausibility floor ----------------------------------------------------


def test_implausibly_deep_underround_is_rejected():
    """A 40% underround on a 'complementary' pair is a bad price read, not
    free money. Both the single-family and cross-family paths refuse it."""
    single = [
        sel("1x2", "Home", 5.00, 0.45),
        sel("1x2", "Draw", 5.00, 0.28),
        sel("1x2", "Away", 5.00, 0.27),
    ]
    assert sum(1 / s.odds for s in single) < MIN_PLAUSIBLE_IMPLIED
    assert find_surething_arbs(single) == []

    cross = [
        sel("1x2", "Home", 5.00, 0.45),
        sel("double_chance", "Draw or Away", 2.50, 0.55),
    ]
    assert are_complementary("1x2", "Home", "double_chance", "Draw or Away")
    assert 1 / 5.00 + 1 / 2.50 < MIN_PLAUSIBLE_IMPLIED
    assert find_cross_family_arbs(cross) == []


def test_a_shallow_underround_just_above_the_floor_is_reported():
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("double_chance", "Draw or Away", 2.00, 0.55),
    ]
    # 1/2.10 + 1/2.00 = 0.9762 -> plausible, report it.
    assert 1 / 2.10 + 1 / 2.00 > MIN_PLAUSIBLE_IMPLIED
    arbs = find_cross_family_arbs(sels)
    assert len(arbs) == 1
    assert arbs[0].implied_total == pytest.approx(1 / 2.10 + 1 / 2.00)


def test_min_edge_filters_out_marginal_arbs():
    sels = [
        sel("1x2", "Home", 2.50, 0.45),
        sel("1x2", "Draw", 3.30, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
    ]
    # 1/2.5 + 1/3.3 + 1/3.6 = 0.9808 -> a 1.9% edge, above the floor.
    assert find_surething_arbs(sels, min_edge=0.005)
    # A 5% minimum edge must filter it out.
    assert find_surething_arbs(sels, min_edge=0.05) == []


# --- leg identity ----------------------------------------------------------


def test_identity_set_keys_legs_by_market_not_selection_text():
    """'No' exists in a dozen unrelated families; the key must include the
    market, or a superedge-only filter matches the wrong bets."""
    sels = [
        sel("btts", "No", 2.05, 0.50),
        sel("btts", "Yes", 2.05, 0.50),
    ]
    arb = find_surething_arbs(sels)[0]
    assert arb.identity_set() == {("btts", "No"), ("btts", "Yes")}
    assert all(leg.market_key == "btts" for leg in arb.legs)
    # A clean-sheet 'No' is a different bet and must not match.
    assert ("home_clean_sheet", "No") not in arb.identity_set()
    # The legacy 3-tuple stakes() shape is unchanged.
    assert all(len(row) == 3 for row in arb.stakes())


def test_normalized_identity_set_does_not_admit_a_name_only_match():
    """The superedge-only filter's actual contract.

    An arb on btts must not admit 'No' from any other family, and must
    admit its own legs even when case/whitespace differ.
    """
    arb = find_surething_arbs(
        [sel("btts", "No", 2.05, 0.50), sel("btts", "Yes", 2.05, 0.50)]
    )[0]
    allowed = normalized_identity_set([arb])
    assert allowed == {("btts", "no"), ("btts", "yes")}
    # The exact over-match the old name-only filter made:
    assert ("home_clean_sheet", "no") not in allowed
    assert ("away_win_to_nil", "no") not in allowed
    # Normalization: the board and the leg need not agree on case.
    assert ("btts", "  NO  ".strip().lower()) in allowed


def test_best_arbs_is_sorted_by_profit():
    sels = [
        sel("1x2", "Home", 2.10, 0.45),
        sel("1x2", "Draw", 3.60, 0.28),
        sel("1x2", "Away", 3.60, 0.27),
    ]
    arbs = best_arbs(sels)
    profits = [a.profit_pct for a in arbs]
    assert profits == sorted(profits, reverse=True)
