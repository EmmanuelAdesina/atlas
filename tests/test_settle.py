"""Tests for settlement from the final score (atlas.live.settle).

Settlement drives the P&L that drives the hard daily stop, so a wrong
settlement is a money bug, not a cosmetic one. Anything the rules cannot
decide must be UNKNOWN, never a guess.
"""

import pytest

from atlas.live.settle import LOST, UNKNOWN, VOID, WON, settle_from_final_score


def s(key, sel, h=1, a=0, stake=100.0, odds=2.0, **kw):
    return settle_from_final_score(key, sel, h, a, stake, odds, **kw)


# --- 1X2 -----------------------------------------------------------------

def test_1x2_home_wins():
    r = s("1x2", "Home", h=2, a=1)
    assert r.status == WON
    assert r.payout_ngn == pytest.approx(100.0)


def test_1x2_draw_is_a_loss():
    r = s("1x2", "Home", h=1, a=1)
    assert r.status == LOST


def test_1x2_away():
    r = s("1x2", "Away", h=0, a=3)
    assert r.status == WON


# --- Double Chance -------------------------------------------------------

def test_double_chance_home_or_draw():
    assert s("double_chance", "Home or Draw", h=1, a=1).status == WON
    assert s("double_chance", "Home or Draw", h=0, a=1).status == LOST
    assert s("double_chance", "Draw or Away", h=0, a=1).status == WON


# --- Draw No Bet ---------------------------------------------------------

def test_dnb_pushes_on_a_draw():
    r = s("dnb", "Home", h=1, a=1)
    assert r.status == VOID
    assert r.payout_ngn == pytest.approx(100.0)


def test_dnb_wins_and_loses():
    assert s("dnb", "Home", h=2, a=0).status == WON
    assert s("dnb", "Home", h=0, a=2).status == LOST


# --- Over/Under ----------------------------------------------------------

def test_over_under():
    assert s("ou_2_5", "Over 2.5", h=2, a=1).status == WON
    assert s("ou_2_5", "Under 2.5", h=2, a=1).status == LOST


def test_over_under_pushes_on_the_line():
    # A .5 line never pushes; a whole line does.
    r = s("ou_2", "Over 2", h=2, a=0)
    assert r.status == VOID
    assert r.payout_ngn == pytest.approx(100.0)
    assert s("ou_2_5", "Over 2.5", h=2, a=0).status == LOST


def test_team_over_under():
    assert s("home_ou_1_5", "Over 1.5", h=3, a=0).status == WON
    assert s("away_ou_1_5", "Over 1.5", h=3, a=0).status == LOST
    assert s("away_ou_1_5", "Over 1.5", h=3, a=2).status == WON


def test_quarter_line_is_unknown_not_a_guess():
    r = s("ou_2_25", "Over 2.25", h=2, a=0)
    assert r.status == UNKNOWN


# --- Both teams to score -------------------------------------------------

def test_btts():
    assert s("btts", "Yes", h=1, a=1).status == WON
    assert s("btts", "No", h=1, a=0).status == WON
    assert s("btts", "Yes", h=1, a=0).status == LOST


# --- Parity, clean sheet, win to nil -------------------------------------

def test_odd_even():
    assert s("odd_even", "Odd", h=1, a=0).status == WON
    assert s("odd_even", "Even", h=1, a=0).status == LOST
    assert s("home_odd_even", "Odd", h=3, a=0).status == WON


def test_clean_sheet():
    assert s("home_clean_sheet", "Yes", h=2, a=0).status == WON
    assert s("home_clean_sheet", "Yes", h=2, a=1).status == LOST
    assert s("away_clean_sheet", "Yes", h=2, a=0).status == LOST


def test_win_to_nil():
    assert s("home_win_to_nil", "Yes", h=2, a=0).status == WON
    assert s("home_win_to_nil", "Yes", h=2, a=1).status == LOST


# --- Exact goals and correct score ---------------------------------------

def test_exact_goals():
    assert s("home_exact_goals", "2", h=2, a=0).status == WON
    assert s("home_exact_goals", "2", h=3, a=0).status == LOST
    assert s("home_exact_goals", "3+", h=4, a=0).status == WON


def test_correct_score():
    assert s("correct_score", "2:1", h=2, a=1).status == WON
    assert s("correct_score", "2:1", h=1, a=1).status == LOST


# --- Handicaps ------------------------------------------------------------

def test_asian_handicap_and_push():
    assert s("ah_-0.5", "Home (-0.5)", h=1, a=0).status == WON
    assert s("ah_0", "Home (0)", h=1, a=0).status == WON
    r = s("ah_0", "Home (0)", h=2, a=2)
    assert r.status == VOID
    assert r.payout_ngn == pytest.approx(100.0)


def test_european_handicap():
    # 'Handicap 1:0' adds one goal to the HOME score, matching the pricer's
    # convention (which the live board's prices confirm: 'Handicap 0:1' has
    # Away at 1.04, i.e. the away side is the one being handicapped).
    assert s("handicap_1_0", "Home (1:0)", h=2, a=0).status == WON
    # A draw needs the adjusted scores level: (h+1) == a.
    assert s("handicap_1_0", "Draw (1:0)", h=0, a=1).status == WON
    assert s("handicap_1_0", "Away (1:0)", h=0, a=0).status == LOST
    assert s("handicap_0_1", "Away (0:1)", h=0, a=2).status == WON


# --- Path families: unknown without the path -----------------------------

def test_1up_without_the_observed_path_is_unknown():
    r = s("1x2_1up", "Home", h=3, a=0)
    assert r.status == UNKNOWN
    assert "path" in r.reason


def test_1up_with_the_observed_path_settles():
    r = s("1x2_1up", "Home", h=3, a=0, observed_max_lead={"home": 2})
    assert r.status == WON
    r2 = s("1x2_1up", "Home", h=0, a=1, observed_max_lead={"home": 0})
    assert r2.status == LOST


def test_lead_by_family_is_unknown_without_the_path():
    r = s("lead_by_2_home", "Yes", h=3, a=0)
    assert r.status == UNKNOWN


# --- Unknown markets ------------------------------------------------------

def test_unknown_market_is_unknown_not_a_loss():
    r = s("some_new_family", "Whatever", h=1, a=0)
    assert r.status == UNKNOWN


def test_unrecognised_selection_is_unknown():
    r = s("1x2", "Sideways", h=1, a=0)
    assert r.status == UNKNOWN
