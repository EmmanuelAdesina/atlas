"""Live-board coverage: how much of the book can ATLAS actually price?

`tests/fixtures/live_board_markets.json` is the market header list captured
from a real SportyBet live match on 2026-10-19: 92 headers and 2,794 odds
for ONE match. Coverage was 1.9% of odds before the path engine and the
new families were added; it is now 67%.

This test guards that number, so a refactor cannot silently drop the
pricer's reach back to a handful of families.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atlas.live.estimate import LiveStateSpaceModel
from atlas.live.extract import _match_header, _team_over_under_key
from atlas.live.math_engine import (
    build_remaining_score_matrix,
    compute_matrix_marginals,
    price_from_joint_matrix,
)
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.live.paths import GoalPathModel

BOARD = Path(__file__).parent / "fixtures" / "live_board_markets.json"

# The match the board was captured from.
HOME_TEAM = "Beitar Jerusalem FC"
AWAY_TEAM = "Maccabi Tel Aviv FC"

REPRESENTATIVE_SELECTION = {
    "1x2": "Home",
    "1x2_1h": "Home",
    "1x2_2h": "Home",
    "1x2_1up": "Home",
    "1x2_2up": "Home",
    "1x2_never_down": "Home",
    "rest_of_match_1x2": "Home",
    "double_chance": "Home or Draw",
    "1h_double_chance": "Home or Draw",
    "2h_double_chance": "Home or Draw",
    "dnb": "Home",
    "1h_dnb": "Home",
    "2h_dnb": "Home",
    "btts": "Yes",
    "1h_btts": "Yes",
    "2h_btts": "Yes",
    "next_goal": "Home",
    "odd_even": "Odd",
    "home_odd_even": "Odd",
    "away_odd_even": "Odd",
    "home_clean_sheet": "Yes",
    "away_clean_sheet": "Yes",
    "home_win_to_nil": "Yes",
    "away_win_to_nil": "Yes",
    "home_exact_goals": "1",
    "away_exact_goals": "1",
    "correct_score": "2:2",
    "combo_1x2__ou_1_5": "Home & Over",
    "combo_1x2__ou_2_5": "Home & Over",
    "combo_1x2__btts": "Home & Yes",
    "combo_ou_2_5__btts": "Over & Yes",
    "combo_double_chance__ou_1_5": "Home or Draw & Over",
    "combo_double_chance__btts": "Home or Draw & Yes",
    "combo_1x2_2h__ou_1_5": "Home & Over",
    "combo_1x2_2h__btts": "Home & Yes",
    "combo_2h_double_chance__ou_1_5": "Home or Draw & Over",
    "combo_2h_double_chance__btts": "Home or Draw & Yes",
    "btts_2plus": "Yes",
    "btts_no_draw": "Yes",
    "odd_even_2h": "Odd",
    "home_odd_even_2h": "Odd",
    "away_odd_even_2h": "Odd",
    "rest_of_match_total_goals": "Over 2.5",
    "rest_of_match_asian_handicap": "Home (-0.5)",
    "1x2_to_85": "Home",
    "1x2_to_80": "Home",
    "1x2_to_75": "Home",
    "ou_to_85": "Over 2.5",
    "ou_to_80": "Over 2.5",
    "ou_to_75": "Over 2.5",
}


def _resolve(title: str):
    """Market key for a live board title, the way the extractor builds it.

    The extractor resolves the line from the selection text for the
    families whose title has no line ('Over/Under', 'Home Team Over/Under'),
    so those placeholders are resolved here too. The captured board has no
    selection names, so a representative line is used.
    """
    hdr = _match_header(title)
    key = None
    if hdr is not None:
        key = hdr[0]
        if not key:
            # A bare 'Over/Under' takes its line from the selection text.
            if title.strip() == "Over/Under":
                key = "ou_2_5"
            elif title.strip() == "2nd Half - Over/Under":
                key = "ou_2h_2_5"
    else:
        key = _team_over_under_key(title, HOME_TEAM, AWAY_TEAM)
    if key and key.endswith("_pending"):
        key = key[: -len("_pending")] + "_2_5"
    return key


def _representative(key: str | None):
    if key is None:
        return None
    if key in REPRESENTATIVE_SELECTION:
        return REPRESENTATIVE_SELECTION[key]
    if key.startswith("ou_") or "ou_pending" in key or "ou_2h" in key:
        return "Over 2.5"
    if key.startswith("home_ou") or key.startswith("away_ou"):
        return "Over 2.5"
    if key.startswith("ah_") or key.endswith("ah_pending"):
        return "Home (-0.5)"
    if key.startswith("handicap_"):
        parts = key.split("_")
        return f"Home ({parts[1]}:{parts[2]})"
    if key.startswith("lead_by_") or key.startswith("goals_in_row_"):
        return "Yes"
    if key.startswith("combo_"):
        comp_a, comp_b = key[len("combo_"):].split("__")
        first = {
            "1x2": "Home", "double_chance": "Home or Draw",
            "2h_double_chance": "Home or Draw", "1x2_2h": "Home",
        }.get(comp_a)
        if first is None and comp_a.startswith("ou_"):
            first = "Over"
        second = "Yes" if comp_b == "btts" else ("Over" if comp_b.startswith("ou_") else None)
        if first and second:
            return f"{first} & {second}"
        return None
    return None


@pytest.fixture(scope="module")
def board():
    return json.loads(BOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pricing_env():
    state = LiveMatchState(
        match_id="m1",
        home=TeamStats(name=HOME_TEAM, goals=1),
        away=TeamStats(name=AWAY_TEAM, goals=1),
        period="2h",
        clock_minute=70.0,
        half_home_goals=0,
        half_away_goals=1,
        observed_at=datetime.now(timezone.utc),
        source="coverage-fixture",
    )
    mu_h, mu_a = 0.45, 0.35
    mat = build_remaining_score_matrix(mu_h, mu_a, max_goals=8)
    return state, mu_h, mu_a, mat, compute_matrix_marginals(mat)


def test_captured_board_is_the_expected_size(board):
    assert len(board) == 92
    assert sum(len(m["odds"]) for m in board) == 2794


def test_pricer_covers_most_of_a_live_board(board, pricing_env):
    """Coverage of the real board must not regress."""
    state, mu_h, mu_a, mat, marg = pricing_env
    path = GoalPathModel.from_intensities(mu_h, mu_a)

    priced_odds = 0
    total_odds = 0
    for m in board:
        n = len(m["odds"])
        total_odds += n
        key = _resolve(m["market"])
        sel = _representative(key)
        if not key or not sel:
            continue
        out = price_from_joint_matrix(
            mat, state, key, sel, mu_h, mu_a, marginals=marg, path_model=path
        )
        if out is not None:
            priced_odds += n

    coverage = priced_odds / total_odds
    assert coverage >= 0.95, (
        f"live-board coverage fell to {coverage:.1%} "
        f"({priced_odds}/{total_odds} odds) — the pricer lost reach"
    )


def test_path_families_are_priceable(board, pricing_env):
    """The 'at any time' families must price from the path engine."""
    state, mu_h, mu_a, mat, marg = pricing_env
    path = GoalPathModel.from_intensities(mu_h, mu_a)
    titles = {
        "Any Team to lead by 2 Goals at any time",
        "Home Team to lead by 1 Goal at any time",
        "Away Team to lead by 3 Goals at any time",
        "Any Team To Score 3 or More Goals in a Row",
        "Home Team To Score 2 or More Goals in a Row",
    }
    seen = 0
    for m in board:
        if m["market"] not in titles:
            continue
        key = _resolve(m["market"])
        assert key is not None, f"{m['market']} did not resolve to a market key"
        out = price_from_joint_matrix(
            mat, state, key, "Yes", mu_h, mu_a, marginals=marg, path_model=path
        )
        assert out is not None, f"{m['market']} ({key}) was not priced"
        assert 0.0 <= out.p_win <= 1.0
        seen += 1
    assert seen >= 4, f"expected several path families on the board, saw {seen}"


def test_path_families_need_the_path_model(board, pricing_env):
    """Without a path model the 'at any time' families must abstain.

    They cannot be priced from the end-state matrix, so silently falling
    back to it would manufacture edges.
    """
    state, mu_h, mu_a, mat, marg = pricing_env
    key = "lead_by_2_home"
    assert (
        price_from_joint_matrix(
            mat, state, key, "Yes", mu_h, mu_a, marginals=marg, path_model=None
        )
        is None
    )


def test_model_prices_the_captured_board_without_abstaining(board, pricing_env):
    """End to end: the model prices the board's families via the estimate API."""
    state, mu_h, mu_a, mat, marg = pricing_env
    model = LiveStateSpaceModel(now=datetime.now(timezone.utc))
    priced = 0
    for m in board:
        key = _resolve(m["market"])
        sel = _representative(key)
        if not key or not sel:
            continue
        est = model.estimate(state, key, sel)
        if not est.abstained and est.p is not None:
            priced += 1
    assert priced >= 30, f"only {priced} board families priced through the model"
