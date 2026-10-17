"""Tests for the quantitative state-space engine, de-vigging, full snapshot extraction,
cross-market coherence laws, calibration forensics, and Slow-Plane Boss ratchet invariants.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atlas.live.boss import (
    BossController,
    RiskRatchetViolationError,
    diagnose_live_operation,
)
from atlas.live.calibration import grade_decisions
from atlas.live.decisions import (
    DecisionLog,
    PolicyConfig,
    compute_recommended_stake_ngn,
)
from atlas.live.estimate import (
    Estimate,
    LiveStateSpaceModel,
    NoMoreGoalsBaseline,
    check_implications,
)
from atlas.live.extract import extract_live_match_state, extract_markets
from atlas.live.math_engine import (
    compute_posterior_intensities,
    devig_power,
    devig_proportional,
    devig_shin,
    effective_hazard_fraction,
    estimate_remaining_minutes,
    expected_value_detailed,
    fractional_kelly_stake,
    infer_prematch_lambdas_from_1x2,
)
from atlas.live.loop import Opportunity, selection_correlation_groups
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.mappers.sportybet_mapper import map_sport_markets

FIXDIR = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 16, 18, 42, tzinfo=timezone.utc)


def _sample_midgame_state(
    clock_minute: float = 60.0,
    home_goals: int = 1,
    away_goals: int = 1,
    home_reds: int = 0,
    away_reds: int = 0,
    home_sot: int = 4,
    away_sot: int = 3,
) -> LiveMatchState:
    return LiveMatchState(
        match_id="test-match-001",
        competition="Premier League",
        home=TeamStats(
            name="Arsenal",
            goals=home_goals,
            possession_pct=55.0,
            shots_on_target=home_sot,
            shots_total=10,
            corners=5,
            yellow_cards=1,
            red_cards=home_reds,
        ),
        away=TeamStats(
            name="Chelsea",
            goals=away_goals,
            possession_pct=45.0,
            shots_on_target=away_sot,
            shots_total=7,
            corners=3,
            yellow_cards=2,
            red_cards=away_reds,
        ),
        period="2h",
        clock_minute=clock_minute,
        clock_label=f"{int(clock_minute)}:00",
        half_home_goals=1,
        half_away_goals=0,
        substitutions_count=4,
        remaining_unknown=False,
        observed_at=NOW,
        source="sportybet_browser",
    )


# ---------------------------------------------------------------------------
# 1. Cross-Market Probability Coherence & No-Arbitrage Laws
# ---------------------------------------------------------------------------


def test_joint_surface_partition_and_cross_market_identities():
    state = _sample_midgame_state(clock_minute=60.0, home_goals=1, away_goals=0)
    model = LiveStateSpaceModel(now=NOW)

    # 1X2 partition sums to 1.0
    p_h = model.estimate(state, "1x2", "Home").p
    p_d = model.estimate(state, "1x2", "Draw").p
    p_a = model.estimate(state, "1x2", "Away").p
    assert p_h is not None and p_d is not None and p_a is not None
    assert (p_h + p_d + p_a) == pytest.approx(1.0, abs=1e-6)

    # 2nd Half 1X2 partition sums to 1.0
    p_2h_h = model.estimate(state, "1x2_2h", "Home").p
    p_2h_d = model.estimate(state, "1x2_2h", "Draw").p
    p_2h_a = model.estimate(state, "1x2_2h", "Away").p
    assert (p_2h_h + p_2h_d + p_2h_a) == pytest.approx(1.0, abs=1e-6)

    # Double Chance exact additive coherence with 1X2
    p_1x = model.estimate(state, "double_chance", "Home or Draw").p
    p_x2 = model.estimate(state, "double_chance", "Draw or Away").p
    p_12 = model.estimate(state, "double_chance", "Home or Away").p
    assert p_1x == pytest.approx(p_h + p_d, abs=1e-6)
    assert p_x2 == pytest.approx(p_d + p_a, abs=1e-6)
    assert p_12 == pytest.approx(p_h + p_a, abs=1e-6)

    # Draw No Bet win + push partition
    dnb_h = model.estimate(state, "dnb", "Home")
    dnb_a = model.estimate(state, "dnb", "Away")
    assert dnb_h.p == pytest.approx(p_h, abs=1e-6)
    assert dnb_a.p == pytest.approx(p_a, abs=1e-6)
    assert dnb_h.p_push == pytest.approx(p_d, abs=1e-6)
    assert (dnb_h.p + dnb_a.p + dnb_h.p_push) == pytest.approx(1.0, abs=1e-6)

    # Over/Under partitions and monotonicity across lines
    unders = []
    for line_str in ("1_5", "2_5", "3_5", "4_5"):
        line_dot = line_str.replace("_", ".")
        po = model.estimate(state, f"ou_{line_str}", f"Over {line_dot}").p
        pu = model.estimate(state, f"ou_{line_str}", f"Under {line_dot}").p
        assert (po + pu) == pytest.approx(1.0, abs=1e-6)
        unders.append(pu)
    # Monotonicity: P(Under 1.5) < P(Under 2.5) < P(Under 3.5) < P(Under 4.5)
    assert unders[0] < unders[1] < unders[2] < unders[3]

    # Next Goal partition & exact identity with Under 1.5 when current score is 1:0
    ng_h = model.estimate(state, "next_goal", "Home").p
    ng_a = model.estimate(state, "next_goal", "Away").p
    ng_none = model.estimate(state, "next_goal", "None").p
    assert (ng_h + ng_a + ng_none) == pytest.approx(1.0, abs=1e-6)
    assert ng_none == pytest.approx(unders[0], abs=1e-6)

    # BTTS partition
    btts_y = model.estimate(state, "btts", "Yes").p
    btts_n = model.estimate(state, "btts", "No").p
    assert (btts_y + btts_n) == pytest.approx(1.0, abs=1e-6)


def test_asian_handicap_half_whole_and_quarter_lines():
    state = _sample_midgame_state(clock_minute=55.0, home_goals=0, away_goals=0)
    model = LiveStateSpaceModel(now=NOW)

    # At 0:0, AH -0.5 Home == 1X2 Home, AH +0.5 Away == Double Chance X2
    p_1x2_h = model.estimate(state, "1x2", "Home").p
    p_dc_x2 = model.estimate(state, "double_chance", "Draw or Away").p
    ah_minus_half = model.estimate(state, "ah_-0.5", "Home (-0.5)")
    ah_plus_half = model.estimate(state, "ah_0.5", "Away (+0.5)")
    assert ah_minus_half.p == pytest.approx(p_1x2_h, abs=1e-6)
    assert ah_plus_half.p == pytest.approx(p_dc_x2, abs=1e-6)

    # Quarter line AH -0.25 at 0:0: full win on Home win, half-loss on Draw
    p_draw = model.estimate(state, "1x2", "Draw").p
    ah_qtr = model.estimate(state, "ah_-0.25", "Home (-0.25)")
    assert ah_qtr.p == pytest.approx(p_1x2_h, abs=1e-6)
    assert ah_qtr.p_half_loss == pytest.approx(p_draw, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Stoppage-Time Survival, Red-Card Multipliers & Live Shrinkage
# ---------------------------------------------------------------------------


def test_stoppage_survival_never_collapses_to_zero_after_95_minutes():
    state_96 = _sample_midgame_state(clock_minute=96.5, home_goals=0, away_goals=1)
    stop = estimate_remaining_minutes(state_96, period_scope="2h")
    assert stop.rem_mean > 0.25
    assert stop.rem_lower > 0.15
    assert stop.rem_upper > stop.rem_mean

    baseline = NoMoreGoalsBaseline(now=NOW)
    est_base = baseline.estimate(state_96, "ou_1_5", "Under 1.5")
    assert est_base.p is not None and 0.80 < est_base.p < 1.0

    ss_model = LiveStateSpaceModel(now=NOW)
    est_ss = ss_model.estimate(state_96, "ou_1_5", "Under 1.5")
    assert est_ss.p is not None and 0.80 < est_ss.p < 1.0
    assert est_ss.p_lower is not None and est_ss.p_lower < est_ss.p < est_ss.p_upper


def test_red_card_and_live_stats_shift_posterior_probabilities():
    model = LiveStateSpaceModel(now=NOW)
    even_state = _sample_midgame_state(clock_minute=55.0, home_reds=0, away_reds=0)
    home_red_state = _sample_midgame_state(clock_minute=55.0, home_reds=1, away_reds=0)

    p_home_even = model.estimate(even_state, "1x2", "Home").p
    p_home_10men = model.estimate(home_red_state, "1x2", "Home").p
    p_away_even = model.estimate(even_state, "1x2", "Away").p
    p_away_vs_10 = model.estimate(home_red_state, "1x2", "Away").p

    assert p_home_10men < p_home_even
    assert p_away_vs_10 > p_away_even

    # Live shot dominance shifts intensity
    dom_home = _sample_midgame_state(clock_minute=65.0, home_sot=9, away_sot=0)
    dom_away = _sample_midgame_state(clock_minute=65.0, home_sot=0, away_sot=9)
    assert model.estimate(dom_home, "1x2", "Home").p > model.estimate(dom_away, "1x2", "Home").p


# ---------------------------------------------------------------------------
# 3. De-Vigging Microstructure (Shin 1993, Power, Pre-Match Inversion)
# ---------------------------------------------------------------------------


def test_shin_and_power_devig_correct_longshot_bias():
    # Recorded Fortuna vs Stuttgart 2H 1X2 odds: Away 1.05, Draw 8.00, Home 61.00
    odds = {"Away": 1.05, "Draw": 8.00, "Home": 61.00}
    prop = devig_proportional(odds)
    power = devig_power(odds)
    shin, z = devig_shin(odds)

    assert sum(prop.values()) == pytest.approx(1.0)
    assert sum(power.values()) == pytest.approx(1.0)
    assert sum(shin.values()) == pytest.approx(1.0)
    assert 0.01 < z < 0.25

    # Shin & Power penalize the 61.00 longshot more than naive proportional de-vigging
    assert shin["Home"] < prop["Home"]
    assert power["Home"] < prop["Home"]
    assert shin["Away"] > prop["Away"]


def test_infer_prematch_lambdas_from_1x2():
    lh, la = infer_prematch_lambdas_from_1x2(1.61, 3.60, 5.60)
    assert lh > la
    assert 1.2 < lh < 2.2
    assert 0.5 < la < 1.3

    # Strong-favourite markets must not pin a genuinely weak side to the old
    # 0.20 lower grid bound.
    lh_extreme, la_extreme = infer_prematch_lambdas_from_1x2(1.02, 12.0, 50.0)
    assert lh_extreme > 2.0
    assert 0.05 <= la_extreme < 0.20


def test_pre_match_1x2_odds_are_wired_into_the_live_prior():
    state = _sample_midgame_state(clock_minute=0.0, home_goals=0, away_goals=0)
    state = state.model_copy(update={
        "period": "1h",
        "prematch_1x2_odds": {"Home": 1.02, "Draw": 12.0, "Away": 50.0},
        "home": state.home.model_copy(update={"shots_on_target": None, "shots_total": None, "corners": None}),
        "away": state.away.model_copy(update={"shots_on_target": None, "shots_total": None, "corners": None}),
    })
    model = LiveStateSpaceModel(now=NOW)
    est = model.estimate(state, "1x2", "Home")
    assert est.diagnostics["prior_source"] == "prematch_1x2"
    assert est.diagnostics["lambda_home_90"] > 2.0
    assert est.diagnostics["lambda_away_90"] < 0.20


def test_full_match_hazard_is_normalized_to_full_match_prior():
    state = _sample_midgame_state(clock_minute=0.0, home_goals=0, away_goals=0)
    state = state.model_copy(update={
        "period": "1h",
        "home": state.home.model_copy(update={"goals": 0}),
        "away": state.away.model_copy(update={"goals": 0}),
    })
    remaining = estimate_remaining_minutes(state, period_scope="full")
    assert effective_hazard_fraction("1h", 0.0, remaining.rem_mean) == pytest.approx(1.0)
    est = LiveStateSpaceModel(now=NOW).estimate(state, "ou_2_5", "Over 2.5")
    assert est.diagnostics["mu_home_rem"] == pytest.approx(1.42)
    assert est.diagnostics["mu_away_rem"] == pytest.approx(1.18)


def test_first_half_stoppage_uncertainty_is_propagated_to_full_match():
    state = _sample_midgame_state(clock_minute=47.0, home_goals=0, away_goals=0)
    state = state.model_copy(update={
        "period": "1h",
        "remaining_unknown": True,
        "added_minutes_announced": None,
        "clock_label": "45:00 +02:00",
    })
    current_half = estimate_remaining_minutes(state, period_scope="1h")
    full_match = estimate_remaining_minutes(state, period_scope="full")
    assert current_half.in_stoppage and current_half.stoppage_uncertainty_high
    assert full_match.in_stoppage and full_match.stoppage_uncertainty_high

    estimate = LiveStateSpaceModel(now=NOW).estimate(state, "ou_2_5", "Over 2.5")
    assert estimate.confidence == "low"
    assert "unverified stoppage duration" in (estimate.abstain_reason or "")


def test_surface_cache_tracks_all_pressure_and_stoppage_inputs():
    state = _sample_midgame_state(clock_minute=62.0, home_goals=1, away_goals=1)
    model = LiveStateSpaceModel(now=NOW)
    base = model.estimate(state, "ou_2_5", "Over 2.5")

    variants = [
        state.model_copy(update={
            "home": state.home.model_copy(update={"shots_total": 20}),
        }),
        state.model_copy(update={
            "home": state.home.model_copy(update={"possession_pct": 30.0}),
            "away": state.away.model_copy(update={"possession_pct": 70.0}),
        }),
        state.model_copy(update={"substitutions_count": 10}),
    ]
    for changed in variants:
        updated = model.estimate(changed, "ou_2_5", "Over 2.5")
        assert updated.diagnostics["mu_home_rem"] != pytest.approx(base.diagnostics["mu_home_rem"])


def test_score_state_is_symmetric_and_possession_alone_does_not_zero_pressure():
    base = _sample_midgame_state(clock_minute=75.0, home_goals=0, away_goals=0)
    for margin in (1, 2):
        home_leads = base.model_copy(update={
            "home": base.home.model_copy(update={"goals": margin}),
            "away": base.away.model_copy(update={"goals": 0}),
        })
        away_leads = base.model_copy(update={
            "home": base.home.model_copy(update={"goals": 0}),
            "away": base.away.model_copy(update={"goals": margin}),
        })
        ph = compute_posterior_intensities(home_leads)
        pa = compute_posterior_intensities(away_leads)
        assert (ph.score_state_multiplier_home, ph.score_state_multiplier_away) == pytest.approx((0.94, 1.12))
        assert (pa.score_state_multiplier_home, pa.score_state_multiplier_away) == pytest.approx((1.12, 0.94))

    possession_only = base.model_copy(update={
        "home": base.home.model_copy(update={
            "shots_on_target": None, "shots_total": None, "corners": None,
            "possession_pct": 55.0,
        }),
        "away": base.away.model_copy(update={
            "shots_on_target": None, "shots_total": None, "corners": None,
            "possession_pct": 45.0,
        }),
    })
    post = compute_posterior_intensities(possession_only)
    assert post.shrinkage_weight == 0.0
    assert post.lambda_home_90 == pytest.approx(1.42)
    assert post.lambda_away_90 == pytest.approx(1.18)

    ten_v_ten = base.model_copy(update={
        "home": base.home.model_copy(update={"red_cards": 1}),
        "away": base.away.model_copy(update={"red_cards": 1}),
    })
    reds = compute_posterior_intensities(ten_v_ten)
    assert reds.red_card_multiplier_home == pytest.approx(1.0)
    assert reds.red_card_multiplier_away == pytest.approx(1.0)


def test_quarter_line_ev_and_kelly_use_full_settlement_distribution():
    est = Estimate(
        p=0.35,
        confidence="high",
        model_name="test",
        p_lower=0.30,
        p_half_win=0.20,
        p_half_loss=0.10,
        p_push=0.05,
    )
    odds = 2.20
    expected = expected_value_detailed(
        odds, p_win=0.35, p_push=0.05, p_half_win=0.20, p_half_loss=0.10
    )
    assert est.expected_value(odds) == pytest.approx(expected)
    opp = Opportunity(
        match_id="m", market_key="ah_-0.25", selection="Home (-0.25)",
        odds=odds, model_p=0.35, p_lower=0.30, p_push=0.05,
        p_half_win=0.20, p_half_loss=0.10,
    )
    assert opp.ev == pytest.approx(expected)
    assert opp.ev_lower == pytest.approx(expected_value_detailed(
        odds, p_win=0.30, p_push=0.05, p_half_win=0.20, p_half_loss=0.10
    ))

    fractional = fractional_kelly_stake(
        0.35, odds, p_push=0.05, fraction=0.25,
        p_half_win=0.20, p_half_loss=0.10,
    )
    assert fractional > 0.0
    # Verify fractional sizing is at one quarter of a full-Kelly optimum under
    # the actual split settlement returns, not a binary approximation.
    returns = (odds - 1.0, 0.5 * (odds - 1.0), 0.0, -0.5, -1.0)
    probs = (0.35, 0.20, 0.05, 0.10, 0.30)
    f_grid = [i / 10000 for i in range(1, 10000)]
    best = max(
        f_grid,
        key=lambda f: sum(p * math.log1p(f * r) for p, r in zip(probs, returns)),
    )
    assert fractional == pytest.approx(0.25 * best, abs=2e-4)


def test_state_space_quarter_line_probability_is_full_win_not_effective_win():
    state = _sample_midgame_state(clock_minute=60.0, home_goals=0, away_goals=0)
    est = LiveStateSpaceModel(now=NOW).estimate(state, "ou_2_25", "Under 2.25")
    assert est.p is not None
    assert est.p + est.p_push + est.p_half_win + est.p_half_loss <= 1.0 + 1e-8
    assert est.effective_win_probability == pytest.approx(est.p + 0.5 * est.p_half_win)
    assert est.expected_value(2.0) == pytest.approx(expected_value_detailed(
        2.0, p_win=est.p, p_push=est.p_push,
        p_half_win=est.p_half_win, p_half_loss=est.p_half_loss,
    ))


def test_conservative_ev_uses_one_complete_worst_case_settlement():
    est = Estimate(
        p=0.50, confidence="high", model_name="test", p_lower=0.40,
        p_half_win=0.10, p_half_loss=0.10,
        settlement_scenarios=[
            (0.40, 0.00, 0.30, 0.00),
            (0.50, 0.00, 0.00, 0.30),
        ],
    )
    # The second scenario has a higher full-win probability but worse exact EV.
    assert est.conservative_settlement(2.0) == pytest.approx((0.50, 0.00, 0.00, 0.30))
    assert est.conservative_expected_value(2.0) == pytest.approx(0.15)


def test_correlation_groups_cover_double_chance_and_period_markets():
    home_1x2 = set(selection_correlation_groups("1x2", "Home"))
    dc_1x = set(selection_correlation_groups("double_chance", "Home or Draw"))
    assert home_1x2 & dc_1x
    assert "low_tempo_under" in dc_1x
    assert selection_correlation_groups("1h_dnb", "Home")
    assert selection_correlation_groups("dnb_1h", "Away")
    assert "total_goals" in selection_correlation_groups("2h_ou_2_5", "Over 2.5")
    assert "low_tempo_under" in selection_correlation_groups("ou_1h_2_5", "Under 2.5")


# ---------------------------------------------------------------------------
# 4. Full Live Match State & Suspended O/U Line Extraction
# ---------------------------------------------------------------------------


def test_extract_full_live_match_state_from_recorded_snapshot():
    texts = json.loads((FIXDIR / "fortuna_snapshot_texts.json").read_text(encoding="utf-8"))
    state = extract_live_match_state(
        texts, match_id="live-fortuna-stuttgart-20261016", observed_at=NOW
    )
    assert state.home.name == "Fortuna Cologne"
    assert state.away.name == "VfB Stuttgart II"
    assert state.competition == "3. Liga | Matchday 10"
    assert state.period == "2h"
    assert state.clock_minute == pytest.approx(93.7)
    assert state.home.goals == 0 and state.away.goals == 1
    assert state.half_home_goals == 0 and state.half_away_goals == 0
    assert state.home.red_cards == 1 and state.home.yellow_cards == 1
    assert state.away.red_cards == 0 and state.away.yellow_cards == 5
    assert state.home.corners == 3 and state.away.corners == 1
    assert state.home.shots_on_target == 3 and state.away.shots_on_target == 1
    assert state.home.shots_total == 6 and state.away.shots_total == 4
    assert state.home.saves == 0 and state.away.saves == 3
    assert state.home.fouls == 18 and state.away.fouls == 20
    assert state.home.free_kicks == 20 and state.away.free_kicks == 18
    assert state.home.offsides == 4 and state.away.offsides == 2
    assert state.home.possession_pct == 44.0 and state.away.possession_pct == 56.0
    assert state.substitutions_count == 9
    assert state.account_balance_ngn == pytest.approx(166.11)
    assert state.remaining_unknown is True


def test_extract_locked_over_under_still_sets_line_and_key():
    tokens = [
        "All", "Main", "Goals", "Half",
        "Over/Under", "Over 3.5", "Under 3.5",
        "Betslip",
    ]
    mkts = extract_markets(tokens, "m1")
    assert len(mkts) == 1
    assert mkts[0]["market_key"] == "ou_3_5"
    assert mkts[0]["line"] == 3.5
    assert mkts[0]["status"] == "suspended"


# ---------------------------------------------------------------------------
# 5. SportyBet NG Risk Gate, Correlation Blocking & Calibration Forensics
# ---------------------------------------------------------------------------


def test_decision_log_blocks_correlated_canonical_markets_and_sizes_kelly(tmp_path):
    policy = PolicyConfig(
        allow_execution=True,
        min_ev=0.04,
        min_ev_lower=0.01,
        max_bets_per_match=3,
        max_correlated_in_group=1,
        auto_size_kelly=True,
        bankroll_ngn=20_000.0,
        max_stake_per_bet=2_000.0,
    )
    log = DecisionLog(tmp_path / "decisions.jsonl", policy)

    est_home = Estimate(0.65, "high", "m", p_lower=0.60, p_upper=0.70)
    r1 = log.decide("m1", "1x2", "Home", 1.85, "open", est_home)
    assert r1.decision == "recommend"
    assert r1.stake >= 100.0 and r1.stake == int(r1.stake)

    # Correlated market in 'home_supremacy' / 'home_goals' must be blocked
    est_ah = Estimate(0.65, "high", "m", p_lower=0.60, p_upper=0.70)
    r2 = log.decide("m1", "ah_-0.5", "Home (-0.5)", 1.85, "open", est_ah)
    assert r2.decision == "pass"
    assert "correlated" in r2.reason

    # Uncorrelated market ('home_corners') is permitted
    est_cor = Estimate(0.64, "high", "m", p_lower=0.58, p_upper=0.70)
    r3 = log.decide("m1", "home_corners_5_5", "Over 5.5", 1.90, "open", est_cor)
    assert r3.decision == "recommend"


def test_calibration_grades_settled_fortuna_decisions():
    fx = json.loads((FIXDIR / "fortuna_stuttgart_90plus.json").read_text(encoding="utf-8"))
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    st = fx["state"]
    state = LiveMatchState(
        match_id=fx["match_id"],
        competition=st["competition"],
        home=st["home"],
        away=st["away"],
        period=st["period"],
        clock_minute=st["clock_minute"],
        clock_label=st["clock_label"],
        half_home_goals=st["half_home_goals"],
        half_away_goals=st["half_away_goals"],
        remaining_unknown=st["remaining_unknown"],
        observed_at=observed_at,
        source="sportybet_browser",
    )
    model = LiveStateSpaceModel(now=observed_at)
    recs, _ = map_sport_markets(fx["markets"], source_url="", collected_at=observed_at)

    raw_decisions = []
    for r in recs:
        for s in r.selections:
            est = model.estimate(state, r.market_key, s.name)
            raw_decisions.append(
                {
                    "match_id": fx["match_id"],
                    "market_key": r.market_key,
                    "selection": s.name,
                    "odds": s.odds,
                    "model_p": est.p,
                    "p_lower": est.p_lower,
                    "ev": est.expected_value(s.odds),
                    "ev_lower": est.conservative_expected_value(s.odds),
                    "confidence": est.confidence,
                    "decision": "pass",
                    "stake": 0.0,
                }
            )

    graded, summary = grade_decisions(
        raw_decisions, settled_outcomes=fx["outcome"]["settled"]
    )
    assert summary.settled_records == 5
    assert summary.brier_score < 0.01  # High accuracy on settled 90+3 state
    assert summary.log_loss < 0.08
    assert "ou_1_5" in summary.by_market_family
    assert "1x2_2h" in summary.by_market_family


# ---------------------------------------------------------------------------
# 6. Slow-Plane Boss Asymmetric Ratchet & Sub-Millisecond Hot Path
# ---------------------------------------------------------------------------


def test_boss_asymmetric_ratchet_enforces_hard_safety_floors(tmp_path):
    boss = BossController(tmp_path / "boss_state.json")

    # Boss can tighten min_ev, reduce kelly_fraction, set bounded priors, and block markets
    boss.set_risk_overrides(
        min_ev=0.08,
        kelly_fraction=0.10,
        max_exposure_ngn=1500.0,
        min_ev_lower=0.0,
        rationale="Feed jitter observed in 2H",
    )
    # Repeated Boss updates may tighten, but can never loosen an earlier ratchet.
    boss.set_risk_overrides(
        min_ev=0.05, kelly_fraction=0.20, max_exposure_ngn=2000.0,
        min_ev_lower=-0.50, rationale="later weaker diagnostic",
    )
    boss.set_prior_adjustment(
        "m1", delta_home_lambda=-0.15, delta_away_lambda=0.10, rationale="Home striker injured"
    )
    boss.block_market("ou_0_5", reason="illiquid")

    tightened = boss.apply_to_policy(PolicyConfig(min_ev=0.05, kelly_fraction=0.25))
    assert tightened.min_ev == pytest.approx(0.08)
    assert tightened.kelly_fraction == pytest.approx(0.10)
    assert tightened.max_exposure_per_match == pytest.approx(1500.0)
    assert tightened.min_ev_lower == pytest.approx(0.0)
    assert "ou_0_5" in tightened.blocked_markets

    # Boss CANNOT enable live execution autonomously
    with pytest.raises(RiskRatchetViolationError, match="allow_execution"):
        boss.set_risk_overrides(allow_execution=True)

    # Boss CANNOT drop min_ev below HARD_MIN_EV_FLOOR (0.02)
    with pytest.raises(RiskRatchetViolationError, match="safety floor"):
        boss.set_risk_overrides(min_ev=0.005)

    # Boss CANNOT raise kelly_fraction above HARD_MAX_KELLY_FRACTION (0.25)
    with pytest.raises(RiskRatchetViolationError, match="cap"):
        boss.set_risk_overrides(kelly_fraction=0.50)

    # Boss CANNOT shift priors beyond [-0.25, +0.25]
    with pytest.raises(RiskRatchetViolationError, match="Prior shift"):
        boss.set_prior_adjustment("m1", delta_home_lambda=0.60, delta_away_lambda=0.0, rationale="hype")


def test_fast_plane_surface_pricing_runs_under_two_milliseconds():
    state = _sample_midgame_state(clock_minute=62.0, home_goals=1, away_goals=1)
    model = LiveStateSpaceModel(now=NOW)
    selections = [
        ("1x2", "Home"), ("1x2", "Draw"), ("1x2", "Away"),
        ("1x2_2h", "Home"), ("1x2_2h", "Draw"), ("1x2_2h", "Away"),
        ("double_chance", "Home or Draw"), ("double_chance", "Draw or Away"),
        ("dnb", "Home"), ("dnb", "Away"),
        ("ou_2_5", "Over 2.5"), ("ou_2_5", "Under 2.5"),
        ("ou_3_5", "Over 3.5"), ("ou_3_5", "Under 3.5"),
        ("btts", "Yes"), ("btts", "No"),
        ("next_goal", "Home"), ("next_goal", "Away"), ("next_goal", "None"),
        ("ah_-0.25", "Home (-0.25)"), ("ah_-0.25", "Away (+0.25)"),
    ]
    # Materialize the shared joint/scenario surface first; then time 21 cached
    # selection prices, matching the documented after-setup throughput target.
    model.estimate(state, *selections[0])
    t0 = time.perf_counter_ns()
    out = [model.estimate(state, m, s) for m, s in selections]
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    assert all(not e.abstained for e in out)
    assert elapsed_us < 2000.0  # Full 21-selection surface in < 2ms after setup


def test_calibration_grades_push_half_outcomes_and_recommendation_roi():
    dnb_push = {
        "match_id": "push-match", "market_key": "dnb", "selection": "Home",
        "odds": 1.90, "model_p": 0.40, "p_push": 0.20,
        "p_lower": 0.35, "ev": 0.16, "ev_lower": 0.10,
        "confidence": "high", "decision": "recommend", "stake": 100.0,
    }
    half_loss = {
        "match_id": "quarter-match", "market_key": "ou_2_25", "selection": "Over 2.25",
        "odds": 2.00, "model_p": 0.30, "p_push": 0.0,
        "p_half_win": 0.20, "p_half_loss": 0.10,
        "p_lower": 0.25, "ev": 0.05, "ev_lower": -0.05,
        "confidence": "medium", "decision": "recommend", "stake": 200.0,
    }
    graded, summary = grade_decisions(
        [dnb_push, half_loss],
        settled_outcomes={
            ("push-match", "dnb"): "Draw",
            ("quarter-match", "ou_2_25"): {
                "selection_outcomes": {"Over 2.25": "half_loss"},
            },
        },
    )
    push, half = graded
    assert push.settlement == "push"
    assert push.realized_unit_pnl == 0.0 and push.realized_stake_pnl == 0.0
    assert half.settlement == "half_loss"
    assert half.realized_unit_pnl == 0.0
    assert half.realized_stake_pnl == 0.0
    assert half.paper_unit_pnl == -0.5
    assert summary.realized_unit_roi == 0.0  # recommendations were not executed
    assert summary.realized_stake_pnl == 0.0
    assert summary.recommended_unit_roi == pytest.approx(-0.25)
    assert summary.recommended_stake_pnl == pytest.approx(-100.0)
    assert summary.paper_unit_roi == pytest.approx(-0.25)

    executed = {**half_loss, "executed": True}
    _, actual_summary = grade_decisions(
        [executed],
        {("quarter-match", "ou_2_25"): {
            "selection_outcomes": {"Over 2.25": "half_loss"},
        }},
    )
    assert actual_summary.executed_count == 1
    assert actual_summary.realized_unit_roi == pytest.approx(-0.5)
    assert actual_summary.realized_stake_pnl == pytest.approx(-100.0)


def test_calibration_separates_realized_recommendations_and_paper_roi():
    records = [
        {
            "match_id": "m", "market_key": "1x2", "selection": "Home",
            "odds": 2.0, "model_p": 0.60, "decision": "pass", "stake": 0.0,
        },
        {
            "match_id": "m", "market_key": "1x2", "selection": "Away",
            "odds": 2.0, "model_p": 0.40, "decision": "recommend", "stake": 100.0,
        },
    ]
    graded, summary = grade_decisions(records, {("m", "1x2"): "Home"})
    assert len(graded) == 2
    assert summary.realized_unit_roi == 0.0  # no bet was reported executed
    assert summary.realized_stake_pnl == 0.0
    assert summary.paper_unit_roi == pytest.approx(0.0)
    assert summary.recommended_unit_roi == pytest.approx(-1.0)
    assert summary.recommended_stake_pnl == pytest.approx(-100.0)


def test_calibration_adds_proper_categorical_1x2_scores():
    records = [
        {"match_id": "m", "market_key": "1x2", "selection": "Home", "odds": 2.0,
         "model_p": 0.50, "decision": "pass"},
        {"match_id": "m", "market_key": "1x2", "selection": "Draw", "odds": 3.0,
         "model_p": 0.30, "decision": "pass"},
        {"match_id": "m", "market_key": "1x2", "selection": "Away", "odds": 4.0,
         "model_p": 0.20, "decision": "pass"},
    ]
    _, summary = grade_decisions(records, {("m", "1x2"): "Home"})
    assert summary.multiclass_market_count == 1
    assert summary.multiclass_log_loss == pytest.approx(-math.log(0.50), abs=1e-4)
    assert summary.multiclass_brier_score == pytest.approx(0.19, abs=1e-4)
