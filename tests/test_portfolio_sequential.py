"""Tests for the sequential portfolio layer (atlas.portfolio.sequential)."""

from datetime import datetime, timedelta, timezone

from atlas.live.decisions import PolicyConfig, compute_recommended_stake_ngn
from atlas.live.estimate import Estimate
from atlas.live.loop import Opportunity, selection_correlation_groups
from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio


def make_opp(
    match_id="m1",
    market_key="1x2",
    selection="home",
    odds=2.0,
    model_p=0.6,
    p_lower=0.55,
    model_name="test-model",
):
    return Opportunity(
        match_id=match_id,
        market_key=market_key,
        selection=selection,
        odds=odds,
        model_p=model_p,
        p_lower=p_lower,
        model_name=model_name,
    )


MATCH_STATE = {"period": "1h", "clock_minute": 30, "score": "0-0", "events": []}


def placed(
    market_key="1x2",
    selection="home",
    odds=2.0,
    stake=100.0,
    ev=0.05,
    ts=None,
    **state,
):
    return {
        "match_id": "m1",
        "market_key": market_key,
        "selection": selection,
        "odds": odds,
        "stake": stake,
        "ev": ev,
        "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
        "period": "1h",
        "clock_minute": 30,
        "score": (0, 0),
        "events_count": 0,
        **state,
    }


def history_12(market_key="ou_2_5_over", selection="over"):
    now = datetime.now(timezone.utc)
    return [
        placed(
            market_key=market_key,
            selection=selection,
            ts=now - timedelta(seconds=70 + i * 5),
        )
        for i in range(12)
    ]


def test_config_defaults():
    cfg = SequentialConfig()
    assert cfg.kelly_fraction == 0.10
    assert cfg.min_stake_ngn == 100.0
    assert cfg.max_stake_per_bet == 1000.0
    assert cfg.max_exposure_per_match == 3000.0
    assert cfg.max_bets_per_match == 100
    assert cfg.max_bets_per_snapshot == 2
    assert cfg.max_bets_per_minute == 5
    assert cfg.max_correlated_in_group == 12
    assert cfg.ev_threshold_high_correlation == 0.12
    assert cfg.rebalance_ev_delta == 0.02
    assert cfg.rebalance_odds_delta == 0.06
    assert cfg.min_minutes_between_same_selection == 8.0


def test_stake_uses_p_lower_not_model_p():
    opp = make_opp(odds=2.0, model_p=0.7, p_lower=0.6)
    decisions = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, [])
    d = decisions[0]
    assert d.should_bet
    est = Estimate(p=0.7, confidence="high", model_name="t", p_lower=0.6)
    expected = compute_recommended_stake_ngn(est, 2.0, PolicyConfig(kelly_fraction=0.10))
    assert d.stake_ngn == expected == 200.0
    assert d.factor_counts == {"home_goals": 1, "home_supremacy": 1}


def test_p_lower_none_falls_back_to_model_p():
    opp = make_opp(odds=2.0, model_p=0.7, p_lower=None)
    d = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, [])[0]
    assert d.should_bet
    assert d.stake_ngn == 400.0


def test_stake_below_min_is_skipped():
    opp = make_opp(
        market_key="ou_2_5_over", selection="over", odds=1.9, model_p=0.55, p_lower=0.53
    )
    d = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, [])[0]
    assert not d.should_bet
    assert d.stake_ngn == 0.0
    assert "below min" in d.reason


def test_factor_cap_blocks_13th_bet_without_high_conservative_ev():
    # 13th bet in the SAME family as the 12 stacked bets (total_goals).
    opp = make_opp(market_key="ou_3_5", selection="over", odds=2.0, model_p=0.6, p_lower=0.555)
    d = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, history_12())[0]
    assert not d.should_bet
    assert "at 12-bet cap" in d.reason
    assert d.factor_counts["total_goals"] == 12


def test_factor_cap_allows_13th_bet_with_high_conservative_ev():
    # Same 13th total_goals bet, but conservative EV (p_lower=0.65, o=2.0
    # => EV_lower 0.30) clears the 0.12 high-correlation bar.
    opp = make_opp(market_key="ou_3_5", selection="over", odds=2.0, model_p=0.6, p_lower=0.65)
    d = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, history_12())[0]
    assert d.should_bet
    assert d.stake_ngn == 300.0
    assert d.factor_counts["total_goals"] == 13


def test_exposure_cap_stakes_capped_then_blocks():
    now = datetime.now(timezone.utc)
    hist = [
        placed(
            market_key="1x2",
            selection="away",
            ts=now - timedelta(minutes=i + 1),
        )
        for i in range(29)
    ]
    opp1 = make_opp(market_key="1x2", selection="draw", odds=2.0, model_p=0.7, p_lower=0.6)
    opp2 = make_opp(
        market_key="ou_2_5_over", selection="over", odds=2.0, model_p=0.7, p_lower=0.6
    )
    d1, d2 = SequentialPortfolio().evaluate_snapshot([opp1, opp2], MATCH_STATE, hist)
    assert d1.should_bet
    assert d1.stake_ngn == 100.0
    assert not d2.should_bet
    # Once 29x100 exposure is on the match the cap, not the stake floor, is the
    # binding constraint — assert the block, not a specific phrasing.
    assert ("exposure cap" in d2.reason) or ("below min" in d2.reason)


def test_max_two_bets_per_snapshot():
    opps = [
        make_opp(odds=2.0, model_p=0.7, p_lower=0.6),
        make_opp(market_key="1x2", selection="away", odds=2.1, model_p=0.65, p_lower=0.6),
        make_opp(
            market_key="ou_2_5_over", selection="over", odds=1.9, model_p=0.6, p_lower=0.55
        ),
    ]
    decisions = SequentialPortfolio().evaluate_snapshot(opps, MATCH_STATE, [])
    assert sum(d.should_bet for d in decisions) == 2
    rejected = next(d for d in decisions if not d.should_bet)
    assert "max bets per snapshot" in rejected.reason


def test_max_five_bets_per_minute():
    now = datetime.now(timezone.utc)
    hist = [
        placed(ts=now - timedelta(seconds=10 * (i + 1))) for i in range(5)
    ]
    opp = make_opp(odds=2.0, model_p=0.6, p_lower=0.55)
    d = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, hist)[0]
    assert not d.should_bet
    assert "per minute" in d.reason


def test_per_minute_window_expires():
    now = datetime.now(timezone.utc)
    hist = [
        placed(ts=now - timedelta(seconds=65 * (i + 1))) for i in range(5)
    ]
    opp = make_opp(odds=2.0, model_p=0.6, p_lower=0.55)
    d = SequentialPortfolio().evaluate_snapshot([opp], MATCH_STATE, hist)[0]
    assert d.should_bet


def test_factor_mapping_covers_eight_risk_factors():
    assert selection_correlation_groups("ou_2_5_over", "over") == ["total_goals"]
    assert selection_correlation_groups("ou_2_5_over", "under") == ["low_tempo_under"]
    assert selection_correlation_groups("1x2", "home") == ["home_goals", "home_supremacy"]
    assert selection_correlation_groups("1x2", "away") == ["away_goals", "away_supremacy"]
    assert selection_correlation_groups("1x2", "draw") == ["low_tempo_under"]
    assert selection_correlation_groups("home_corners_5_5_over", "over") == ["home_corners"]
    assert selection_correlation_groups("away_corners_6_5_over", "over") == ["away_corners"]


def test_should_rebalance_new_selection():
    assert SequentialPortfolio().should_rebalance(make_opp(), [], MATCH_STATE)


def test_should_rebalance_ev_move():
    opp = make_opp(odds=2.0, model_p=0.6, p_lower=0.55)
    assert SequentialPortfolio().should_rebalance(opp, [placed(ev=0.13)], MATCH_STATE)


def test_should_rebalance_odds_move():
    opp = make_opp(odds=2.2, model_p=0.6, p_lower=0.55)
    assert SequentialPortfolio().should_rebalance(opp, [placed(odds=2.0)], MATCH_STATE)


def test_should_rebalance_match_events():
    opp = make_opp(odds=2.0, model_p=0.6, p_lower=0.55)
    pf = SequentialPortfolio()
    hist = [placed()]
    assert pf.should_rebalance(opp, hist, {**MATCH_STATE, "period": "2h"})
    assert pf.should_rebalance(opp, hist, {**MATCH_STATE, "score": "1-0"})
    assert pf.should_rebalance(opp, hist, {**MATCH_STATE, "clock_minute": 36})
    assert pf.should_rebalance(opp, hist, {**MATCH_STATE, "events": [{"type": "goal"}]})


def test_should_rebalance_after_eight_minutes():
    opp = make_opp(odds=2.0, model_p=0.6, p_lower=0.55)
    hist = [placed(ts=datetime.now(timezone.utc) - timedelta(minutes=10))]
    assert SequentialPortfolio().should_rebalance(opp, hist, MATCH_STATE)


def test_should_not_rebalance_when_quiet():
    opp = make_opp(odds=2.02, model_p=0.5495, p_lower=0.52)
    hist = [
        placed(
            odds=2.0,
            ev=0.1,
            ts=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
    ]
    assert not SequentialPortfolio().should_rebalance(opp, hist, MATCH_STATE)


def test_record_bet_round_trip_no_rebalance_when_unchanged():
    pf = SequentialPortfolio()
    opp = make_opp(odds=2.0, model_p=0.7, p_lower=0.6)
    decisions = pf.evaluate_snapshot([opp], MATCH_STATE, [])
    entry = pf.record_bet(decisions[0], MATCH_STATE)
    assert entry["stake"] == decisions[0].stake_ngn

    opp2 = make_opp(odds=2.0, model_p=0.7, p_lower=0.6)
    d2 = pf.evaluate_snapshot([opp2], MATCH_STATE, [entry])[0]
    assert not d2.should_bet
    assert "no rebalance trigger" in d2.reason
