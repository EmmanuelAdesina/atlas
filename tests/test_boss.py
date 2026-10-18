"""Tests for the Boss LLM decision protocol (atlas.ops.boss).

The Boss never invents probabilities — tests feed it model outputs and
board data only, and assert on decisions, risk ratchets, and drift flags.
"""

from datetime import datetime, timedelta, timezone

import pytest

from atlas.live.decisions import PolicyConfig
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.ops.boss import (
    DAILY_LOSS_LIMIT_NGN,
    BetLedger,
    Boss,
    BossDecision,
    MatchCandidate,
)


def make_state(
    period: str = "2h",
    observed_at: datetime | None = None,
    match_id: str = "m1",
) -> LiveMatchState:
    return LiveMatchState(
        match_id=match_id,
        home=TeamStats(name="A"),
        away=TeamStats(name="B"),
        period=period,
        clock_minute=55.0,
        observed_at=observed_at or datetime.now(timezone.utc),
        source="test",
    )


def make_board() -> list[dict]:
    return [
        {
            "match_id": "deep-2h",
            "home_team": "Riverside",
            "away_team": "Coastal",
            "score": "1-0",
            "minute": 55.0,
            "period": "2h",
            "num_markets": 20,
            "total_volume": 1_000_000.0,
            "edges": [
                {"market_key": "ou_2_5", "selection": "Over 2.5",
                 "odds": 1.8, "model_p": 0.6, "ev": 0.08},
                {"market_key": "btts", "selection": "Yes",
                 "odds": 1.7, "model_p": 0.58, "ev": 0.026},
            ],
        },
        {
            "match_id": "thin-1h",
            "home_team": "Hilltop",
            "away_team": "Valley",
            "score": "0-0",
            "minute": 30.0,
            "period": "1h",
            "num_markets": 6,
            "total_volume": 50_000.0,
            "edges": [
                {"market_key": "1x2", "selection": "Home",
                 "odds": 2.1, "model_p": 0.5, "ev": 0.05},
            ],
        },
        {
            "match_id": "late-2h",
            "home_team": "Northgate",
            "away_team": "Southport",
            "score": "2-2",
            "minute": 88.0,
            "period": "2h",
            "num_markets": 12,
            "total_volume": 200_000.0,
            "edges": [
                {"market_key": "ou_3_5", "selection": "Over 3.5",
                 "odds": 2.4, "model_p": 0.45, "ev": 0.08},
            ],
        },
    ]


def make_boss(ledger: BetLedger | None = None, **policy_kwargs) -> Boss:
    policy = PolicyConfig(**policy_kwargs)
    return Boss(ledger if ledger is not None else BetLedger(), policy)


# --- Risk governor ---


@pytest.mark.asyncio
async def test_kill_switch_breaches_daily_loss_limit():
    ledger = BetLedger()
    ledger.record_pending(
        {"match_id": "m1", "market_key": "ou_1_5", "selection": "Under 1.5",
         "stake_ngn": 600.0, "odds": 1.5, "model_p": 0.7}
    )
    ledger.settle("m1", "ou_1_5", "Under 1.5", "loss", payout_ngn=0.0)
    boss = make_boss(ledger=ledger)
    decision = await boss.evaluate_operation(
        "m1", make_state(), [], []
    )
    assert decision.action == "stop"
    assert "500" in decision.reason
    assert ledger.daily_pnl_ngn == pytest.approx(-600.0)


@pytest.mark.asyncio
async def test_continue_when_operation_is_healthy():
    boss = make_boss()
    decision = await boss.evaluate_operation(
        "m1", make_state(), [], []
    )
    assert decision.action == "continue"
    assert decision.reason


@pytest.mark.asyncio
async def test_policy_kill_switch_stops_loop():
    boss = make_boss(kill_switch=True)
    decision = await boss.evaluate_operation(
        "m1", make_state(), [], []
    )
    assert decision.action == "stop"
    assert "kill_switch" in decision.reason


# --- Anomaly handler ---


@pytest.mark.asyncio
async def test_stale_feed_pauses():
    stale_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    boss = make_boss()
    decision = await boss.evaluate_operation(
        "m1", make_state(observed_at=stale_at), [], []
    )
    assert decision.action == "pause"
    assert "stale_feed" in decision.reason
    assert decision.risk_override is not None


@pytest.mark.asyncio
async def test_session_expired_alerts():
    boss = make_boss()
    placed = [
        {"match_id": "m1", "market_key": "ou_1_5", "selection": "Under 1.5",
         "stake_ngn": 100.0, "odds": 1.5, "model_p": 0.7,
         "session_expired": True}
    ]
    decision = await boss.evaluate_operation("m1", make_state(), placed, [])
    assert decision.action == "alert"
    assert "session_expired" in decision.reason


@pytest.mark.asyncio
async def test_price_mismatch_alerts():
    boss = make_boss()
    placed = [
        {"match_id": "m1", "market_key": "ou_1_5", "selection": "Under 1.5",
         "stake_ngn": 100.0, "odds": 1.5, "model_p": 0.7,
         "clicked_odds": 2.10, "accepted_odds": 2.00}
    ]
    decision = await boss.evaluate_operation("m1", make_state(), placed, [])
    assert decision.action == "alert"
    assert "price_mismatch" in decision.reason


@pytest.mark.asyncio
async def test_odds_disappeared_with_exposure_alerts():
    boss = make_boss()
    open_bet = {"match_id": "m1", "market_key": "ou_1_5",
                "selection": "Under 1.5", "stake_ngn": 500.0,
                "odds": 1.5, "model_p": 0.7, "status": "open"}
    boss._track_markets([open_bet])
    dead_bet = {**open_bet, "status": "suspended"}
    decision = await boss.evaluate_operation("m1", make_state(), [dead_bet], [])
    assert decision.action == "alert"
    assert "odds_disappeared" in decision.reason


@pytest.mark.asyncio
async def test_odds_disappeared_without_exposure_switches():
    boss = make_boss()
    open_bet = {"match_id": "m1", "market_key": "ou_1_5",
                "selection": "Under 1.5", "stake_ngn": 0.0,
                "odds": 1.5, "model_p": 0.7, "status": "open"}
    boss._track_markets([open_bet])
    dead_bet = {**open_bet, "status": "suspended"}
    decision = await boss.evaluate_operation(
        "m1", make_state(), [dead_bet], make_board()
    )
    assert decision.action == "switch_match"
    assert decision.target_match_id == "deep-2h"


# --- Match selector ---


def test_select_next_match_prefers_depth_liquidity_and_2h():
    boss = make_boss()
    candidate = boss.select_next_match(make_board())
    assert isinstance(candidate, MatchCandidate)
    assert candidate.match_id == "deep-2h"
    assert candidate.num_markets == 20
    assert candidate.total_volume == pytest.approx(1_000_000.0)
    assert len(candidate.top_edges) == 2
    assert candidate.top_edges[0]["ev"] == pytest.approx(0.08)


def test_select_next_match_phase_preference_when_else_equal():
    boss = make_boss()
    board = [
        {"match_id": "a-2h", "home_team": "A", "away_team": "B", "score": "0-0",
         "minute": 50.0, "period": "2h", "num_markets": 10,
         "total_volume": 100_000.0,
         "edges": [{"market_key": "1x2", "selection": "Home", "odds": 2.0,
                    "model_p": 0.55, "ev": 0.10}]},
        {"match_id": "b-1h", "home_team": "C", "away_team": "D", "score": "0-0",
         "minute": 50.0, "period": "1h", "num_markets": 10,
         "total_volume": 100_000.0,
         "edges": [{"market_key": "1x2", "selection": "Home", "odds": 2.0,
                    "model_p": 0.55, "ev": 0.10}]},
    ]
    candidate = boss.select_next_match(board)
    assert candidate.match_id == "a-2h"


def test_select_next_match_returns_none_on_empty_or_thin_board():
    boss = make_boss()
    assert boss.select_next_match([]) is None
    thin = [{"match_id": "x", "home_team": "A", "away_team": "B",
             "score": "0-0", "minute": 10.0, "period": "1h",
             "num_markets": 1, "total_volume": 1000.0, "edges": []}]
    assert boss.select_next_match(thin) is None


@pytest.mark.asyncio
async def test_ended_match_switches_to_best_candidate():
    boss = make_boss()
    decision = await boss.evaluate_operation(
        "m1", make_state(period="ft"), [], make_board()
    )
    assert decision.action == "switch_match"
    assert decision.target_match_id == "deep-2h"


@pytest.mark.asyncio
async def test_ended_match_with_no_board_pauses():
    boss = make_boss()
    decision = await boss.evaluate_operation(
        "m1", make_state(period="ft"), [], []
    )
    assert decision.action == "pause"


# --- Stake sizing ---


def test_kelly_stake_within_policy_caps():
    boss = make_boss(bankroll_ngn=10_000.0, kelly_fraction=0.25,
                     max_stake_per_bet=1_000.0, max_exposure_per_match=3_000.0,
                     min_stake_ngn=100.0)
    stake = boss.kelly_stake(0.6, 2.0)
    assert stake in (499.0, 500.0)
    assert stake <= boss.policy.max_stake_per_bet


def test_kelly_stake_capped_by_max_stake():
    boss = make_boss(bankroll_ngn=10_000.0, kelly_fraction=0.25,
                     max_stake_per_bet=1_000.0)
    assert boss.kelly_stake(0.9, 10.0) == pytest.approx(1_000.0)


def test_kelly_stake_capped_by_remaining_exposure():
    boss = make_boss(bankroll_ngn=10_000.0, kelly_fraction=0.25,
                     max_stake_per_bet=1_000.0, max_exposure_per_match=3_000.0)
    assert boss.kelly_stake(0.6, 2.0, current_match_exposure=2_500.0) in (
        499.0, 500.0
    )


def test_kelly_stake_zero_below_min_stake_or_no_edge():
    boss = make_boss(bankroll_ngn=10_000.0, kelly_fraction=0.25,
                     min_stake_ngn=100.0)
    assert boss.kelly_stake(0.679, 1.5) == 0.0
    assert boss.kelly_stake(0.5, 1.9) == 0.0
    assert boss.kelly_stake(None, 2.0) == 0.0


# --- Post-match review ---


def test_review_match_flags_drift_when_realized_trails_predicted():
    ledger = BetLedger()
    for _ in range(2):
        ledger.record_pending(
            {"match_id": "m1", "market_key": "ou_1_5", "selection": "Under 1.5",
             "stake_ngn": 500.0, "odds": 1.8, "model_p": 0.7}
        )
        ledger.settle("m1", "ou_1_5", "Under 1.5", "loss", payout_ngn=0.0)
    boss = Boss(ledger, PolicyConfig())
    review = boss.review_match("m1")
    assert review["settled_bets"] == 2
    assert review["realized_pnl_ngn"] == pytest.approx(-1_000.0)
    assert review["drift_flag"] is True
    assert "realized ROI" in review["drift_reason"]
    assert review["action"] == "monitor_model"


def test_review_match_ok_when_realized_matches_prediction():
    ledger = BetLedger()
    ledger.record_pending(
        {"match_id": "m1", "market_key": "ou_1_5", "selection": "Under 1.5",
         "stake_ngn": 500.0, "odds": 1.1, "model_p": 0.6}
    )
    ledger.settle("m1", "ou_1_5", "Under 1.5", "win", payout_ngn=550.0)
    boss = Boss(ledger, PolicyConfig())
    review = boss.review_match("m1")
    assert review["drift_flag"] is False
    assert review["action"] == "ok"


# --- Risk ratchet: tighten-only ---


@pytest.mark.asyncio
async def test_tightening_never_loosens_policy():
    boss = make_boss(min_ev=0.05, max_bets_per_match=3, kelly_fraction=0.25)
    stale_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    decision = await boss.evaluate_operation(
        "m1", make_state(observed_at=stale_at), [], []
    )
    override = decision.risk_override
    assert override is not None
    assert override["min_ev"] >= boss.policy.min_ev
    assert override["max_bets_per_match"] <= boss.policy.max_bets_per_match
    assert override["kelly_fraction"] <= boss.policy.kelly_fraction

    tightened = boss.apply_to_policy(boss.policy)
    assert tightened.min_ev >= boss.policy.min_ev
    assert tightened.max_bets_per_match <= boss.policy.max_bets_per_match
    assert tightened.kelly_fraction <= boss.policy.kelly_fraction


@pytest.mark.asyncio
async def test_negative_pnl_ratchets_risk_without_loosening():
    ledger = BetLedger()
    ledger.record_pending(
        {"match_id": "m1", "market_key": "ou_1_5", "selection": "Under 1.5",
         "stake_ngn": 200.0, "odds": 1.5, "model_p": 0.7}
    )
    ledger.settle("m1", "ou_1_5", "Under 1.5", "loss", payout_ngn=0.0)
    boss = make_boss(ledger=ledger)
    decision = await boss.evaluate_operation("m1", make_state(), [], [])
    assert decision.action == "continue"
    assert decision.risk_override is not None
    assert decision.risk_override["min_ev"] > boss.policy.min_ev


# --- Decision logging ---


@pytest.mark.asyncio
async def test_decisions_are_logged_with_reasons(tmp_path):
    log_path = tmp_path / "boss_decisions.jsonl"
    boss = Boss(BetLedger(), PolicyConfig(), decision_log_path=log_path)
    await boss.evaluate_operation("m1", make_state(), [], [])
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    import json

    entry = json.loads(lines[0])
    assert entry["action"] == "continue"
    assert entry["reason"]
    assert len(boss.decisions) == 1
