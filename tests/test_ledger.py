"""Tests for atlas.ops.ledger — the NGN bet ledger and daily loss stop.

These exercise the REAL place_bet/settle_bet path. An earlier version used a
non-reentrant lock: place_bet held the lock while calling is_daily_stop_active
-> daily_pnl, which re-acquired it and deadlocked the whole process.
"""

from datetime import datetime, timezone

import pytest

from atlas.ops.ledger import BetLedger, BetRecord, LedgerConfig


def make_bet(bet_id="b1", stake=100.0, odds=2.1, match_id="m1"):
    return BetRecord(
        bet_id=bet_id,
        match_id=match_id,
        market_key="ou_2_5",
        selection="Over 2.5",
        odds=odds,
        stake_ngn=stake,
        model_p=0.55,
        p_lower=0.50,
        ev=0.155,
        confidence="high",
        correlation_groups=["total_goals"],
    )


def test_place_bet_does_not_deadlock(tmp_path):
    """Regression: place_bet -> is_daily_stop_active -> daily_pnl re-enters lock."""
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    assert led.place_bet(make_bet("b1")) is True
    assert led.summary()["placed"] == 1


def test_exposure_counts_unsettled_only(tmp_path):
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    led.place_bet(make_bet("b1", stake=100.0))
    led.place_bet(make_bet("b2", stake=200.0))
    assert led.match_exposure("m1") == 300.0
    led.settle_bet("b1", "lost")
    assert led.match_exposure("m1") == 200.0


def test_pnl_math_win_loss_void(tmp_path):
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    led.place_bet(make_bet("b1", stake=100.0, odds=2.1))
    led.place_bet(make_bet("b2", stake=100.0, odds=2.1))
    led.place_bet(make_bet("b3", stake=100.0, odds=2.1))
    led.settle_bet("b1", "won")
    led.settle_bet("b2", "lost")
    led.settle_bet("b3", "void")
    # win: stake*(odds-1) = 110 ; loss: -100 ; void: 0  =>  +10
    assert led.daily_pnl() == pytest.approx(10.0)


def test_daily_loss_stop_trips_at_minus_500(tmp_path):
    led = BetLedger(LedgerConfig(daily_loss_limit=-500.0), log_path=str(tmp_path / "l.jsonl"))
    for i in range(5):
        led.place_bet(make_bet("b%d" % i, stake=100.0))
    for i in range(5):
        led.settle_bet("b%d" % i, "lost")
    assert led.daily_pnl() == pytest.approx(-500.0)
    assert led.is_daily_stop_active() is True
    assert led.can_place_bet(100.0) is False
    # Hard stop: even a new bet cannot be placed.
    assert led.place_bet(make_bet("b9", stake=100.0)) is False


def test_daily_stop_lifts_once_recovered(tmp_path):
    led = BetLedger(LedgerConfig(daily_loss_limit=-500.0), log_path=str(tmp_path / "l.jsonl"))
    for i in range(6):
        led.place_bet(make_bet("b%d" % i, stake=100.0))
    for i in range(5):
        led.settle_bet("b%d" % i, "lost")
    assert led.is_daily_stop_active() is True
    led.settle_bet("b5", "won")  # +110 -> -390
    assert led.is_daily_stop_active() is False
    assert led.can_place_bet(100.0) is True


def test_stake_bounds_enforced(tmp_path):
    led = BetLedger(LedgerConfig(min_stake=100.0, max_stake=1000.0), log_path=str(tmp_path / "l.jsonl"))
    assert led.place_bet(make_bet("small", stake=50.0)) is False
    assert led.place_bet(make_bet("big", stake=5000.0)) is False
    assert led.place_bet(make_bet("ok", stake=500.0)) is True


def test_invalid_settle_status_rejected(tmp_path):
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    led.place_bet(make_bet("b1"))
    with pytest.raises(ValueError):
        led.settle_bet("b1", "definitely_won")


def test_ledger_reloads_from_jsonl(tmp_path):
    p = str(tmp_path / "l.jsonl")
    led = BetLedger(LedgerConfig(), log_path=p)
    led.place_bet(make_bet("b1", stake=300.0))
    led.settle_bet("b1", "won")
    reopened = BetLedger(LedgerConfig(), log_path=p)
    # win on 300 @ 2.1 => 300 * (2.1 - 1) = 330
    assert reopened.match_pnl("m1") == pytest.approx(330.0)
    assert reopened.daily_pnl() == pytest.approx(330.0)


def test_summary_shape(tmp_path):
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    led.place_bet(make_bet("b1", stake=100.0))
    led.place_bet(make_bet("b2", stake=100.0))
    led.settle_bet("b1", "won")
    s = led.summary()
    assert s["total_bets"] == 2
    assert s["won"] == 1
    assert s["placed"] == 1
    assert s["unrealized_exposure_ngn"] == 100.0
    assert s["currency"] == "NGN"