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


def test_external_bet_settled_event_updates_the_placed_bet(tmp_path):
    """Bet History recon writes `bet_settled` events with the BOOK's truth.

    Regression 2026-11-01: the loader constructed a BetRecord from every
    event's record, so the first external settlement crashed every later
    loop start with TypeError: unexpected keyword argument 'settle_ngn'.
    The ledger must ingest them as settlement UPDATES on the placed bet:
    the book's status and PnL are authoritative over anything local, and
    the recon's 'win'/'loss' vocabulary must be normalized to the
    ledger's 'won'/'lost'.
    """
    import json as _json

    p = tmp_path / "l.jsonl"
    led = BetLedger(LedgerConfig(), log_path=str(p))
    led.place_bet(make_bet("b1", stake=10.0, odds=2.3))
    settled_ev = {
        "event": "bet_settled",
        "timestamp": "2026-11-01T10:00:00+00:00",
        "record": {
            "bet_id": "b1",
            "match_id": "m1",
            "market_key": "ou_2_5",
            "selection": "Over 2.5",
            "stake_ngn": 10.0,
            "settle_ngn": 14.2,
            "pnl_ngn": 4.2,
            "status": "win",
            "book_time": "01/11/2026 01:14",
            "book_status": "Won",
            "source": "bet_history_recon",
        },
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(settled_ev) + "\n")
    reopened = BetLedger(LedgerConfig(), log_path=str(p))
    s = reopened.summary()
    assert s["won"] == 1, s
    assert s["total_pnl_ngn"] == pytest.approx(4.2)
    assert s["total_staked_ngn"] == pytest.approx(10.0)
    bet = reopened._bets["b1"]
    assert bet.status == "won"
    assert bet.settle_ngn == pytest.approx(14.2)
    assert bet.book_status == "Won"
    assert bet.settle_source == "bet_history_recon"
    assert bet.settled_at is not None


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