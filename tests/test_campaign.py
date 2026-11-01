"""Tests for the pre-registered campaign verdict (atlas.ops.campaign).

The verdict is a self-terminating rule for the real-money campaign:
collect -> observe (freeze real money) | proven (edge confirmed) | watch.
These tests pin the PRE-REGISTERED thresholds: a verdict that could be
silently re-tuned after the fact is not a pre-registration.
"""

import json

import pytest

from atlas.ops import campaign
from atlas.ops.campaign import campaign_start_ts, cohort, verdict
from atlas.ops.ledger import BetLedger, BetRecord, LedgerConfig


def make_bet(bet_id="b1", stake=10.0, odds=2.0, placed="2026-11-01T13:00:00+00:00"):
    return BetRecord(
        bet_id=bet_id,
        match_id="m1",
        market_key="ou_2_5",
        selection="Over 2.5",
        odds=odds,
        stake_ngn=stake,
        model_p=0.55,
        p_lower=0.50,
        ev=0.10,
        confidence="high",
        correlation_groups=["total_goals"],
        placed_at=placed,
    )


def settle(led, bet_id, status, pnl):
    """Settle with an explicit book PnL (external settlement shape)."""
    led.settle_bet(bet_id, status)
    led._bets[bet_id].pnl_ngn = pnl
    led._bets[bet_id].settled_at = "2026-11-02T10:00:00+00:00"


def make_ledger_with(tmp_path, outcomes, placed="2026-11-01T13:00:00+00:00"):
    """outcomes: iterable of pnl values (won +odds-1 / lost -1 shape)."""
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    for i, pnl in enumerate(outcomes):
        bet = make_bet(f"b{i}", placed=placed)
        led.place_bet(bet)
        # place_bet stamps placed_at with 'now' (the real invariant); the
        # fixture's cohort timestamps are restored so the cutoff logic
        # under test is exercised, not the stamping.
        bet.placed_at = placed
        settle(led, f"b{i}", "won" if pnl > 0 else "lost", pnl)
    return led


# ---- cohort -------------------------------------------------------------

def test_cohort_excludes_pre_campaign_bets(tmp_path):
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    old = make_bet("old", placed="2026-10-31T09:00:00+00:00")
    led.place_bet(old)
    old.placed_at = "2026-10-31T09:00:00+00:00"
    settle(led, "old", "won", 10.0)  # pre-gate churn must NOT be judged
    new = make_bet("new")
    led.place_bet(new)
    settle(led, "new", "lost", -10.0)
    ids = {b.bet_id for b in cohort(led)}
    assert ids == {"new"}


# ---- verdict: the pre-registered rule -----------------------------------

def test_small_cohort_collects(tmp_path):
    led = make_ledger_with(tmp_path, [-10.0] * 30)  # losing hard, but n < 100
    v = verdict(cohort(led))
    assert v["action"] == "collect"
    assert v["n"] == 30


def test_confident_loser_freezes(tmp_path, monkeypatch):
    """roi - 1.96*se well below 0 AND roi + 1.96*se < stop floor -> observe."""
    monkeypatch.setattr(campaign, "MIN_N", 20)
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    # Losing 80% of flat bets: mean per-bet ROI -0.7, sd ~0.8, se ~0.18.
    outcomes = [-10.0, 10.0, -10.0, -10.0, -10.0] * 8  # 40 settled, 8/40 won
    led = make_ledger_with(tmp_path, outcomes)
    v = verdict(cohort(led))
    assert v["action"] == "observe"
    assert v["ci95_pct"] is not None and v["ci95_pct"][1] < 0


def test_proven_requires_ci_and_clv(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "MIN_N", 20)
    # No decision-log evidence at all -> CLV None -> the rule never 'proves'.
    monkeypatch.setattr(
        campaign, "DECISIONS", tmp_path / "absent_decisions.jsonl"
    )
    # 90% winners at odds 2.0: roi +0.8, tiny se -> CI excludes zero,
    # but decision-log CLV is None here -> verdict stays 'watch'.
    outcomes = [10.0] * 36 + [-10.0] * 4
    led = make_ledger_with(tmp_path, outcomes)
    v = verdict(cohort(led))
    assert v["action"] == "watch"  # no CLV evidence -> never 'proven'


def test_proven_with_clv(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "MIN_N", 20)
    outcomes = [10.0] * 36 + [-10.0] * 4
    led = make_ledger_with(tmp_path, outcomes)
    # Fabricate a decision log whose CLV is exactly 0 (placed == last tick).
    ddir = tmp_path / "dec"
    ddir.mkdir()
    rows = []
    for i in range(40):
        rows.append(
            {
                "at": "2026-11-01T14:00:00+00:00",
                "market_key": "ou_2_5",
                "selection": "Over 2.5",
                "odds": 2.0,
            }
        )
    (ddir / "loop_decisions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    monkeypatch.setattr(campaign, "DECISIONS", ddir / "loop_decisions.jsonl")
    v = verdict(cohort(led))
    assert v["action"] == "proven"
    assert v["clv_pct"] == 0.0


def test_frozen_is_fail_closed_on_any_non_proven_state(tmp_path, monkeypatch):
    """MISSING / UNREADABLE / collect / watch / observe — every state that
    is not an explicit 'proven' verdict freezes real money. The old
    explicitly-open rule is how this session bet real NGN at collect
    n=0; this test pins the inversion."""
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    assert campaign.frozen() is True  # missing file
    campaign.STATE_FILE.write_text("{not json", encoding="utf-8")
    assert campaign.frozen() is True  # unreadable file
    for action in ("collect", "watch", "observe"):
        campaign.STATE_FILE.write_text(
            json.dumps({"action": action}), encoding="utf-8"
        )
        assert campaign.frozen() is True, action
    campaign.STATE_FILE.write_text(
        json.dumps({"action": "proven"}), encoding="utf-8"
    )
    assert campaign.frozen() is False  # the only open state


def test_refresh_state_file_below_min_n_returns_collect(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    refresh_v = campaign.refresh_state_file(
        make_ledger_with(tmp_path / "led", [-10.0] * 5)
    )
    assert refresh_v["action"] == "collect"


def test_refresh_state_file_persists_frozen_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(campaign, "MIN_N", 10)
    outcomes = [-10.0, 10.0, -10.0, -10.0, -10.0] * 4
    led = make_ledger_with(tmp_path, outcomes)
    v = campaign.refresh_state_file(led)
    assert v["action"] == "observe"
    assert campaign.frozen() is True


# ---- enforcement at the last wall ---------------------------------------

def test_ledger_refuses_placement_when_frozen(tmp_path, monkeypatch):
    """The freeze is enforced in BetLedger.place_bet, the LAST wall."""
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    campaign.STATE_FILE.write_text(
        json.dumps({"action": "observe"}), encoding="utf-8"
    )
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    assert led.can_place_bet(10.0) is False
    bet = make_bet("b1")
    assert led.place_bet(bet) is False
    assert bet.bet_id not in led._bets  # journaled, never a live position


def test_ledger_refuses_placement_when_verdict_watch(tmp_path, monkeypatch):
    """FAIL-CLOSED: under the old rule 'watch' traded real money while
    collecting; under the inversion only 'proven' trades. Collect-phase
    evidence is the paper stream, not the wallet."""
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    campaign.STATE_FILE.write_text(
        json.dumps({"action": "watch"}), encoding="utf-8"
    )
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    assert led.can_place_bet(10.0) is False
    bet = make_bet("b1")
    assert led.place_bet(bet) is False
    assert bet.bet_id not in led._bets


def test_ledger_allows_placement_when_proven(tmp_path, monkeypatch):
    """The only state that opens the wall: the pre-registered verdict
    reads 'proven' — edge confirmed at the MIN_N bar."""
    monkeypatch.setattr(campaign, "STATE_FILE", tmp_path / "state.json")
    campaign.STATE_FILE.write_text(
        json.dumps({"action": "proven"}), encoding="utf-8"
    )
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    assert led.can_place_bet(10.0) is True
    assert led.place_bet(make_bet("b1")) is True


def test_missing_state_file_freezes_placement(tmp_path, monkeypatch):
    """Fail-closed on the ledger wall too: no campaign state file means
    the wall is SHUT, not open."""
    monkeypatch.setattr(
        campaign, "STATE_FILE", tmp_path / "absent-state.json"
    )
    led = BetLedger(LedgerConfig(), log_path=str(tmp_path / "l.jsonl"))
    led.config.daily_loss_limit = -10_000  # isolate the campaign gate
    assert led.can_place_bet(10.0) is False
    bet = make_bet("b1")
    assert led.place_bet(bet) is False
    assert bet.bet_id not in led._bets
