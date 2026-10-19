"""Tests for the betslip driver contract (atlas.browser.betslip).

The DOM contract was established against the live board; these tests lock in
the safety properties and the pure logic, without needing a browser.
"""

import json

import pytest

from atlas.browser.betslip import (
    DEFAULT_STAKE_NGN,
    MIN_STAKE_NGN,
    BetslipDriver,
    BetslipError,
    SubmissionNotAllowed,
    validate_stake,
)


class FakeAdapter:
    """Minimal stand-in for WebBridgeAdapter: returns canned JSON strings."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.codes: list[str] = []

    def evaluate(self, code: str) -> str:
        self.codes.append(code)
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({"ok": False, "error": "no canned response"})


def test_minimum_stake_is_ten_ngn():
    assert MIN_STAKE_NGN == 10.0
    assert DEFAULT_STAKE_NGN == 100.0


def test_validate_stake_rejects_below_minimum():
    assert "below minimum" in validate_stake(9.99)
    assert validate_stake(10.0) == ""
    assert validate_stake(100.0) == ""


def test_validate_stake_rejects_more_than_balance():
    reason = validate_stake(100.0, balance_ngn=7.66)
    assert "exceeds available balance" in reason
    # With a NGN 7.66 balance even the NGN 10 floor is unaffordable.
    assert "exceeds available balance" in validate_stake(10.0, balance_ngn=7.66)
    assert validate_stake(100.0, balance_ngn=500.0) == ""


def test_place_bet_is_blocked_without_explicit_opt_in():
    driver = BetslipDriver(FakeAdapter())
    with pytest.raises(SubmissionNotAllowed):
        driver.place_bet()


def test_place_bet_never_runs_when_slip_not_placeable():
    adapter = FakeAdapter([json.dumps({"seg": [], "stake": "100", "canPlace": False})])
    driver = BetslipDriver(adapter, allow_submit=True)
    with pytest.raises(BetslipError):
        driver.place_bet()
    # Only the read was evaluated: no click payload was ever sent.
    assert len(adapter.codes) == 1
    assert "btn.click()" not in adapter.codes[0]


def test_set_stake_refuses_below_minimum():
    driver = BetslipDriver(FakeAdapter())
    with pytest.raises(BetslipError):
        driver.set_stake(5.0)


def test_add_odds_targets_the_exact_decimal_price():
    adapter = FakeAdapter([json.dumps({"ok": True, "odds": "5.20"})])
    driver = BetslipDriver(adapter)
    out = driver.add_odds(5.2)
    assert out["ok"] is True
    code = adapter.codes[0]
    assert ".m-table-row.m-outcome" in code
    assert "'5.20'" in code


def test_clear_never_clicks_place_bet():
    adapter = FakeAdapter([json.dumps({"removed": 2})])
    driver = BetslipDriver(adapter)
    out = driver.clear()
    assert out["removed"] == 2
    assert "place bet" not in adapter.codes[0].lower()


def test_read_slip_parses_stake():
    adapter = FakeAdapter(
        [json.dumps({"seg": ["Total Stake", "Potential Win"], "stake": "150.00", "canPlace": True})]
    )
    driver = BetslipDriver(adapter)
    slip = driver.read_slip()
    assert slip.total_stake_ngn == 150.0
    assert slip.can_place_bet is True
    assert "Total Stake" in slip.raw_text
