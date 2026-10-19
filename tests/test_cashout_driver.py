"""Tests for the cashout driver (atlas.browser.cashout)."""

import json

import pytest

from atlas.browser.cashout import CashoutDriver


class FakeAdapter:
    def __init__(self, read_payload=None, tab_ok=True):
        self.read_payload = read_payload
        self.tab_ok = tab_ok
        self.codes = []

    def evaluate(self, code: str):
        self.codes.append(code)
        if "m-cashout-item" in code and "no cashout button" not in code:
            if self.read_payload is None:
                return {"data": {"value": json.dumps({"items": [], "oneCut": False})}}
            return {"data": {"value": json.dumps(self.read_payload)}}
        if "no cashout tab" in code or "/^Cashout/i" in code:
            return {"data": {"value": json.dumps({"ok": self.tab_ok})}}
        return {"data": {"value": json.dumps({"ok": True})}}


def test_reads_a_cashout_offer():
    payload = {
        "items": [
            {
                "i": 0,
                "txt": "Multiple (x1) | F | 18+ of 19 | Frosinone v Sassuolo | "
                "Bet Stake | 160.00 | Pot. Win | 38,498.89 | Full Cashout 246.53",
                "btns": ["Cashout now|"],
            }
        ],
        "oneCut": False,
    }
    driver = CashoutDriver(FakeAdapter(payload))
    offers = driver.read_offers()
    assert len(offers) == 1
    o = offers[0]
    assert o.offered_ngn == pytest.approx(246.53)
    assert o.stake_ngn == pytest.approx(160.0)
    assert o.potential_win_ngn == pytest.approx(38498.89)
    assert o.legs_known == 18
    assert o.legs_total == 19


def test_no_offers_is_not_an_error():
    driver = CashoutDriver(FakeAdapter())
    assert driver.read_offers() == []


def test_a_broken_adapter_does_not_raise():
    class Broken:
        def evaluate(self, code):
            raise RuntimeError("browser gone")

    driver = CashoutDriver(Broken())
    assert driver.read_offers() == []
    assert driver.partial_cashout_available() is False
    assert driver.execute(0)["ok"] is False


def test_execution_is_refused_without_an_explicit_opt_in():
    driver = CashoutDriver(FakeAdapter())
    out = driver.execute(0)
    assert out["ok"] is False
    assert "disabled" in out["error"]


def test_execution_clicks_the_cashout_button_when_enabled():
    adapter = FakeAdapter()
    driver = CashoutDriver(adapter, allow_execute=True)
    out = driver.execute(0)
    assert out["ok"] is True
    assert any("no cashout button" in c for c in adapter.codes)


def test_one_cut_detection():
    payload = {"items": [], "oneCut": True}
    driver = CashoutDriver(FakeAdapter(payload))
    assert driver.partial_cashout_available() is True
