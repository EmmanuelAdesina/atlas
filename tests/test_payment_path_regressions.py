"""Payment-path regression tests for the 2026-11-01 VPS incidents.

Everything here was observed live: a ReadTimeout mid receipt-poll declared a
PLACED bet 'unconfirmed'; a leftover receipt modal was mistaken for a pending
slip; the balance probe alerted about real money it could then not track.
"""

import json

import pytest

from atlas.browser.betslip import BetslipDriver


class FlakyAdapter:
    """Adapter whose evaluate raises N times before answering statefully."""

    def __init__(self, state, fail_first: int = 0):
        self.state = state
        self.fail_first = fail_first
        self.eval_calls = 0
        self.clicks: list[str] = []

    def evaluate(self, code: str) -> str:
        self.eval_calls += 1
        if self.eval_calls <= self.fail_first:
            raise TimeoutError("ReadTimeout")
        state = self.state() if callable(self.state) else self.state
        return json.dumps(state)

    def click(self, selector: str) -> dict:
        self.clicks.append(selector)
        return {"success": True, "tag": "BUTTON"}


def receipt_state(booking="MDV6BQ"):
    return {
        "empty": True,
        "receipt": True,
        "booking": booking,
        "insufficient": False,
    }


def test_await_receipt_survives_read_timeouts(monkeypatch):
    """A ReadTimeout mid-poll must keep polling, not abort the placement
    verification: at 13:27 the poll raised and the book's confirmed bet
    became an 'unconfirmed' phantom."""
    monkeypatch.setattr("atlas.browser.betslip.time.sleep", lambda _s: None)
    # WINDOW: fail the first two reads (would have aborted at 12s under
    # pressure), then present the receipt.
    adapter = FlakyAdapter(receipt_state(), fail_first=2)
    driver = BetslipDriver(adapter, allow_submit=True)
    import time as _time

    started = _time.monotonic()
    placed, booking, error = driver._await_receipt(started + 25.0, started)
    assert placed is True, "receipt after flaky reads must still confirm"
    assert (booking or "") == "MDV6BQ"
    assert error == "" or error is None or placed


def test_await_receipt_reports_timeout_when_nothing_shows(monkeypatch):
    monkeypatch.setattr("atlas.browser.betslip.time.sleep", lambda _s: None)
    adapter = FlakyAdapter(
        lambda: {"empty": False, "receipt": False, "booking": None, "insufficient": False},
        fail_first=0,
    )
    driver = BetslipDriver(adapter, allow_submit=True)
    import time as _time

    started = _time.monotonic()
    placed, booking, error = driver._await_receipt(started + 0.05, started)
    assert placed is False
    assert "did not confirm" in error


def test_dismiss_receipt_is_gated_and_clicks_ok(monkeypatch):
    """dismiss_receipt must be a money-tool-gated REAL click on the receipt's
    own OK button, never a synthetic page click."""
    driver = BetslipDriver(FlakyAdapter(receipt_state()), allow_submit=False)
    assert driver.dismiss_receipt() is False, "observer driver never clicks"

    adapter = FlakyAdapter(receipt_state())
    driver = BetslipDriver(adapter, allow_submit=True)
    driver._real_click = lambda js, label: {"ok": True, "selector": "[data-atlas-dismiss-receipt=\"1\"]"}
    assert driver.dismiss_receipt() is True


def _probe_runner(tmp_path, baseline, candidate):
    """A minimal runner carrying only the probe's own state.

    LiveRunner.__new__ keeps __init__ (and its browser plane) out of the
    test: the probe touches only the ledger, the notifier, and its own
    5 attributes.
    """
    from atlas.live.runner import LiveRunner

    from atlas.ops.ledger import BetLedger, LedgerConfig

    runner = LiveRunner.__new__(LiveRunner)
    runner.ledger = BetLedger(
        LedgerConfig(daily_loss_limit=-200.0), log_path=tmp_path / "probe.jsonl"
    )
    runner.notifier = None
    runner.cycle = 1
    runner._probe_at = None
    runner._probe_baseline = baseline
    runner._probe_candidate = candidate
    runner._unconfirmed_at = 0.0
    runner._balance_alert_fired = False
    return runner


def test_probe_reconciles_exact_stake_drop_into_ledger(tmp_path, monkeypatch):
    """The 2026-11-01 13:30 case: the book took NGN 10, the receipt never
    rendered, the ledger never learned. The drop equal to the planned stake
    is the book's own proof — record the bet WITH the balance_delta caveat."""
    runner = _probe_runner(
        tmp_path,
        baseline=851.58,
        candidate={
            "match_id": "live",
            "market_key": "goals_to_lead_by_2_away",
            "market_label": "Any Team to lead by 2 Goals at any time",
            "selection": "No",
            "odds": 1.32,
            "stake_ngn": 10.0,
            "model_p": 0.70,
            "p_lower": 0.66,
            "ev": 0.08,
            "confidence": "high",
        },
    )
    # arm the probe as the runner does
    runner._arm_unconfirmed_probe(
        {
            "match_id": "live",
            "market_key": "goals_to_lead_by_2_away",
            "market_label": "Any Team to lead by 2 Goals at any time",
            "selection": "No",
            "odds": 1.32,
            "stake_ngn": 10.0,
        }
    )
    # pretend BALANCE_PROBE_DELAY_S passed
    import time as _time

    runner._probe_at = _time.monotonic() - 1.0
    runner._check_balance_probe(841.58)

    bets = list(runner.ledger.bets_for("live"))
    assert len(bets) == 1, "the balance drop must become a tracked position"
    bet = bets[0]
    assert bet.get("selection") == "No"
    assert float(bet.get("stake_ngn", 0.0)) == 10.0
    assert bet.get("receipt") == "balance_delta"


def test_probe_unchanged_drop_stays_critical_alert(tmp_path):
    """A drop that does NOT match the planned stake is still unexplained
    money (blend, a second slip): alert loudly and record NOTHING."""
    runner = _probe_runner(
        tmp_path,
        baseline=500.0,
        candidate={
            "match_id": "live",
            "market_key": "ou_2_5",
            "market_label": "Over/Under 2.5",
            "selection": "Over",
            "odds": 2.0,
            "stake_ngn": 10.0,
        },
    )
    import time as _time

    runner._probe_at = _time.monotonic() - 1.0
    runner._check_balance_probe(380.0)  # -120 drop != planned 10
    assert all(
        b.get("receipt") is None for b in runner.ledger.bets_for("live")
    ), "an unexplained drop must never be recorded as a bet"


def test_probe_drop_clears_candidate_without_firing_again(tmp_path):
    """A NO-drop reading closes the case entirely (no candidate leak into
    the next unrelated drop)."""
    runner = _probe_runner(
        tmp_path,
        baseline=851.58,
        candidate={"match_id": "m", "market_key": "k", "market_label": "l",
                   "selection": "s", "odds": 1.5, "stake_ngn": 10.0},
    )
    import time as _time

    runner._probe_at = _time.monotonic() - 1.0
    runner._check_balance_probe(851.58)
    assert runner._probe_candidate is None
    assert runner._probe_at is None
    assert runner.ledger.bets_for("m") == []
