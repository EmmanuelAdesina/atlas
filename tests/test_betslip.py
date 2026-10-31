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
        self.clicks: list[str] = []

    def click(self, selector: str) -> dict:
        self.clicks.append(selector)
        return {"success": True, "tag": "BUTTON", "text": selector}

    def evaluate(self, code: str) -> str:
        self.codes.append(code)
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({"ok": False, "error": "no canned response"})


def test_minimum_stake_is_ten_ngn():
    assert MIN_STAKE_NGN == 10.0
    assert DEFAULT_STAKE_NGN == 100.0


class SweepStateAdapter:
    """Stateful fake of the book's slip: REAL clicks remove one card each.

    The previous scripted-response versions depended on the exact number of
    page polls; clear() polls (0.1s / 0.6s), so a fixed queue drifts out of
    step and the tests fail for the wrong reason (2026-10-29).
    """

    def __init__(self, removals_needed: int = 2):
        self.remaining = removals_needed
        self.clicks: list[str] = []
        self.sweep_marks = 0

    def evaluate(self, code: str) -> str:
        # ORDER MATTERS: the marker payload itself mentions 'Remove All'
        # (it looks for that control), so it must be classified BEFORE
        # JS_CLEAR or every sweep call is mistaken for the legacy pass.
        if "data-atlas-slip-remove" in code:  # JS_MARK_SLIP_REMOVE
            self.sweep_marks += 1
            return json.dumps(
                {"ok": True, "selector": '[data-atlas-slip-remove="1"]'}
            )
        if "Submission Successful" in code:  # JS_SLIP_EMPTY
            return json.dumps(
                {
                    "empty": self.remaining <= 0,
                    "receipt": False,
                    "booking": None,
                    "insufficient": False,
                }
            )
        if "Remove All" in code:  # JS_CLEAR: synthetic pass the book ignores
            return json.dumps(
                {"ok": True, "how": "per-selection", "removed": 0}
            )
        return json.dumps({"ok": False, "error": "unexpected script"})

    def click(self, selector: str) -> dict:
        self.clicks.append(selector)
        self.remaining -= 1
        return {"success": True, "tag": "BUTTON"}


def test_clear_sweeps_unavailable_selection_with_real_clicks():
    """Ground truth 2026-10-28, FC Goa v Sporting Club Delhi: a suspended
    market leaves the slip holding one selection marked Unavailable behind
    an 'Accept Changes' button. Synthetic JS clicks are IGNORED by this
    build, so clear() must fall through to the REAL-click sweep and drain
    the card. The sweep stops as soon as the slip is empty."""
    adapter = SweepStateAdapter(removals_needed=2)
    driver = BetslipDriver(adapter, allow_submit=False)
    result = driver.clear()
    assert result.get("ok") is True
    assert result.get("swept_clicks", 0) >= 1
    assert len(adapter.clicks) == 2, "two cards, two REAL clicks"
    # REAL clicks went through the extension on the marked selector:
    assert all("[data-atlas-slip-remove" in c for c in adapter.clicks), (
        "sweep must use REAL clicks on the marked remove control"
    )


def test_clear_reports_failure_when_slip_never_empties():
    adapter = SweepStateAdapter(removals_needed=99)
    driver = BetslipDriver(adapter, allow_submit=False)
    result = driver.clear()
    assert result.get("ok") is False
    assert len(adapter.clicks) <= 10, "sweep is bounded (SWEEP_MAX_CLICKS)"


def test_sweep_marker_without_selector_is_rejected():
    """The Unavailable-sweep bug (2026-10-28): the marker JS reported
    {'ok': True} without a selector, so _real_click asked the extension to
    click None and the slip stayed wedged across cycles. A marker that
    cannot name an element must never reach the click."""
    adapter = FakeAdapter([json.dumps({"ok": True, "how": "per-selection"})])
    driver = BetslipDriver(adapter, allow_submit=False)
    out = driver._mark("(() => 1)()", "slip remove")
    assert out["ok"] is False
    assert "no selector" in out["error"]
    assert adapter.clicks == []


def test_sweep_script_every_success_branch_returns_a_selector():
    """Lock the DOM contract: a branch that marks an element must return
    the marked selector, or the sweep is a silent no-op."""
    from atlas.browser.slipclear import JS_MARK_SLIP_REMOVE

    assert JS_MARK_SLIP_REMOVE.count("return mark(") >= 3, (
        "all / remove / accept branches must return a marked selector"
    )
    assert "selector:" in JS_MARK_SLIP_REMOVE
    assert "data-atlas-slip-remove" in JS_MARK_SLIP_REMOVE


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
    # The slip was read, but the Place Bet button was never clicked.
    assert adapter.codes, "the slip must have been read"
    assert all("data-atlas-place-bet" not in c for c in adapter.codes)


def test_place_bet_refuses_a_simulation_slip():
    """The book's SIM (virtual-money) slip must never move real money."""
    adapter = FakeAdapter([
        json.dumps({"seg": ["Over 2.5", "Total Stake", "100.00"], "stake": "100",
                    "canPlace": True, "sim": True}),
        # after a best-effort REAL switch, the slip is read again and is
        # STILL a simulation slip
        json.dumps({"ok": False, "error": "no mode switch"}),
        json.dumps({"seg": ["Over 2.5", "Total Stake", "100.00"], "stake": "100",
                    "canPlace": True, "sim": True}),
        json.dumps({"seg": ["Over 2.5", "Total Stake", "100.00"], "stake": "100",
                    "canPlace": True, "sim": True}),
    ])
    driver = BetslipDriver(adapter, allow_submit=True)
    result = driver.place_bet()
    assert result == {"ok": False, "error": "slip is in SIM (simulation) mode"}
    assert all("data-atlas-place-bet" not in c for c in adapter.codes)


def test_place_bet_confirms_on_receipt_and_returns_booking_code():
    """Ground truth (2026-10-28 live book): a placed bet does NOT empty the
    slip — it shows the receipt 'Submission Successful | ... | Booking Code'.
    The poll must accept the receipt as confirmation and surface the code."""
    poll = json.dumps({"empty": False, "receipt": True, "booking": "GZ9LG6"})
    marker = json.dumps({"ok": True, "selector": "[data-atlas-x='1']"})
    adapter = FakeAdapter([
        json.dumps({"seg": ["Over 2.5", "Total Stake", "10.00"], "stake": "10",
                    "canPlace": True, "sim": False}),   # read after mode check
        marker,                                          # JS_PLACE_BET marker
        marker,                                          # JS_MARK_CONFIRM marker
        poll,                                            # post-confirm poll (receipt -> break)
        json.dumps({}),                                  # marker cleanup
        json.dumps({}),
        json.dumps({"ok": True, "selector": "[r]"}),     # receipt dismiss mark
    ])
    driver = BetslipDriver(adapter, allow_submit=True)
    result = driver.place_bet()
    assert result == {"ok": True, "booking": "GZ9LG6", "error": None}
    # the receipt was acknowledged with a REAL click on the OK button
    assert "[r]" in adapter.clicks


def test_place_bet_refuses_when_never_confirmed():
    """No receipt, never empty: the book never confirmed — NOT a placed bet."""
    poll = json.dumps({"empty": False, "receipt": False, "booking": None})
    marker = json.dumps({"ok": True, "selector": "[data-atlas-x='1']"})
    adapter = FakeAdapter([
        json.dumps({"seg": ["Over 2.5", "Total Stake", "10.00"], "stake": "10",
                    "canPlace": True, "sim": False}),
        marker, marker,
        poll, poll, poll, poll,
        json.dumps({}),
        json.dumps({}),
    ])
    driver = BetslipDriver(adapter, allow_submit=True)
    result = driver.place_bet()
    assert result["ok"] is False
    assert "not confirm" in result["error"]
    # never confirmed -> no receipt -> no EXTRA click beyond place + confirm
    assert len(adapter.clicks) == 2



def test_failed_receipt_dismissal_still_reports_a_placed_bet():
    """The dismissal is best-effort ergonomics; the bet is already live."""
    marker = json.dumps({"ok": True, "selector": "[data-atlas-x='1']"})
    poll = json.dumps({"empty": False, "receipt": True, "booking": "GZ9LG6"})
    adapter = FakeAdapter([
        json.dumps({"seg": ["Total Stake", "10.00"], "stake": "10",
                    "canPlace": True, "sim": False}),
        marker, marker, poll,
        json.dumps({}), json.dumps({}),
        json.dumps({"ok": False, "error": "no receipt OK button"}),
    ])
    driver = BetslipDriver(adapter, allow_submit=True)
    result = driver.place_bet()
    assert result == {"ok": True, "booking": "GZ9LG6", "error": None}


def test_place_bet_reports_insufficient_balance_dialog_specifically():
    """Ground truth (2026-10-28, Overtime U2.5): the book can pop the funding
    dialog AFTER Confirm when the slip's total is not what was planned. The
    poll must stop waiting for a receipt that can never come and report the
    dialog as the failure."""
    marker = json.dumps({"ok": True, "selector": "[data-atlas-x='1']"})
    insufficient = json.dumps({
        "empty": False, "receipt": False, "booking": None, "insufficient": True,
    })
    adapter = FakeAdapter([
        json.dumps({"seg": ["Overtime - Over/Under", "Total Stake", "100.00"],
                    "stake": "100", "canPlace": True, "sim": False}),
        marker, marker,
        insufficient,
        json.dumps({}),
        json.dumps({}),
    ])
    driver = BetslipDriver(adapter, allow_submit=True)
    result = driver.place_bet()
    assert result["ok"] is False
    assert "insufficient balance" in result["error"]


def test_slip_is_empty_treats_receipt_as_no_pending_selection():
    """After a placement the receipt still shows a Total Stake; the slip is
    nonetheless free of SELECTIONS and must not wedge the pre-add gate."""
    poll = json.dumps({"empty": True, "receipt": True, "booking": "GZ9LG6"})
    driver = BetslipDriver(FakeAdapter([poll, poll]))
    assert driver.slip_is_empty() is True
    assert driver.slip_state()["receipt"] is True


def test_slip_is_empty_does_not_hallucinate_when_page_unreadable():
    driver = BetslipDriver(FakeAdapter([]))
    assert driver.slip_is_empty() is False
    assert driver.slip_state() == {
        "empty": False, "receipt": False, "booking": None, "insufficient": False,
    }


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


def test_read_slip_records_button_diagnostics():
    """A not-placeable read must be explainable: was the button absent
    (selection never landed) or disabled (book validating / suspended)?"""
    adapter = FakeAdapter(
        [
            json.dumps(
                {
                    "seg": ["Everton", "Under 4", "Total Stake", "Unavailable"],
                    "stake": "10.00",
                    "canPlace": False,
                    "hasPlaceButton": True,
                    "unavailable": True,
                    "placeInfo": [
                        {"text": "Place Bet", "dis": True, "vis": True}
                    ],
                }
            )
        ]
    )
    driver = BetslipDriver(adapter)
    slip = driver.read_slip(wait_s=0.0)
    assert slip.can_place_bet is False
    assert slip.has_place_button is True
    assert slip.unavailable is True
    assert slip.diagnostic == [{"text": "Place Bet", "dis": True, "vis": True}]


def test_read_slip_reports_an_absent_button():
    """No place-shaped element at all: the selection never landed."""
    adapter = FakeAdapter(
        [json.dumps({"seg": ["Total Stake"], "stake": None, "canPlace": False})]
    )
    driver = BetslipDriver(adapter)
    slip = driver.read_slip(wait_s=0.0)
    assert slip.has_place_button is False
    assert slip.diagnostic == {}
