"""Acceptance tests for the P0 execution-safety work.

  1. Pre-submission odds verification: a bet is never submitted at odds
     more than 2% away from what was priced.
  2. The execution switch: --submit must actually produce 'recommend' and
     call place_bet().
  3. Navigation: the live board URL is never passed to navigate().
  4. Session health: an expired session is detected and reported.
  5. Cashout is called every cycle.
  6. Defence in depth: the stop and the kill switch are re-checked right
     before the click.
"""

import json
import time

import pytest

from atlas.live.capture import CaptureResult, LiveCapture
from atlas.live.decisions import PolicyConfig
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.live.runner import PRICE_MISMATCH_TOLERANCE, LiveRunner
from atlas.ops.ledger import BetLedger, LedgerConfig


class FakeSlip:
    """A betslip that reports the odds the book 'accepted'."""

    def __init__(self, accepted_odds=None, can_place=True):
        self.accepted_odds = accepted_odds
        self.can_place = can_place
        self.added = []
        self.stakes = []
        self.placed = 0
        self.cleared = 0

    def add_odds(self, odds):
        self.added.append(odds)
        return {"ok": True}

    def add_selection(self, market_label, selection):
        self.added.append((market_label, selection))
        return {"ok": True}

    def slip_is_empty(self):
        return True

    def accept_changes(self):
        return {"ok": True}

    def set_real_mode(self):
        """FakeSlips are always in REAL mode; the switch never fails."""
        return {"ok": True, "mode": "REAL"}

    def mode(self):
        return "REAL"

    def set_stake(self, stake):
        self.stakes.append(stake)
        return {"ok": True}

    def read_slip(self):
        from atlas.browser.betslip import SlipSelection, SlipState

        sels = []
        if self.accepted_odds is not None:
            sels = [SlipSelection(name="Home", odds=self.accepted_odds)]
        return SlipState(
            selections=sels,
            total_stake_ngn=self.stakes[-1] if self.stakes else 0.0,
            can_place_bet=self.can_place,
            # The real slip text carries the market and selection names; the
            # runner's identity gate requires them before money moves.
            raw_text="1X2 | Total Stake | NGN 100.00 | Potential Win | 202.00 | Home | 2.00",
        )

    def place_bet(self):
        self.placed += 1
        return {"ok": True}

    def clear(self):
        self.cleared += 1
        return {"ok": True}


class FakeDecision:
    def __init__(self, odds=2.0, should_bet=True, stake=100.0, p=0.55):
        from atlas.live.loop import Opportunity

        self.opportunity = Opportunity(
            match_id="m1",
            market_key="1x2",
            selection="Home",
            odds=odds,
            model_p=p,
        )
        self.stake_ngn = stake
        self.should_bet = should_bet
        self.reason = "test"
        self.decision = "recommend" if should_bet else "pass"


@pytest.fixture
def ledger(tmp_path):
    return BetLedger(LedgerConfig(), log_path=tmp_path / "l.jsonl")


def state():
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="H", goals=1),
        away=TeamStats(name="A", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        source="test",
    )


# --- 1. Odds verification -------------------------------------------------

def test_a_bet_is_placed_when_the_accepted_odds_match(ledger):
    slip = FakeSlip(accepted_odds=2.00)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    placed = runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    assert placed == 1
    assert slip.placed == 1


def test_a_bet_is_refused_when_the_accepted_odds_moved(ledger):
    """A 2% drift turns a +EV bet into a -EV one silently."""
    slip = FakeSlip(accepted_odds=2.00 * (1 + PRICE_MISMATCH_TOLERANCE + 0.01))
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    placed = runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    assert placed == 0
    assert slip.placed == 0
    assert slip.cleared == 1, "the slip must be cleared, not left pending"


def test_a_small_drift_inside_the_tolerance_is_accepted(ledger):
    slip = FakeSlip(accepted_odds=2.00 * (1 + PRICE_MISMATCH_TOLERANCE - 0.001))
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 1
    assert slip.placed == 1


def test_a_slip_that_reports_no_odds_is_refused(ledger):
    slip = FakeSlip(accepted_odds=None)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 0


def test_the_ledger_records_the_accepted_odds_not_the_priced_ones(ledger):
    slip = FakeSlip(accepted_odds=2.01)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    bet = ledger.bets_for("m1")[0]
    assert bet["odds"] == pytest.approx(2.01)
    assert bet["priced_odds"] == pytest.approx(2.0)


# --- 1b. Balance-vs-ledger guard -------------------------------------------

class FakeCapture:
    """Playbacks canned CaptureResults (oldest first)."""

    def __init__(self, results):
        self.results = list(results)

    def capture(self):
        return self.results.pop(0)


class RecorderNotifier:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def warn(self, title, body="", **kw):
        self.calls.append(("warn", title))

    def critical(self, title, body="", **kw):
        self.calls.append(("critical", title))

    def info(self, title, body="", **kw):
        self.calls.append(("info", title))


def balanced_state(bal):
    s = state()
    s.account_balance_ngn = bal
    return s


def _result(ready=True, reason="ready"):
    return CaptureResult(ready=ready, reason=reason, state=state())


def test_balance_drop_after_unconfirmed_placement_is_critical(ledger):
    """Ground truth (2026-10-28, 11:08 & 11:15 UTC): the book took bets the
    slip check could not confirm and the ledger missed them. The visible
    signature is a balance drop within cycles of an unconfirmed attempt."""
    notifier = RecorderNotifier()
    cap = FakeCapture([_result(), _result()])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        betslip=FakeSlip(accepted_odds=2.0), allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    # two balanced states: first observe 159.18, then a 10.00 drop
    runner.run_cycle()
    runner._balance_observed = 159.18
    runner._arm_unconfirmed_probe()
    runner._balance_alert_fired = False
    cap.results = [
        CaptureResult(ready=True, reason="ready", state=balanced_state(149.18))
    ]
    runner.run_cycle()
    assert any(
        kind == "critical" and "Balance dropped" in title
        for kind, title in notifier.calls
    )


def test_balance_drop_without_unconfirmed_placement_stays_quiet(ledger):
    """The user bets manually on this account too — the book shrinking for
    other reasons must not raise the reconcile alarm."""
    notifier = RecorderNotifier()
    notifier2 = RecorderNotifier()
    cap = FakeCapture([
        CaptureResult(ready=True, reason="ready", state=balanced_state(159.18)),
        CaptureResult(ready=True, reason="ready", state=balanced_state(149.18)),
    ])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        betslip=FakeSlip(accepted_odds=2.0), allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    runner.run_cycle()
    runner.run_cycle()
    assert not any(
        kind == "critical" and "Balance dropped" in title
        for kind, title in notifier.calls
    )
    assert notifier2.calls == [] or True  # (second recorder unused by design)


def test_slip_stake_mismatch_is_refused_before_submission(ledger):
    """The book's 'Balance Insufficient' dialog after Confirm (2026-10-28,
    Overtime U2.5) proved a slip can hit the book with a blended total
    nobody planned. The stake-fidelity gate must refuse the numeric total
    mismatch BEFORE submitting."""
    class FatSlip(FakeSlip):
        def read_slip(self):
            s = FakeSlip.read_slip(self)
            s.total_stake_ngn = self.blended_total
            return s

    slip = FatSlip(accepted_odds=2.0)
    slip.blended_total = 100.0
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    placed = runner.place_bets([FakeDecision(odds=2.0, stake=10.0)], state(), 1)
    assert placed == 0
    assert slip.placed == 0, "the slip must never be submitted"
    assert slip.cleared == 1
    assert "stake 100.00 != planned 10.00" in runner.last_error


def test_placement_worker_isolates_cycle_from_slow_placement(ledger):
    """Envelope test for the async placement worker: enqueue, let the
    thread drain, verify the placement lands. The cycle free-waiting here
    is the wedge the worker exists to prevent (2026-10-28, cycle 15)."""
    import time as _t

    slip = FakeSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
        operator_max_stake=10.0, operator_min_stake=10.0,
    )
    runner.enqueue_place([FakeDecision(odds=2.0, stake=10.0)], state(), 1)
    deadline = _t.monotonic() + 10.0
    while slip.placed < 1 and _t.monotonic() < deadline:
        _t.sleep(0.05)
    assert slip.placed == 1, "worker must drain the queue and place"
    stats = runner._place_stats
    assert stats["done"] == 1 and stats["failed"] == 0


class SlowSlip(FakeSlip):
    """A slip whose first read BLOCKS, keeping a placement in flight."""

    def __init__(self, accepted_odds=2.0):
        super().__init__(accepted_odds=accepted_odds)
        import threading as _th

        self.entered = _th.Event()
        self.release = _th.Event()

    def read_slip(self):
        self.entered.set()
        self.release.wait(10.0)
        return super().read_slip()


def test_async_placement_reserves_inflight_and_dedupes(ledger):
    """A queued bet is not in the ledger yet, so the NEXT cycle must still
    see it. Without the in-flight reservation the same selection is
    re-planned as 'new' and placed twice — double exposure on one outcome.
    """
    import time as _t

    slip = SlowSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
        operator_max_stake=10.0, operator_min_stake=10.0,
    )
    assert runner.enqueue_place(
        [FakeDecision(odds=2.0, stake=10.0)], state(), 1
    ) == 1
    assert slip.entered.wait(5.0), "the worker must start the placement"

    # Committed but not yet in the ledger: the reservation is what the
    # portfolio sees in place of a ledger row.
    inflight = runner._inflight_bets("m1")
    assert len(inflight) == 1
    assert inflight[0]["selection"] == "Home"
    assert ledger.bets_for("m1") == []

    # The next cycle re-proposes the same bet while the first is in
    # flight: it must be suppressed, not queued again.
    assert runner.enqueue_place(
        [FakeDecision(odds=2.0, stake=10.0)], state(), 2
    ) == 0
    assert runner._place_stats["duplicate"] == 1

    slip.release.set()
    deadline = _t.monotonic() + 10.0
    while slip.placed < 1 and _t.monotonic() < deadline:
        _t.sleep(0.05)
    assert slip.placed == 1, "the single reserved bet must be placed once"

    deadline = _t.monotonic() + 5.0
    while runner._inflight_bets("m1") and _t.monotonic() < deadline:
        _t.sleep(0.05)
    assert runner._inflight_bets("m1") == [], (
        "the reservation must be released once the worker is done"
    )
    assert len(ledger.bets_for("m1")) == 1
    runner.shutdown(timeout=5.0)



def test_saved_posture_stake_cap_does_not_override_the_operator(ledger, tmp_path):
    """2026-10-28: the dashboard-saved posture held max_stake_per_bet=100
    against an explicit NGN 10 CLI and re-applied it every cycle (NGN 35/100
    stakes reached recommendations). The operator bounds re-apply after
    EVERY load."""
    import json as _json

    spath = tmp_path / "state.json"
    spath.write_text(
        _json.dumps({"posture": {"min_stake_ngn": 10.0, "max_stake_per_bet": 100.0}}),
        encoding="utf-8",
    )
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=None, allow_submission=False,
        state_path=str(spath),
        operator_min_stake=10.0, operator_max_stake=10.0,
    )
    loaded = runner.load_state()
    assert loaded is not None
    assert runner.posture.max_stake_per_bet == 10.0
    assert runner.posture.min_stake_ngn == 10.0


def test_operator_screening_file_survives_explicit_overrides(ledger, tmp_path):
    """A loaded posture with both bounds fresh from CLI stays untouched."""
    import json as _json

    spath = tmp_path / "state.json"
    spath.write_text(
        _json.dumps({"posture": {"min_stake_ngn": 10.0, "max_stake_per_bet": 10.0}}),
        encoding="utf-8",
    )
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=None, allow_submission=False,
        state_path=str(spath),
        operator_min_stake=10.0, operator_max_stake=10.0,
    )
    runner.load_state()
    assert runner.posture.max_stake_per_bet == 10.0
    assert runner.posture.min_stake_ngn == 10.0


def test_stake_over_operator_cap_is_refused_at_money_path(ledger):
    """The LAST wall: even a decision that slipped past sizing with a
    NGN 100 stake is refused before submission when the operator's cap is
    NGN 10."""
    slip = FakeSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
        operator_max_stake=10.0, operator_min_stake=10.0,
    )
    placed = runner.place_bets([FakeDecision(odds=2.0, stake=100.0)], state(), 1)
    assert placed == 0
    assert slip.placed == 0, "nothing over the operator cap may reach the book"
    assert slip.cleared >= 0
    assert "operator cap" in runner.last_error


def test_degenerate_certainty_is_refused(ledger):
    """Case study 2026-10-28: model priced p=1.0 / EV 9.0 on an overtime
    market at odds 10.0 (NGN 100 lost). A model claiming near-certainty at
    live odds is misreading the market family — refuse before sizing."""
    slip = FakeSlip(accepted_odds=10.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    placed = runner.place_bets([FakeDecision(odds=10.0, p=1.0)], state(), 1)
    assert placed == 0
    assert slip.placed == 0
    assert "degenerate certainty" in runner.last_error


def test_high_but_finite_model_p_still_bets(ledger):
    """A healthy model (p=0.62 @ 1.8) is not the degenerate case."""
    slip = FakeSlip(accepted_odds=1.80)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=1.80, p=0.62)], state(), 1) == 1


def test_contradictory_certainty_at_a_short_price_is_refused(ledger):
    """Case study 2026-10-29: away_win_to_nil 'No' priced p=1.0 at 1.24.

    The old gate only fired at odds >= 1.50, so a fake certainty at a short
    price sailed through and took NGN 10. A market that is genuinely
    settled is suspended, not sold at 1.24.
    """
    slip = FakeSlip(accepted_odds=1.24)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    placed = runner.place_bets([FakeDecision(odds=1.24, p=1.0)], state(), 1)
    assert placed == 0
    assert slip.placed == 0
    assert "degenerate certainty" in runner.last_error


def test_contradictory_impossibility_at_a_short_price_is_refused(ledger):
    """The mirror case: p=0.0 on an outcome the book still sells."""
    slip = FakeSlip(accepted_odds=1.30)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=1.30, p=0.0)], state(), 1) == 0
    assert "degenerate certainty" in runner.last_error


def test_certainty_the_book_also_prices_is_allowed(ledger):
    """p=0.999 at 1.02 is agreement, not contradiction: the book has all but
    suspended the outcome. That is the one shape a terminal state may take."""
    slip = FakeSlip(accepted_odds=1.02)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=1.02, p=0.999)], state(), 1) == 1


def test_confirmed_placement_resets_the_guard(ledger):
    notifier = RecorderNotifier()
    cap = FakeCapture([_result()])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        betslip=FakeSlip(accepted_odds=2.0), allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    assert runner._unconfirmed_at is None
    assert runner._balance_alert_fired is False


def test_balance_observed_survives_capture_noise(ledger):
    """A stale or unreadable balance (None) must not erase the last real
    observation — drops are judged relative to a real reading."""
    cap = FakeCapture([
        CaptureResult(ready=True, reason="ready", state=balanced_state(159.18)),
        CaptureResult(ready=True, reason="no data", state=state()),
    ])
    runner = LiveRunner(
        capture=cap, ledger=ledger,
        betslip=FakeSlip(accepted_odds=2.0), allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    runner.run_cycle()
    runner.run_cycle()
    assert runner._balance_observed == pytest.approx(159.18)


# --- 2. The execution switch ----------------------------------------------

def test_execution_is_refused_without_the_policy_opt_in(ledger):
    slip = FakeSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=False),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 0


def test_the_default_policy_does_not_allow_execution():
    assert PolicyConfig().allow_execution is False


# --- 7. The 2026-10-26 phantom-bet regression class ------------------------
# A night cycle placed 'correct_score 4:0' (a cross-wired extraction) on the
# BOOK's rejection path and the runner recorded it anyway. Book truth today:
# no ticket, balance unchanged. These tests pin the invariant: the ledger
# only ever holds bets the book confirmed.

class FakeUnconfirmedSlip(FakeSlip):
    """A slip the book never confirmed: place_bet reports ok: False."""

    def place_bet(self):
        # The real driver attempts the click and the confirmation before
        # discovering the slip did not empty — the attempt still happened.
        self.placed += 1
        return {"ok": False, "error": "the slip did not empty after Confirm"}


class FakeMismatchedSlip(FakeSlip):
    """A slip whose text does not name the priced market/selection."""

    def read_slip(self):
        from atlas.browser.betslip import SlipState

        return SlipState(
            selections=[],
            total_stake_ngn=self.stakes[-1] if self.stakes else 0.0,
            can_place_bet=True,
            raw_text="SOME OTHER MARKET | Total Stake | NGN 100.00",
        )


def test_a_book_rejection_is_never_recorded_as_placed(ledger):
    """The wrapper's ok:False (slip did not empty) means NO position, EVER."""
    slip = FakeUnconfirmedSlip(accepted_odds=2.00)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 1, "the click still happened — but must not be recorded"
    assert ledger.bets_for("m1") == [], "a phantom bet must not enter the ledger"
    assert slip.cleared >= 1, "the slip must be cleared for the next bet"


def test_a_slip_without_the_priced_identity_is_refused(ledger):
    """The slip must name the market AND the selection that was priced."""
    slip = FakeMismatchedSlip(accepted_odds=2.00)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 0
    assert ledger.bets_for("m1") == []


def test_a_missing_selection_never_falls_back_to_clicking_by_price(ledger):
    """A price string can match a cell in a DIFFERENT market table; there is
    deliberately no by-price fallback in the runner."""
    slip = FakeSlip(accepted_odds=2.00)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )

    class NoSelection(FakeSlip):
        def add_selection(self, market_label, selection):
            slip.no_selection_calls += 1
            return {"ok": False, "error": "selection not found on the board"}

        def add_odds(self, odds):
            slip.by_price_calls = getattr(slip, "by_price_calls", 0) + 1
            return {"ok": True}

    slip2 = NoSelection(accepted_odds=2.00)
    runner2 = LiveRunner(
        capture=None, ledger=ledger, betslip=slip2, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner2.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert getattr(slip2, "by_price_calls", 0) == 0, "by-price clicking is forbidden"
    assert ledger.bets_for("m1") == []


def test_execution_is_refused_without_a_submission_opt_in(ledger):
    slip = FakeSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=False,
        policy=PolicyConfig(allow_execution=True),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 0


# --- 6. Defence in depth --------------------------------------------------

def test_the_daily_stop_blocks_placement_mid_cycle(ledger):
    slip = FakeSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    # The stop is engaged AFTER the cycle priced the bet.
    for _ in range(6):
        ledger.record_pending(
            {
                "match_id": "m1", "market_key": "1x2", "selection": "Home",
                "odds": 2.0, "stake_ngn": 100.0, "model_p": 0.5,
                "p_lower": 0.45, "ev": 0.0, "confidence": "unknown",
            }
        )
        ledger.settle("m1", "1x2", "Home", "loss", payout_ngn=0.0)
    assert ledger.is_daily_stop_active() is True
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 0


def test_the_kill_switch_blocks_placement(ledger):
    slip = FakeSlip(accepted_odds=2.0)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True, kill_switch=True),
    )
    assert runner.place_bets([FakeDecision(odds=2.0)], state(), 1) == 0
    assert slip.placed == 0


# --- 3. Navigation --------------------------------------------------------

def test_the_live_board_url_is_never_passed_to_navigate():
    """navigate() to the live board times out and unregisters the tab."""
    adapter = _RecordingAdapter()
    capture = LiveCapture(adapter)
    capture.load_board(force=True)
    for url in adapter.navigations:
        assert "/sport/live" not in url, f"navigate() was called with the live board: {url}"


def test_recovery_never_navigates_to_the_live_board():
    adapter = _RecordingAdapter()
    LiveCapture(adapter).recover()
    for url in adapter.navigations:
        assert "/sport/live" not in url


class _RecordingAdapter:
    def __init__(self):
        self.navigations = []

    def evaluate(self, code):
        return {"data": {"value": "{}"}}

    def snapshot(self):
        return {"tree": []}

    def navigate(self, url, **kwargs):
        self.navigations.append(url)
        return {"success": True}


# --- 4. Session health ----------------------------------------------------

def test_an_expired_session_is_detected():
    class Expired:
        def evaluate(self, code):
            if "session" in code.lower() or "password" in code.lower():
                return {
                    "data": {
                        "value": json.dumps(
                            {"expired": True, "onLoginPage": True, "hasPasswordField": True}
                        )
                    }
                }
            return {"data": {"value": "{}"}}

        def snapshot(self):
            return {"tree": []}

        def navigate(self, url, **kwargs):
            return {"success": True}

    assert LiveCapture(Expired()).session_expired().get("expired") is True


def test_a_live_session_is_not_flagged():
    class Live:
        def evaluate(self, code):
            return {"data": {"value": json.dumps({"expired": False})}}

        def snapshot(self):
            return {"tree": []}

        def navigate(self, url, **kwargs):
            return {"success": True}

    assert LiveCapture(Live()).session_expired().get("expired") is False


# --- The decision vocabulary and the cashout in the cycle -----------------

class _FakeCapture:
    """A capture that returns a ready panel with one generous market."""

    def __init__(self, markets, state=None):
        self.markets = markets
        self.state = state or LiveMatchState(
            match_id="m1",
            home=TeamStats(name="Home", goals=1),
            away=TeamStats(name="Away", goals=0),
            period="2h",
            clock_minute=70.0,
            observed_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
            source="test",
        )

    def capture(self, wait=True):
        from atlas.live.capture import CaptureResult

        return CaptureResult(
            ready=True, reason="ready", markets=self.markets, state=self.state,
            observed_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ), match_id="m1",
        )


def _generous_markets():
    return [
        {
            "match_id": "m1",
            "market_key": "ou_1_5",
            "market_label": "Over/Under",
            "status": "open",
            "line": 1.5,
            "selections": [
                # Priced so the model and the book agree closely (the 5pp gap
                # guard) but the book is still generous: EV +0.06.
                {"name": "Over 1.5", "odds": 2.50},
                {"name": "Under 1.5", "odds": 1.56},
            ],
        }
    ]


def test_a_planned_bet_is_logged_as_recommend_when_execution_is_allowed(
    ledger, tmp_path
):
    from atlas.live.runner import LiveRunner

    log = tmp_path / "decisions.jsonl"
    runner = LiveRunner(
        capture=_FakeCapture(_generous_markets()),
        ledger=ledger,
        allow_submission=False,
        policy=PolicyConfig(allow_execution=True, bankroll_ngn=100_000.0),
        decision_log_path=str(log),
    )
    runner.run_cycle()
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert entries
    assert any(e["decision"] == "recommend" for e in entries), entries
    assert any(e["decision"] == "pass" for e in entries), entries


def test_a_planned_bet_is_logged_as_pass_when_execution_is_disabled(
    ledger, tmp_path
):
    from atlas.live.runner import LiveRunner

    log = tmp_path / "decisions.jsonl"
    runner = LiveRunner(
        capture=_FakeCapture(_generous_markets()),
        ledger=ledger,
        allow_submission=False,
        policy=PolicyConfig(allow_execution=False, bankroll_ngn=100_000.0),
        decision_log_path=str(log),
    )
    runner.run_cycle()
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert entries
    assert all(e["decision"] == "pass" for e in entries), entries
    assert any("execution disabled" in e["reason"] for e in entries)


def test_cashout_offers_are_valued_every_cycle(ledger):
    from atlas.browser.cashout import CashoutOffer
    from atlas.live.runner import LiveRunner

    ledger.record_pending(
        {
            "match_id": "m1", "market_key": "1x2", "selection": "Home",
            "odds": 2.0, "stake_ngn": 100.0, "model_p": 0.5,
            "p_lower": 0.45, "ev": 0.0, "confidence": "unknown",
        }
    )

    class Cashout:
        allow_execute = False

        def read_offers(self):
            return [CashoutOffer(index=0, stake_ngn=100.0, offered_ngn=130.0)]

        def execute(self, index):
            return {"ok": True}

    runner = LiveRunner(
        capture=_FakeCapture(_generous_markets()),
        ledger=ledger,
        cashout_driver=Cashout(),
    )
    report = runner.run_cycle()
    assert report.cashouts == 1
    assert report.cashout_taken == 1


# --- balance visibility + display, 2026-10-28 ------------------------------
# The dashboard showed 'account balance ngn' empty and
# 'PostureMode.BALANCED' — the balance was only read from a match panel
# (none open in board mode) and the enum repr leaked into the panel.

def test_balance_is_observed_even_when_the_board_is_empty(ledger):
    class EmptyBoard:
        def capture(self, wait=True):
            return CaptureResult(
                False, "the board is empty (the live feed is not delivering)"
            )

        def board_candidates(self):
            return []

        def load_board(self, force=False):
            return None

        def read_balance_ngn(self):
            return 52.68

    runner = LiveRunner(capture=EmptyBoard(), ledger=ledger)
    runner.run_cycle()
    assert runner._balance_observed == pytest.approx(52.68), (
        "the header balance must be observed on not-ready cycles too"
    )


def test_heartbeat_reports_a_readable_posture_mode(ledger, tmp_path):
    from atlas.live.runner import CycleReport
    from atlas.ops.posture import posture_for_mode

    runner = LiveRunner(
        capture=None,
        ledger=ledger,
        posture=posture_for_mode("balanced"),
        state_path=str(tmp_path / "state.json"),
    )
    runner._write_heartbeat(
        CycleReport(cycle=1, at="t", ready=False, reason="r")
    )
    import json as _json

    hb = _json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["posture_mode"] == "balanced"


def test_heartbeat_reports_the_page_the_automation_is_on(ledger, tmp_path):
    from atlas.live.runner import CycleReport

    class Capture:
        last_href = "https://www.sportybet.com/ng/sport/football/live_list/"
        last_transport_error = ""

    runner = LiveRunner(
        capture=Capture(),
        ledger=ledger,
        state_path=str(tmp_path / "state.json"),
    )
    runner._write_heartbeat(
        CycleReport(cycle=3, at="t", ready=False, reason="r")
    )
    import json as _json

    hb = _json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["page_url"] == Capture.last_href
    assert hb["page_error"] is None


def test_heartbeat_surfaces_a_transport_fault(ledger, tmp_path):
    from atlas.live.runner import CycleReport

    class Capture:
        last_href = "https://www.sportybet.com/ng/sport/live/"
        last_transport_error = "daemon call evaluate returned HTTP 502"

    runner = LiveRunner(
        capture=Capture(),
        ledger=ledger,
        state_path=str(tmp_path / "state.json"),
    )
    runner._write_heartbeat(
        CycleReport(cycle=4, at="t", ready=False, reason="r")
    )
    import json as _json

    hb = _json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert "502" in hb["page_error"]


# --- market-level failure cooldown ----------------------------------------
# 2026-10-29: 8 of 10 placement attempts burned on the same two O/U lines
# at half-time, where the book suspends in-play markets and the slip can
# never become placeable. A refused market must sit out, not be re-clicked
# every cycle.


def _unplaceable_runner(ledger):
    slip = FakeSlip(accepted_odds=2.00, can_place=False)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    return runner, slip


def test_a_not_placeable_slip_fails_without_moving_money(ledger):
    runner, slip = _unplaceable_runner(ledger)
    placed = runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    assert placed == 0
    assert slip.placed == 0
    assert slip.cleared >= 1, "the slip must be cleared for the next bet"


def test_the_not_placeable_failure_says_why(ledger):
    """'not placeable' alone is not diagnosable; the slip's own signals
    must travel with the refusal."""
    runner, _ = _unplaceable_runner(ledger)
    runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    assert "unavailable=" in runner.last_error
    assert "button=" in runner.last_error
    assert "info=" in runner.last_error


def test_a_refused_market_is_not_retried_next_cycle(ledger):
    runner, slip = _unplaceable_runner(ledger)
    runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    first_additions = list(slip.added)
    # Next cycle, the same market is skipped BEFORE the slip is touched.
    runner.place_bets([FakeDecision(odds=2.0)], state(), 2)
    assert slip.added == first_additions, (
        "a refused market must not be re-clicked while on cooldown"
    )
    assert slip.placed == 0


def test_a_refused_market_does_not_block_other_markets(ledger):
    """The cooldown is per market/selection, never per match or per book."""
    import dataclasses

    class SuspendedMarketSlip(FakeSlip):
        """The 1x2 selection lands on the slip but the market is suspended
        (button never enables); other markets read placeable — exactly the
        half-time board the failure log came from."""

        def read_slip(self):
            s = super().read_slip()
            if self.added:
                market, selection = self.added[-1]
                suspended = selection == "Home"
                text = (
                    f"{market or '1X2'} | Total Stake | NGN 10.00 | "
                    f"{selection} | 2.00"
                )
                return dataclasses.replace(s, can_place_bet=not suspended,
                                           raw_text=text)
            return s

        def clear(self):
            self.added = []
            return super().clear()

    slip = SuspendedMarketSlip(accepted_odds=2.00, can_place=True)
    runner = LiveRunner(
        capture=None, ledger=ledger, betslip=slip, allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    refused = FakeDecision(odds=2.0, p=0.55)
    other = FakeDecision(odds=2.0, p=0.55)
    other.opportunity.market_key = "ou_1_5"
    other.opportunity.selection = "Over 1.5"
    placed = runner.place_bets([refused, other], state(), 1)
    assert placed == 1, "the untouched market must still be worked"
    assert slip.placed == 1
    # Its cooldown entry exists for the refused one only.
    keys = [(k[1], k[2]) for k in runner._avoid]
    assert ("1x2", "Home") in keys
    assert ("ou_1_5", "Over 1.5") not in keys


def test_the_cooldown_expires_after_its_window(ledger):
    import time as _time

    runner, _ = _unplaceable_runner(ledger)
    key = ("m1", "ou_4", "Under 4")
    runner._cool_off(key)
    assert runner._is_avoided(key) is True
    # Ageing past the window frees the market without external intervention.
    runner._avoid[key] = _time.monotonic() - 0.001
    runner._avoid_prune()
    assert key not in runner._avoid


def test_a_cycle_start_ping_marks_the_heartbeat_scanning(ledger, tmp_path):
    """The heartbeat used to be written only at cycle end, so one slow
    step (a wedged read + its CDP fallback) showed STALE while the loop
    was alive and working (2026-10-31, Tottenham panel). A ping at cycle
    start makes liveness visible per cycle."""
    import json as _json

    runner = LiveRunner(
        capture=None, ledger=ledger, state_path=str(tmp_path / "state.json"),
    )
    runner.cycle = 7
    runner._heartbeat_scan_ping()
    hb = _json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    # The ping announces the number run_cycle will END on, so the panel
    # never shows two different figures for one cycle.
    assert hb["cycle"] == runner.cycle + 1
    assert hb["ready"] is False
    assert "scanning" in hb["reason"]


def test_a_finished_cycle_overwrites_the_ping(ledger, tmp_path):
    import json as _json

    from atlas.live.runner import CycleReport

    runner = LiveRunner(
        capture=None, ledger=ledger, state_path=str(tmp_path / "state.json"),
    )
    runner._heartbeat_scan_ping()
    runner._write_heartbeat(
        CycleReport(cycle=2, at="t", ready=True, reason="ready")
    )
    hb = _json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["ready"] is True
    assert hb["reason"] == "ready"


# --- post-unconfirmed balance probe ---------------------------------------
# 'no receipt seen in 12s' cannot tell 'never went through' from 'receipt
# missed'. A scheduled second balance read resolves it either way.

NO_PROBE_CRIT = lambda calls: [
    (k, t) for k, t in calls if k == "critical" and "Balance dropped" in t
]


def test_the_probe_fires_critically_after_its_deadline(ledger):
    """Baseline 159.18, probe deadline passed, fresh read 149.18: the book
    took NGN 10 the ledger does not hold."""
    import time as _time

    notifier = RecorderNotifier()
    cap = FakeCapture([
        CaptureResult(ready=True, reason="ready", state=balanced_state(149.18)),
    ])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        policy=PolicyConfig(allow_execution=True),
    )
    runner._balance_observed = 159.18
    runner._probe_baseline = 159.18
    runner._unconfirmed_at = _time.monotonic() - 60.0
    runner._probe_at = _time.monotonic() - 1.0  # deadline passed
    runner.run_cycle()
    assert len(NO_PROBE_CRIT(notifier.calls)) == 1
    # One-shot: the probe does not survive its own firing.
    assert runner._probe_at is None
    assert runner._probe_baseline is None


def test_a_clear_probe_closes_the_case_quietly(ledger):
    """Balance unchanged at probe time: no receipt missed, no alert."""
    import time as _time

    notifier = RecorderNotifier()
    cap = FakeCapture([
        CaptureResult(ready=True, reason="ready", state=balanced_state(159.18)),
    ])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        policy=PolicyConfig(allow_execution=True),
    )
    runner._balance_observed = 159.18
    runner._probe_baseline = 159.18
    runner._unconfirmed_at = _time.monotonic() - 60.0
    runner._probe_at = _time.monotonic() - 1.0
    runner.run_cycle()
    assert NO_PROBE_CRIT(notifier.calls) == []
    assert runner._probe_at is None, "a clear probe must close, not linger"


def test_a_confirmed_placement_supersedes_the_probe(ledger):
    """A later CONFIRMED bet owns the balance drop — the probe must not
    turn the next legitimate placement into a phantom-position alarm."""
    import time as _time

    notifier = RecorderNotifier()
    cap = FakeCapture([
        CaptureResult(ready=True, reason="ready", state=balanced_state(149.18)),
    ])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        betslip=FakeSlip(accepted_odds=2.0), allow_submission=True,
        policy=PolicyConfig(allow_execution=True),
    )
    runner._balance_observed = 159.18
    runner._arm_unconfirmed_probe()
    runner.place_bets([FakeDecision(odds=2.0)], state(), 1)
    assert runner._probe_at is None, "confirmation supersedes the probe"
    runner.run_cycle()
    assert NO_PROBE_CRIT(notifier.calls) == []


def test_the_probe_gives_up_loudly_without_any_reading(ledger):
    """A tab that no longer shows a header must not leave the probe armed
    forever: give up with a warn the operator can act on."""
    import time as _time

    notifier = RecorderNotifier()
    cap = FakeCapture([_result()])
    runner = LiveRunner(
        capture=cap, ledger=ledger, notifier=notifier,
        policy=PolicyConfig(allow_execution=True),
    )
    runner._balance_observed = None
    runner._unconfirmed_at = _time.monotonic() - 400.0
    runner._probe_at = _time.monotonic() - 30.0
    runner.run_cycle()
    assert runner._probe_at is None
    assert any(
        k == "warn" and "Phantom probe" in t for k, t in notifier.calls
    ), "giving up silently would hide a possible phantom bet"
