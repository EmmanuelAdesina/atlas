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

import pytest

from atlas.live.capture import LiveCapture
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
        )

    def place_bet(self):
        self.placed += 1
        return {"ok": True}

    def clear(self):
        self.cleared += 1
        return {"ok": True}


class FakeDecision:
    def __init__(self, odds=2.0, should_bet=True, stake=100.0):
        from atlas.live.loop import Opportunity

        self.opportunity = Opportunity(
            match_id="m1",
            market_key="1x2",
            selection="Home",
            odds=odds,
            model_p=0.55,
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
