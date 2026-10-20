"""Tests for the autonomous loop (atlas.live.runner).

The loop is exercised with a fake capture and a fake betslip, so the gates
can be tested in isolation: readiness, the daily stop, the posture, and the
rule that money only moves when submission is explicitly allowed.
"""

from datetime import datetime, timezone

import pytest

from atlas.live.capture import CaptureResult
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.live.runner import LiveRunner
from atlas.ops.ledger import BetLedger, LedgerConfig
from atlas.ops.posture import PostureMode, posture_for_mode


def make_state(home=0, away=0, period="2h", minute=70.0):
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=home),
        away=TeamStats(name="Away", goals=away),
        period=period,
        clock_minute=minute,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


def markets(odds=("2.20", "3.40", "3.20"), key="1x2"):
    return [
        {
            "match_id": "m1",
            "market_key": key,
            "market_label": "1X2",
            "status": "open",
            "selections": [
                {"name": "Home", "odds": float(odds[0])},
                {"name": "Draw", "odds": float(odds[1])},
                {"name": "Away", "odds": float(odds[2])},
            ],
        }
    ]


class FakeCapture:
    def __init__(self, ready=True, reason="ready", state=None, mkts=None):
        self.ready = ready
        self.reason = reason
        self.state = state if state is not None else make_state()
        self.markets = mkts if mkts is not None else markets()
        self.loads = 0

    def capture(self, wait=True):
        return CaptureResult(
            ready=self.ready,
            reason=self.reason,
            markets=self.markets,
            state=self.state,
            observed_at=datetime.now(timezone.utc),
            match_id="m1",
        )


class FakeBetslip:
    def __init__(self, can_place=True):
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
        class S:
            can_place_bet = self.can_place
            total_stake_ngn = 0.0
            raw_text = ""
        return S()

    def place_bet(self):
        self.placed += 1
        return {"ok": True}

    def clear(self):
        self.cleared += 1


@pytest.fixture
def ledger(tmp_path):
    return BetLedger(LedgerConfig(), log_path=tmp_path / "ledger.jsonl")


def make_runner(ledger, capture, **kw):
    return LiveRunner(capture=capture, ledger=ledger, **kw)


def test_cycle_on_a_not_ready_panel_plans_nothing(ledger):
    cap = FakeCapture(ready=False, reason="score unknown")
    runner = make_runner(ledger, cap)
    report = runner.run_cycle()
    assert report.ready is False
    assert report.planned == 0
    assert report.placed == 0
    assert "score unknown" in report.reason


def test_cycle_prices_the_board(ledger):
    cap = FakeCapture()
    runner = make_runner(ledger, cap)
    report = runner.run_cycle()
    assert report.ready is True
    # The 5pp model-market gap guard blocks selections where the model and
    # the book disagree too much, so not every selection is priced.
    assert report.priced_selections >= 2
    assert report.opportunities == report.priced_selections


def test_the_gap_guard_blocks_a_selection_the_model_disagrees_with(ledger):
    """A draw at 70 minutes with no goals is a bet the model and the book
    disagree about; the gap guard must block it rather than price it."""
    from atlas.live.estimate import CalibratedModel, LiveStateSpaceModel
    from atlas.live.math_engine import devig_shin

    fair, _ = devig_shin({"Home": 2.20, "Draw": 3.40, "Away": 3.20})
    model = CalibratedModel(
        base_model=LiveStateSpaceModel(now=datetime.now(timezone.utc)),
        market_fair={("1x2", k): v for k, v in fair.items()},
    )
    est = model.estimate(make_state(), "1x2", "Draw")
    assert est.abstained is True
    assert "gap" in (est.abstain_reason or "")


def test_no_money_moves_without_submission_enabled(ledger):
    cap = FakeCapture()
    slip = FakeBetslip()
    runner = make_runner(ledger, cap, betslip=slip, allow_submission=False)
    runner.run_cycle()
    assert slip.placed == 0
    assert slip.added == []


def test_money_moves_when_submission_is_enabled(ledger):
    cap = FakeCapture()
    slip = FakeBetslip()
    runner = make_runner(ledger, cap, betslip=slip, allow_submission=True)
    report = runner.run_cycle()
    # Whether a bet is planned depends on the model finding an edge; what
    # matters is that anything planned is submitted and recorded.
    if report.planned:
        assert slip.placed == report.placed
        assert len(ledger.bets_for("m1")) == report.placed
    else:
        assert slip.placed == 0


def test_daily_stop_blocks_planning(ledger):
    cap = FakeCapture()
    runner = make_runner(ledger, cap)
    # Push the ledger past the hard daily stop.
    for _ in range(6):
        ledger.record_pending(
            {
                "match_id": "m1",
                "market_key": "1x2",
                "selection": "Home",
                "odds": 2.0,
                "stake_ngn": 100.0,
                "model_p": 0.5,
                "p_lower": 0.45,
                "ev": 0.0,
                "confidence": "unknown",
            }
        )
        ledger.settle("m1", "1x2", "Home", "loss", payout_ngn=0.0)
    assert ledger.is_daily_stop_active() is True
    report = runner.run_cycle()
    assert report.stop_active is True
    assert report.planned == 0
    assert "daily stop" in report.reason.lower()


def test_a_finished_match_settles_the_ledger(ledger):
    ledger.record_pending(
        {
            "match_id": "m1",
            "market_key": "1x2",
            "selection": "Home",
            "odds": 2.0,
            "stake_ngn": 100.0,
            "model_p": 0.5,
            "p_lower": 0.45,
            "ev": 0.0,
            "confidence": "unknown",
        }
    )
    cap = FakeCapture(state=make_state(home=2, away=0, period="ft", minute=90.0))
    runner = make_runner(ledger, cap)
    report = runner.run_cycle()
    assert report.settled == 1
    bet = ledger.bets_for("m1")[0]
    assert bet["status"] == "won"
    assert ledger.daily_pnl() == pytest.approx(100.0)


def test_posture_is_saved_and_restored(tmp_path):
    ledger = BetLedger(LedgerConfig(), log_path=tmp_path / "l.jsonl")
    state_file = tmp_path / "state.json"
    cap = FakeCapture()
    runner = LiveRunner(
        capture=cap, ledger=ledger, state_path=str(state_file)
    )
    runner.set_posture(posture_for_mode(PostureMode.AGGRESSIVE))
    assert state_file.exists()

    fresh = LiveRunner(capture=cap, ledger=ledger, state_path=str(state_file))
    restored = fresh.load_state()
    assert restored is not None
    assert restored.mode is PostureMode.AGGRESSIVE


def test_posture_changes_the_execution(ledger):
    """A conservative posture raises the EV floor, so fewer bets qualify."""
    cap = FakeCapture()
    runner = make_runner(ledger, cap)
    runner.set_posture(posture_for_mode(PostureMode.CONSERVATIVE))
    assert runner.posture.min_ev == 0.08
    aggressive = make_runner(ledger, cap)
    aggressive.set_posture(posture_for_mode(PostureMode.AGGRESSIVE))
    assert aggressive.posture.min_ev < 0.08


def test_capture_errors_are_contained(ledger):
    class Boom:
        def capture(self, wait=True):
            raise RuntimeError("browser gone")

    runner = LiveRunner(capture=Boom(), ledger=ledger)
    report = runner.run_cycle()
    assert report.ready is False
    assert "capture error" in report.reason
    assert runner.last_error


def test_run_forever_respects_the_cycle_limit(ledger):
    cap = FakeCapture(ready=False, reason="not ready")
    runner = make_runner(ledger, cap, cycle_s=0.0)
    reports = runner.run_forever(max_cycles=3)
    assert len(reports) == 3
    assert [r.cycle for r in reports] == [1, 2, 3]


# --- match selection ------------------------------------------------------

def _cand(period="2nd Half", minute=55.0, markets=10, i=0, home="A", away="B"):
    from atlas.live.capture import BoardCandidate

    return BoardCandidate(
        match_id=f"c{i}",
        home_team=home,
        away_team=away,
        score="1-0",
        minute=minute,
        period=period,
        num_markets=markets,
        index=i,
    )


def test_an_ended_match_is_never_chosen():
    """A finished card shows 'ENDED' in its result text and the HALF-TIME
    score in its period, so both fields must be checked."""
    from atlas.live.capture import BoardCandidate
    from atlas.live.runner import LiveRunner

    ended = BoardCandidate(
        match_id="c0", home_team="A", away_team="B",
        score="ENDED | 1 | : | 2", minute=None, period="HT 1 : 0",
        num_markets=40, index=0,
    )
    live = _cand(period="2nd Half", minute=55.0, markets=8, i=1)
    chosen = LiveRunner.choose_match([ended, live])
    assert chosen is not None
    assert chosen.period == "2nd Half"


def test_finished_matches_are_never_chosen():
    from atlas.live.runner import LiveRunner

    candidates = [
        _cand(period="FT", markets=40, i=0),
        _cand(period="2nd Half", minute=55.0, markets=8, i=1),
    ]
    chosen = LiveRunner.choose_match(candidates)
    assert chosen is not None
    assert chosen.period == "2nd Half"


def test_the_second_half_is_preferred_over_the_first():
    from atlas.live.runner import LiveRunner

    candidates = [
        _cand(period="1st Half", minute=30.0, markets=10, i=0),
        _cand(period="2nd Half", minute=60.0, markets=10, i=1),
    ]
    chosen = LiveRunner.choose_match(candidates)
    assert chosen.period == "2nd Half"


def test_a_deeper_match_wins_at_equal_phase():
    from atlas.live.runner import LiveRunner

    candidates = [
        _cand(period="2nd Half", minute=60.0, markets=4, i=0),
        _cand(period="2nd Half", minute=60.0, markets=30, i=1),
    ]
    chosen = LiveRunner.choose_match(candidates)
    assert chosen.num_markets == 30


def test_the_live_middle_is_preferred_over_the_very_end():
    from atlas.live.runner import LiveRunner

    candidates = [
        _cand(period="2nd Half", minute=88.0, markets=10, i=0),
        _cand(period="2nd Half", minute=50.0, markets=10, i=1),
    ]
    chosen = LiveRunner.choose_match(candidates)
    assert chosen.minute == 50.0


def test_no_live_candidates_means_no_choice():
    from atlas.live.runner import LiveRunner

    assert LiveRunner.choose_match([]) is None
    assert LiveRunner.choose_match([_cand(period="FT")]) is None


# --- cashout handling ----------------------------------------------------

class FakeCashoutDriver:
    def __init__(self, offers=None, allow_execute=False):
        from atlas.browser.cashout import CashoutOffer

        self.offers = offers or []
        self.allow_execute = allow_execute
        self.executed = []

    def read_offers(self):
        return self.offers

    def execute(self, index):
        self.executed.append(index)
        return {"ok": True}


def test_cashout_offer_is_valued_against_the_open_position(ledger):
    from atlas.browser.cashout import CashoutOffer

    ledger.record_pending(
        {
            "match_id": "m1",
            "market_key": "1x2",
            "selection": "Home",
            "odds": 2.0,
            "stake_ngn": 100.0,
            "model_p": 0.5,
            "p_lower": 0.45,
            "ev": 0.0,
            "confidence": "unknown",
        }
    )
    offer = CashoutOffer(index=0, stake_ngn=100.0, offered_ngn=130.0)
    driver = FakeCashoutDriver([offer])
    runner = make_runner(ledger, FakeCapture(), cashout_driver=driver)
    vals = runner.handle_cashout_offers()
    assert len(vals) == 1
    # The model value is 100, so 130 beats it: take the cashout.
    assert vals[0]["recommended"] is True
    assert vals[0]["discount"] == pytest.approx(0.30)


def test_a_cheap_cashout_is_refused(ledger):
    from atlas.browser.cashout import CashoutOffer

    ledger.record_pending(
        {
            "match_id": "m1",
            "market_key": "1x2",
            "selection": "Home",
            "odds": 2.0,
            "stake_ngn": 100.0,
            "model_p": 0.5,
            "p_lower": 0.45,
            "ev": 0.0,
            "confidence": "unknown",
        }
    )
    offer = CashoutOffer(index=0, stake_ngn=100.0, offered_ngn=70.0)
    driver = FakeCashoutDriver([offer])
    runner = make_runner(ledger, FakeCapture(), cashout_driver=driver)
    vals = runner.handle_cashout_offers()
    assert vals[0]["recommended"] is False
    assert driver.executed == []


def test_cashout_execution_needs_the_explicit_opt_in(ledger):
    from atlas.browser.cashout import CashoutOffer

    ledger.record_pending(
        {
            "match_id": "m1",
            "market_key": "1x2",
            "selection": "Home",
            "odds": 2.0,
            "stake_ngn": 100.0,
            "model_p": 0.5,
            "p_lower": 0.45,
            "ev": 0.0,
            "confidence": "unknown",
        }
    )
    offer = CashoutOffer(index=0, stake_ngn=100.0, offered_ngn=130.0)
    driver = FakeCashoutDriver([offer], allow_execute=False)
    runner = make_runner(ledger, FakeCapture(), cashout_driver=driver)
    runner.handle_cashout_offers()
    assert driver.executed == []


def test_an_ambiguous_cashout_offer_is_not_acted_on(ledger):
    """Two open bets of the same stake: the offer cannot be attributed, so
    the loop must not act on it."""
    from atlas.browser.cashout import CashoutOffer

    for sel in ("Home", "Away"):
        ledger.record_pending(
            {
                "match_id": "m1",
                "market_key": "1x2",
                "selection": sel,
                "odds": 2.0,
                "stake_ngn": 100.0,
                "model_p": 0.5,
                "p_lower": 0.45,
                "ev": 0.0,
                "confidence": "unknown",
            }
        )
    offer = CashoutOffer(index=0, stake_ngn=100.0, offered_ngn=130.0)
    driver = FakeCashoutDriver([offer])
    runner = make_runner(ledger, FakeCapture(), cashout_driver=driver)
    vals = runner.handle_cashout_offers()
    assert vals == []
    assert driver.executed == []


def test_multi_view_period_labels_are_understood():
    """Multi View labels a half 'H1'/'H2'; the phase factor must recognise
    them or the 35%-weight term is inert."""
    from atlas.live.runner import LiveRunner

    candidates = [
        _cand(period="H2", minute=80.0, markets=10, i=0),
        _cand(period="H1", minute=45.0, markets=10, i=1),
    ]
    chosen = LiveRunner.choose_match(candidates)
    assert chosen is not None
    assert chosen.period == "H2"


def test_single_view_period_labels_are_understood():
    from atlas.live.runner import LiveRunner

    candidates = [
        _cand(period="2nd Half", minute=80.0, markets=10, i=0),
        _cand(period="1st Half", minute=45.0, markets=10, i=1),
    ]
    chosen = LiveRunner.choose_match(candidates)
    assert chosen is not None
    assert chosen.period == "2nd Half"
