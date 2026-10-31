"""Tests for the control dashboard (atlas.live.control).

The dashboard may steer the posture and read state. It must not be able to
place a bet, widen a limit, or bypass the clamp.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from atlas.live.capture import CaptureResult
from atlas.live.observe import LiveMatchState, TeamStats
from atlas.live.runner import LiveRunner
from atlas.ops.ledger import BetLedger, LedgerConfig
from atlas.ops.notify import Notifier
from atlas.ops.posture import PostureMode


def make_state():
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=0),
        away=TeamStats(name="Away", goals=0),
        period="2h",
        clock_minute=70.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


class FakeCapture:
    def capture(self, wait=True):
        return CaptureResult(
            ready=True,
            reason="ready",
            markets=[
                {
                    "match_id": "m1",
                    "market_key": "1x2",
                    "market_label": "1X2",
                    "status": "open",
                    "selections": [
                        {"name": "Home", "odds": 2.20},
                        {"name": "Draw", "odds": 3.40},
                        {"name": "Away", "odds": 3.20},
                    ],
                }
            ],
            state=make_state(),
            observed_at=datetime.now(timezone.utc),
            match_id="m1",
        )


@pytest.fixture
def client(tmp_path):
    ledger = BetLedger(LedgerConfig(), log_path=tmp_path / "l.jsonl")
    notifier = Notifier(log_path=tmp_path / "n.jsonl", console=False)
    runner = LiveRunner(
        capture=FakeCapture(), ledger=ledger, notifier=notifier
    )
    from atlas.live.control import build_app

    with TestClient(build_app(runner, notifier)) as c:
        c.runner = runner
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_status_reports_the_posture_and_the_stop(client):
    r = client.get("/status")
    body = r.json()
    assert body["posture"]["mode"] == "balanced"
    assert body["daily_stop_active"] is False
    assert body["daily_pnl_ngn"] == 0.0
    assert body["submission_enabled"] is False


def test_posture_endpoint_sets_the_posture(client):
    r = client.post("/posture", json={"instruction": "more revenue but low risk"})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"]["mode"] == "conservative"
    assert client.runner.posture.mode is PostureMode.CONSERVATIVE


def test_posture_is_clamped_by_the_dashboard_too(client):
    r = client.post(
        "/posture", json={"instruction": "aggressive, max volume", "use_llm": False}
    )
    applied = r.json()["applied"]
    from atlas.ops.posture import HARD_BOUNDS

    for field_name, (low, high) in HARD_BOUNDS.items():
        assert low <= applied[field_name] <= high


def test_empty_instruction_is_rejected(client):
    r = client.post("/posture", json={"instruction": "   "})
    assert r.status_code == 400


def test_notifications_endpoint_returns_the_feed(client):
    client.post("/posture", json={"instruction": "low risk"})
    r = client.get("/notifications")
    assert r.status_code == 200
    assert r.json()["notifications"]


def test_the_dashboard_cannot_place_a_bets(client):
    """There is no endpoint that submits a bet, and submission is off."""
    body = client.get("/status").json()
    assert body["submission_enabled"] is False
    for path in ("/place", "/bet", "/submit"):
        assert client.post(path, json={}).status_code == 404


def test_cycle_endpoint_runs_one_cycle(client):
    r = client.post("/cycle")
    assert r.status_code == 200
    assert r.json()["ready"] is True
    assert client.runner.cycle == 1


def test_dashboard_only_process_cannot_trigger_cycles(tmp_path):
    """Two processes share one SportyBet tab: the loop (money path, page
    owner) and the dashboard. A cycle in the dashboard process would
    navigate the board behind the loop's back — page locks are
    process-local. A dashboard-only server must refuse /cycle and say why;
    posture steering stays available (2026-10-28)."""
    from atlas.live.control import build_app

    ledger = BetLedger(LedgerConfig(), log_path=tmp_path / "l.jsonl")
    runner = LiveRunner(capture=FakeCapture(), ledger=ledger)
    with TestClient(build_app(runner, None, manual_cycles=False)) as c:
        r = c.post("/cycle")
        assert r.status_code == 409
        assert "observer" in r.json()["detail"]
        assert runner.cycle == 0, "no cycle may run in the observer process"
        # steering is NOT lost:
        p = c.post("/posture", json={"instruction": "low risk"})
        assert p.status_code == 200
