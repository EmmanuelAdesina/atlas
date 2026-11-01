"""Tests for the `atlas.main plan` CLI — the full snapshot pipeline.

Observe-only: it prices every open market, runs the sequential portfolio, and
records accepted bets in the canonical ledger. Nothing is ever submitted.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path("tests/fixtures/fortuna_stuttgart_90plus.json")


def run_plan(ledger: Path, fixture: Path = FIXTURE, extra: list[str] | None = None):
    cmd = [
        sys.executable,
        "-m",
        "atlas.main",
        "plan",
        "--fixture",
        str(fixture),
        "--ledger",
        str(ledger),
    ] + (extra or [])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def generous_fixture(tmp_path: Path) -> Path:
    """A fixture with an edge that survives every honest gate.

    A prior-only model against a book that prices with live statistics
    never qualifies: the 5pp model-market gap guard and the calibration
    shrink mean a fairly priced market carries no edge. And the previous
    version of this fixture cheated in the OTHER direction — 'Over 1.5'
    at 70' priced 2.20 while the model saturated at p≈1.0 (needs one goal
    in 20 minutes with live shots), a +120% 'edge' that is really a
    mislabeled near-certainty: exactly the shape the absurd-EV ceiling
    now refuses (2026-11-01 Perth: 'Over 3.5' claiming +46%).

    So the test builds the ONE shape of edge the engine is allowed to
    recommend: a graded-high estimate on a semantics-verified family whose
    price is inside the [min_ev, max_ev] band. Nine league-standard
    corners by 70' — the corner model needs 2 more for Over 10.5 — the
    book hangs 1.30 against the model's 0.82: +6.6% claimed EV.
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["state"]["clock_minute"] = 70.0
    data["state"]["clock_label"] = "70:00"
    data["state"]["home"]["goals"] = 0
    data["state"]["home"]["corners"] = 6
    data["state"]["away"]["corners"] = 3
    data["state"]["home"]["shots_on_target"] = 3
    data["state"]["home"]["shots_total"] = 8
    data["state"]["home"]["possession_pct"] = 55.0
    data["state"]["away"]["shots_on_target"] = 2
    data["state"]["away"]["shots_total"] = 6
    data["state"]["away"]["possession_pct"] = 45.0
    for market in data.get("markets", []):
        if market.get("market_key") == "ou_1_5":
            market["market_key"] = "corners_ou_10_5"
            market["market_label"] = "Corners O/U 10.5"
            market["line"] = 10.5
            market["selections"] = [
                {"name": "Over 10.5", "odds": 1.30},
                {"name": "Under 10.5", "odds": 4.60},
            ]
    out = tmp_path / "generous.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def test_plan_prices_the_board_and_finds_no_edge_on_a_fair_market(tmp_path):
    """A market priced fairly has no edge after calibration, and the plan
    must say so rather than inventing one."""
    ledger = tmp_path / "ledger.jsonl"
    proc = run_plan(ledger)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "clear the conservative EV floor" in proc.stdout
    assert "No sure-thing arb" in proc.stdout
    if ledger.exists():
        lines = [
            l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        placed = [
            json.loads(l) for l in lines if json.loads(l)["event"] == "bet_placed"
        ]
        assert not placed, "a fairly priced market must not produce a bet"


def test_plan_records_a_bet_when_there_is_a_real_edge(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    fixture = generous_fixture(tmp_path)
    proc = run_plan(ledger, fixture=fixture)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Plan: 1 bet(s)" in proc.stdout

    lines = [l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    placed = [json.loads(l) for l in lines if json.loads(l)["event"] == "bet_placed"]
    assert len(placed) == 1
    rec = placed[0]["record"]
    assert rec["market_key"] == "corners_ou_10_5"
    assert rec["selection"] == "Over 10.5"
    assert rec["stake_ngn"] >= 100
    assert rec["stake_ngn"] <= 1000


def test_second_run_does_not_stack_identical_bets(tmp_path):
    """Second snapshot on the same ledger must respect prior-bet accounting."""
    ledger = tmp_path / "ledger.jsonl"
    fixture = generous_fixture(tmp_path)
    first = run_plan(ledger, fixture=fixture)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "Plan: 1 bet(s)" in first.stdout

    second = run_plan(ledger, fixture=fixture)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "1 prior bet(s) on this match" in second.stdout
    # The same selection must not be re-bet without a rebalance trigger.
    assert "Plan: 0 bet(s) in this snapshot" in second.stdout

    lines = [l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    placed = [json.loads(l) for l in lines if json.loads(l)["event"] == "bet_placed"]
    assert len(placed) == 1


def test_daily_stop_suppresses_betting(tmp_path):
    """A ledger already at the daily loss limit must block new bets."""
    ledger = tmp_path / "ledger.jsonl"
    fixture = generous_fixture(tmp_path)
    proc = run_plan(ledger, fixture=fixture, extra=["--daily-loss-limit", "0"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "HARD STOP active" in proc.stdout
    assert "Plan: 0 bet(s) in this snapshot" in proc.stdout
    if ledger.exists():
        lines = [l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        placed = [json.loads(l) for l in lines if json.loads(l)["event"] == "bet_placed"]
        assert not placed


def test_missing_fixture_fails_cleanly(tmp_path):
    proc = run_plan(tmp_path / "ledger.jsonl", fixture=tmp_path / "nope.json")
    assert proc.returncode == 1
    assert "Cannot read fixture" in proc.stdout


def test_posture_changes_the_ev_floor(tmp_path):
    """A conservative posture must demand a higher edge than an aggressive one."""
    ledger = tmp_path / "ledger.jsonl"
    fixture = generous_fixture(tmp_path)
    conservative = run_plan(
        ledger, fixture=fixture, extra=["--instruction", "low risk, protect the bank"]
    )
    assert conservative.returncode == 0, conservative.stdout + conservative.stderr
    assert "conservative" in conservative.stdout
    assert "EV floor 0.08" in conservative.stdout
