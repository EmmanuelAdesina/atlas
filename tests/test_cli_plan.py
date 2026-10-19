"""Tests for the `atlas.main plan` CLI — the full snapshot pipeline.

Observe-only: it prices every open market, runs the sequential portfolio, and
records accepted bets in the canonical ledger. Nothing is ever submitted.
"""

import json
import subprocess
import sys
from pathlib import Path

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


def test_plan_prices_and_writes_to_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    proc = run_plan(ledger)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "clear the conservative EV floor" in proc.stdout
    assert ledger.exists()

    lines = [l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines
    entries = [json.loads(l) for l in lines]
    assert any(e["event"] == "bet_placed" for e in entries)
    rec = [e for e in entries if e["event"] == "bet_placed"][0]["record"]
    assert rec["match_id"]
    assert rec["stake_ngn"] >= 100
    assert rec["odds"] > 1.0


def test_second_run_does_not_stack_identical_bets(tmp_path):
    """Second snapshot on the same ledger must respect prior-bet accounting."""
    ledger = tmp_path / "ledger.jsonl"
    first = run_plan(ledger)
    assert first.returncode == 0, first.stdout + first.stderr

    second = run_plan(ledger)
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
    proc = run_plan(ledger, extra=["--daily-loss-limit", "0"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "HARD STOP active" in proc.stdout
    assert "Plan: 0 bet(s) in this snapshot" in proc.stdout
    # No bets may be recorded. The ledger file may not even be created.
    if ledger.exists():
        lines = [l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        placed = [json.loads(l) for l in lines if json.loads(l)["event"] == "bet_placed"]
        assert not placed


def test_missing_fixture_fails_cleanly(tmp_path):
    proc = run_plan(tmp_path / "ledger.jsonl", fixture=tmp_path / "nope.json")
    assert proc.returncode == 1
    assert "Cannot read fixture" in proc.stdout
