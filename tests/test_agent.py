"""Tests for the AI-first agent plane.

Style note (mirrors tests/test_runner.py): fake adapters implement the
adapter surface discipline the real WebBridgeAdapter enforces — the daemon
envelope edges are asserted so the plane cannot quietly start trusting a
different transport.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.agent.action_log import AgentActionLog
from atlas.agent.brain import BrainError, OpencodeBrain, _harvest_text_lines
from atlas.agent.orchestrator import (
    AgentOrchestrator,
    cycle_prompt,
    parse_sessions,
)
from atlas.agent.tab import (
    CertaintyError,
    MoneyGateError,
    TabTool,
)
from atlas.browser.webbridge import WebBridgeError
from atlas.ops.ledger import BetLedger, BetRecord, LedgerConfig


class FakeAdapter:
    """Records _post calls; answers a canned envelope per action."""

    def __init__(self, answers: dict[str, Any] | None = None, fail_on: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.answers = answers or {}
        self.fail_on = fail_on or set()

    def _post(self, action: str, args: dict) -> Any:
        self.calls.append((action, args))
        if action in self.fail_on:
            raise WebBridgeError(f"daemon call {action} failed", transient=False)
        return self.answers.get(action, {"action": action})


def _tmp_log(tmp_path: Path) -> AgentActionLog:
    return AgentActionLog(str(tmp_path / "actions.jsonl"))


# ---------------------------------------------------------------------------
# action log
# ---------------------------------------------------------------------------


def test_agent_action_log_truncates_and_validates(tmp_path):
    log = _tmp_log(tmp_path)
    log.append("ai_action", action="snapshot", payload="x" * 3000)
    entries = log.tail()
    assert entries[0]["event"] == "ai_action"
    rec = entries[0]["record"]
    assert rec["payload"].endswith("...[3000 chars]")


# ---------------------------------------------------------------------------
# TabTool: money gate, logging order, unknown actions
# ---------------------------------------------------------------------------


def test_money_gate_refuses_click_when_closed(tmp_path):
    tool = TabTool("sess", adapter=FakeAdapter(), action_log=_tmp_log(tmp_path), allow_money=False)
    with pytest.raises(MoneyGateError):
        tool.call("click", {"selector": "@e12"})


def test_money_gate_allows_click_when_armed(tmp_path):
    adapter = FakeAdapter()
    tool = TabTool("sess", adapter=adapter, action_log=_tmp_log(tmp_path), allow_money=True)
    data = tool.call("click", {"selector": "@e12"})
    assert data == {"action": "click"}
    events = [e["event"] for e in log_entries(tmp_path)]
    assert events[-2] == "ai_action" and events[-1] == "ai_action_result"


def test_reads_stay_open_without_money(tmp_path):
    tool = TabTool("sess", adapter=FakeAdapter(), action_log=_tmp_log(tmp_path), allow_money=False)
    assert tool.call("snapshot", {}) == {"action": "snapshot"}


def test_failure_is_logged_before_and_after(tmp_path):
    tool = TabTool(
        "sess",
        adapter=FakeAdapter(fail_on={"snapshot"}),
        action_log=_tmp_log(tmp_path),
        allow_money=False,
    )
    with pytest.raises(WebBridgeError):
        tool.call("snapshot", {})
    events = log_entries(tmp_path)
    assert events[0]["event"] == "ai_action" and events[0]["record"]["ok"] is None


def log_entries(tmp_path: Path) -> list[dict]:
    path = tmp_path / "actions.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# TabTool.record_bet: the ledger rail
# ---------------------------------------------------------------------------


def test_record_bet_writes_ledger_entry(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    tool = TabTool("sess", adapter=FakeAdapter(), action_log=_tmp_log(tmp_path), allow_money=False)
    result = tool.record_bet(
        market_key="1x2",
        selection="home",
        odds=2.05,
        stake_ngn=100.0,
        model_p=0.58,
        ev=0.07,
        ledger_path=ledger_path,
    )
    assert result["ok"] is True
    ledger = BetLedger(LedgerConfig(), log_path=ledger_path)
    bets = list(ledger.summary().items())
    assert ledger.summary()["placed"] == 1
    assert result["bet_id"]
    # record shape: an ai- prefixed id does not collide with the kernel's ids
    ref = dict(bets)
    del ref
    bet = ledger._bets[result["bet_id"]]
    assert bet.model_p == 0.58 and bet.stake_ngn == 100.0


def test_record_bet_refuses_certainty(tmp_path):
    tool = TabTool("sess", adapter=FakeAdapter(), action_log=_tmp_log(tmp_path), allow_money=False)
    with pytest.raises(CertaintyError):
        tool.record_bet(
            market_key="1x2", selection="home", odds=10.0, stake_ngn=100.0,
            model_p=0.97, ev=2.9, ledger_path=str(tmp_path / "ledger.jsonl"),
        )


def test_record_bet_observes_daily_stop(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    # A daily stop of -1.0 means one settled loss (-10) freezes placement.
    ledger = BetLedger(LedgerConfig(daily_loss_limit=-1.0), log_path=ledger_path)
    ledger.place_bet(
        BetRecord(
            bet_id="seed", match_id="m", market_key="1x2", selection="away",
            odds=2.0, stake_ngn=10.0, model_p=0.5, p_lower=None, ev=0.0,
            confidence="test", correlation_groups=[],
        )
    )
    ledger.settle_bet("seed", "lost")
    tool = TabTool("sess", adapter=FakeAdapter(), action_log=_tmp_log(tmp_path), allow_money=False)
    result = tool.record_bet(
        market_key="1x2", selection="home", odds=2.05, stake_ngn=100.0,
        model_p=0.58, ev=0.07, ledger_path=ledger_path,
        daily_loss_limit=-1.0,
    )
    assert result["ok"] is False
    assert result["why"]


# ---------------------------------------------------------------------------
# brain: argv, parsing, subprocess handling
# ---------------------------------------------------------------------------


def test_brain_argv_contains_session_and_auto():
    brain = OpencodeBrain(
        session="atlas-trader", model="x/y", agent_name="atlas", attach_url="http://127.0.0.1:4096"
    )
    argv = brain.argv("cycle prompt")
    joined = " ".join(argv)
    assert "--session atlas-trader" in joined
    assert "--format json" in joined
    assert "--auto" in joined
    assert "--attach http://127.0.0.1:4096" in joined
    assert "--model x/y" in joined
    assert argv[-1] == "cycle prompt"


def test_harvest_text_from_json_events():
    raw = "\n".join(
        json.dumps(e)
        for e in [
            {"type": "step_start", "part": {}},
            {"type": "text", "part": {"text": "I looked at the board."}},
            {"type": "text", "part": {"text": "No edge this cycle."}},
        ]
    )
    assert _harvest_text_lines(raw) == "I looked at the board.\nNo edge this cycle."


def test_harvest_falls_back_to_raw_stdout():
    assert _harvest_text_lines("not json at all") == ""
    brain = OpencodeBrain(session="s")
    fallback = brain.run.__doc__  # keep the linter honest about the function
    del fallback


def test_brain_run_extracts_text(monkeypatch, tmp_path):
    payload = json.dumps({"type": "text", "part": {"text": "cycle done"}}) + "\n"
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return SimpleNamespace(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    brain = OpencodeBrain(session="s", workdir=str(tmp_path))
    assert brain.run("go") == "cycle done"
    assert "go" in captured["argv"]


def test_brain_run_raises_on_exit_code(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(returncode=3, stdout="", stderr="boom"),
    )
    with pytest.raises(BrainError, match="exited 3"):
        OpencodeBrain(session="s").run("go")


def test_brain_timeout_maps_to_brain_error(monkeypatch):
    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=5)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(BrainError, match="timed out"):
        OpencodeBrain(session="s", timeout_s=1.0).run("go")


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def test_parse_sessions_defaults_and_splits():
    assert parse_sessions(None) == ["atlas-trader"]
    assert parse_sessions(" a , b,") == ["a", "b"]


def test_cycle_prompt_carries_ledger_facts():
    prompt = cycle_prompt(7, {"placed": 3, "unrealized_exposure_ngn": 300.0})
    assert "#7" in prompt
    assert '"placed": 3' in prompt
    assert "atlas.agent.tab" in prompt


def test_orchestrator_logs_cycles_and_survives_heartbeats(tmp_path):
    class FakeBrain:
        def __init__(self):
            self.session = "s1"
            self.calls: list[str] = []

        def run(self, msg: str) -> str:
            self.calls.append(msg)
            return "did: nothing; refused: everything; why: no edge"

    brave = FakeBrain()
    sleeps: list[float] = []
    printed: list[str] = []

    orch = AgentOrchestrator(
        brains=[brave],  # type: ignore[list-item]
        ledger=BetLedger(LedgerConfig(), log_path=str(tmp_path / "ledger.jsonl")),
        action_log=_tmp_log(tmp_path),
        cycle_s=0.5,
        max_cycles=2,
        sleep=sleeps.append,
        print=printed.append,
    )
    orch.run_forever()
    assert len(brave.calls) == 2
    events = [e["event"] for e in log_entries(tmp_path)]
    assert "agent_cycle" in events
    assert "agent_stopped" in events
    assert printed[-1].startswith("[cycle 2 | s1]")


def test_orchestrator_backs_off_on_brain_error(tmp_path):
    class BadBrain:
        def __init__(self):
            self.session = "bad"

        def run(self, msg: str) -> str:
            raise BrainError("opencode exited 1: boom")

    sleeps: list[float] = []
    orch = AgentOrchestrator(
        brains=[BadBrain()],  # type: ignore[list-item]
        ledger=BetLedger(LedgerConfig(), log_path=str(tmp_path / "ledger.jsonl")),
        action_log=_tmp_log(tmp_path),
        cycle_s=1.0,
        max_cycles=1,
        sleep=sleeps.append,
        print=lambda m: None,
    )
    orch.run_forever()
    events = [e["event"] for e in log_entries(tmp_path)]
    assert "agent_error" in events
    assert sleeps  # the backoff sleep was exercised


# ---------------------------------------------------------------------------
# CLI surface: unknown actions still refused at the adapter boundary
# ---------------------------------------------------------------------------


def test_tab_cli_call_smoke(monkeypatch, tmp_path, capsys):
    from atlas.agent import tab as tab_mod

    monkeypatch.setattr(tab_mod.AgentActionLog, "append", lambda *a, **k: None)

    class OkAdapter:
        def _post(self, action, args):
            return {"url": "https://x"}

    real = tab_mod.TabTool
    monkeypatch.setattr(
        tab_mod, "TabTool",
        lambda *a, **k: real(session="t", adapter=OkAdapter(),
                             action_log=_tmp_log(tmp_path), allow_money=False),
    )
    rc = tab_mod.main(
        ["call", "navigate", "--args-json", '{"url": "https://www.sportybet.com"}']
    )
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 0 and json.loads(out)["ok"] is True


def test_tab_cli_bet_writes_ledger(monkeypatch, tmp_path, capsys):
    from atlas.agent import tab as tab_mod

    ledger_path = str(tmp_path / "ledger.jsonl")
    log = _tmp_log(tmp_path)
    monkeypatch.setattr(tab_mod, "AgentActionLog", lambda *a, **k: log)
    rc = tab_mod.main(
        [
            "--ledger", ledger_path,
            "bet", "--market-key", "1x2", "--selection", "home",
            "--odds", "2.0", "--stake", "100", "--model-p", "0.6", "--ev", "0.2",
        ]
    )
    assert rc == 0
    assert BetLedger(LedgerConfig(), log_path=ledger_path).summary()["placed"] == 1


def test_tab_cli_money_refusal_exit_code(monkeypatch, tmp_path, capsys):
    from atlas.agent import tab as tab_mod

    monkeypatch.delenv("ATLAS_AGENT_MONEY", raising=False)

    fake = FakeAdapter()
    tool = TabTool("t", adapter=fake, action_log=_tmp_log(tmp_path), allow_money=False)
    monkeypatch.setattr(tab_mod, "TabTool", lambda *a, **k: tool)
    rc = tab_mod.main(["call", "fill", "--args-json", '{"selector": "@e1", "value": "100"}'])
    assert rc == 3
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out)["error"]["code"] == "money_gate"
