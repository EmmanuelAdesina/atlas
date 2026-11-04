"""The orchestrator — keeps the AI-first loop alive on the VPS.

Cycle shape (per session):

  refresh ledger -> build cycle context -> opencode run --session ... -> log

The orchestrator owns NOTHING about the match: no pricing, no market list,
no event parsing. It supplies two things every cycle:

1. FACTS from the auditable store: the ledger summary (day P&L, open
   exposure, stops), the current UTC time, and the cycle number. The agent
   does NOT need to trust its own memory for these.
2. THE CADENCE: one prompt per cycle keeps the brain moving when the page
   is fine, and a backoff keeps the machine from spinning when opencode or
   the daemon is down.

The reasoning, the market exploration, the match selection, and the bet
placement all happen INSIDE the opencode session, through the tab tool
(atlas/agent/tab.py) and the WebBridge daemon. This file is the process
supervisor, not the mind.

SESSIONS: ATLAS_AGENT_SESSIONS is a comma-separated list (default one:
"atlas-trader"). Each session gets its own opencode conversation, so
tracking two live matches means two sessions, each with its own memory.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from atlas.agent.action_log import AgentActionLog, DEFAULT_PATH
from atlas.agent.brain import BrainError, OpencodeBrain
from atlas.ops.ledger import BetLedger, LedgerConfig

SESSIONS_ENV = "ATLAS_AGENT_SESSIONS"
DEFAULT_SESSIONS = "atlas-trader"
CYCLE_SECONDS_ENV = "ATLAS_AGENT_CYCLE_S"
DEFAULT_CYCLE_S = 90.0

# A failed cycle must not spin. Walls the deterministic era paid for are
# gone; the plan is evidence-driven, so a support failure is handled by
# cadence, not by a gate.
ERROR_BACKOFF_S = 120.0
OUTPUT_TAIL = 2000


def parse_sessions(raw: Optional[str]) -> list[str]:
    sessions = [s.strip() for s in (raw or "").split(",") if s.strip()]
    return sessions or [DEFAULT_SESSIONS]


def cycle_prompt(cycle_n: int, ledger_summary: dict) -> str:
    now = datetime.now(timezone.utc).isoformat()
    return (
        f"ATLAS agent cycle #{cycle_n} at {now}.\n"
        "Ledger facts (authoritative, just refreshed):\n"
        f"{json.dumps(ledger_summary, ensure_ascii=False)}\n\n"
        "Your operating rhythm:\n"
        "1. Use the tab tool (python -m atlas.agent.tab) on the live venue "
        "pages to explore markets and read the live match statistics. "
        "Sessions persist — you already know your tracked matches.\n"
        "2. Price what you explored, decide what is worth staking, and "
        "place it through the tab. Keep a stake budget per match and never "
        "duplicate the same selection.\n"
        "3. After the venue accepts a submission, record the position "
        "immediately via `atlas.agent.tab bet ...` — a submitted ticket "
        "missing from the ledger is an unattributable void in the record.\n"
        "4. Verify what the previous cycles left behind: open exposure vs "
        "the facts above, receipts in the betslip, page health.\n"
        "State in your final text what you did, what you refused, and why. "
        "'No edge this cycle' is a valid, useful answer."
    )


@dataclass
class AgentOrchestrator:
    brains: list[OpencodeBrain]
    ledger: BetLedger
    action_log: AgentActionLog
    cycle_s: float = DEFAULT_CYCLE_S
    max_cycles: Optional[int] = None
    # Tests inject a sleep and a printer; production keeps defaults.
    sleep: Callable[[float], None] = time.sleep
    print: Callable[[str], None] = lambda msg: None

    def _cycle(self, brain: OpencodeBrain, n: int) -> str:
        self.ledger.refresh()
        prompt = cycle_prompt(n, self.ledger.summary())
        text = brain.run(prompt)
        self.action_log.append(
            "agent_cycle",
            session=brain.session,
            cycle=n,
            output_tail=text[-OUTPUT_TAIL:],
        )
        return text

    def run_forever(self) -> None:
        n = 0
        try:
            while self.max_cycles is None or n < self.max_cycles:
                n += 1
                for brain in self.brains:
                    try:
                        text = self._cycle(brain, n)
                        self.print(
                            f"[cycle {n} | {brain.session}] {text[-300:]}"
                        )
                    except BrainError as e:
                        self._log_error(brain, n, str(e))
                        self.sleep(ERROR_BACKOFF_S)
                    except Exception as e:  # noqa: BLE001 — the loop must survive
                        self._log_error(
                            brain, n, f"{e}\n{traceback.format_exc(-3)}"
                        )
                        self.sleep(ERROR_BACKOFF_S)
                if self.max_cycles is None or n < self.max_cycles:
                    self.sleep(self.cycle_s)
        finally:
            self.action_log.append("agent_stopped", cycles=n)

    def _log_error(self, brain: OpencodeBrain, n: int, detail: str) -> None:
        self.action_log.append(
            "agent_error", session=brain.session, cycle=n, error=detail
        )
        self.print(f"[cycle {n} | {brain.session}] ERROR: {detail[:200]}")


def build_from_env(
    sessions: Optional[list[str]] = None,
    ledger_path: str = "data/ops/ledger.jsonl",
    action_log_path: Optional[str] = None,
    cycle_s: Optional[float] = None,
) -> tuple[list[OpencodeBrain], BetLedger, AgentActionLog, float]:
    names = sessions or parse_sessions(os.environ.get(SESSIONS_ENV))
    brains = [OpencodeBrain.from_env(session=name) for name in names]
    ledger = BetLedger(LedgerConfig(), log_path=ledger_path)
    log = AgentActionLog(action_log_path or DEFAULT_PATH)
    seconds = cycle_s
    if seconds is None:
        seconds = float(os.environ.get(CYCLE_SECONDS_ENV, DEFAULT_CYCLE_S))
    for b in brains:
        b.workdir = os.getcwd()
    return brains, ledger, log, seconds
