"""The tab tool — the agent's finite, logged browser surface.

The opencode agent controls the venue page LIVE through this CLI:

    python -m atlas.agent.tab call snapshot
    python -m atlas.agent.tab call navigate --args-json '{"url": "https://www.sportybet.com/ng/m/ongoing-events", "newTab": true}'
    python -m atlas.agent.tab call evaluate --args-json '{"code": "(() => JSON.stringify(document.querySelector(...)?.innerText))()"}'
    python -m atlas.agent.tab list-tabs
    python -m atlas.agent.tab bet --market-key 1x2 --selection home --odds 2.05 \
        --stake 100 --model-p 0.58 --ev 0.07

Every call is envelope JSON on stdout ({"ok": true, "data": ...} shape) and
every call is written to the append-only agent action log first——BEFORE the
daemon answers, so a wedge is still audited.

Two rails survive the AI-first switch, and both live here:

1. Observe-only by construction. click/fill/upload are refused unless the
   OPERATOR exported ATLAS_AGENT_MONEY=1. The agent cannot arm its own
   money path; a page it could not pay through is a page it can still fully
   explore (navigate/snapshot/evaluate/network all stay open).
2. No silent actions. Every attempt — including failures — is logged with
   its args. The evidence trail outlives whatever the page did next.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from atlas.agent.action_log import AgentActionLog, DEFAULT_PATH
from atlas.browser.webbridge import DEFAULT_DAEMON_URL, WebBridgeAdapter, WebBridgeError
from atlas.ops.ledger import BetRecord, BetLedger, LedgerConfig

# The actions that can move money in a real logged-in session. Everything
# else (navigate, snapshot, evaluate, network, list_tabs, screenshots) is
# read/navigation and stays open in observe mode.
MONEY_ACTIONS = frozenset({"click", "fill", "upload"})

MONEY_ENV = "ATLAS_AGENT_MONEY"

# The WebBridge session the agent drives. A separate session name from the
# deterministic loop's keeps the agent's tab group distinct and auditable.
AGENT_SESSION_ENV = "ATLAS_WEBBRIDGE_AGENT_SESSION"
DEFAULT_AGENT_SESSION = "atlas-agent"

# Certainty refusal runs here too, at a level a little above the kernel's
# 0.95: agent probability claims at/above this are recorded defects.
CERTAINTY_CEILING = 0.95


class MoneyGateError(RuntimeError):
    """The operator has not armed money interaction for the agent."""


class CertaintyError(RuntimeError):
    """A recorded bet claims p >= ceiling. That is a defect, not a wager."""


def _env_allow_money() -> bool:
    return os.environ.get(MONEY_ENV, "").strip() == "1"


class TabTool:
    """One session-bound adapter + action log + money gate."""

    def __init__(
        self,
        session: str = DEFAULT_AGENT_SESSION,
        daemon_url: Optional[str] = None,
        adapter: Optional[WebBridgeAdapter] = None,
        action_log: Optional[AgentActionLog] = None,
        allow_money: Optional[bool] = None,
    ) -> None:
        self.session = session
        self.adapter = adapter or WebBridgeAdapter(
            session=session, daemon_url=daemon_url or DEFAULT_DAEMON_URL
        )
        self.action_log = action_log or AgentActionLog()
        self.allow_money = (
            _env_allow_money() if allow_money is None else allow_money
        )

    def call(self, action: str, args: dict[str, Any]) -> Any:
        if action in MONEY_ACTIONS and not self.allow_money:
            self.action_log.append(
                "ai_action_refused",
                action=action,
                reason="money_gate_closed",
                hint=f"export {MONEY_ENV}=1 to arm",
            )
            raise MoneyGateError(
                f"action {action!r} can move money and ATLAS_AGENT_MONEY != 1. "
                "The arm switch belongs to the operator, not the agent."
            )
        # Audit BEFORE the daemon call: a wedge must still appear in the log.
        self.action_log.append("ai_action", action=action, args=args, ok=None)
        data = self.adapter._post(action, args)
        self.action_log.append("ai_action_result", action=action, ok=True)
        return data

    def record_bet(
        self,
        *,
        market_key: str,
        selection: str,
        odds: float,
        stake_ngn: float,
        model_p: float,
        ev: float,
        ledger_path: str = "data/ops/ledger.jsonl",
        match_id: str = "live",
        match_label: Optional[str] = None,
        event_id: Optional[str] = None,
        confidence: str = "agent",
        p_lower: Optional[float] = None,
        daily_loss_limit: Optional[float] = None,
    ) -> dict:
        """Obligation hook: write a placed position into the SAME ledger the
        deterministic era settled from. Without this entry the reconciliation
        job cannot attribute the outcome, and the agent's edge claim is
        unverifiable — so the runbook makes this call mandatory after every
        submission the venue accepted.

        The daily stop is not persisted inside the ledger file, so it
        travels here from the operator's environment:
        ATLAS_AGENT_DAILY_LOSS_LIMIT (default -500, the kernel-era default).
        """
        if not 0.0 < model_p < CERTAINTY_CEILING:
            raise CertaintyError(
                f"model_p must obey 0 < p < {CERTAINTY_CEILING}; got {model_p!r}. "
                "A recorded >=0.95 is a defect report, not a stake."
            )
        limit = daily_loss_limit
        if limit is None:
            limit = float(os.environ.get("ATLAS_AGENT_DAILY_LOSS_LIMIT", -500.0))
        ledger = BetLedger(LedgerConfig(daily_loss_limit=limit), log_path=ledger_path)
        stake = float(stake_ngn)
        if not ledger.can_place_bet(stake):
            return {
                "ok": False,
                "why": (
                    "daily stop or campaign freeze active or stake out of "
                    f"range [{ledger.config.min_stake}, {ledger.config.max_stake}]"
                ),
                "summary": ledger.summary(),
            }
        bet = BetRecord(
            bet_id=f"ai-{uuid4().hex[:12]}",
            match_id=match_id,
            market_key=market_key,
            selection=selection,
            odds=float(odds),
            stake_ngn=round(stake, 2),
            model_p=float(model_p),
            p_lower=p_lower,
            ev=float(ev),
            confidence=confidence,
            correlation_groups=[],
            match_label=match_label,
            event_id=event_id,
        )
        accepted = ledger.place_bet(bet)
        self.action_log.append(
            "ai_bet_recorded", bet_id=bet.bet_id, accepted=accepted, stake_ngn=bet.stake_ngn
        )
        return {
            "ok": accepted,
            "bet_id": bet.bet_id if accepted else None,
            "summary": ledger.summary(),
        }


# ---------------------------------------------------------------------------
# CLI: the agent's day-1 interface. Argument schema mirrors the daemon's
# envelope ({"action", "args"}): the model reads this docstring or the
# runbook, calls the binary, and gets JSON back.
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atlas.agent.tab",
        description=(
            "Drive the live venue tab via the WebBridge daemon. "
            "Output: one JSON envelope per call. click/fill/upload require "
            f"{MONEY_ENV}=1."
        ),
    )
    p.add_argument(
        "--session",
        default=os.environ.get(AGENT_SESSION_ENV, DEFAULT_AGENT_SESSION),
        help="WebBridge task session (= the agent's browser tab group)",
    )
    p.add_argument(
        "--daemon-url",
        default=None,
        help=f"Daemon URL (default: {DEFAULT_DAEMON_URL})",
    )
    p.add_argument(
        "--action-log",
        default=os.environ.get("ATLAS_AGENT_ACTION_LOG", DEFAULT_PATH),
        help="Append-only action log path",
    )
    p.add_argument("--ledger", default="data/ops/ledger.jsonl", help="Ledger JSONL path")
    p.add_argument("--allow-money", action="store_true", default=None,
                   help="Override the env money gate for this single call")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("call", help="One WebBridge action, args as JSON")
    c.add_argument("action", help="snapshot | navigate | evaluate | network | ...")
    c.add_argument(
        "--args-json",
        default="{}",
        help='JSON object of args, e.g. \'{"url": "...", "newTab": true}\'',
    )

    lt = sub.add_parser("list-tabs", help="List tabs in this session")

    b = sub.add_parser("bet", help="Record an AI-placed bet in the ledger")
    b.add_argument("--market-key", required=True)
    b.add_argument("--selection", required=True)
    b.add_argument("--odds", type=float, required=True)
    b.add_argument("--stake", type=float, required=True, help="NGN")
    b.add_argument("--model-p", type=float, required=True, help="Agent's probability, < 0.95")
    b.add_argument("--p-lower", type=float, default=None, help="Low estimate if the agent states one")
    b.add_argument("--ev", type=float, default=0.0, help="Agent's EV estimate")
    b.add_argument("--match-id", default="live")
    b.add_argument("--match-label", default=None)
    b.add_argument("--event-id", default=None)
    b.add_argument("--confidence", default="agent")
    b.add_argument(
        "--daily-loss-limit",
        type=float,
        default=None,
        help="Hard daily stop for this recording session (default: ATLAS_AGENT_DAILY_LOSS_LIMIT or -500)",
    )
    return p



def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    tool = TabTool(
        session=args.session,
        daemon_url=args.daemon_url,
        action_log=AgentActionLog(args.action_log),
        allow_money=args.allow_money,
    )
    try:
        if args.cmd == "list-tabs":
            data = tool.call("list_tabs", {})
        elif args.cmd == "call":
            try:
                parsed = json.loads(args.args_json)
            except json.JSONDecodeError as e:
                print(json.dumps({"ok": False, "error": f"bad --args-json: {e}"}))
                return 2
            if not isinstance(parsed, dict):
                print(json.dumps({"ok": False, "error": "--args-json must be an object"}))
                return 2
            data = tool.call(args.action, parsed)
        elif args.cmd == "bet":
            return _record_bet_cli(tool, args)
    except MoneyGateError as e:
        print(json.dumps({"ok": False, "error": {"code": "money_gate", "message": str(e)}}))
        return 3
    except WebBridgeError as e:
        print(json.dumps({"ok": False, "error": {"code": "webbridge", "message": str(e)}}))
        return 1
    print(json.dumps({"ok": True, "data": data}))
    return 0


def _record_bet_cli(tool: TabTool, args: argparse.Namespace) -> int:
    try:
        result = tool.record_bet(
            market_key=args.market_key,
            selection=args.selection,
            odds=args.odds,
            stake_ngn=args.stake,
            model_p=args.model_p,
            ev=args.ev,
            ledger_path=args.ledger,
            match_id=args.match_id,
            match_label=args.match_label,
            event_id=args.event_id,
            confidence=args.confidence,
            p_lower=args.p_lower,
            daily_loss_limit=args.daily_loss_limit,
        )
    except CertaintyError as e:
        print(json.dumps({"ok": False, "error": {"code": "certainty_ceiling", "message": str(e)}}))
        return 2
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

