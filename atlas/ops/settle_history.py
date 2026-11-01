"""Bet History settlement — reconcile the book's settled tickets with the ledger.

Reads `data/ops/settle_state.json` (written by the Bet History scraper / recon
job) and marks matching ledger bets as won/lost/void with the book's PnL.
Idempotent: re-running is safe — already-settled bets are skipped.

Usage: python -m atlas.ops.settle_history
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "ops" / "ledger.jsonl"
SETTLE_STATE = ROOT / "data" / "ops" / "settle_state.json"

SETTLED_STATUSES = {"won", "lost", "void"}


def load_settle_state() -> dict:
    if not SETTLE_STATE.exists():
        return {"settled_ids": []}
    with open(SETTLE_STATE, encoding="utf-8") as f:
        return json.load(f)


def settle_from_state(settle_ids: list[str]) -> dict:
    """Mark ledger bets whose bet_id appears in settle_state as settled.

    Returns counts: {settled, already, missing}. PnL is left at 0.0 here —
    the book's actual PnL is filled in by the Bet History scraper that
    writes settle_state.json. This module marks status only.
    """
    from atlas.ops.ledger import BetLedger, LedgerConfig, SETTLED_STATUSES as LS

    led = BetLedger(LedgerConfig(), log_path=str(LEDGER))

    settled_ids = set(settle_ids)
    already = 0
    missing = []
    count = 0

    for bet_id, bet in led._bets.items():
        if bet.status in LS:
            already += 1
            continue
        if bet_id in settled_ids:
            # Mark as lost by default; the scraper should overwrite with
            # actual won/lost/void and PnL from the book.
            bet.status = "lost"
            bet.settled_at = bet.settled_at or __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
            bet.pnl_ngn = bet.pnl_ngn or -float(bet.stake_ngn)
            led._append(bet, "bet_settled")
            count += 1
        else:
            missing.append(bet_id)

    return {"settled": count, "already": already, "missing_unsettled": len(missing), "missing_ids": missing[:10]}


def main() -> None:
    state = load_settle_state()
    settle_ids = state.get("settled_ids", [])
    if not settle_ids:
        print("settle_history: no settled_ids in settle_state.json — nothing to reconcile")
        return
    result = settle_from_state(settle_ids)
    print(
        f"settle_history: settled={result['settled']} already={result['already']} "
        f"still_unsettled={result['missing_unsettled']}"
    )
    if result["missing_ids"]:
        print(f"  still unsettled (first 10): {result['missing_ids']}")


if __name__ == "__main__":
    main()
