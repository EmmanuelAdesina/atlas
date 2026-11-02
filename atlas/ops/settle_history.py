"""Bet History reconcile — settle the ledger from the book's own truth.

Reads `data/ops/settle_state.json` (written by the Bet History scraper,
`atlas.live.bet_history_scrape`) and matches the book's settled tickets
to the ledger's unresolved bets. Matched bets are settled with the BOOK's
outcome and amounts — pnl = what the book paid minus the stake, exactly
the ledger's payout path — via `BetLedger.record_book_settle`.

Matching rule (fail-closed, no guesses):

  A ledger bet matches a book ticket when the ticket's placement
  timestamp (createTime, epoch ms — the same anchor that is embedded in
  the orderId), the stake and the odds all agree within fixed tolerances:

    |createTime − placed_at| ≤ MATCH_WINDOW_S (150s)
    |totalStake − stake_ngn| ≤ STAKE_TOLERANCE (NGN 0.005)
    |selection odds − odds|  ≤ ODDS_TOLERANCE   (0.02)

  Candidate pairs are assigned bijectively: smallest time-delta first,
  ties broken by bet_id then order_id — for the observed ambiguous
  buckets (e.g. two NGN-27 @1.37 tickets seconds apart) this equals the
  sorted(bet_id)↔sorted(order_id) pairing that placement order implies.

Each ledger bet stays UNSETTLED unless exactly one book ticket proves
it; everything unmatched, or of unknown book status, is REPORTED, never
guessed. Idempotent: reruns skip already-settled bets and change nothing
when nothing matches.

This module never places money. It only records outcomes the book has
already reported, so the campaign wall and the daily stop see real PnL.

Usage:
  python -m atlas.ops.settle_history [--scrape] [--strict]
        [--ledger PATH] [--state PATH]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "ops" / "ledger.jsonl"
SETTLE_STATE = ROOT / "data" / "ops" / "settle_state.json"

# Fixed, pre-registered tolerances. Placement is claimed client-side a few
# seconds before the book creates the ticket; 150s covers clock skew
# between the ledger clock and the ticket clock without spanning two
# placement rounds (closest distinct rounds observed were minutes apart).
MATCH_WINDOW_S = 150.0
STAKE_TOLERANCE = 0.005
ODDS_TOLERANCE = 0.02

# Book winningStatus enum — probe-verified over the whole 264-ticket
# account history (2026-11-01): 20=won, 30=lost, 40=void/push. Anything
# else is 'unknown:<raw>' and is never settled from here.
WINNING_STATUS_MAP = {20: "won", 30: "lost", 40: "void"}

SETTLED_STATUSES = {"won", "lost", "void"}
SETTLE_SOURCE = "sportybet_bet_history"


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            # The ledger always writes UTC-aware iso strings; a naive
            # fallback would only appear if a foreign entry slipped in.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _ticket_time(ticket: dict) -> Optional[datetime]:
    """Ticket placement time. createTime (epoch ms) is authoritative; the
    pre-normalized create_time ISO is a fallback."""
    ms = ticket.get("create_time_ms")
    if ms:
        try:
            return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    return _parse_iso(ticket.get("create_time"))


def book_status_of(ticket: dict) -> Optional[str]:
    """'won'/'lost'/'void' for statuses ATLAS settles from; None for any
    unrecognized winningStatus (reported, never guessed)."""
    if ticket.get("status") in SETTLED_STATUSES:
        return ticket["status"]
    return None


def book_payout_ngn(ticket: dict) -> Optional[float]:
    """What the book reports as returned/paid for this ticket.

    won   -> totalWinnings (full return; ledger pnl = winnings − stake)
    lost  -> 0.00
    void  -> the stake (push; probe-verified totalWinnings == totalStake)
    None for unknown statuses or unparseable amounts — no settle.
    """
    status = book_status_of(ticket)
    try:
        stake = float(ticket.get("stake_ngn") or 0.0)
        winnings = float(ticket.get("winnings_ngn") or 0.0)
    except (TypeError, ValueError):
        return None
    if status == "won":
        return winnings
    if status == "lost":
        return 0.0
    if status == "void":
        return stake
    return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def match_candidates(
    bets: list[dict], tickets: list[dict]
) -> list[tuple[dict, dict, float]]:
    """All within-window (bet, ticket, |Δt| seconds) candidate pairs.

    Fail-closed details:
      - a bet without placed_at or stake can never match;
      - a ticket without usable createTime or odds can never match;
      - stake/odds/window tolerances are all mandatory.
    """
    pairs: list[tuple[dict, dict, float]] = []
    for bet in bets:
        placed = _parse_iso(bet.get("placed_at"))
        stake = _float_or_none(bet.get("stake_ngn"))
        odds = _float_or_none(bet.get("odds"))
        if placed is None or stake is None:
            continue
        for ticket in tickets:
            created = _ticket_time(ticket)
            if created is None:
                continue
            ticket_odds = _float_or_none(ticket.get("odds"))
            if ticket_odds is None:
                # Multiples/booster rows carry no single-leg odds we can
                # trust for identity — never matched.
                continue
            ticket_stake = _float_or_none(ticket.get("stake_ngn"))
            if ticket_stake is None:
                continue
            if abs(ticket_odds - odds) > ODDS_TOLERANCE:
                continue
            if abs(ticket_stake - stake) > STAKE_TOLERANCE:
                continue
            delta_s = (created - placed).total_seconds()
            if abs(delta_s) > MATCH_WINDOW_S:
                continue
            pairs.append((bet, ticket, abs(delta_s)))
    return pairs


def assign_bijective(
    pairs: list[tuple[dict, dict, float]],
) -> tuple[list[tuple[dict, dict]], dict]:
    """Greedy bijection; avoid any ledger bet settling twice or any book
    ticket settling two ledger bets. Ties: (Δt, bet_id, order_id)."""
    ordered = sorted(
        pairs,
        key=lambda p: (
            round(p[2], 3),
            str(p[0].get("bet_id")),
            str(p[1].get("order_id")),
        ),
    )
    used_bets: set[str] = set()
    used_tickets: set[str] = set()
    matched: list[tuple[dict, dict]] = []
    for bet, ticket, _delta in ordered:
        bet_id = str(bet.get("bet_id"))
        order_id = str(ticket.get("order_id"))
        if bet_id in used_bets or order_id in used_tickets:
            continue
        used_bets.add(bet_id)
        used_tickets.add(order_id)
        matched.append((bet, ticket))
    # Ambiguity seen in the raw candidate graph (pre-resolution): how many
    # bets and tickets had more than one candidate — honesty metric only.
    bet_counts: dict[str, int] = {}
    ticket_counts: dict[str, int] = {}
    for bet, ticket, _d in pairs:
        bet_counts[str(bet.get("bet_id"))] = bet_counts.get(str(bet.get("bet_id")), 0) + 1
        ticket_counts[str(ticket.get("order_id"))] = ticket_counts.get(str(ticket.get("order_id")), 0) + 1
    diag = {
        "bets_with_multiple_candidates": sum(1 for c in bet_counts.values() if c > 1),
        "tickets_with_multiple_candidates": sum(1 for c in ticket_counts.values() if c > 1),
        "max_candidates_per_bet": max(bet_counts.values()) if bet_counts else 0,
    }
    return matched, diag


def reconcile(
    ledger_path: Path = LEDGER,
    state_path: Path = SETTLE_STATE,
) -> dict:
    """One reconcile pass; writes settlements through the ledger payout
    path and returns a fail-closed report."""
    if not Path(state_path).exists():
        return {
            "scrape_state_missing": True,
            "message": "settle_state.json not found — run `python -m atlas.ops.settle_history --scrape`",
            "settled": {"won": 0, "lost": 0, "void": 0},
            "changed": 0,
            "open_bets": 0,
        }

    from atlas.ops.ledger import BetLedger, LedgerConfig  # delayed: import cycle

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    src = state.get("source") or {}
    tickets = state.get("tickets") or []

    ledger = BetLedger(LedgerConfig(), log_path=str(ledger_path))
    all_bets = ledger.all_bets()
    open_bets = [b for b in all_bets if b.get("status") not in SETTLED_STATUSES]

    pairs = match_candidates(open_bets, tickets)
    matched, amb = assign_bijective(pairs)

    report: dict[str, Any] = {
        "state_fetched_at": state.get("fetched_at"),
        "state_source": src,
        "open_bets": len(open_bets),
        "tickets_in_state": len(tickets),
        "candidate_pairs": len(pairs),
        "ambiguity": amb,
        "settled": {"won": 0, "lost": 0, "void": 0},
        "changed": 0,
        "matched": [],
        "unmatched_bets": [],
        "unknown_status_tickets": [],
    }

    matched_bet_ids: set[str] = set()
    matched_order_ids: set[str] = set()

    for bet, ticket in matched:
        bet_id = str(bet.get("bet_id"))
        order_id = str(ticket.get("order_id"))
        matched_bet_ids.add(bet_id)
        matched_order_ids.add(order_id)

        status = book_status_of(ticket)
        if status is None:
            report["unknown_status_tickets"].append(
                {
                    "order_id": order_id,
                    "winning_status": ticket.get("winning_status"),
                    "money_status": ticket.get("status"),
                    "bet_id": bet_id,
                    "note": "matched but unrecognized book status — left unsettled",
                }
            )
            continue
        payout = book_payout_ngn(ticket)
        if payout is None:
            report["unknown_status_tickets"].append(
                {
                    "order_id": order_id,
                    "winning_status": ticket.get("winning_status"),
                    "bet_id": bet_id,
                    "note": "matched but book amounts unparseable — left unsettled",
                }
            )
            continue

        ledger.record_book_settle(
            bet_id=bet_id,
            status=status,
            payout_ngn=payout,
            book_status=f"order:{ticket.get('winning_status')}:{order_id}",
            source=SETTLE_SOURCE,
        )
        report["settled"][status] += 1
        report["changed"] += 1
        placed = _parse_iso(bet.get("placed_at"))
        created = _ticket_time(ticket)
        report["matched"].append(
            {
                "bet_id": bet_id,
                "order_id": order_id,
                "status": status,
                "stake_ngn": bet.get("stake_ngn"),
                "payout_ngn": round(payout, 2),
                "pnl_ngn": round(payout - float(bet.get("stake_ngn") or 0.0), 2),
                "delta_s": round((created - placed).total_seconds(), 3)
                if placed and created
                else None,
            }
        )

    for bet in open_bets:
        if str(bet.get("bet_id")) in matched_bet_ids:
            continue
        report["unmatched_bets"].append(
            {
                "bet_id": str(bet.get("bet_id")),
                "placed_at": bet.get("placed_at"),
                "stake_ngn": bet.get("stake_ngn"),
                "odds": bet.get("odds"),
                "market_key": bet.get("market_key"),
                "selection": bet.get("selection"),
                "status": bet.get("status"),
                "note": "no settled Bet History ticket inside the match window — left unsettled",
            }
        )

    # The book's list is account-wide (user history outside ATLAS, ~264
    # tickets vs 86 ledger bets). Unowned = settled tickets that matched
    # no ledger bet — an estimate of the account's external history.
    report["tickets_unowned_estimate"] = max(0, len(set(t.get("order_id") for t in tickets)) - len(matched_order_ids))
    report["ledger_status_counts"] = {
        k: sum(1 for b in ledger.all_bets() if b.get("status") == k)
        for k in ("won", "lost", "void", "placed", "pending")
    }
    return report


def _print_report(report: dict, cap: int = 15) -> None:
    print("=" * 68)
    print("BET HISTORY RECONCILE")
    if report.get("scrape_state_missing"):
        print(report["message"])
        return
    src = report.get("state_source") or {}
    print(
        f"  state fetched: {report.get('state_fetched_at')} "
        f"(api {src.get('api')}, session {src.get('session')})"
    )
    print(f"  open bets entering: {report['open_bets']}")
    print(f"  candidate pairs: {report['candidate_pairs']}  ambiguity: {json.dumps(report['ambiguity'], sort_keys=True)}")
    print(f"  settled: {json.dumps(report['settled'], sort_keys=True)}  changed={report['changed']}")
    matched = report["matched"]
    print(f"  matched {len(matched)}:")
    for item in matched[:cap]:
        print(
            f"    + {str(item['bet_id'])[:8]}…  {item['status']:<4} "
            f"stake={item['stake_ngn']} payout={item['payout_ngn']} "
            f"pnl={item['pnl_ngn']} Δt={item['delta_s']}s "
            f"order=…{str(item['order_id'])[-14:]}"
        )
    if len(matched) > cap:
        print(f"    … and {len(matched) - cap} more")
    unmatched = report["unmatched_bets"]
    if unmatched:
        print(f"  UNMATCHED (left unsettled, no book proof yet): {len(unmatched)}")
        for b in unmatched[:cap]:
            print(
                f"    - {str(b['bet_id'])[:8]}… placed={b['placed_at']} "
                f"stake={b['stake_ngn']} odds={b['odds']} "
                f"{b['market_key']} | {b['selection']}"
            )
        if len(unmatched) > cap:
            print(f"    … and {len(unmatched) - cap} more")
    unknown = report["unknown_status_tickets"]
    if unknown:
        print(f"  UNKNOWN BOOK STATUS (left unsettled): {len(unknown)}")
        for t in unknown[:cap]:
            print(f"    ? {json.dumps(t)}")
    print(
        f"  ticket snapshot: account-wide {report['tickets_in_state']}, "
        f"unowned (pre-ATLAS external history)≈{report['tickets_unowned_estimate']}"
    )
    print(
        f"  ledger status counts now: "
        f"{json.dumps(report['ledger_status_counts'], sort_keys=True)}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ledger_path = LEDGER
    state_path = SETTLE_STATE
    scrape_first = False
    strict = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--scrape":
            scrape_first = True
            i += 1
        elif arg == "--strict":
            strict = True
            i += 1
        elif arg == "--ledger" and i + 1 < len(argv):
            ledger_path = Path(argv[i + 1])
            i += 2
        elif arg == "--state" and i + 1 < len(argv):
            state_path = Path(argv[i + 1])
            i += 2
        else:
            print(f"settle_history: unknown argument {arg!r}")
            return 2

    if scrape_first:
        from atlas.live.bet_history_scrape import WebBridgeError
        from atlas.live.bet_history_scrape import scrape as scrape_book

        try:
            state = scrape_book()
            print(
                f"settle_history: scraped {state['count']} settled tickets "
                f"(enum {json.dumps(state['ticket_status_enum'], sort_keys=True)})"
            )
        except WebBridgeError as exc:
            if strict:
                raise
            # Environmental skip — the browser stack is restarting or the
            # extension is wedged. A later hourly pass converges; the
            # oneshot unit must not page the operator for this.
            print(
                f"settle_history: SCRAPE SKIPPED (browser unavailable): "
                f"{str(exc)[:240]}"
            )

    report = reconcile(ledger_path=ledger_path, state_path=state_path)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
