"""Settle source: the book's settled-ticket list over direct HTTP.

`GET /api/ng/orders/order/v2/realbetlist?isSettled=10&pageSize=50&pageNo=N`
— confirmed against two archived responses (totalNum=159 and 264; see
tests/fixtures/api/realbetlist_page1.json and data/ops/realbetlist_2026-10-28.json).

Output is the SAME settle_state.json shape that atlas.live.bet_history_scrape
writes and atlas.ops.settle_history consumes — a different road to the same
file, not a new format. The winningStatus enum mapping is preserved verbatim:
20=won, 30=lost, 40=void; anything else is reported as unknown:<raw> and is
never settled on.

Fail-closed rules:

  * bizCode != 10000          -> RejectedError (cycle keeps the wall shut)
  * page short-read           -> SchemaDriftError (raw quarantined)
  * fetched != totalNum       -> incomplete_fetch=True in the state doc
  * unparseable ticket rows   -> counted, kept, marked status=unknown
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atlas.transport import snapshots
from atlas.transport.client import TransportClient
from atlas.transport.errors import (
    RejectedError,
    SchemaDriftError,
    SessionExpiredError,
    TransportError,
)
from atlas.transport.schemas import RealbetlistEnvelope

API_PATH = "/api/ng/orders/order/v2/realbetlist"
PAGE_SIZE = 50
MAX_PAGES_HARD_CAP = 20  # 50 tickets/page * 20 = 1000 tickets per settle run

WINNING_STATUS = {20: "won", 30: "lost", 40: "void"}

HISTORY_REFERER = (
    "https://www.sportybet.com/ng/my_accounts/bet_history/sport_bets?isSettled=10"
)


def _iso_from_ms(ms) -> Optional[str]:
    try:
        secs, rem = divmod(int(ms), 1000)
        dt = datetime.fromtimestamp(secs, tz=timezone.utc)
        if rem:
            return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{rem:03d}+00:00"
        return dt.isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def normalize_ticket(row: dict) -> dict:
    """Exact mirror of atlas.live.bet_history_scrape.parse_ticket.

    Kept shape-identical so atlas.ops.settle_history consumes either writer
    unchanged (guarded by tests/test_transport_settle.py parity assertions).
    Adds only additive keys: staked_at (alias of create_time) — the book's
    own timestamp of when the stake was placed.
    """
    raw_status = row.get("winningStatus")
    status = WINNING_STATUS.get(raw_status, f"unknown:{raw_status}")
    try:
        stake = round(float(row.get("totalStake") or 0.0), 2)
        winnings = round(float(row.get("totalWinnings") or 0.0), 2)
    except (TypeError, ValueError):
        stake = None
        winnings = None

    create_ms = row.get("createTime")
    selections = []
    for sel in row.get("selections") or []:
        odds_raw = sel.get("odds")
        try:
            odds = float(odds_raw) if odds_raw is not None else None
        except (TypeError, ValueError):
            odds = None
        score = None
        for key in ("setScore", "gameScore"):
            if sel.get(key):
                score = sel.get(key)
                break
        selections.append({
            "event_id": sel.get("eventId"),
            "market_desc": sel.get("marketDesc"),
            "outcome_desc": sel.get("outcomeDesc"),
            "correct_outcome": sel.get("correctOutcome"),
            "odds": odds,
            "home": sel.get("home"),
            "away": sel.get("away"),
            "leg_status": sel.get("status"),
            "match_status": sel.get("matchStatus"),
            "event_status": sel.get("eventStatus"),
            "score": score,
            "specifier": sel.get("specifier"),
            "tournament": sel.get("tournamentName"),
        })

    first = (row.get("selections") or [{}])[0]
    try:
        first_odds = float(first.get("odds")) if first.get("odds") is not None else None
    except (TypeError, ValueError):
        first_odds = None

    return {
        "order_id": row.get("orderId"),
        "short_id": row.get("shortId"),
        "create_time_ms": create_ms,
        "create_time": _iso_from_ms(create_ms) if create_ms else None,
        "staked_at": _iso_from_ms(create_ms) if create_ms else None,
        "winning_status": raw_status,
        "status": status,
        "stake_ngn": stake,
        "winnings_ngn": winnings,
        "currency": row.get("currency"),
        "selection_size": row.get("selectionSize"),
        "odds": first_odds,
        "selections": selections,
    }


def fetch_page(client: TransportClient, page_no: int) -> tuple[list[dict], int, int]:
    """Fetch one page. Returns (raw_rows, total_num, pages_total_hint)."""
    resp = client.get(
        API_PATH,
        params={"isSettled": "10", "pageSize": str(PAGE_SIZE), "pageNo": str(page_no)},
        referer=HISTORY_REFERER,
        retries=2,
    )
    archive_path = snapshots.archive_raw(
        "realbetlist",
        resp.body,
        meta={"page_no": page_no, "status_code": resp.status_code},
    )
    try:
        payload = json.loads(resp.body)
        env = RealbetlistEnvelope.model_validate(payload)
        env.require_ok()
    except (json.JSONDecodeError, ValueError) as exc:
        snapshots.quarantine_raw("realbetlist", resp.body, f"schema: {exc}")
        raise SchemaDriftError(
            f"realbetlist page {page_no} failed validation; raw archived at {archive_path}"
        ) from exc

    data = env.data
    if data is None:  # require_ok() guarantees this; belt and braces
        raise RejectedError("realbetlist envelope carries no data payload")
    rows = [t.model_dump() for t in data.entityList]
    return rows, data.totalNum, data.pageNo or page_no


def fetch_settled(client: TransportClient, max_pages: int = 10) -> dict:
    """Fetch all settled pages; assemble settle_state.json content."""
    max_pages = max(1, min(int(max_pages), MAX_PAGES_HARD_CAP))
    rows: list[dict] = []
    total_num: Optional[int] = None
    pages_fetched = 0
    for page_no in range(1, max_pages + 1):
        page_rows, page_total, _ = fetch_page(client, page_no)
        total_num = page_total
        rows.extend(page_rows)
        pages_fetched += 1
        if len(rows) >= total_num:
            break

    tickets = [normalize_ticket(r) for r in rows]
    seen: set[str] = set()
    unique: list[dict] = []
    for t in tickets:
        oid = t.get("order_id") or ""
        if oid:
            if oid in seen:
                continue
            seen.add(oid)
        unique.append(t)

    enum_counts: dict[str, int] = {}
    for t in unique:
        enum_counts[t["status"]] = enum_counts.get(t["status"], 0) + 1

    fetched = len(rows)
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "api": API_PATH,
            "host": "www.sportybet.com",
            "filter": "isSettled=10",
            "transport": "http",
            "session": "direct",
            "pages_fetched": pages_fetched,
            "page_size": PAGE_SIZE,
        },
        "total_num": total_num,
        "count": len(unique),
        "incomplete_fetch": total_num is not None and fetched < total_num,
        "ticket_status_enum": enum_counts,
        "tickets": unique,
    }


def write_settle_state(state: dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from atlas.transport.session import load_vault

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--out", default="data/ops/settle_state.json")
    args = parser.parse_args(argv)

    try:
        session = load_vault()
    except (FileNotFoundError, ValueError) as exc:
        print(f"settle: vault unreadable: {exc} (seed with python -m atlas.transport.seed)")
        return 2
    with TransportClient(session) as client:
        try:
            state = fetch_settled(client, max_pages=args.max_pages)
        except SessionExpiredError:
            print("settle: session expired — re-seed the vault (atlas.transport.session)")
            return 2
        except RejectedError as exc:
            print(f"settle: rejected by the book: {exc}")
            return 3
        except TransportError as exc:
            print(f"settle: transport failure (fail closed) — {exc}")
            return 4
    write_settle_state(state, Path(args.out))
    print(
        f"settle: {state['count']} tickets "
        f"(totalNum={state['total_num']}, incomplete={state['incomplete_fetch']}) "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
