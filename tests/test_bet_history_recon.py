"""Bet History reconcile — the fail-closed contract, no browser needed.

Money truth rules under test:
  - settlement uses the BOOK's amounts (payout − stake), never a local
    recompute;
  - won/lost/void map from the probe-verified winningStatus enum
    (20/30/40); unknown statuses and unmatched bets are REPORTED, never
    guessed;
  - matching is by createTime+stake+odds inside a fixed window, assigned
    bijectively so no bet or ticket settles twice;
  - reruns are idempotent.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from atlas.live.bet_history_scrape import parse_ticket
from atlas.ops.ledger import BetLedger, BetRecord, LedgerConfig
from atlas.ops.settle_history import (
    book_payout_ngn,
    book_status_of,
    match_candidates,
    reconcile,
    assign_bijective,
)

API = "/api/ng/orders/order/v2/realbetlist"


# --------------------------------------------------------------------------
# fixtures


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def make_ledger(tmp_path: Path) -> tuple[BetLedger, Path]:
    log = tmp_path / "ledger.jsonl"
    ledger = BetLedger(LedgerConfig(), log_path=str(log))
    return ledger, log


def add_placed(
    ledger: BetLedger,
    bet_id: str,
    placed_at: str,
    odds: float,
    stake: float,
    market_key: str = "1x2",
    selection: str = "Home",
) -> None:
    """Placement with a pre-set claim time, bypassing the placement walls —
    reconcile tests observe settlement truth, not the money gate."""
    rec = BetRecord(
        bet_id=bet_id,
        match_id="live",
        market_key=market_key,
        selection=selection,
        odds=odds,
        stake_ngn=stake,
        model_p=0.5,
        p_lower=None,
        ev=0.05,
        confidence="medium",
        correlation_groups=[],
        placed_at=placed_at,
        status="placed",
    )
    # Mirror place_bet's two effects (memory insertion + append) without the
    # walls: these tests exercise settlement, not the money gate.
    with ledger._lock:
        ledger._bets[bet_id] = rec
    ledger._append(rec, "bet_placed")


def raw_ticket(
    order_id: str,
    create_ms: int,
    stake: float,
    odds: float,
    winnings: float,
    winning_status: int,
    outcome: str = "Away",
    market: str = "Money Line",
    event_id: str = "sr:match:66299760",
    away: str = "New York Red Bulls",
    home: str = "Nashville SC",
) -> dict:
    """A row in the shape the book API actually returns (probe schema)."""
    return {
        "orderType": 1,
        "shortId": "560120",
        "orderId": order_id,
        "totalStake": f"{stake:.2f}",
        "selectionSize": 1,
        "totalWinnings": f"{winnings:.2f}",
        "winningStatus": winning_status,
        "isPaymentInProgress": False,
        "createTime": create_ms,
        "currency": "NGN",
        "selections": [
            {
                "id": "261101231827sele17476286",
                "eventId": event_id,
                "marketId": "28",
                "outcomeId": "70",
                "status": 1,
                "matchStatus": "Ended",
                "eventStatus": 3,
                "odds": f"{odds}",
                "marketDesc": market,
                "outcomeDesc": outcome,
                "home": home,
                "away": away,
                "tournamentName": "MLS",
            }
        ],
    }


def write_state(tmp_path: Path, raw_rows: list[dict], fetched: str = "2026-11-01T23:59:00+00:00") -> Path:
    tickets = [parse_ticket(r) for r in raw_rows]
    path = tmp_path / "settle_state.json"
    path.write_text(
        json.dumps(
            {
                "fetched_at": fetched,
                "source": {"api": API, "host": "www.sportybet.com", "filter": "isSettled=10", "session": "atlas-recon-history"},
                "total_num": len(tickets),
                "count": len(tickets),
                "ticket_status_enum": {"won": 0, "lost": 0, "void": 0},
                "tickets": tickets,
            }
        ),
        encoding="utf-8",
    )
    return path


def iso_within(ms: int, seconds_before: float = 2.0) -> str:
    """A placed_at a couple of seconds before the book's createTime."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc) - timedelta(seconds=seconds_before)
    return dt.isoformat()


def run_reconcile(ledger: BetLedger, log: Path, state: Path) -> dict:
    """Reconcile through a fresh view: reconcile writes with its own ledger
    instance, so the caller's in-memory view must be re-read from disk."""
    report = reconcile(log, state)
    ledger.refresh()
    return report


# --------------------------------------------------------------------------
# parser


def test_parse_ticket_maps_the_probe_verified_enum() -> None:
    ms = 1793575107049
    t = parse_ticket(raw_ticket("261101231827ord47418247", ms, 15.0, 1.41, 0.0, 30))
    assert t["status"] == "lost"
    assert t["winning_status"] == 30
    assert t["stake_ngn"] == 15.0
    assert t["winnings_ngn"] == 0.0
    assert t["create_time_ms"] == ms
    assert t["create_time"].startswith("2026-11-01T23:18:27")
    assert t["selections"][0]["odds"] == pytest.approx(1.41)
    assert t["selections"][0]["event_id"] == "sr:match:66299760"

    won = parse_ticket(raw_ticket("o2", 1000, 10.0, 1.28, 12.8, 20))
    assert won["status"] == "won"

    void = parse_ticket(raw_ticket("o3", 2000, 10.0, 1.42, 10.0, 40))
    assert void["status"] == "void"
    assert void["winnings_ngn"] == pytest.approx(10.0)

    unknown = parse_ticket(raw_ticket("o4", 3000, 10.0, 1.5, 0.0, 99))
    assert unknown["status"] == "unknown:99"


# --------------------------------------------------------------------------
# settlement semantics


def test_won_settles_with_the_book_pnl_not_a_local_formula(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 23, 18, 27, tzinfo=timezone.utc))
    add_placed(ledger, "b_won", iso_within(create_ms), odds=1.28, stake=10.0)
    state = write_state(
        tmp_path,
        [raw_ticket("260000000000ordA", create_ms, 10.0, 1.28, 12.80, 20)],
    )

    report = run_reconcile(ledger, log, state)

    assert report["settled"] == {"won": 1, "lost": 0, "void": 0}
    assert report["changed"] == 1
    bet = next(b for b in ledger.all_bets() if b["bet_id"] == "b_won")
    assert bet["status"] == "won"
    # the book paid 12.80 — pnl is 12.80 − 10, NOT stake×(odds−1) recomputed
    assert bet["pnl_ngn"] == pytest.approx(2.80)
    assert bet["settle_ngn"] == pytest.approx(12.80)
    assert bet["book_status"] == "order:20:260000000000ordA"
    assert bet["settle_source"] == "sportybet_bet_history"


def test_lost_settles_negative_full_stake(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 23, 13, 24, tzinfo=timezone.utc))
    add_placed(ledger, "b_lost", iso_within(create_ms), odds=1.36, stake=27.0, market_key="ah_0", selection="Away (0)")
    state = write_state(
        tmp_path, [raw_ticket("260000000000ordB", create_ms, 27.0, 1.36, 0.0, 30)]
    )

    report = run_reconcile(ledger, log, state)
    assert report["settled"]["lost"] == 1
    bet = next(b for b in ledger.all_bets() if b["bet_id"] == "b_lost")
    assert bet["status"] == "lost"
    assert bet["pnl_ngn"] == pytest.approx(-27.0)
    assert bet["book_status"] == "order:30:260000000000ordB"


def test_void_is_a_push_stake_returned_pnl_zero(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 17, 6, 3, tzinfo=timezone.utc))
    add_placed(ledger, "b_void", iso_within(create_ms), odds=1.42, stake=10.0, market_key="ou_2", selection="Over 2")
    state = write_state(
        tmp_path, [raw_ticket("260000000000ordC", create_ms, 10.0, 1.42, 10.0, 40)]
    )

    report = run_reconcile(ledger, log, state)
    assert report["settled"]["void"] == 1
    bet = next(b for b in ledger.all_bets() if b["bet_id"] == "b_void")
    assert bet["status"] == "void"
    assert bet["settle_ngn"] == pytest.approx(10.0)
    assert bet["pnl_ngn"] == pytest.approx(0.0)
    assert bet["book_status"].startswith("order:40:")


# --------------------------------------------------------------------------
# matching contract


def test_ambiguous_same_bucket_pairs_bijectively(tmp_path: Path) -> None:
    """The observed Nashville bucket: two 27-ngn 1.37-odds tickets seconds
    apart. Each ledger bet must settle against exactly one distinct order."""
    ledger, log = make_ledger(tmp_path)
    base = datetime(2026, 11, 1, 23, 13, 0, tzinfo=timezone.utc)
    (create_a, create_b) = _ms(base), _ms(base + timedelta(seconds=9))
    # NOTE ledger claim times nearly identical to ticket times; Δt ~2s each
    add_placed(ledger, "bet_1", iso_within(create_a, 2.0), odds=1.37, stake=27.0, market_key="ah_0", selection="Away (0)")
    add_placed(ledger, "bet_2", iso_within(create_b, 2.0), odds=1.37, stake=27.0, market_key="dnb", selection="Away")
    state = write_state(
        tmp_path,
        [
            raw_ticket("26110123132xordAAA00", create_a, 27.0, 1.37, 0.0, 30, outcome="Away (0)"),
            raw_ticket("26110123133xordBBB00", create_b, 27.0, 1.37, 0.0, 30, outcome="Away"),
        ],
    )

    report = run_reconcile(ledger, log, state)
    assert report["changed"] == 2
    assert report["settled"]["lost"] == 2
    settled = {b["bet_id"]: b for b in ledger.all_bets() if b["status"] == "lost"}
    assert len(settled) == 2
    orders = {b["book_status"] for b in settled.values()}
    assert orders == {"order:30:26110123132xordAAA00", "order:30:26110123133xordBBB00"}
    # each bet must own exactly one order — no double-crediting
    assert len(orders) == len(settled)


def test_individual_deltas_drive_the_pairing_when_tolerances_tie(tmp_path: Path) -> None:
    """When both tickets could match by stake+odds, per-bet nearest Δt wins;
    sorted(bet_id)↔sorted(order_id) is the identity when Δt ties exactly."""
    ledger, log = make_ledger(tmp_path)
    base = datetime(2026, 11, 1, 15, 0, 0, tzinfo=timezone.utc)
    c1, c2 = _ms(base), _ms(base + timedelta(seconds=10))
    add_placed(ledger, "aaa", iso_within(c1, 2.0), odds=1.5, stake=10.0)
    add_placed(ledger, "bbb", iso_within(c2, 2.0), odds=1.5, stake=10.0)
    tickets = [
        parse_ticket(raw_ticket("26110115000xord111111", c1, 10.0, 1.5, 0.0, 20)),
        parse_ticket(raw_ticket("26110115010xord222222", c2, 10.0, 1.5, 0.0, 30)),
    ]
    pairs = match_candidates(
        [b for b in ledger.all_bets() if b["status"] == "placed"],
        tickets,
    )
    assert len(pairs) == 4
    matched, diag = assign_bijective(pairs)
    assert len(matched) == 2
    pairing = {b["bet_id"]: t["order_id"] for b, t in matched}
    # aaa (≈2s from c1) must take the c1 ticket, not the far one
    assert pairing["aaa"] == "26110115000xord111111"
    assert pairing["bbb"] == "26110115010xord222222"
    assert diag["bets_with_multiple_candidates"] == 2
    assert diag["tickets_with_multiple_candidates"] == 2


def test_window_mismatch_is_never_matched(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc))
    add_placed(ledger, "b_far", iso_within(create_ms, seconds_before=600.0), odds=1.45, stake=10.0)
    state = write_state(tmp_path, [raw_ticket("260000000000ordD", create_ms, 10.0, 1.45, 0.0, 30)])

    report = run_reconcile(ledger, log, state)
    assert report["changed"] == 0
    assert report["candidate_pairs"] == 0
    assert [b["bet_id"] for b in report["unmatched_bets"]] == ["b_far"]
    assert next(b for b in ledger.all_bets() if b["bet_id"] == "b_far")["status"] == "placed"


def test_unknown_book_status_is_reported_never_settled(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 19, 0, 0, tzinfo=timezone.utc))
    add_placed(ledger, "b_unk", iso_within(create_ms), odds=1.6, stake=10.0)
    state = write_state(tmp_path, [raw_ticket("260000000000ordE", create_ms, 10.0, 1.6, 99.99, 99)])

    report = run_reconcile(ledger, log, state)
    assert report["changed"] == 0
    assert report["settled"] == {"won": 0, "lost": 0, "void": 0}
    assert len(report["unknown_status_tickets"]) == 1
    assert report["unknown_status_tickets"][0]["money_status"] == "unknown:99"
    bet = next(b for b in ledger.all_bets() if b["bet_id"] == "b_unk")
    assert bet["status"] == "placed"
    assert bet["book_status"] is None


def test_stake_disagreement_breaks_the_match(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 21, 0, 0, tzinfo=timezone.utc))
    add_placed(ledger, "b_stake", iso_within(create_ms), odds=1.47, stake=10.0, market_key="home_odd_even")
    state = write_state(tmp_path, [raw_ticket("260000000000ordF", create_ms, 10.5, 1.47, 0.0, 30)])

    report = run_reconcile(ledger, log, state)
    assert report["changed"] == 0


def test_unmatched_older_bets_cohabit_external_history(tmp_path: Path) -> None:
    """The account holds 264 settled tickets, of which only 86 are ATLAS's —
    unmatched account history must not block or distort ATLAS settlement."""
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 23, 18, 27, tzinfo=timezone.utc))
    add_placed(ledger, "b_atlas", iso_within(create_ms), odds=1.41, stake=15.0)
    foreign = [
        raw_ticket("260000000000ordX1", _ms(datetime(2026, 11, 1, 20, 0, 0, tzinfo=timezone.utc)), 100.0, 3.7, 0.0, 30),
        raw_ticket("260000000000ordX2", _ms(datetime(2026, 11, 1, 20, 5, 0, tzinfo=timezone.utc)), 500.0, 2.1, 1050.0, 20),
    ]
    state = write_state(tmp_path, foreign + [raw_ticket("260000000000ordY", create_ms, 15.0, 1.41, 21.15, 20)])

    report = run_reconcile(ledger, log, state)
    assert report["changed"] == 1
    assert report["tickets_in_state"] == 3
    assert report["tickets_unowned_estimate"] == 2
    bet = next(b for b in ledger.all_bets() if b["bet_id"] == "b_atlas")
    assert bet["pnl_ngn"] == pytest.approx(6.15)


def test_reconcile_is_idempotent_no_double_settlement(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    create_ms = _ms(datetime(2026, 11, 1, 23, 18, 27, tzinfo=timezone.utc))
    add_placed(ledger, "b_idem", iso_within(create_ms), odds=1.41, stake=15.0)
    state = write_state(tmp_path, [raw_ticket("260000000000ordZ", create_ms, 15.0, 1.41, 21.15, 20)])

    first = run_reconcile(ledger, log, state)
    assert first["changed"] == 1
    with open(log, encoding="utf-8") as f:
        rows_after_first = sum(1 for line in f if line.strip())

    second = run_reconcile(ledger, log, state)
    assert second["changed"] == 0
    assert second["open_bets"] == 0
    # shutil-level guarantee: no second bet_settled row was appended
    with open(log, encoding="utf-8") as f:
        rows_after_second = sum(1 for line in f if line.strip())
    assert rows_after_second == rows_after_first

    bet = next(b for b in ledger.all_bets() if b["bet_id"] == "b_idem")
    assert bet["pnl_ngn"] == pytest.approx(6.15)


def test_missing_state_file_reports_instead_of_guessing(tmp_path: Path) -> None:
    ledger, log = make_ledger(tmp_path)
    report = reconcile(log, tmp_path / "absent_settle_state.json")
    assert report["scrape_state_missing"] is True
    assert report["changed"] == 0


# --------------------------------------------------------------------------
# the ledger method under the reconcile


def test_record_book_settle_rejects_unknown_statuses(tmp_path: Path) -> None:
    ledger, _ = make_ledger(tmp_path)
    add_placed(ledger, "b", iso_within(_ms(datetime(2026, 11, 1, tzinfo=timezone.utc))), 1.5, 10.0)
    with pytest.raises(ValueError):
        ledger.record_book_settle("b", "unknown", 0.0, "order:99:x")


def test_record_book_settle_is_idempotent_at_the_ledger(tmp_path: Path) -> None:
    log = tmp_path / "ledger.jsonl"
    ledger = BetLedger(LedgerConfig(), log_path=str(log))
    add_placed(ledger, "b2", iso_within(_ms(datetime(2026, 11, 1, tzinfo=timezone.utc))), 1.5, 10.0)

    first = ledger.record_book_settle("b2", "won", 18.5, "order:20:oid")
    assert first["pnl_ngn"] == pytest.approx(8.5)

    again = ledger.record_book_settle("b2", "lost", 0.0, "order:30:oid")
    # second call must silently return the FIRST truth, not overwrite it
    assert again["status"] == "won"
    assert again["pnl_ngn"] == pytest.approx(8.5)

    # one bet_settled row only
    with open(log, encoding="utf-8") as f:
        events = [json.loads(line)["event"] for line in f if line.strip()]
    assert events.count("bet_settled") == 1


# --------------------------------------------------------------------------
# pure payout helper


def test_book_payout_helper() -> None:
    t_lost = {"status": "lost", "winning_status": 30, "stake_ngn": 27.0, "winnings_ngn": 0.0}
    t_won = {"status": "won", "winning_status": 20, "stake_ngn": 10.0, "winnings_ngn": 12.8}
    t_void = {"status": "void", "winning_status": 40, "stake_ngn": 10.0, "winnings_ngn": 10.0}
    t_unknown = {"status": "unknown:77", "winning_status": 77, "stake_ngn": 10.0, "winnings_ngn": 3.0}
    assert book_payout_ngn(t_lost) == 0.0
    assert book_payout_ngn(t_won) == pytest.approx(12.8)
    assert book_payout_ngn(t_void) == pytest.approx(10.0)
    assert book_payout_ngn(t_unknown) is None
    assert book_status_of(t_unknown) is None
    assert book_status_of(t_won) == "won"
