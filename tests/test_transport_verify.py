"""Tests for the realbetlist order verifier (tests fixture-backed)."""

from __future__ import annotations

import json

from pathlib import Path

from atlas.transport.executor import OrderRequest
from atlas.transport.verify import make_realbetlist_verifier

FIXTURE = Path(__file__).parent / "fixtures" / "api" / "realbetlist_page1.json"


def _fixture_rows() -> list[dict]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return list(payload["data"]["entityList"])


def _fixture_order(**overrides) -> OrderRequest:
    ticket = _fixture_rows()[0]
    sel = ticket["selections"][0]
    base = dict(
        event_id=sel["eventId"],
        market_id=str(sel["marketId"]),
        outcome_id=str(sel["outcomeId"]),
        odds=float(sel["odds"]),
        stake_ngn=float(ticket["totalStake"]),
        bet_id="unit-test-bet",
        match_id="live",
    )
    base.update(overrides)
    return OrderRequest(**base)


def _verifier(rows, *, now_ms=None):
    return make_realbetlist_verifier(lambda _page: rows, now_ms=now_ms and (lambda: now_ms))


def test_verified_when_exactly_one_match():
    rows = _fixture_rows()
    ticket = rows[0]
    verifier = _verifier(rows, now_ms=ticket["createTime"] + 30_000)
    result = verifier(_fixture_order(), {})
    assert result is not None
    assert result["order_id"] == ticket["orderId"]
    assert result["short_id"] == ticket["shortId"]
    assert result["create_time_ms"] == ticket["createTime"]
    assert abs(result["odds"] - float(ticket["selections"][0]["odds"])) < 1e-9


def test_stale_ticket_is_not_matched():
    rows = _fixture_rows()
    ticket = rows[0]
    # Verified 16 minutes after creation with default 15-minute lookback.
    verifier = _verifier(rows, now_ms=ticket["createTime"] + 16 * 60 * 1000)
    assert verifier(_fixture_order(), {}) is None


def test_wrong_stake_never_matches():
    rows = _fixture_rows()
    verifier = _verifier(rows, now_ms=rows[0]["createTime"] + 30_000)
    assert verifier(_fixture_order(stake_ngn=1.00), {}) is None


def test_wrong_event_never_matches():
    rows = _fixture_rows()
    verifier = _verifier(rows, now_ms=rows[0]["createTime"] + 30_000)
    assert verifier(_fixture_order(event_id="sr:match:000000"), {}) is None


def test_wrong_odds_never_matches():
    rows = _fixture_rows()
    verifier = _verifier(rows, now_ms=rows[0]["createTime"] + 30_000)
    assert verifier(_fixture_order(odds=2.50), {}) is None


def test_duplicate_match_is_ambiguous_and_refused():
    rows = _fixture_rows()
    doubled = rows[:1] + rows[:1]
    verifier = _verifier(doubled, now_ms=rows[0]["createTime"] + 30_000)
    assert verifier(_fixture_order(), {}) is None


def test_fetch_failure_fails_closed():
    def boom(_page):
        raise RuntimeError("vault locked")

    verifier = make_realbetlist_verifier(boom, now_ms=lambda: 1)
    assert verifier(_fixture_order(), {}) is None


def test_garbage_rows_are_skipped():
    rows = [{"createTime": "not-an-int"}, {"createTime": 123}, None, "junk"]
    verifier = make_realbetlist_verifier(lambda _p: rows, now_ms=lambda: 123 + 30_000)
    assert verifier(_fixture_order(), {}) is None


def test_order_id_missing_is_unverifiable():
    row = dict(_fixture_rows()[0])
    row.pop("orderId", None)
    verifier = _verifier([row], now_ms=row["createTime"] + 30_000)
    assert verifier(_fixture_order(), {}) is None
