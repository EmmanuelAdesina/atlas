"""Transport layer contract tests (no network, no browser).

Uses httpx.MockTransport. The archived realbetlist fixture is the parity
oracle against the legacy browser scraper.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from atlas.transport.client import TransportClient
from atlas.transport.errors import (
    OutcomeUnknownError,
    PreSendError,
    RejectedError,
    SchemaDriftError,
    SessionExpiredError,
)
from atlas.transport.session import Session
from atlas.transport.sources import balance, settle
from atlas.live.bet_history_scrape import parse_ticket

FIXTURES = Path(__file__).parent / "fixtures" / "api"
PAGE1 = json.loads((FIXTURES / "realbetlist_page1.json").read_text())


def _session() -> Session:
    return Session(
        cookies={"sb_country": "ng", "locale": "en"},
        seeded_at=datetime.now(timezone.utc).isoformat(),
    )


def _mock(handler):
    return TransportClient(_session(), transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- settle ---
def test_normalize_ticket_parity_with_browser_scraper():
    """API normalization must agree with the legacy parser on the same rows."""
    for raw in PAGE1["data"]["entityList"]:
        legacy = parse_ticket(raw)
        new = settle.normalize_ticket(raw)
        assert new["order_id"] == legacy["order_id"]
        assert new["short_id"] == legacy["short_id"]
        assert new["create_time_ms"] == legacy["create_time_ms"]
        assert new["create_time"] == legacy["create_time"]
        assert new["status"] == legacy["status"]
        assert new["winning_status"] == legacy["winning_status"]
        assert new["stake_ngn"] == legacy["stake_ngn"]
        assert new["winnings_ngn"] == legacy["winnings_ngn"]
        assert new["currency"] == legacy["currency"]
        assert new["selection_size"] == legacy["selection_size"]
        assert new["odds"] == legacy["odds"]
        assert [s["odds"] for s in new["selections"]] == [
            s["odds"] for s in legacy["selections"]
        ]


def test_fetch_settled_uses_archived_fixture(monkeypatch):
    monkeypatch.setattr(
        settle.snapshots, "archive_raw", lambda *a, **k: Path("/tmp/archived.json")
    )
    body = json.dumps(PAGE1).encode()

    def handler(request):
        assert request.url.params["isSettled"] == "10"
        assert request.headers["cookie"] == "sb_country=ng; locale=en"
        return httpx.Response(200, content=body)

    with _mock(handler) as client:
        state = settle.fetch_settled(client, max_pages=1)

    assert state["count"] == 2
    assert state["total_num"] == 159
    assert state["incomplete_fetch"] is True  # only one page fetched of 159
    assert state["tickets"][0]["create_time"]
    assert state["tickets"][0]["staked_at"] == state["tickets"][0]["create_time"]
    assert state["source"]["transport"] == "http"


def test_fetch_page_rejects_bizcode_failure(monkeypatch):
    monkeypatch.setattr(settle.snapshots, "archive_raw", lambda *a, **k: Path("/tmp/a.json"))
    monkeypatch.setattr(settle.snapshots, "quarantine_raw", lambda *a, **k: Path("/tmp/q.bin"))

    def handler(request):
        return httpx.Response(200, json={"bizCode": 10001, "message": "forbidden-ish", "data": None})

    with _mock(handler) as client:
        with pytest.raises(RejectedError):
            settle.fetch_page(client, 1)


def test_fetch_page_quarantines_malformed_json(monkeypatch, tmp_path):
    calls = {}

    def fake_quarantine(kind, body, reason, root=None):
        calls["kind"] = kind
        calls["reason"] = reason
        return tmp_path / "q.bin"

    monkeypatch.setattr(settle.snapshots, "archive_raw", lambda *a, **k: tmp_path / "a.json")
    monkeypatch.setattr(settle.snapshots, "quarantine_raw", fake_quarantine)

    def handler(request):
        return httpx.Response(200, content=b"<html>Sorry, something went wrong</html>")

    with _mock(handler) as client:
        with pytest.raises(SchemaDriftError):
            settle.fetch_page(client, 1)
    assert calls["kind"] == "realbetlist"


def test_total_num_zero_is_incomplete_not_crash(monkeypatch):
    monkeypatch.setattr(settle.snapshots, "archive_raw", lambda *a, **k: Path("/tmp/a.json"))

    def handler(request):
        return httpx.Response(200, json={
            "bizCode": 10000, "isAvailable": True, "message": "Success",
            "data": {"pageSize": 50, "pageNo": 1, "totalNum": 0, "entityList": []},
        })

    with _mock(handler) as client:
        state = settle.fetch_settled(client, max_pages=1)
    # totalNum=0 means "not one settled ticket" — NOT an error, but the
    # caller must still see the book's own total, never a fabricated count.
    assert state["total_num"] == 0
    assert state["count"] == 0
    assert state["incomplete_fetch"] is False


# ---------------------------------------------------------------- client ---
def test_get_5xx_retries_then_raises(monkeypatch):
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        return httpx.Response(503)

    monkeypatch.setattr(TransportClient, "_backoff", staticmethod(lambda *a, **k: None))
    with _mock(handler) as client:
        with pytest.raises(RejectedError):
            client.get("/x")
    assert hits["n"] == 3  # 1 + 2 retries


def test_get_connect_error_is_presend_and_bounded(monkeypatch):
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    monkeypatch.setattr(TransportClient, "_backoff", staticmethod(lambda *a, **k: None))
    with _mock(handler) as client:
        with pytest.raises(PreSendError):
            client.get("/x")
    assert hits["n"] == 3


def test_get_read_timeout_is_outcome_unknown_no_retry():
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    with _mock(handler) as client:
        with pytest.raises(OutcomeUnknownError):
            client.get("/x")
    assert hits["n"] == 1  # after send: never retried


def test_post_5xx_is_outcome_unknown_single_attempt():
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        return httpx.Response(500)

    with _mock(handler) as client:
        with pytest.raises(OutcomeUnknownError):
            client.post("/orders", body={"a": "b"})
    assert hits["n"] == 1


def test_401_is_session_expired():
    def handler(request):
        return httpx.Response(401)

    with _mock(handler) as client:
        with pytest.raises(SessionExpiredError):
            client.get("/x")


def test_429_honors_single_retry_then_gives_up(monkeypatch):
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "0"})

    monkeypatch.setattr(TransportClient, "_backoff", staticmethod(lambda *a, **k: None))
    with _mock(handler) as client:
        with pytest.raises(RejectedError):
            client.get("/x", retries=1)
    assert hits["n"] == 2


# ---------------------------------------------------------------- balance ---
USER_BAL = json.loads((FIXTURES / "user_bal_ngn.json").read_text())


def test_read_balance_parses_avlbal_envelope(monkeypatch):
    """Observed live NGN pocket shape (bizCode 10000, data.avlBal) parses."""
    monkeypatch.setattr(balance.snapshots, "archive_raw", lambda *a, **k: Path("/tmp/a.json"))

    def handler(request):
        return httpx.Response(200, content=json.dumps(USER_BAL).encode())

    with _mock(handler) as client:
        reading = balance.read_balance_ngn(client)

    assert reading.balance_ngn == pytest.approx(123.45)
    assert reading.source_key == "avlBal"


def test_read_balance_refuses_coins_only_payload(monkeypatch):
    """avlCoins is a coins balance, NOT the NGN balance — refuse to guess."""
    monkeypatch.setattr(balance.snapshots, "archive_raw", lambda *a, **k: Path("/tmp/a.json"))
    monkeypatch.setattr(balance.snapshots, "quarantine_raw", lambda *a, **k: Path("/tmp/q.bin"))
    payload = {
        "bizCode": 10000,
        "message": "Success",
        "data": {"avlCoins": 7.0, "coinsStatus": 1, "userFinAccNotExist": False},
    }

    def handler(request):
        return httpx.Response(200, content=json.dumps(payload).encode())

    with _mock(handler) as client:
        with pytest.raises(SchemaDriftError):
            balance.read_balance_ngn(client)
