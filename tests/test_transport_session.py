"""Session vault, executor boundary and ledger stake-time guarantee tests."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone

import httpx
import pytest

from atlas.transport.client import TransportClient
from atlas.transport.errors import OutcomeUnknownError, OrderNotConfiguredError, RejectedError
from atlas.transport.executor import DirectOrderExecutor, OrderRequest, SubmissionNotAllowed
from atlas.transport.session import Session, load_vault, redact_headers, seed_vault
from atlas.ops.ledger import BetLedger


# ---------------------------------------------------------------- vault ---
@pytest.fixture(autouse=True)
def _walls_open(monkeypatch):
    """Tests must not depend on this working tree's live campaign verdict.

    The campaign wall is exactly the kind of thing that must FAIL CLOSED in
    production — so the tests pin it open explicitly, never by accident.
    """
    monkeypatch.setattr("atlas.ops.campaign.frozen", lambda: False)


def test_vault_roundtrip_and_permissions(tmp_path):
    path = tmp_path / "session.json"
    cookies = {"sb_country": "ng", "usrId": "secret-user", "locale": "en"}
    seed_vault(cookies, path)
    if os.name == "posix":
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600  # Windows has no 0600 equivalent; prod is POSIX
    loaded = load_vault(path)
    assert loaded.cookies == cookies
    assert "secret-user" not in json.dumps(loaded.redacted_headers())
    assert redact_headers({"cookie": "sb_country=ng; usrId=secret-user"})["cookie"] == "[redacted]"


def test_vault_refuses_empty_seed(tmp_path):
    with pytest.raises(ValueError):
        seed_vault({}, tmp_path / "session.json")


def test_vault_missing_raises_not_invents(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vault(tmp_path / "absent.json")


# ------------------------------------------------------------- executor ---
def _session() -> Session:
    return Session(cookies={"sb_country": "ng"}, seeded_at=datetime.now(timezone.utc).isoformat())


def test_executor_refuses_without_explicit_allow(tmp_path):
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))
    with TransportClient(_session(), transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with pytest.raises(SubmissionNotAllowed):
            DirectOrderExecutor(client, ledger)


def test_executor_unconfigured_without_recipe(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_ORDER_RECIPE", raising=False)
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))

    def handler(request):  # pragma: no cover — must never be reached
        raise AssertionError("network touched before recipe check")

    with TransportClient(_session(), transport=httpx.MockTransport(handler)) as client:
        ex = DirectOrderExecutor(client, ledger, allow_submit=True)
        with pytest.raises(OrderNotConfiguredError):
            ex.place(OrderRequest(event_id="e", market_id="m", outcome_id="o", odds=1.5, stake_ngn=10.0))
    assert ledger.all_bets() == []


def _recipe(tmp_path):
    recipe = {"body_template": {"eventId": "{{eventId}}", "marketId": "{{marketId}}",
                                "outcomeId": "{{outcomeId}}", "odds": "{{odds}}",
                                "stake": "{{stake}}"},
              "note": "test recipe"}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return str(path)


def test_executor_rejected_records_no_position(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_REFUSAL_LOG", str(tmp_path / "refusals.jsonl"))
    monkeypatch.setattr(
        "atlas.transport.snapshots.archive_raw", lambda *a, **k: tmp_path / "o.json"
    )
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))

    def handler(request):
        return httpx.Response(200, json={"bizCode": 10004, "message": "rejected"})

    with TransportClient(_session(), transport=httpx.MockTransport(handler)) as client:
        ex = DirectOrderExecutor(client, ledger, allow_submit=True,
                                 recipe_path=_recipe(tmp_path))
        with pytest.raises(RejectedError):
            ex.place(OrderRequest(event_id="e", market_id="m", outcome_id="o", odds=1.5, stake_ngn=10.0))
    assert ledger.all_bets() == []
    refusal = (tmp_path / "refusals.jsonl").read_text().strip()
    assert "order_refused_http" in refusal


def test_executor_accepted_unverified_counts_exposure_not_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atlas.transport.snapshots.archive_raw", lambda *a, **k: tmp_path / "o.json"
    )
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))

    def handler(request):
        return httpx.Response(200, json={"bizCode": 10000, "data": {"orderId": "ord123"}})

    with TransportClient(_session(), transport=httpx.MockTransport(handler)) as client:
        ex = DirectOrderExecutor(client, ledger, allow_submit=True,
                                 recipe_path=_recipe(tmp_path), verifier=None)
        out = ex.place(OrderRequest(event_id="e", market_id="m", outcome_id="o",
                                    odds=1.5, stake_ngn=10.0, bet_id="bet1"))
    assert out.state == "accepted_unverified"
    bets = ledger.all_bets()
    assert len(bets) == 1
    assert bets[0]["receipt"] == "unverified"
    assert bets[0]["placed_at_epoch_ms"] is not None


def test_executor_verified_stamps_book_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atlas.transport.snapshots.archive_raw", lambda *a, **k: tmp_path / "o.json"
    )
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))
    book_create_ms = 1793186108668  # the real ticket time observed in archives

    def verifier(order, payload):
        return {"order_id": "ord999", "short_id": "s1", "create_time_ms": book_create_ms,
                "odds": 1.85}

    def handler(request):
        return httpx.Response(200, json={"bizCode": 10000, "data": {"orderId": "ord999"}})

    with TransportClient(_session(), transport=httpx.MockTransport(handler)) as client:
        ex = DirectOrderExecutor(client, ledger, allow_submit=True,
                                 recipe_path=_recipe(tmp_path), verifier=verifier)
        out = ex.place(OrderRequest(event_id="e", market_id="m", outcome_id="o",
                                    odds=1.5, stake_ngn=10.0, bet_id="bet2"))
    assert out.state == "verified"
    bet = [b for b in ledger.all_bets() if b["bet_id"] == "bet2"][0]
    assert bet["placed_at_epoch_ms"] == book_create_ms
    assert datetime.fromtimestamp(book_create_ms / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds") == bet["placed_at"]
    assert bet["receipt"] == "verified:ord999/s1"
    assert bet["odds"] == 1.85  # book's accepted price wins over our priced one


def test_executor_unparseable_200_is_outcome_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atlas.transport.snapshots.quarantine_raw", lambda *a, **k: tmp_path / "q.bin"
    )
    monkeypatch.setattr(
        "atlas.transport.snapshots.archive_raw", lambda *a, **k: tmp_path / "o.json"
    )
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))

    def handler(request):
        return httpx.Response(200, text="<html>Sorry, something went wrong</html>")

    with TransportClient(_session(), transport=httpx.MockTransport(handler)) as client:
        ex = DirectOrderExecutor(client, ledger, allow_submit=True,
                                 recipe_path=_recipe(tmp_path))
        with pytest.raises(OutcomeUnknownError):
            ex.place(OrderRequest(event_id="e", market_id="m", outcome_id="o",
                                  odds=1.5, stake_ngn=10.0))
    assert ledger.all_bets() == []


# --------------------------------------------------------------- ledger ---
def test_ledger_placed_at_epoch_ms_and_reload(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = BetLedger(log_path=str(path))
    ok = ledger.record_pending({
        "match_id": "live", "market_key": "ou_2_5|over", "selection": "Over 2.5",
        "odds": 1.9, "stake_ngn": 10.0, "model_p": 0.55, "ev": 0.07,
        "confidence": "medium",
    })
    assert ok
    bet = ledger.all_bets()[0]
    assert isinstance(bet["placed_at_epoch_ms"], int)
    assert bet["placed_at_epoch_ms"] > 0
    # epoch ms must agree with the ISO claim clock
    iso_ms = int(datetime.fromisoformat(bet["placed_at"]).timestamp() * 1000)
    assert abs(iso_ms - bet["placed_at_epoch_ms"]) < 2000

    # reload from disk: the new field survives the JSONL roundtrip
    ledger2 = BetLedger(log_path=str(path))
    bet2 = ledger2.all_bets()[0]
    assert bet2["placed_at_epoch_ms"] == bet["placed_at_epoch_ms"]


def test_mark_verified_placed_rejects_bad_timestamps(tmp_path):
    ledger = BetLedger(log_path=str(tmp_path / "ledger.jsonl"))
    ledger.record_pending({
        "bet_id": "b1", "match_id": "live", "market_key": "m", "selection": "s",
        "odds": 1.5, "stake_ngn": 10.0, "model_p": 0.5, "ev": 0.05,
    })
    with pytest.raises(ValueError):
        ledger.mark_verified_placed("b1", order_id="o", short_id=None, create_time_ms=0)
    with pytest.raises(KeyError):
        ledger.mark_verified_placed("nope", order_id="o", short_id=None, create_time_ms=1793186108668)
