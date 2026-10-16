"""Tests for the Polymarket Gamma vertical slice.

All network access is mocked: these tests run offline on any platform
(including ARMv7/Termux). Live end-to-end verification is a separate,
explicit manual step (`discover --limit N`), never part of the suite.
"""

import json
from datetime import datetime, timezone

import httpx
import pytest

from atlas.collectors.polymarket import PolymarketGammaCollector
from atlas.mappers.polymarket_mapper import map_market, map_markets
from atlas.models.market import MarketRecord
from atlas.schemas.collection import CollectionStatus
from atlas.services.snapshots import save_raw_snapshot

COLLECTED_AT = datetime(2026, 10, 12, tzinfo=timezone.utc)
SOURCE_URL = "https://gamma-api.polymarket.com/markets?closed=false&limit=2"

# Mirrors the real Gamma wire format: outcomes/prices are JSON-encoded strings.
RAW_MARKET = {
    "id": "123",
    "question": "Xi Jinping out before 2027?",
    "slug": "xi-jinping-out-before-2027",
    "description": "Resolves Yes if ...",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.0275", "0.9725"]',
    "volumeNum": 14785276.6,
    "liquidityNum": 598384.0,
    "bestBid": 0.027,
    "bestAsk": 0.028,
    "spread": 0.001,
    "lastTradePrice": 0.0275,
    "takerBaseFee": 0.0,
    "endDateIso": "2026-12-31T00:00:00Z",
    "resolutionSource": "https://example.com/resolution",
    "closed": False,
    "archived": False,
    "events": [{"id": "99"}],
    "conditionId": "0xabc",
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    """Stand-in for AtlasHTTPClient: records calls, returns canned payload."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    async def get(self, url, params=None, **kwargs):
        self.calls.append((url, params))
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


# --- Mapper ---


def test_map_market_string_encoded_outcomes():
    r = map_market(dict(RAW_MARKET), source_url=SOURCE_URL, collected_at=COLLECTED_AT)
    assert isinstance(r, MarketRecord)
    assert r.platform == "polymarket"
    assert r.market_id == "123"
    assert r.event_id == "99"
    assert r.outcomes == ["Yes", "No"]
    assert r.outcome_prices == pytest.approx([0.0275, 0.9725])
    assert r.status == "open"
    assert r.volume == pytest.approx(14785276.6)
    assert r.liquidity == pytest.approx(598384.0)
    assert r.parser_version == "polymarket_gamma_v1"
    assert r.end_time is not None and r.end_time.year == 2026


def test_map_market_real_list_outcomes():
    raw = dict(RAW_MARKET, outcomes=["Yes", "No"], outcomePrices=[0.5, 0.5], closed=True)
    r = map_market(raw, source_url=SOURCE_URL, collected_at=COLLECTED_AT)
    assert r.outcome_prices == [0.5, 0.5]
    assert r.status == "closed"


def test_map_market_rejects_mismatched_prices():
    raw = dict(RAW_MARKET, outcomePrices='["0.5"]')
    with pytest.raises(ValueError, match="length mismatch"):
        map_market(raw, source_url=SOURCE_URL, collected_at=COLLECTED_AT)


def test_map_market_rejects_missing_question():
    raw = dict(RAW_MARKET)
    del raw["question"]
    with pytest.raises(ValueError, match="question"):
        map_market(raw, source_url=SOURCE_URL, collected_at=COLLECTED_AT)


def test_map_markets_reports_skipped_not_crash():
    raws = [dict(RAW_MARKET), {"id": "bad"}]
    records, skipped = map_markets(raws, source_url=SOURCE_URL, collected_at=COLLECTED_AT)
    assert len(records) == 1
    assert len(skipped) == 1
    assert skipped[0]["market_id"] == "bad"


def test_market_record_rejects_out_of_range_price():
    with pytest.raises(ValueError):
        MarketRecord(
            platform="polymarket", market_id="1", question="Q?",
            outcomes=["Yes", "No"], outcome_prices=[1.5, -0.5],
            status="open", source="t", source_url="u",
            collected_at=COLLECTED_AT, parser_version="v",
        )


# --- Snapshots ---


def test_save_raw_snapshot_envelope(tmp_path):
    payload = [{"id": "1"}]
    path = save_raw_snapshot(
        source="polymarket_gamma", endpoint="https://x/markets",
        payload=payload, raw_data_path=tmp_path, collected_at=COLLECTED_AT,
    )
    assert path.exists()
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["source"] == "polymarket_gamma"
    assert envelope["payload"] == payload
    assert len(envelope["payload_sha256"]) == 64
    assert envelope["collected_at"] == COLLECTED_AT.isoformat()


# --- Collector (mocked transport) ---


@pytest.mark.asyncio
async def test_discover_markets_success(tmp_path):
    fake = FakeClient(payload=[dict(RAW_MARKET), dict(RAW_MARKET, id="124")])
    collector = PolymarketGammaCollector(client=fake, raw_data_path=str(tmp_path))
    result = await collector.discover_markets(limit=2)
    assert result.status == CollectionStatus.SUCCESS
    assert result.is_success
    assert result.record_count == 2
    assert result.metadata["count"] == 2
    assert "snapshot_path" in result.metadata
    assert fake.calls[0][0].endswith("/markets")
    assert fake.calls[0][1]["limit"] == 2


@pytest.mark.asyncio
async def test_discover_markets_empty(tmp_path):
    fake = FakeClient(payload=[])
    collector = PolymarketGammaCollector(client=fake, raw_data_path=str(tmp_path))
    result = await collector.discover_markets(limit=2)
    assert result.status == CollectionStatus.EMPTY
    assert not result.is_success


@pytest.mark.asyncio
async def test_discover_markets_rate_limited(tmp_path):
    req = httpx.Request("GET", "https://gamma-api.polymarket.com/markets")
    resp = httpx.Response(429, request=req)
    fake = FakeClient(error=httpx.HTTPStatusError("rate limited", request=req, response=resp))
    collector = PolymarketGammaCollector(client=fake, raw_data_path=str(tmp_path))
    result = await collector.discover_markets(limit=2)
    assert result.status == CollectionStatus.RATE_LIMITED
    assert result.error is not None


@pytest.mark.asyncio
async def test_discover_markets_network_error(tmp_path):
    fake = FakeClient(error=httpx.ConnectError("dns boom"))
    collector = PolymarketGammaCollector(client=fake, raw_data_path=str(tmp_path))
    result = await collector.discover_markets(limit=2)
    assert result.status == CollectionStatus.NETWORK_ERROR
