"""Tests for the SportyBet V1 vertical slice (offline, mocked browser input).

Fixture odds mirror the Boca Juniors vs Talleres PDF:
  1X2 1.61/3.60/5.60, O2.5 2.10/1.67, GG 2.10/1.64,
  Boca corners O5.5 1.70 / U5.5 2.08.
"""

import json
from datetime import datetime, timezone

import pytest

from atlas.collectors.sportybet import SportyBetCollector
from atlas.live.loop import LoopConfig, Opportunity, breakeven, expected_value, rank_and_filter
from atlas.mappers.sportybet_mapper import map_sport_market, map_sport_markets
from atlas.models.sport import SportMarket
from atlas.schemas.collection import CollectionStatus
from atlas.services.snapshots import save_raw_snapshot

COLLECTED_AT = datetime(2026, 10, 12, tzinfo=timezone.utc)
SOURCE_URL = "https://www.sportybet.com/ng/sport/football/Argentina/Primera_LPF/Boca_Juniors_vs_CA_Talleres_de_Cordoba/sr:match:73265408"
MATCH_ID = "sr:match:73265408"

RAW_1X2 = {
    "match_id": MATCH_ID, "market_key": "1x2", "market_label": "1X2",
    "selections": [
        {"name": "Home", "odds": 1.61},
        {"name": "Draw", "odds": 3.60},
        {"name": "Away", "odds": 5.60},
    ],
}
RAW_O25 = {
    "match_id": MATCH_ID, "market_key": "ou_2_5", "market_label": "Over/Under 2.5",
    "selections": [
        {"name": "Over 2.5", "odds": 2.10},
        {"name": "Under 2.5", "odds": 1.67},
    ],
}


def test_map_1x2_overround_and_fair():
    r = map_sport_market(dict(RAW_1X2), source_url=SOURCE_URL, collected_at=COLLECTED_AT)
    assert isinstance(r, SportMarket)
    assert r.platform == "sportybet"
    assert r.overround == pytest.approx(1.0775, abs=1e-3)
    fair = r.fair_probs()
    assert fair["Home"] == pytest.approx(0.5764, abs=1e-3)
    assert sum(fair.values()) == pytest.approx(1.0)


def test_map_rejects_sub_unit_odds():
    bad = {"match_id": MATCH_ID, "market_key": "x", "selections": [{"name": "A", "odds": 0.9}, {"name": "B", "odds": 1.5}]}
    with pytest.raises(ValueError, match=">= 1.0"):
        map_sport_market(bad, source_url=SOURCE_URL, collected_at=COLLECTED_AT)


def test_map_batch_reports_skipped():
    records, skipped = map_sport_markets(
        [dict(RAW_1X2), {"market_key": "broken"}],
        source_url=SOURCE_URL, collected_at=COLLECTED_AT,
    )
    assert len(records) == 1 and len(skipped) == 1


def test_collector_archives_snapshot(tmp_path):
    c = SportyBetCollector(raw_data_path=str(tmp_path))
    result = c.ingest_snapshot([dict(RAW_1X2), dict(RAW_O25)], source_url=SOURCE_URL, collected_at=COLLECTED_AT)
    assert result.status == CollectionStatus.SUCCESS
    assert result.metadata["count"] == 2
    assert "snapshot_path" in result.metadata
    envelope = json.loads(open(result.metadata["snapshot_path"], encoding="utf-8").read())
    assert envelope["payload"][0]["market_key"] == "1x2"


def test_collector_rejects_empty(tmp_path):
    c = SportyBetCollector(raw_data_path=str(tmp_path))
    result = c.ingest_snapshot([], source_url=SOURCE_URL)
    assert result.status == CollectionStatus.EMPTY


def test_breakeven_and_ev_math():
    assert breakeven(3.00) == pytest.approx(0.3333, abs=1e-3)
    assert expected_value(0.40, 3.00) == pytest.approx(0.20)
    assert expected_value(0.30, 3.00) == pytest.approx(-0.10)


def test_correlation_guard_blocks_stacked_home_goals_bets():
    opps = [
        Opportunity(MATCH_ID, "1x2_home", "Home", 1.61, 0.70, "live_v1"),
        Opportunity(MATCH_ID, "ou_2_5_over", "Over 2.5", 2.10, 0.55, "live_v1"),
        Opportunity(MATCH_ID, "btts_yes", "Yes", 2.10, 0.55, "live_v1"),
        Opportunity(MATCH_ID, "home_corners_5_5_over", "Over 5.5", 1.70, 0.65, "live_v1"),
    ]
    accepted, rejected = rank_and_filter(opps, LoopConfig(min_ev=0.05))
    keys = [o.market_key for o in accepted]
    assert "home_corners_5_5_over" in keys  # uncorrelated angle survives
    assert len([k for k in keys if k in {"1x2_home", "ou_2_5_over", "btts_yes"}]) <= 1
    assert any("correlated" in r["reason"] for r in rejected)


def test_weak_ev_rejected():
    opps = [Opportunity(MATCH_ID, "ou_2_5_over", "Over 2.5", 2.10, 0.44, "live_v1")]
    accepted, rejected = rank_and_filter(opps, LoopConfig(min_ev=0.05))
    assert accepted == [] and len(rejected) == 1
