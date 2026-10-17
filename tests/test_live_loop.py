"""Tests for the Live Match Intelligence Loop increment.

Uses the recorded Fortuna Cologne vs VfB Stuttgart II observation
(tests/fixtures/fortuna_stuttgart_90plus.json + fortuna_snapshot_texts.json):
real bridge-read data, nothing fabricated. Timestamps replay as-of the
recorded observation time.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from atlas.browser.webbridge import WebBridgeAdapter, WebBridgeError
from atlas.live.decisions import DecisionLog, PolicyConfig
from atlas.live.estimate import Estimate, LiveStateSpaceModel, NoMoreGoalsBaseline, check_implications
from atlas.live.extract import extract_clock_score, extract_markets, iter_texts
from atlas.live.loop import LoopConfig, Opportunity, expected_value, rank_and_filter
from atlas.live.observe import LiveMatchState
from atlas.mappers.sportybet_mapper import map_sport_markets

FIXDIR = Path(__file__).parent / "fixtures"


def load_fixture():
    return json.loads((FIXDIR / "fortuna_stuttgart_90plus.json").read_text(encoding="utf-8"))


def make_state(fixture, observed_at):
    st = fixture["state"]
    return LiveMatchState(
        match_id=fixture["match_id"], competition=st.get("competition"),
        home=st["home"], away=st["away"], period=st.get("period", "unknown"),
        clock_minute=st.get("clock_minute"), clock_label=st.get("clock_label"),
        half_home_goals=st.get("half_home_goals"),
        half_away_goals=st.get("half_away_goals"),
        remaining_unknown=st.get("remaining_unknown", False),
        observed_at=observed_at, source="sportybet_browser", source_url="",
    )


# --- Extraction on the real snapshot texts ---


def test_extract_real_snapshot_markets_and_locks():
    texts = json.loads((FIXDIR / "fortuna_snapshot_texts.json").read_text(encoding="utf-8"))
    mk = extract_markets(texts, "m")
    by_key = {m["market_key"]: m for m in mk}
    ou = by_key["ou_1_5"]
    assert ou["status"] == "open" and ou["line"] == 1.5
    assert ou["selections"] == [
        {"name": "Over 1.5", "odds": 3.75},
        {"name": "Under 1.5", "odds": 1.23},
    ]
    h2 = by_key["1x2_2h"]
    assert h2["status"] == "open"
    assert {s["name"]: s["odds"] for s in h2["selections"]} == {
        "Home": 61.0, "Draw": 8.0, "Away": 1.05}
    assert by_key["1x2"]["status"] == "suspended"
    assert by_key["ou_2_5"]["status"] == "suspended"


def test_extract_real_clock_and_score():
    texts = json.loads((FIXDIR / "fortuna_snapshot_texts.json").read_text(encoding="utf-8"))
    info = extract_clock_score(texts)
    assert info == {"period": "2h", "clock_label": "90:00 +03:42",
                    "clock_minute": pytest.approx(93.7),
                    "home_goals": 0, "away_goals": 1}


def test_iter_texts_order_and_glyph_filter():
    tree = [{"role": "StaticText", "name": "A"},
            {"role": "StaticText", "name": "B",
             "children": [{"role": "link", "name": "C"}]}]
    assert iter_texts(tree) == ["A", "B", "C"]


# --- Observation freshness ---


def test_state_freshness_and_reds():
    fx = load_fixture()
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    s = make_state(fx, observed_at)
    assert s.is_live and s.score_total == 1
    assert s.home_reds == 1 and s.away_reds == 0
    assert s.is_fresh(60.0, now=observed_at)
    assert not s.is_fresh(60.0, now=datetime(2026, 10, 17, tzinfo=timezone.utc))


# --- Baseline model on the recorded state ---


def test_baseline_under_positive_ev_but_low_confidence():
    fx = load_fixture()
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    s = make_state(fx, observed_at)
    model = NoMoreGoalsBaseline(now=observed_at)
    est = model.estimate(s, "ou_1_5", "Under 1.5")
    assert est.p == pytest.approx(0.9618, abs=1e-3)
    assert est.confidence == "low"
    assert "stoppage" in (est.abstain_reason or "")
    assert expected_value(est.p, 1.23) > 0.05  # EV passes, confidence does not


def test_baseline_abstains_outside_family_stale_and_ft():
    fx = load_fixture()
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    s = make_state(fx, observed_at)
    model = NoMoreGoalsBaseline(now=observed_at)
    assert model.estimate(s, "1x2_2h", "Away").abstained
    stale = NoMoreGoalsBaseline(now=datetime(2026, 10, 17, tzinfo=timezone.utc))
    assert "stale" in (stale.estimate(s, "ou_1_5", "Under 1.5").abstain_reason or "")
    ft = s.model_copy(update={"period": "ft"})
    assert "not live" in (model.estimate(ft, "ou_1_5", "Under 1.5").abstain_reason or "")


# --- Implications on recorded fair probs ---


def test_implications_consistent_on_recorded_odds():
    fx = load_fixture()
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    s = make_state(fx, observed_at)
    recs, skipped = map_sport_markets(
        fx["markets"], source_url="", collected_at=observed_at)
    fair = {}
    for r in recs:
        fair.update({(r.market_key, k): v for k, v in r.fair_probs().items()})
    out = { (c["rule"].antecedent, c["rule"].consequent): c["status"] for c in check_implications(s, fair)}
    assert out[(("ou_1_5", "Under 1.5"), ("1x2_2h", "Away"))] == "consistent"
    assert out[(("ou_1_5", "Under 1.5"), ("ou_2_5", "Under 2.5"))] == "skipped-suspended"
    assert any("suspended" in x["reason"] for x in skipped)


# --- Decision policy ---


def test_policy_passes_low_confidence_and_suspended(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", PolicyConfig())
    est = Estimate(0.96, "low", "m", abstain_reason="low confidence: test")
    rec = log.decide("m", "ou_1_5", "Under 1.5", 1.23, "open", est)
    assert rec.decision == "pass" and "confidence" in rec.reason
    rec2 = log.decide("m", "1x2", "Home", 59.0, "suspended", est)
    assert rec2.decision == "pass" and "not open" in rec2.reason
    lines = (tmp_path / "d.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2 and json.loads(lines[0])["decision"] == "pass"


def test_policy_blocks_duplicates_and_caps_exposure(tmp_path):
    policy = PolicyConfig(allow_execution=True, min_ev=0.01,
                          max_bets_per_match=1, max_exposure_per_match=500.0)
    log = DecisionLog(tmp_path / "d.jsonl", policy)
    est = Estimate(0.60, "high", "m")
    r1 = log.decide("m", "ou_1_5", "Under 1.5", 2.00, "open", est, stake=400.0)
    assert r1.decision == "recommend"
    assert log.decide("m", "ou_1_5", "Under 1.5", 2.00, "open", est).decision == "pass"
    assert "duplicate" in log.decide("m", "ou_1_5", "Under 1.5", 2.00, "open", est).reason
    r3 = log.decide("m", "1x2_2h", "Away", 1.05, "open", Estimate(0.99, "high", "m"), stake=200.0)
    assert r3.decision == "pass"  # max 1 bet per match


# --- Mapper line/period + suspended reason ---


def test_mapper_line_period_and_suspended_reason():
    fx = load_fixture()
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    recs, skipped = map_sport_markets(fx["markets"], source_url="", collected_at=observed_at)
    by_key = {r.market_key: r for r in recs}
    assert by_key["ou_1_5"].line == 1.5 and by_key["ou_1_5"].period == "full"
    assert by_key["1x2_2h"].period == "2h"
    assert any("suspended" in x["reason"] for x in skipped)


# --- WebBridge adapter (mocked transport) ---


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_webbridge_snapshot_passthrough():
    tree = [{"role": "StaticText", "name": "x"}]
    client = _mock_client(lambda req: httpx.Response(
        200, json={"ok": True, "data": {"url": "u", "tree": tree}}))
    out = WebBridgeAdapter(session="t", client=client).snapshot()
    assert out["tree"] == tree


def test_webbridge_rejects_undocumented_action():
    adapter = WebBridgeAdapter(session="t", client=_mock_client(lambda req: httpx.Response(200, json={})))
    with pytest.raises(WebBridgeError, match="undocumented"):
        adapter._post("place_bet", {})


def test_webbridge_raises_on_daemon_error():
    client = _mock_client(lambda req: httpx.Response(
        200, json={"ok": False, "error": {"code": "extension_error", "message": "nope"}}))
    with pytest.raises(WebBridgeError, match="extension_error"):
        WebBridgeAdapter(session="t", client=client).snapshot()



def test_confidence_uses_probability_scale_aware_interval():
    fx = load_fixture()
    observed_at = datetime.fromisoformat(fx["meta"]["observed_at"])
    state = make_state(fx, observed_at)
    estimate = LiveStateSpaceModel(now=observed_at).estimate(state, "1x2_2h", "Home")
    assert estimate.p is not None and estimate.p < 0.001
    assert estimate.confidence == "low"
    assert estimate.diagnostics["logit_uncertainty_width"] > 2.0


def test_observe_only_decisions_apply_shadow_correlation_and_exposure_caps(tmp_path):
    policy = PolicyConfig(
        min_ev=0.01,
        max_bets_per_match=2,
        max_correlated_in_group=1,
        max_exposure_per_match=250.0,
        allow_execution=False,
    )
    log = DecisionLog(tmp_path / "shadow.jsonl", policy)
    first = log.decide(
        "m", "1x2", "Home", 2.0, "open", Estimate(0.80, "high", "model")
    )
    assert first.decision == "pass"
    assert "shadow risk limits recorded" in first.reason
    assert first.shadow_stake == pytest.approx(250.0)
    assert log._shadow_exposure["m"] == pytest.approx(250.0)

    correlated = log.decide(
        "m", "double_chance", "Home or Draw", 1.20, "open",
        Estimate(0.95, "high", "model"), stake=100.0,
    )
    assert correlated.decision == "pass"
    assert "correlated" in correlated.reason
    exposure_limited = log.decide(
        "m", "home_corners_5_5", "Over 5.5", 2.0,
        "open", Estimate(0.80, "high", "model"), stake=100.0,
    )
    assert exposure_limited.decision == "pass"
    assert "exposure cap" in exposure_limited.reason
    assert exposure_limited.shadow_stake == pytest.approx(100.0)
    assert log._taken == set() and log._exposure == {}
    assert len(log._shadow_taken) == 1
    assert log._shadow_exposure["m"] == pytest.approx(250.0)


def test_opportunity_ranks_quarter_lines_with_detailed_ev():
    quarter = Opportunity(
        match_id="m", market_key="ou_2_25", selection="Over 2.25",
        odds=2.0, model_p=0.35, p_lower=0.30,
        p_half_win=0.40, p_half_loss=0.10,
    )
    # Simple p*odds-1 is -0.30; exact split-settlement EV is +0.35.
    assert quarter.ev == pytest.approx(0.35)
    accepted, rejected = rank_and_filter([quarter], LoopConfig(min_ev=0.20))
    assert accepted == [quarter]
    assert rejected == []

    conservative = Opportunity(
        match_id="m", market_key="ah_-0.25", selection="Home (-0.25)",
        odds=2.0, model_p=0.50, p_lower=0.40,
        p_half_win=0.10, p_half_loss=0.10,
        settlement_scenarios=[
            (0.40, 0.00, 0.30, 0.00),
            (0.50, 0.00, 0.00, 0.30),
        ],
    )
    assert conservative.ev == pytest.approx(0.20)
    assert conservative.ev_lower == pytest.approx(0.15)
    accepted, rejected = rank_and_filter(
        [conservative], LoopConfig(min_ev=0.10, min_ev_lower=0.18)
    )
    assert accepted == []
    assert "EV_lower" in rejected[0]["reason"]


def test_policy_rejects_manual_stake_below_minimum(tmp_path):
    log = DecisionLog(
        tmp_path / "minimum.jsonl",
        PolicyConfig(allow_execution=True, min_ev=0.01, min_stake_ngn=100.0),
    )
    rec = log.decide(
        "m", "1x2", "Home", 2.0, "open", Estimate(0.8, "high", "model"), stake=50.0
    )
    assert rec.decision == "pass"
    assert "below SportyBet NG min" in rec.reason
