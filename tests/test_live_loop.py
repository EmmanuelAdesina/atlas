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
from atlas.live.estimate import Estimate, NoMoreGoalsBaseline, check_implications
from atlas.live.extract import extract_clock_score, extract_markets, iter_texts
from atlas.live.loop import expected_value
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


# --- WebBridge transient-failure policy (2026-10-29) ----------------------
# A wedged extension answers 502 until it recovers. One 502 used to abort a
# cycle and surface as "the board is empty"; reads retry bounded, money-path
# clicks never do (an ambiguous click can place a second bet).

def test_webbridge_retries_a_transient_502_and_recovers():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json={"ok": True, "data": {"tree": []}})

    adapter = WebBridgeAdapter(
        session="t", client=_mock_client(handler), retry_backoff_s=0.0
    )
    assert adapter.snapshot() == {"tree": []}
    assert calls["n"] == 3


def test_webbridge_gives_up_after_bounded_transient_retries():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    adapter = WebBridgeAdapter(
        session="t", client=_mock_client(handler), retry_backoff_s=0.0
    )
    with pytest.raises(WebBridgeError) as ei:
        adapter.snapshot()
    assert ei.value.transient is True
    assert calls["n"] == 3, "bounded: attempts=3"


def test_webbridge_never_retries_a_click():
    """A timed-out click may already have happened: retrying can double it."""
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        raise httpx.ConnectError("daemon down")

    adapter = WebBridgeAdapter(
        session="t",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_backoff_s=0.0,
    )
    with pytest.raises(WebBridgeError):
        adapter.click('[data-atlas-place-bet="1"]')
    assert calls["n"] == 1


def test_webbridge_does_not_retry_an_application_error():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(
            200, json={"ok": False, "error": {"code": "no_tab", "message": "x"}}
        )

    adapter = WebBridgeAdapter(
        session="t", client=_mock_client(handler), retry_backoff_s=0.0
    )
    with pytest.raises(WebBridgeError) as ei:
        adapter.snapshot()
    assert ei.value.transient is False
    assert calls["n"] == 1


def test_evaluate_safe_falls_back_to_cdp_when_the_extension_hangs():
    """The renderer dies to chrome-error on the heavy list; CDP still runs."""
    def handler(req):
        body = json.loads(req.content)
        if body["action"] == "evaluate":
            return httpx.Response(502, text="bad gateway")
        if body["action"] == "cdp":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "result": {
                            "result": {"type": "string", "value": "cdp-ok"}
                        }
                    },
                },
            )
        return httpx.Response(404)

    adapter = WebBridgeAdapter(
        session="t", client=_mock_client(handler), retry_backoff_s=0.0
    )
    out = adapter.evaluate_safe("location.href")
    assert out["value"] == "cdp-ok"


def test_evaluate_safe_does_not_replay_an_application_error():
    """ok:false is a deterministic answer, not a transport fault."""
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(
            200, json={"ok": False, "error": {"code": "no_tab", "message": "x"}}
        )

    adapter = WebBridgeAdapter(
        session="t", client=_mock_client(handler), retry_backoff_s=0.0
    )
    with pytest.raises(WebBridgeError):
        adapter.evaluate_safe("1")
    assert calls["n"] == 1


def test_read_actions_use_the_short_timeout_navigate_keeps_the_long_one():
    """A wedged page answers reads never; waiting 45s for each read (+45s
    for the CDP fallback) is how cycles stretched past five minutes
    (2026-10-31). Reads get 15s; money-path actions keep the default."""
    seen = {}

    def handler(request: httpx.Request):
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200, json={"ok": True, "data": {"type": "number", "value": 1}}
        )

    adapter = WebBridgeAdapter(
        session="t", client=_mock_client(handler), retry_backoff_s=0.0
    )
    adapter.evaluate("1 + 1")
    assert seen["timeout"]["read"] == 15.0, (
        "evaluate must fail fast on a wedged page"
    )
    adapter.navigate("https://example.com")
    assert seen["timeout"]["read"] == 45.0, (
        "cold navigation must keep the generous default"
    )
    adapter.click("sel")
    assert seen["timeout"]["read"] == 45.0, (
        "the money path must keep the generous default"
    )
