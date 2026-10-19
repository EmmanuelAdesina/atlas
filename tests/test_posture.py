"""Tests for the posture layer and the notifier (atlas.ops.posture / notify).

The safety property under test: an LLM may PROPOSE a posture, but the
deterministic layer clamps it, and it can never touch a probability or a
stake directly.
"""

import json

import pytest

from atlas.ops.notify import Notifier, Severity
from atlas.ops.posture import (
    HARD_BOUNDS,
    LLMProposer,
    Posture,
    PostureMode,
    parse_instruction_rule_based,
    posture_for_mode,
    propose_posture,
)
from atlas.portfolio.sequential import SequentialConfig


# --- rule-based proposer -------------------------------------------------

def test_low_risk_instruction_maps_to_conservative():
    p = parse_instruction_rule_based("I want more revenue and low risk bets")
    assert p.mode is PostureMode.CONSERVATIVE
    assert p.min_ev >= posture_for_mode(PostureMode.BALANCED).min_ev
    assert p.kelly_fraction < posture_for_mode(PostureMode.BALANCED).kelly_fraction


def test_high_risk_instruction_maps_to_aggressive():
    p = parse_instruction_rule_based("go high risk, max volume")
    assert p.mode is PostureMode.AGGRESSIVE
    assert p.min_ev < posture_for_mode(PostureMode.BALANCED).min_ev
    assert p.max_exposure_per_match > posture_for_mode(PostureMode.BALANCED).max_exposure_per_match


def test_superedge_instruction_requires_guaranteed_profit():
    p = parse_instruction_rule_based("only superedge plays from now on")
    assert p.mode is PostureMode.SUPEREDGE_ONLY
    assert p.require_guaranteed_profit is True


def test_empty_instruction_falls_back_to_default():
    p = parse_instruction_rule_based("")
    assert p.mode is PostureMode.BALANCED


def test_rule_based_proposer_is_deterministic():
    a = parse_instruction_rule_based("more revenue, low risk")
    b = parse_instruction_rule_based("more revenue, low risk")
    assert a.to_dict() == b.to_dict()


# --- clamping ------------------------------------------------------------

def test_posture_is_clamped_to_hard_bounds():
    p = Posture(
        mode=PostureMode.AGGRESSIVE,
        min_ev=0.99,
        kelly_fraction=5.0,
        max_exposure_per_match=10_000_000.0,
        max_bets_per_snapshot=500,
        max_stake_per_bet=1_000_000.0,
    ).clamped()
    for field_name, (low, high) in HARD_BOUNDS.items():
        value = getattr(p, field_name)
        assert low <= value <= high, f"{field_name} escaped its bounds: {value}"
    assert p.clamps


def test_clamp_records_what_it_changed():
    p = Posture(min_ev=0.9).clamped()
    assert any("min_ev" in c for c in p.clamps)


def test_valid_posture_is_not_clamped():
    p = posture_for_mode(PostureMode.BALANCED)
    assert p.clamped().clamps == ()


# --- LLM proposer --------------------------------------------------------

def test_llm_proposal_is_clamped():
    """A model asking for absurd numbers gets clamped, not obeyed."""
    reply = json.dumps(
        {
            "mode": "aggressive",
            "min_ev": 0.9,
            "kelly_fraction": 10.0,
            "max_exposure_per_match": 10_000_000,
            "max_bets_per_snapshot": 999,
            "max_stake_per_bet": 5_000_000,
            "require_guaranteed_profit": False,
            "rationale": "because I said so",
        }
    )
    p = LLMProposer.parse_posture(reply)
    for field_name, (low, high) in HARD_BOUNDS.items():
        value = getattr(p, field_name)
        assert low <= value <= high, f"{field_name} escaped: {value}"
    assert p.clamps
    assert p.source == "llm"


def test_llm_unparseable_reply_falls_back_to_rules():
    p = LLMProposer.parse_posture("I think you should go high risk, honestly")
    assert p.mode is PostureMode.AGGRESSIVE
    assert p.source == "rules"


def test_llm_markdown_fence_is_parsed():
    reply = "```json\n" + json.dumps({"mode": "conservative", "min_ev": 0.07}) + "\n```"
    p = LLMProposer.parse_posture(reply)
    assert p.mode is PostureMode.CONSERVATIVE


def test_llm_unknown_mode_falls_back_to_balanced():
    p = LLMProposer.parse_posture(json.dumps({"mode": "yolo"}))
    assert p.mode is PostureMode.BALANCED


def test_propose_posture_without_llm_uses_rules():
    p = propose_posture("low risk please", use_llm=False)
    assert p.mode is PostureMode.CONSERVATIVE


def test_llm_without_api_key_is_not_configured():
    assert LLMProposer(api_key="").configured is False


# --- posture -> portfolio config -----------------------------------------

def test_posture_drives_the_portfolio_config():
    p = posture_for_mode(PostureMode.CONSERVATIVE)
    cfg = p.apply_to(SequentialConfig())
    assert cfg.kelly_fraction == p.kelly_fraction
    assert cfg.max_exposure_per_match == p.max_exposure_per_match
    assert cfg.max_bets_per_snapshot == p.max_bets_per_snapshot
    assert cfg.max_stake_per_bet == p.max_stake_per_bet


def test_posture_does_not_loosen_the_safety_caps():
    """Aggressive posture must not raise the correlation cap or cooldown."""
    aggressive = posture_for_mode(PostureMode.AGGRESSIVE).apply_to(SequentialConfig())
    base = SequentialConfig()
    assert aggressive.max_correlated_in_group == base.max_correlated_in_group
    assert aggressive.min_minutes_between_same_selection == base.min_minutes_between_same_selection
    assert aggressive.rebalance_ev_delta == base.rebalance_ev_delta
    assert aggressive.max_bets_per_match == base.max_bets_per_match


# --- notifier ------------------------------------------------------------

def test_notifier_writes_to_the_audit_log(tmp_path):
    log = tmp_path / "notify.jsonl"
    n = Notifier(log_path=log, console=False)
    n.good("Sure-thing found", "1X2 covers the space", profit=0.019)
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["severity"] == "good"
    assert entry["payload"]["profit"] == 0.019


def test_notifier_survives_a_broken_sink(tmp_path):
    """A failing sink must not stop the others or raise."""
    log = tmp_path / "notify.jsonl"
    n = Notifier(log_path=log, console=False)
    n.webhook_url = "http://127.0.0.1:1/never-listening"
    n.critical("Daily stop", "breached")
    assert log.exists()
    assert len(log.read_text(encoding="utf-8").strip().split("\n")) == 1


def test_notifier_records_what_it_sent(tmp_path):
    n = Notifier(log_path=tmp_path / "n.jsonl", console=False)
    n.info("a", "b")
    n.warn("c", "d")
    assert [x.severity for x in n.sent] == [Severity.INFO, Severity.WARN]
