"""Operational posture: what the Boss is allowed to want.

The user steers ATLAS in natural language ("more revenue and low risk", "go
high risk", "only superedges"). Something has to turn that sentence into
numbers the portfolio obeys.

The split is deliberate:

  - The LLM (slow plane) READS the instruction and PROPOSES a posture. It
    never sees a probability and never sets a stake.
  - This module VALIDATES and CLAMPS the proposal against hard invariants.
    A proposal that asks for something outside the bounds is clamped, not
    obeyed, and the clamp is recorded.
  - The portfolio then executes the posture deterministically.

So the AI interprets; the quant layer decides. An LLM that misreads a
sentence can move the posture within bounds, and can never price a bet,
invent an edge, or raise a limit above its ceiling.

Two proposers ship:
  - `RuleBasedProposer` (default): deterministic keyword parsing. Works with
    no API key and no network, so the system is never blocked on a model.
  - `LLMProposer`: an OpenAI-compatible chat client, configured through
    ATLAS_BOSS_API_KEY / ATLAS_BOSS_BASE_URL / ATLAS_BOSS_MODEL. The model
    output is parsed as JSON and then goes through exactly the same
    validation as the rule-based one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional


class PostureMode(str, Enum):
    """How much risk the operator is willing to carry."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    SUPEREDGE_ONLY = "superedge_only"


# Hard invariants. A posture may move freely inside these and nowhere else.
HARD_BOUNDS = {
    "min_ev": (0.01, 0.25),
    "kelly_fraction": (0.01, 0.25),
    "max_exposure_per_match": (500.0, 20_000.0),
    "max_bets_per_snapshot": (1, 10),
    "max_stake_per_bet": (100.0, 10_000.0),
    # How much BETTER than the model value a cashout offer must be before it
    # is taken. Negative means a risk-seeking posture accepts a discounted
    # floor; positive means it demands a premium to sell.
    "cashout_min_discount": (-0.50, 1.00),
}

# The default posture: what ATLAS runs with when nobody has said otherwise.
DEFAULT_POSTURE = "balanced"

_MODE_PRESETS: dict[PostureMode, dict[str, float]] = {
    PostureMode.CONSERVATIVE: {
        "min_ev": 0.08,
        "kelly_fraction": 0.05,
        "max_exposure_per_match": 1_500.0,
        "max_bets_per_snapshot": 1,
        "max_stake_per_bet": 500.0,
        "cashout_min_discount": 0.05,
    },
    PostureMode.BALANCED: {
        "min_ev": 0.05,
        "kelly_fraction": 0.10,
        "max_exposure_per_match": 3_000.0,
        "max_bets_per_snapshot": 2,
        "max_stake_per_bet": 1_000.0,
        "cashout_min_discount": 0.0,
    },
    PostureMode.AGGRESSIVE: {
        "min_ev": 0.02,
        "kelly_fraction": 0.18,
        "max_exposure_per_match": 8_000.0,
        "max_bets_per_snapshot": 4,
        "max_stake_per_bet": 3_000.0,
        "cashout_min_discount": -0.10,
    },
    PostureMode.SUPEREDGE_ONLY: {
        "min_ev": 0.02,
        "kelly_fraction": 0.10,
        "max_exposure_per_match": 2_000.0,
        "max_bets_per_snapshot": 1,
        "max_stake_per_bet": 2_000.0,
        "cashout_min_discount": 0.0,
    },
}


@dataclass(frozen=True)
class Posture:
    """The validated operating posture the portfolio executes."""

    mode: PostureMode = PostureMode.BALANCED
    min_ev: float = 0.05
    kelly_fraction: float = 0.10
    max_exposure_per_match: float = 3_000.0
    max_bets_per_snapshot: int = 2
    max_stake_per_bet: float = 1_000.0
    cashout_min_discount: float = 0.0
    require_guaranteed_profit: bool = False
    rationale: str = "default posture"
    source: str = "default"
    clamps: tuple[str, ...] = ()

    def clamped(self) -> "Posture":
        """Return a copy clamped to the hard bounds, recording what changed."""
        notes: list[str] = []
        values = {}
        for field_name, (low, high) in HARD_BOUNDS.items():
            raw = getattr(self, field_name)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                notes.append(f"{field_name} invalid -> {low}")
                values[field_name] = low
                continue
            if raw < low:
                notes.append(f"{field_name} {raw} below floor {low} -> {low}")
                values[field_name] = low
            elif raw > high:
                notes.append(f"{field_name} {raw} above ceiling {high} -> {high}")
                values[field_name] = high
            else:
                values[field_name] = raw
        if not notes:
            return self
        return replace(self, clamps=tuple(self.clamps) + tuple(notes), **values)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "min_ev": self.min_ev,
            "kelly_fraction": self.kelly_fraction,
            "max_exposure_per_match": self.max_exposure_per_match,
            "max_bets_per_snapshot": self.max_bets_per_snapshot,
            "max_stake_per_bet": self.max_stake_per_bet,
            "cashout_min_discount": self.cashout_min_discount,
            "require_guaranteed_profit": self.require_guaranteed_profit,
            "rationale": self.rationale,
            "source": self.source,
            "clamps": list(self.clamps),
        }

    def describe(self) -> str:
        return (
            f"{self.mode.value}: min_ev {self.min_ev:.3f}, kelly "
            f"{self.kelly_fraction:.2f}, exposure cap NGN "
            f"{self.max_exposure_per_match:.0f}, "
            f"{self.max_bets_per_snapshot} bet(s)/snapshot, "
            f"cashout needs {self.cashout_min_discount:+.0%} over model"
            + (
                ", superedge-only"
                if self.require_guaranteed_profit
                else ""
            )
            + (f" | clamped: {'; '.join(self.clamps)}" if self.clamps else "")
        )

    def apply_to(self, config: Any) -> Any:
        """Return a portfolio config that obeys this posture.

        The posture only sets the levers the quant layer already understands:
        the EV floor, the Kelly fraction, the exposure cap, the per-snapshot
        bet count and the per-bet stake cap. Everything else in the config
        (correlation caps, rebalance thresholds, cooldowns) is left alone,
        because those protect the operator regardless of appetite.
        """
        from atlas.portfolio.sequential import SequentialConfig

        return SequentialConfig(
            kelly_fraction=self.kelly_fraction,
            min_stake_ngn=config.min_stake_ngn,
            max_stake_per_bet=self.max_stake_per_bet,
            max_exposure_per_match=self.max_exposure_per_match,
            max_bets_per_match=config.max_bets_per_match,
            max_bets_per_snapshot=self.max_bets_per_snapshot,
            max_bets_per_minute=config.max_bets_per_minute,
            max_correlated_in_group=config.max_correlated_in_group,
            ev_threshold_high_correlation=config.ev_threshold_high_correlation,
            rebalance_ev_delta=config.rebalance_ev_delta,
            rebalance_odds_delta=config.rebalance_odds_delta,
            min_minutes_between_same_selection=config.min_minutes_between_same_selection,
        )


def posture_for_mode(mode: str | PostureMode) -> Posture:
    """The preset posture for a named mode."""
    try:
        m = mode if isinstance(mode, PostureMode) else PostureMode(str(mode).lower())
    except ValueError:
        m = PostureMode.BALANCED
    preset = dict(_MODE_PRESETS[m])
    return Posture(mode=m, rationale=f"preset:{m.value}", source="preset", **preset)


# ---------------------------------------------------------------------------
# Rule-based proposer (no API key required)
# ---------------------------------------------------------------------------

_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"super\s*edge|superedge|guaranteed|sure[- ]?thing|arb", re.I),
     PostureMode.SUPEREDGE_ONLY.value),
    (re.compile(r"conservativ|low risk|safe|protect|preserv", re.I),
     PostureMode.CONSERVATIVE.value),
    (re.compile(r"aggressiv|high risk|maxim|volume|more revenue|yolo", re.I),
     PostureMode.AGGRESSIVE.value),
    (re.compile(r"balanced|normal|default|standard", re.I),
     PostureMode.BALANCED.value),
)


def parse_instruction_rule_based(instruction: str) -> Posture:
    """Deterministically turn an instruction into a posture.

    This is the fallback that keeps ATLAS usable with no API key: the same
    instruction always yields the same posture, and it is always clamped.
    """
    text = (instruction or "").strip()
    mode = DEFAULT_POSTURE
    for pattern, candidate in _RULES:
        if pattern.search(text):
            mode = candidate
            break
    posture = posture_for_mode(mode)
    if not text:
        posture = replace(
            posture, rationale="empty instruction -> default posture", source="rules"
        )
    else:
        posture = replace(
            posture, rationale=f"instruction matched mode '{mode}'", source="rules"
        )
    # An explicit superedge instruction also demands guaranteed profit.
    if mode == PostureMode.SUPEREDGE_ONLY.value:
        posture = replace(posture, require_guaranteed_profit=True)
    return posture.clamped()


# ---------------------------------------------------------------------------
# LLM proposer (OpenAI-compatible)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You translate an operator's instruction about a live betting system into "
    "a JSON operating posture. Reply with JSON only, no prose.\n"
    "Fields: mode (conservative|balanced|aggressive|superedge_only), "
    "min_ev (0.01-0.25), kelly_fraction (0.01-0.25), "
    "max_exposure_per_match (500-20000), max_bets_per_snapshot (1-10), "
    "max_stake_per_bet (100-10000), require_guaranteed_profit (bool), "
    "rationale (one short sentence).\n"
    "Higher revenue means a lower min_ev and a larger exposure cap. "
    "Lower risk means the opposite. Never ask for a value outside the ranges."
)


class LLMProposer:
    """Proposes a posture with an OpenAI-compatible chat completion API.

    The model output is parsed as JSON and then clamped by `Posture.clamped`,
    so a misbehaving model cannot push the system past its hard bounds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ATLAS_BOSS_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("ATLAS_BOSS_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.environ.get("ATLAS_BOSS_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def propose(self, instruction: str, context: Optional[dict] = None) -> Posture:
        if not self.configured:
            raise RuntimeError(
                "no API key: set ATLAS_BOSS_API_KEY (or use RuleBasedProposer)"
            )
        import httpx

        user = instruction.strip()
        if context:
            user += "\n\nCurrent state:\n" + json.dumps(context, default=str)[:2000]
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return self.parse_posture(content)

    @staticmethod
    def parse_posture(content: str) -> Posture:
        """Parse a model reply into a clamped posture.

        Anything unparseable falls back to the deterministic rule-based
        parse, so a bad model reply degrades to a safe posture instead of an
        error.
        """
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return parse_instruction_rule_based(content)
        if not isinstance(data, dict):
            return parse_instruction_rule_based(content)
        mode_raw = str(data.get("mode", DEFAULT_POSTURE)).lower()
        try:
            mode = PostureMode(mode_raw)
        except ValueError:
            mode = PostureMode.BALANCED
        posture = Posture(
            mode=mode,
            min_ev=float(data.get("min_ev", 0.05) or 0.05),
            kelly_fraction=float(data.get("kelly_fraction", 0.10) or 0.10),
            max_exposure_per_match=float(
                data.get("max_exposure_per_match", 3000.0) or 3000.0
            ),
            max_bets_per_snapshot=int(data.get("max_bets_per_snapshot", 2) or 2),
            max_stake_per_bet=float(data.get("max_stake_per_bet", 1000.0) or 1000.0),
            cashout_min_discount=float(
                data.get("cashout_min_discount", 0.0) or 0.0
            ),
            require_guaranteed_profit=bool(data.get("require_guaranteed_profit", False)),
            rationale=str(data.get("rationale", "proposed by model"))[:400],
            source="llm",
        )
        return posture.clamped()


def propose_posture(
    instruction: str,
    context: Optional[dict] = None,
    use_llm: bool = False,
) -> Posture:
    """Propose a posture, preferring the LLM when it is configured."""
    if use_llm:
        proposer = LLMProposer()
        if proposer.configured:
            try:
                return proposer.propose(instruction, context)
            except Exception:
                # A model failure must not stop the operator's command.
                fallback = parse_instruction_rule_based(instruction)
                return replace(
                    fallback,
                    rationale="LLM unavailable -> rule-based fallback: "
                    + fallback.rationale,
                    source="rules-fallback",
                )
    return parse_instruction_rule_based(instruction)
