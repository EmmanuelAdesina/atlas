"""Live Match Intelligence Loop — statistical core (no browser calls here).

Layering (per mission: explainable, traceable, reproducible):
  1. Math engine (this file): breakeven, EV, de-vig, correlation guard.
     Pure functions — no LLM, no network, fully testable.
  2. Perception (browser/WebBridge): supplies live stats + odds snapshots.
  3. Judgement (LLM): interprets match state, picks which markets deserve
     a model estimate, rejects weak evidence. It NEVER invents probabilities.
  4. Execution controller: acts only when EV > threshold AND exposure caps
     pass; records every decision for later calibration.

Single-match rule: all functions take one match_id. Multi-match scanning
is out of scope for this loop.
"""

from dataclasses import dataclass, field
from typing import Optional

# Markets whose outcomes move together. Individual EVs in the same group
# must NOT be treated as independent opportunities.
CORRELATED_GROUPS: dict[str, set[str]] = {
    "home_goals": {
        "1x2_home", "ou_2_5_over", "btts_yes", "home_over_1_5",
        "home_2nd_half_over", "home_win_to_nil",
    },
    "total_goals": {"ou_2_5_over", "ou_1_5_over", "goal_range_2_3", "btts_yes"},
    "home_corners": {"home_corners_5_5_over", "home_corners_6_5_over"},
}


def breakeven(odds: float) -> float:
    """Break-even probability for decimal odds: p = 1 / o."""
    if odds < 1.0:
        raise ValueError(f"odds must be >= 1.0, got {odds}")
    return 1.0 / odds


def expected_value(model_p: float, odds: float) -> float:
    """Expected return per unit staked: EV = p * o - 1."""
    if not 0.0 <= model_p <= 1.0:
        raise ValueError(f"model_p must be in [0, 1], got {model_p}")
    return model_p * odds - 1.0


def devig(selections: dict[str, float]) -> dict[str, float]:
    """De-vig decimal-odds dict {name: odds} -> {name: fair prob}."""
    implied = {k: breakeven(o) for k, o in selections.items()}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


@dataclass
class Opportunity:
    match_id: str
    market_key: str
    selection: str
    odds: float
    model_p: float
    model_name: str = "unspecified"
    ev: float = field(init=False)
    breakeven_p: float = field(init=False)

    def __post_init__(self) -> None:
        self.breakeven_p = breakeven(self.odds)
        self.ev = expected_value(self.model_p, self.odds)


@dataclass
class LoopConfig:
    min_ev: float = 0.05
    min_model_p: float = 0.0
    max_bets_per_match: int = 3
    max_correlated_in_group: int = 1


def rank_and_filter(
    opps: list[Opportunity], config: LoopConfig | None = None
) -> tuple[list[Opportunity], list[dict]]:
    """Keep only EV >= min_ev, rank by EV desc, enforce correlation caps.

    Returns (accepted, rejected) where rejected entries carry reasons,
    so every pass/reject is auditable.
    """
    cfg = config or LoopConfig()
    rejected: list[dict] = []
    candidates: list[Opportunity] = []
    for o in opps:
        if o.model_p < cfg.min_model_p:
            rejected.append({"opp": o, "reason": "model_p below floor / weak evidence"})
        elif o.ev < cfg.min_ev:
            rejected.append({"opp": o, "reason": f"EV {o.ev:.3f} < min {cfg.min_ev}"})
        else:
            candidates.append(o)
    candidates.sort(key=lambda o: o.ev, reverse=True)

    accepted: list[Opportunity] = []
    group_hits: dict[str, int] = {}
    for o in candidates:
        if len(accepted) >= cfg.max_bets_per_match:
            rejected.append({"opp": o, "reason": "max bets per match reached"})
            continue
        groups = [g for g, keys in CORRELATED_GROUPS.items() if o.market_key in keys]
        blocked = next(
            (g for g in groups if group_hits.get(g, 0) >= cfg.max_correlated_in_group), None
        )
        if blocked:
            rejected.append({"opp": o, "reason": f"correlated with accepted bet in '{blocked}'"})
            continue
        accepted.append(o)
        for g in groups:
            group_hits[g] = group_hits.get(g, 0) + 1
    return accepted, rejected
