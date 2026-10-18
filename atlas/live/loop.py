"""Live Match Intelligence Loop — statistical core (no browser calls here).

Layering (per mission: explainable, traceable, reproducible):
  1. Math engine (this file + math_engine.py): breakeven, EV, de-vig (Proportional,
     Power, Shin 1993), factor-based correlation guard, and fractional Kelly.
     Pure functions — no LLM, no network, fully testable.
  2. Perception (browser/WebBridge): supplies live stats + odds snapshots.
  3. Judgement / Control Plane (Slow Plane Boss): monitors portfolio, tightens
     risk ratchets, and diagnoses live operation anomalies. NEVER invents probabilities.
  4. Execution controller: acts only when EV > threshold AND exposure/correlation caps
     pass; records every decision for later calibration.

Single-match rule: all functions take one match_id. Multi-match scanning
is out of scope for this loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from atlas.live.math_engine import (
    devig_power,
    devig_proportional,
    devig_shin,
    expected_value_detailed,
    fractional_kelly_stake,
)

# Markets whose outcomes move together. Individual EVs in the same group
# must NOT be treated as independent opportunities.
CORRELATED_GROUPS: dict[str, set[str]] = {
    "home_goals": {
        "1x2_home",
        "ou_2_5_over",
        "btts_yes",
        "home_over_1_5",
        "home_2nd_half_over",
        "home_win_to_nil",
    },
    "total_goals": {"ou_2_5_over", "ou_1_5_over", "goal_range_2_3", "btts_yes"},
    "home_corners": {"home_corners_5_5_over", "home_corners_6_5_over"},
}


def selection_correlation_groups(market_key: str, selection: str = "") -> list[str]:
    """Map both legacy shorthand keys and canonical (market_key, selection) pairs to risk factors."""
    groups: set[str] = {
        g for g, keys in CORRELATED_GROUPS.items() if market_key in keys
    }
    mkey = market_key.strip().lower()
    sel = selection.strip().lower()

    # Canonical SportyBet factor mapping
    if mkey in ("1x2", "1x2_1h", "1x2_2h", "rest_of_match", "rest_of_match_1x2", "dnb", "draw_no_bet"):
        if sel in ("home", "1"):
            groups.add("home_goals")
            groups.add("home_supremacy")
        elif sel in ("away", "2"):
            groups.add("away_goals")
            groups.add("away_supremacy")
        elif sel in ("draw", "x"):
            groups.add("low_tempo_under")

    elif mkey.startswith(("ou_", "home_ou_", "away_ou_")):
        if sel.startswith("over"):
            groups.add("total_goals")
            if mkey.startswith("home_ou_"):
                groups.add("home_goals")
            elif mkey.startswith("away_ou_"):
                groups.add("away_goals")
        elif sel.startswith("under"):
            groups.add("low_tempo_under")

    elif mkey in ("btts", "gg_ng", "1h_btts", "2h_btts"):
        if sel in ("yes", "gg"):
            groups.add("total_goals")
            groups.add("home_goals")
            groups.add("away_goals")
        elif sel in ("no", "ng"):
            groups.add("low_tempo_under")

    elif mkey == "next_goal":
        if sel in ("home", "1"):
            groups.add("home_goals")
            groups.add("total_goals")
        elif sel in ("away", "2"):
            groups.add("away_goals")
            groups.add("total_goals")
        elif sel in ("none", "no goal"):
            groups.add("low_tempo_under")

    elif mkey.startswith(("ah_", "asian_handicap_", "handicap_")):
        if sel.startswith("home"):
            groups.add("home_supremacy")
            groups.add("home_goals")
        elif sel.startswith("away"):
            groups.add("away_supremacy")
            groups.add("away_goals")

    elif "corner" in mkey:
        if "home" in mkey:
            groups.add("home_corners")
        elif "away" in mkey:
            groups.add("away_corners")
        else:
            groups.add("total_corners")

    return sorted(groups)


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


def devig(
    selections: dict[str, float], method: str = "proportional"
) -> dict[str, float]:
    """De-vig decimal-odds dict {name: odds} -> {name: fair prob}.

    Supports 'proportional' (default), 'shin' (Shin 1993 insider model), and 'power'.
    """
    if method == "shin":
        probs, _ = devig_shin(selections)
        return probs
    if method == "power":
        return devig_power(selections)
    return devig_proportional(selections)


@dataclass
class Opportunity:
    match_id: str
    market_key: str
    selection: str
    odds: float
    model_p: float
    model_name: str = "unspecified"
    p_lower: Optional[float] = None
    p_push: float = 0.0
    ev: float = field(init=False)
    ev_lower: float = field(init=False)
    breakeven_p: float = field(init=False)
    kelly_fraction: float = field(init=False)

    def __post_init__(self) -> None:
        self.breakeven_p = breakeven(self.odds)
        if self.p_push > 0.0:
            self.ev = expected_value_detailed(
                self.odds, p_win=self.model_p, p_push=self.p_push
            )
        else:
            self.ev = expected_value(self.model_p, self.odds)
        p_low = self.p_lower if self.p_lower is not None else self.model_p
        if self.p_push > 0.0:
            self.ev_lower = expected_value_detailed(
                self.odds, p_win=p_low, p_push=self.p_push
            )
        else:
            self.ev_lower = expected_value(p_low, self.odds)
        self.kelly_fraction = fractional_kelly_stake(
            p_low, self.odds, p_push=self.p_push, fraction=0.25
        )


@dataclass
class LoopConfig:
    min_ev: float = 0.05
    min_ev_lower: Optional[float] = None
    min_model_p: float = 0.0
    max_bets_per_match: int = 3
    max_correlated_in_group: int = 1


def rank_and_filter(
    opps: list[Opportunity], config: LoopConfig | None = None
) -> tuple[list[Opportunity], list[dict]]:
    """Keep only EV >= min_ev (and ev_lower >= min_ev_lower when set), rank by EV desc, enforce correlation caps.

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
        elif cfg.min_ev_lower is not None and o.ev_lower < cfg.min_ev_lower:
            rejected.append(
                {
                    "opp": o,
                    "reason": f"uncertainty-penalized EV_lower {o.ev_lower:.3f} < min {cfg.min_ev_lower}",
                }
            )
        else:
            candidates.append(o)
    candidates.sort(key=lambda o: o.ev, reverse=True)

    accepted: list[Opportunity] = []
    group_hits: dict[str, int] = {}
    for o in candidates:
        if len(accepted) >= cfg.max_bets_per_match:
            rejected.append({"opp": o, "reason": "max bets per match reached"})
            continue
        groups = selection_correlation_groups(o.market_key, o.selection)
        blocked = next(
            (g for g in groups if group_hits.get(g, 0) >= cfg.max_correlated_in_group),
            None,
        )
        if blocked:
            rejected.append(
                {"opp": o, "reason": f"correlated with accepted bet in '{blocked}'"}
            )
            continue
        accepted.append(o)
        for g in groups:
            group_hits[g] = group_hits.get(g, 0) + 1
    return accepted, rejected
