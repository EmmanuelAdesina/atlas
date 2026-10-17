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
import re
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
    """Map legacy and canonical market/selection pairs to shared risk factors."""
    mkey = market_key.strip().lower()
    sel = selection.strip().lower()
    compact_sel = re.sub(r"[^a-z0-9]+", "", sel)
    groups: set[str] = {
        group
        for group, keys in CORRELATED_GROUPS.items()
        if mkey in {key.lower() for key in keys}
    }

    dnb_keys = {"dnb", "draw_no_bet", "1h_dnb", "2h_dnb", "dnb_1h", "dnb_2h"}
    one_x2_keys = {
        "1x2", "1x2_1h", "1x2_2h", "rest_of_match", "rest_of_match_1x2",
        *dnb_keys,
    }
    if mkey in one_x2_keys:
        if compact_sel in {"home", "1"}:
            groups.update(("home_goals", "home_supremacy"))
            if mkey in dnb_keys:
                groups.add("low_tempo_under")
        elif compact_sel in {"away", "2"}:
            groups.update(("away_goals", "away_supremacy"))
            if mkey in dnb_keys:
                groups.add("low_tempo_under")
        elif compact_sel in {"draw", "x"}:
            groups.add("low_tempo_under")

    elif mkey in {
        "double_chance", "dc", "doublechance", "1h_double_chance",
        "2h_double_chance", "double_chance_1h", "double_chance_2h",
        "1h_dc", "2h_dc",
    }:
        if compact_sel in {"homeordraw", "draworhome", "1x"}:
            groups.update(("home_goals", "home_supremacy", "low_tempo_under"))
        elif compact_sel in {"draworaway", "awayordraw", "x2"}:
            groups.update(("away_goals", "away_supremacy", "low_tempo_under"))
        elif compact_sel in {"homeoraway", "awayorhome", "12"}:
            groups.update(("home_supremacy", "away_supremacy"))

    elif mkey.startswith(("ou_", "home_ou_", "away_ou_", "1h_ou_", "2h_ou_")):
        if sel.startswith("over"):
            groups.add("total_goals")
            if mkey.startswith("home_ou_"):
                groups.add("home_goals")
            elif mkey.startswith("away_ou_"):
                groups.add("away_goals")
        elif sel.startswith("under"):
            groups.add("low_tempo_under")

    elif mkey in ("btts", "gg_ng", "1h_btts", "2h_btts", "btts_1h", "btts_2h"):
        if compact_sel in {"yes", "gg"}:
            groups.update(("total_goals", "home_goals", "away_goals"))
        elif compact_sel in {"no", "ng"}:
            groups.add("low_tempo_under")

    elif mkey == "next_goal":
        if compact_sel in {"home", "1"}:
            groups.update(("home_goals", "total_goals"))
        elif compact_sel in {"away", "2"}:
            groups.update(("away_goals", "total_goals"))
        elif compact_sel in {"none", "nogoal"}:
            groups.add("low_tempo_under")

    elif mkey.startswith(("ah_", "asian_handicap_", "handicap_")):
        if sel.startswith("home"):
            groups.update(("home_supremacy", "home_goals"))
        elif sel.startswith("away"):
            groups.update(("away_supremacy", "away_goals"))

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
    p_half_win: float = 0.0
    p_half_loss: float = 0.0
    settlement_scenarios: list[tuple[float, float, float, float]] = field(
        default_factory=list, repr=False
    )
    ev: float = field(init=False)
    ev_lower: float = field(init=False)
    breakeven_p: float = field(init=False)
    kelly_fraction: float = field(init=False)

    def __post_init__(self) -> None:
        """`model_p` and `p_lower` are full-win probabilities."""
        self.breakeven_p = breakeven(self.odds)
        self.ev = expected_value_detailed(
            self.odds,
            p_win=self.model_p,
            p_push=self.p_push,
            p_half_win=self.p_half_win,
            p_half_loss=self.p_half_loss,
        )
        p_low = self.p_lower if self.p_lower is not None else self.model_p
        scenarios = self.settlement_scenarios or [
            (p_low, self.p_push, self.p_half_win, self.p_half_loss)
        ]
        conservative = min(
            scenarios,
            key=lambda item: expected_value_detailed(
                self.odds,
                p_win=item[0],
                p_push=item[1],
                p_half_win=item[2],
                p_half_loss=item[3],
            ),
        )
        self.ev_lower = expected_value_detailed(
            self.odds,
            p_win=conservative[0],
            p_push=conservative[1],
            p_half_win=conservative[2],
            p_half_loss=conservative[3],
        )
        self.kelly_fraction = fractional_kelly_stake(
            conservative[0],
            self.odds,
            p_push=conservative[1],
            fraction=0.25,
            p_half_win=conservative[2],
            p_half_loss=conservative[3],
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
