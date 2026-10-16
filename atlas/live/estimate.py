"""Probability interface + narrow baseline model family.

The AI never invents probabilities. A MarketModel receives the current
LiveMatchState plus the observed market and returns an Estimate carrying
the probability, a confidence grade, evidence freshness, and — crucially —
an abstain reason whenever the evidence cannot support a number.

Baseline scope (deliberately narrow): late-match 'no more goals' for
Under selections. Constant-rate Poisson with an explicitly labeled assumed
rate, confidence capped at 'low' whenever the rate is assumed rather than
match-measured or remaining time is uncertain. A 'low' confidence can
never clear the betting policy in decisions.py — the baseline informs and
calibrates, it does not trigger stakes by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from atlas.live.observe import LiveMatchState


def _second_half_score(state: LiveMatchState) -> Optional[tuple[int, int]]:
    if (state.home.goals is None or state.away.goals is None
            or state.half_home_goals is None or state.half_away_goals is None):
        return None
    return (state.home.goals - state.half_home_goals,
            state.away.goals - state.half_away_goals)


@dataclass
class ImplicationRule:
    """If antecedent occurs then consequent must occur (given `when`)."""
    antecedent: tuple[str, str]
    consequent: tuple[str, str]
    when: str  # human-readable validity condition


def _rule_conditions() -> list[tuple[ImplicationRule, Any]]:
    return [
        (ImplicationRule(("ou_1_5", "Under 1.5"), ("1x2_2h", "Away"),
                         "2h with away leading the half: no more goals keeps 2H 0:1"),
         lambda s: s.period == "2h" and (_second_half_score(s) or (0, 0))[1] > (_second_half_score(s) or (0, 0))[0]),
        (ImplicationRule(("ou_1_5", "Under 1.5"), ("ou_2_5", "Under 2.5"),
                         "total <= 1 implies total <= 2 (no state needed)"),
         lambda s: True),
    ]


def check_implications(state: LiveMatchState,
                       fair: dict[tuple[str, str], float],
                       tol: float = 0.02) -> list[dict]:
    """Verify fair P(consequent) >= fair P(antecedent) for each valid rule.

    `fair` maps (market_key, selection) -> de-vigged probability. Markets
    that are suspended or missing are skipped, never filled in.
    """
    results: list[dict] = []
    for rule, cond in _rule_conditions():
        try:
            valid = bool(cond(state))
        except Exception:
            valid = False
        if not valid:
            results.append({"rule": rule, "status": "not-applicable"})
            continue
        if rule.antecedent not in fair or rule.consequent not in fair:
            results.append({"rule": rule, "status": "skipped-suspended"})
            continue
        pa, pc = fair[rule.antecedent], fair[rule.consequent]
        results.append({
            "rule": rule,
            "status": "consistent" if pc + tol >= pa else "VIOLATION",
            "p_antecedent": pa,
            "p_consequent": pc,
        })
    return results

Confidence = str  # 'high' | 'medium' | 'low'


@dataclass
class Estimate:
    p: Optional[float]
    confidence: Confidence
    model_name: str
    evidence_age_s: Optional[float] = None
    abstain_reason: Optional[str] = None

    @property
    def abstained(self) -> bool:
        return self.p is None


class MarketModel(Protocol):
    model_name: str

    def estimate(self, state: LiveMatchState, market_key: str,
                 selection: str) -> Estimate:
        ...


# Selections this baseline family may evaluate: {market_key: {selection}}.
# Under X.5 (full time) wins iff no more goals take the total past the line.
BASELINE_UNDER_SELECTIONS = {
    "ou_0_5": {"Under 0.5": 0},
    "ou_1_5": {"Under 1.5": 1},
    "ou_2_5": {"Under 2.5": 2},
    "ou_3_5": {"Under 3.5": 3},
    "ou_4_5": {"Under 4.5": 4},
}


@dataclass
class NoMoreGoalsBaseline:
    """P(total stays under line) = P(Poisson(rate * t) <= goals_allowed).

    Rate is an explicitly assumed league-average goals-per-minute unless a
    measured live rate is supplied. Confidence is 'low' whenever the rate
    is assumed or stoppage/remaining time is uncertain — i.e. almost always
    live. This model exists to force the abstention path and to calibrate
    against, not to find edges.
    """

    model_name: str = "no_more_goals_baseline_v1"
    assumed_rate_per_min: float = 0.030
    max_evidence_age_s: float = 60.0
    now: Optional[datetime] = None  # replay hook: evaluate freshness as-of this time

    def estimate(self, state: LiveMatchState, market_key: str,
                 selection: str) -> Estimate:
        allowed_map = BASELINE_UNDER_SELECTIONS.get(market_key)
        if allowed_map is None or selection not in allowed_map:
            return Estimate(None, "low", self.model_name,
                            abstain_reason=f"market {market_key}/{selection} outside baseline family")
        if not state.is_live:
            return Estimate(None, "low", self.model_name,
                            abstain_reason=f"match not live (period={state.period})")
        age = state.age_seconds(self.now or datetime.now(timezone.utc))
        if age > self.max_evidence_age_s:
            return Estimate(None, "low", self.model_name, evidence_age_s=age,
                            abstain_reason=f"stale evidence ({age:.0f}s > {self.max_evidence_age_s:.0f}s)")
        if state.clock_minute is None or state.score_total is None:
            return Estimate(None, "low", self.model_name, evidence_age_s=age,
                            abstain_reason="clock or score unknown")
        goals_allowed = allowed_map[selection] - state.score_total
        if goals_allowed < 0:
            return Estimate(0.0, "high", self.model_name, evidence_age_s=age)
        mu = self.assumed_rate_per_min * max(0.0, 90.0 + 5.0 - state.clock_minute)
        # Remaining-time guess (90+5') is itself uncertain -> cap confidence.
        p = sum(math.exp(-mu) * mu**k / math.factorial(k)
                for k in range(goals_allowed + 1))
        reasons = ["assumed league-average rate", "stoppage length unknown"]
        if state.home_reds or state.away_reds:
            reasons.append("red-card state shifts rate (unmodeled)")
        return Estimate(min(1.0, max(0.0, p)), "low", self.model_name,
                        evidence_age_s=age,
                        abstain_reason="low confidence: " + "; ".join(reasons))
