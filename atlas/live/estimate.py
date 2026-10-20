"""Probability interface + narrow baseline + coherent state-space goal surface model.

The AI never invents probabilities. A MarketModel receives the current
LiveMatchState plus the observed market and returns an Estimate carrying
the probability, uncertainty bounds [p_lower, p_upper], a confidence grade,
evidence freshness, and — crucially — an abstain reason whenever the evidence
cannot support a number.

Models provided:
  1. `NoMoreGoalsBaseline` — deliberately narrow Under-only constant-rate
     Poisson baseline (confidence capped at 'low' whenever rate or stoppage
     length is assumed).
  2. `LiveStateSpaceModel` — coherent time-inhomogeneous Bivariate Poisson /
     Dixon-Coles state-space surface model with piecewise clock hazard,
     Bayesian live-statistics shrinkage, Markov red-card & score-state
     multipliers, and 7-point sigma uncertainty propagation across 14+
     SportyBet market families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from atlas.live.math_engine import (
    is_path_market,
    _smooth_positive_residual,
    build_remaining_score_matrix,
    compute_matrix_marginals,
    compute_posterior_intensities,
    effective_hazard_fraction,
    estimate_remaining_minutes,
    estimate_xg_from_stats,
    expected_value_detailed,
    price_from_joint_matrix,
)
from atlas.live.observe import LiveMatchState
from atlas.live.corners import CornerModel
from atlas.live.paths import GoalPathModel


def _second_half_score(state: LiveMatchState) -> Optional[tuple[int, int]]:
    if (
        state.home.goals is None
        or state.away.goals is None
        or state.half_home_goals is None
        or state.half_away_goals is None
    ):
        return None
    return (
        state.home.goals - state.half_home_goals,
        state.away.goals - state.half_away_goals,
    )


@dataclass
class ImplicationRule:
    """If antecedent occurs then consequent must occur (given `when`)."""

    antecedent: tuple[str, str]
    consequent: tuple[str, str]
    when: str  # human-readable validity condition


def _rule_conditions() -> list[tuple[ImplicationRule, Any]]:
    return [
        (
            ImplicationRule(
                ("ou_1_5", "Under 1.5"),
                ("1x2_2h", "Away"),
                "2h with away leading the half: no more goals keeps 2H 0:1",
            ),
            lambda s: s.period == "2h"
            and (_second_half_score(s) or (0, 0))[1]
            > (_second_half_score(s) or (0, 0))[0],
        ),
        (
            ImplicationRule(
                ("ou_1_5", "Under 1.5"),
                ("ou_2_5", "Under 2.5"),
                "total <= 1 implies total <= 2 (no state needed)",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("ou_0_5", "Under 0.5"),
                ("ou_1_5", "Under 1.5"),
                "total == 0 implies total <= 1",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("ou_2_5", "Under 2.5"),
                ("ou_3_5", "Under 3.5"),
                "total <= 2 implies total <= 3",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("ou_3_5", "Under 3.5"),
                ("ou_4_5", "Under 4.5"),
                "total <= 3 implies total <= 4",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("ou_2_5", "Over 2.5"),
                ("ou_1_5", "Over 1.5"),
                "total >= 3 implies total >= 2",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("btts", "Yes"),
                ("ou_1_5", "Over 1.5"),
                "both teams scoring >= 1 implies total goals >= 2",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("ou_1_5", "Under 1.5"),
                ("1x2", "Away"),
                "score 0:1: staying under 1.5 implies Away wins FT",
            ),
            lambda s: s.home.goals == 0 and s.away.goals == 1,
        ),
        (
            ImplicationRule(
                ("ou_1_5", "Under 1.5"),
                ("next_goal", "None"),
                "score total == 1: staying under 1.5 implies no 2nd goal",
            ),
            lambda s: s.score_total == 1,
        ),
        (
            ImplicationRule(
                ("1x2", "Home"),
                ("double_chance", "Home or Draw"),
                "Home win implies Home or Draw (1X)",
            ),
            lambda s: True,
        ),
        (
            ImplicationRule(
                ("1x2", "Away"),
                ("double_chance", "Draw or Away"),
                "Away win implies Draw or Away (X2)",
            ),
            lambda s: True,
        ),
    ]


def check_implications(
    state: LiveMatchState, fair: dict[tuple[str, str], float], tol: float = 0.02
) -> list[dict]:
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
        results.append(
            {
                "rule": rule,
                "status": "consistent" if pc + tol >= pa else "VIOLATION",
                "p_antecedent": pa,
                "p_consequent": pc,
            }
        )
    return results


Confidence = str  # 'high' | 'medium' | 'low'


@dataclass
class Estimate:
    p: Optional[float]
    confidence: Confidence
    model_name: str
    evidence_age_s: Optional[float] = None
    abstain_reason: Optional[str] = None
    p_lower: Optional[float] = None
    p_upper: Optional[float] = None
    p_push: float = 0.0
    p_half_win: float = 0.0
    p_half_loss: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.p is None

    def expected_value(self, odds: float) -> Optional[float]:
        if self.p is None:
            return None
        return expected_value_detailed(
            odds,
            p_win=self.p,
            p_push=self.p_push,
            p_half_win=self.p_half_win,
            p_half_loss=self.p_half_loss,
        )

    def conservative_expected_value(self, odds: float) -> Optional[float]:
        p_low = self.p_lower if self.p_lower is not None else self.p
        if p_low is None:
            return None
        return expected_value_detailed(
            odds,
            p_win=p_low,
            p_push=self.p_push,
            p_half_win=self.p_half_win,
            p_half_loss=self.p_half_loss,
        )


class MarketModel(Protocol):
    model_name: str

    def estimate(
        self, state: LiveMatchState, market_key: str, selection: str
    ) -> Estimate: ...


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

    def estimate(
        self, state: LiveMatchState, market_key: str, selection: str
    ) -> Estimate:
        allowed_map = BASELINE_UNDER_SELECTIONS.get(market_key)
        if allowed_map is None or selection not in allowed_map:
            return Estimate(
                None,
                "low",
                self.model_name,
                abstain_reason=f"market {market_key}/{selection} outside baseline family",
            )
        if not state.is_live:
            return Estimate(
                None,
                "low",
                self.model_name,
                abstain_reason=f"match not live (period={state.period})",
            )
        age = state.age_seconds(self.now or datetime.now(timezone.utc))
        if age > self.max_evidence_age_s:
            return Estimate(
                None,
                "low",
                self.model_name,
                evidence_age_s=age,
                abstain_reason=f"stale evidence ({age:.0f}s > {self.max_evidence_age_s:.0f}s)",
            )
        if state.clock_minute is None or state.score_total is None:
            return Estimate(
                None,
                "low",
                self.model_name,
                evidence_age_s=age,
                abstain_reason="clock or score unknown",
            )
        goals_allowed = allowed_map[selection] - state.score_total
        if goals_allowed < 0:
            return Estimate(
                0.0,
                "high",
                self.model_name,
                evidence_age_s=age,
                p_lower=0.0,
                p_upper=0.0,
            )
        # Preserve exact 95.0 - clock_minute linear formula before 94.5' for
        # backward-compatible baseline tests, while preventing zero collapse
        # if clock_minute >= 95.0 via smooth positive residual tail.
        raw_rem = 90.0 + 5.0 - state.clock_minute
        rem_mins = raw_rem if raw_rem >= 0.5 else _smooth_positive_residual(raw_rem, 0.5)
        mu = self.assumed_rate_per_min * rem_mins
        p = sum(
            math.exp(-mu) * mu**k / math.factorial(k)
            for k in range(goals_allowed + 1)
        )
        # Compute conservative lower/upper bounds under stoppage uncertainty [rem - 1.5, rem + 2.0]
        mu_hi = self.assumed_rate_per_min * (rem_mins + 2.0)
        mu_lo = self.assumed_rate_per_min * max(0.2, rem_mins - 1.0)
        p_low = sum(
            math.exp(-mu_hi) * mu_hi**k / math.factorial(k)
            for k in range(goals_allowed + 1)
        )
        p_high = sum(
            math.exp(-mu_lo) * mu_lo**k / math.factorial(k)
            for k in range(goals_allowed + 1)
        )
        reasons = ["assumed league-average rate", "stoppage length unknown"]
        if state.home_reds or state.away_reds:
            reasons.append("red-card state shifts rate (unmodeled)")
        return Estimate(
            min(1.0, max(0.0, p)),
            "low",
            self.model_name,
            evidence_age_s=age,
            abstain_reason="low confidence: " + "; ".join(reasons),
            p_lower=min(1.0, max(0.0, p_low)),
            p_upper=min(1.0, max(0.0, p_high)),
        )


@dataclass
class LiveStateSpaceModel:
    """Coherent Bivariate Poisson / Dixon-Coles live state-space surface model.

    Prices 14+ SportyBet market families from a single joint remaining-goals
    distribution P(Delta G_H = i, Delta G_A = j) with:
      - Non-linear piecewise clock hazard Lambda(t),
      - Smooth survival stoppage-time estimation,
      - Empirical Bayes live-stat shrinkage (shots on target, attempts, corners, possession),
      - Markov red-card manpower & score-state hazard multipliers,
      - 7-point sigma uncertainty propagation producing [p_lower, p, p_upper].
    """

    model_name: str = "live_state_space_bivariate_dc_v2"
    prior_home_lambda: float = 1.42
    prior_away_lambda: float = 1.18
    prior_adj_home: float = 0.0
    prior_adj_away: float = 0.0
    dixon_coles_rho: float = -0.06
    shared_tempo_frac: float = 0.05
    intensity_rel_sigma: float = 0.16
    max_evidence_age_s: float = 60.0
    now: Optional[datetime] = None
    _surface_cache: dict[tuple, Any] = field(default_factory=dict, repr=False)

    def _get_surface(self, state: LiveMatchState, period_scope: str) -> tuple:
        eff_scope = "2h" if state.period == "2h" else period_scope
        cache_key = (
            state.match_id,
            state.period,
            state.clock_minute,
            state.home.goals,
            state.away.goals,
            state.half_home_goals,
            state.half_away_goals,
            state.home_reds,
            state.away_reds,
            state.home.shots_on_target,
            state.away.shots_on_target,
            state.home.corners,
            state.away.corners,
            state.remaining_unknown,
            state.added_minutes_announced,
            eff_scope,
            self.prior_home_lambda,
            self.prior_away_lambda,
            self.prior_adj_home,
            self.prior_adj_away,
        )
        cached = self._surface_cache.get(cache_key)
        if cached is not None:
            return cached

        assert state.clock_minute is not None
        stoppage = estimate_remaining_minutes(state, period_scope=eff_scope)
        xg_est = None
        if state.home.shots_on_target is not None or state.away.shots_on_target is not None:
            xg_est = estimate_xg_from_stats({
                "shots_on_target_home": state.home.shots_on_target,
                "shots_on_target_away": state.away.shots_on_target,
                "shots_total_home": state.home.shots_total,
                "shots_total_away": state.away.shots_total,
                "corners_home": state.home.corners,
                "corners_away": state.away.corners,
            })
        post = compute_posterior_intensities(
            state,
            prior_home_lambda=self.prior_home_lambda,
            prior_away_lambda=self.prior_away_lambda,
            prior_adj_home=self.prior_adj_home,
            prior_adj_away=self.prior_adj_away,
            xg_estimate=xg_est,
        )

        haz_mean = effective_hazard_fraction(
            state.period, state.clock_minute, stoppage.rem_mean
        )
        haz_lo = effective_hazard_fraction(
            state.period, state.clock_minute, stoppage.rem_lower
        )
        haz_hi = effective_hazard_fraction(
            state.period, state.clock_minute, stoppage.rem_upper
        )

        mu_h = post.lambda_home_90 * haz_mean
        mu_a = post.lambda_away_90 * haz_mean
        max_g = 6 if (mu_h + mu_a) < 0.6 else 10

        mat_nom = build_remaining_score_matrix(
            mu_h,
            mu_a,
            rho=self.dixon_coles_rho,
            shared_tempo_frac=self.shared_tempo_frac,
            max_goals=max_g,
        )
        marg_nom = compute_matrix_marginals(mat_nom)
        rel_s = self.intensity_rel_sigma
        scenarios = [
            (post.lambda_home_90 * haz_lo, post.lambda_away_90 * haz_lo),
            (post.lambda_home_90 * haz_hi, post.lambda_away_90 * haz_hi),
            (mu_h * (1.0 + rel_s), mu_a * (1.0 + rel_s)),
            (mu_h * (1.0 - rel_s), mu_a * (1.0 - rel_s)),
            (mu_h * (1.0 + rel_s), mu_a * (1.0 - rel_s)),
            (mu_h * (1.0 - rel_s), mu_a * (1.0 + rel_s)),
        ]
        scenario_mats = []
        for s_mu_h, s_mu_a in scenarios:
            s_mat = build_remaining_score_matrix(
                s_mu_h,
                s_mu_a,
                rho=self.dixon_coles_rho,
                shared_tempo_frac=self.shared_tempo_frac,
                max_goals=max_g,
            )
            scenario_mats.append(
                (s_mu_h, s_mu_a, s_mat, compute_matrix_marginals(s_mat))
            )
        bundle = (stoppage, post, mu_h, mu_a, mat_nom, marg_nom, scenario_mats)
        self._surface_cache[cache_key] = bundle
        return bundle

    def estimate(
        self, state: LiveMatchState, market_key: str, selection: str
    ) -> Estimate:
        if not state.is_live:
            return Estimate(
                None,
                "low",
                self.model_name,
                abstain_reason=f"match not live (period={state.period})",
            )
        age = state.age_seconds(self.now or datetime.now(timezone.utc))
        if age > self.max_evidence_age_s:
            return Estimate(
                None,
                "low",
                self.model_name,
                evidence_age_s=age,
                abstain_reason=f"stale evidence ({age:.0f}s > {self.max_evidence_age_s:.0f}s)",
            )
        if state.clock_minute is None or state.score_total is None:
            return Estimate(
                None,
                "low",
                self.model_name,
                evidence_age_s=age,
                abstain_reason="clock or score unknown",
            )

        mkey = market_key.strip().lower()
        period_scope = "1h" if "1h" in mkey else "2h" if "2h" in mkey else "full"
        stoppage, post, mu_h, mu_a, mat_nom, marg_nom, scenario_mats = (
            self._get_surface(state, period_scope)
        )

        # The path model is only needed by the 'at any time' families; building
        # it for every selection would tax the fast plane for nothing.
        needs_path = is_path_market(market_key)
        needs_corners = market_key.strip().lower().startswith("corners_")
        corner_model = None
        if needs_corners:
            rem = estimate_remaining_minutes(state, period_scope="full").rem_mean
            corner_model = CornerModel.from_state(state, rem)
        nom_path = (
            GoalPathModel.from_intensities(mu_h, mu_a, self.shared_tempo_frac)
            if needs_path
            else None
        )
        nom_out = price_from_joint_matrix(
            mat_nom,
            state,
            market_key,
            selection,
            mu_h,
            mu_a,
            marginals=marg_nom,
            path_model=nom_path,
            corner_model=corner_model,
        )
        if nom_out is None:
            return Estimate(
                None,
                "low",
                self.model_name,
                evidence_age_s=age,
                abstain_reason=f"market {market_key}/{selection} outside supported state-space families or missing period state",
            )

        win_samples = [nom_out.effective_win_prob]
        for s_mu_h, s_mu_a, s_mat, s_marg in scenario_mats:
            s_out = price_from_joint_matrix(
                s_mat,
                state,
                market_key,
                selection,
                s_mu_h,
                s_mu_a,
                marginals=s_marg,
                path_model=(
                    GoalPathModel.from_intensities(
                        s_mu_h, s_mu_a
                    )
                    if needs_path
                    else None
                ),
                corner_model=corner_model,
            )
            if s_out is not None:
                win_samples.append(s_out.effective_win_prob)

        p_mean = min(1.0, max(0.0, nom_out.effective_win_prob))
        p_lower = min(1.0, max(0.0, min(win_samples)))
        p_upper = min(1.0, max(0.0, max(win_samples)))
        width = p_upper - p_lower

        # Determine confidence grade & transparent diagnostics
        reasons: list[str] = []
        if stoppage.stoppage_uncertainty_high and width > 0.005:
            reasons.append(
                f"unverified stoppage duration (rem={stoppage.rem_mean:.2f}m [{stoppage.rem_lower:.2f}m..{stoppage.rem_upper:.2f}m])"
            )
        if post.shrinkage_weight == 0.0:
            reasons.append("no live shot/corner stats observed (prior-only intensity)")
        if width > 0.16:
            reasons.append(f"wide uncertainty interval ({width:.3f})")

        if reasons:
            confidence: Confidence = "low"
            abstain_note: Optional[str] = "low confidence: " + "; ".join(reasons)
        elif width <= 0.07 and post.shrinkage_weight >= 0.25:
            confidence = "high"
            abstain_note = None
        else:
            confidence = "medium"
            abstain_note = None

        diagnostics = {
            "mu_home_rem": round(mu_h, 4),
            "mu_away_rem": round(mu_a, 4),
            "lambda_home_90": round(post.lambda_home_90, 4),
            "lambda_away_90": round(post.lambda_away_90, 4),
            "rem_minutes_mean": round(stoppage.rem_mean, 2),
            "rem_minutes_bounds": [round(stoppage.rem_lower, 2), round(stoppage.rem_upper, 2)],
            "red_card_mult": [round(post.red_card_multiplier_home, 3), round(post.red_card_multiplier_away, 3)],
            "shrinkage_weight": round(post.shrinkage_weight, 3),
            "uncertainty_width": round(width, 4),
        }

        return Estimate(
            p=p_mean,
            confidence=confidence,
            model_name=self.model_name,
            evidence_age_s=age,
            abstain_reason=abstain_note,
            p_lower=p_lower,
            p_upper=p_upper,
            p_push=nom_out.p_push,
            p_half_win=nom_out.p_half_win,
            p_half_loss=nom_out.p_half_loss,
            diagnostics=diagnostics,
        )


def _lo(calibrated_p: float, base_lower: Optional[float]) -> Optional[float]:
    """Lower bound of the calibrated interval.

    The shrink pulls the probability toward the market, which can take it
    BELOW the base model's own lower bound. The interval must stay
    coherent, or the 'conservative' EV is computed from a higher
    probability than the headline and the betting gate inverts.
    """
    if base_lower is None:
        return calibrated_p
    return min(calibrated_p, base_lower)


def _hi(calibrated_p: float, base_upper: Optional[float]) -> Optional[float]:
    """Upper bound of the calibrated interval."""
    if base_upper is None:
        return calibrated_p
    return max(calibrated_p, base_upper)


@dataclass
class CalibratedModel:
    """Wraps a base MarketModel with market-implied calibration.

    When the base model's confidence is low, probabilities are shrunk toward
    the market-implied (Shin de-vigged) probabilities. This counteracts the
    systematic overconfidence observed in backtests (model more extreme than
    market in both directions).

    The shrinkage weight is proportional to (1 - confidence_score):
      confidence high   → 0% shrink (trust the model)
      confidence medium → 50% shrink
      confidence low    → 75% shrink (mostly trust the market)

    The market-implied probabilities come from the 'Rest of the Match' market
    when available (it is the market's own conditional estimate given the
    current score), otherwise from the selection's own market.
    """

    base_model: MarketModel
    market_fair: dict[tuple[str, str], float] = field(default_factory=dict)
    model_name: str = "calibrated_v1"

    _SHRINK = {"high": 0.0, "medium": 0.5, "low": 0.75}

    def estimate(
        self, state: LiveMatchState, market_key: str, selection: str
    ) -> Estimate:
        base = self.base_model.estimate(state, market_key, selection)
        if base.abstained or base.p is None:
            return base

        shrink = self._SHRINK.get(base.confidence, 0.75)
        if shrink == 0.0:
            return Estimate(
                p=base.p, confidence=base.confidence,
                model_name=self.model_name,
                evidence_age_s=base.evidence_age_s,
                p_lower=base.p_lower, p_upper=base.p_upper,
                p_push=base.p_push, p_half_win=base.p_half_win,
                p_half_loss=base.p_half_loss,
                diagnostics={**base.diagnostics, "shrink_pct": 0.0},
            )

        # Calibrate using Rest-of-Match only when the two are the SAME event.
        # 'Rest of the Match (current score 1:0)' settles on the result of
        # the REMAINING time treated as a fresh match, so its 'Away' is
        # P(the away side leads in the remaining time) — which is NOT
        # P(the away side wins the full-time 1X2) unless the score is level.
        # Calibrating a full-time 1X2 against it at 1-0 made the model
        # believe a side had ~9% to win at 65.00, a +4.5 EV phantom edge. It
        # is also a full-match market, so it says nothing about a half-time
        # 1X2.
        mkt_key = market_key
        rest_key = None
        score_level = (
            state.home.goals is not None
            and state.away.goals is not None
            and state.home.goals == state.away.goals
        )
        if market_key == "1x2" and score_level:
            for rk in ("rest_of_match_1x2", "1x2_rest"):
                if (rk, selection) in self.market_fair:
                    rest_key = rk
                    break
        if rest_key:
            mkt_key = rest_key

        market_p = self.market_fair.get((mkt_key, selection))
        if market_p is None:
            market_p = self.market_fair.get((market_key, selection))
        if market_p is None:
            return Estimate(
                p=base.p, confidence=base.confidence,
                model_name=self.model_name,
                evidence_age_s=base.evidence_age_s,
                abstain_reason="calibration skipped: no market-implied prob available",
                p_lower=base.p_lower, p_upper=base.p_upper,
                diagnostics={**base.diagnostics, "shrink_pct": shrink * 100},
            )

        calibrated_p = base.p * (1.0 - shrink) + market_p * shrink
        calibrated_p = min(1.0, max(0.0, calibrated_p))

        gap_pp = abs(calibrated_p - market_p) * 100.0
        if gap_pp > 5.0:
            return Estimate(
                p=None,
                confidence=base.confidence,
                model_name=self.model_name,
                evidence_age_s=base.evidence_age_s,
                abstain_reason=f"model-market gap too large ({gap_pp:.1f}pp > 5pp), blocking bet",
                p_lower=_lo(calibrated_p, base.p_lower),
                p_upper=_hi(calibrated_p, base.p_upper),
                p_push=base.p_push, p_half_win=base.p_half_win,
                p_half_loss=base.p_half_loss,
                diagnostics={
                    **base.diagnostics,
                    "shrink_pct": shrink * 100,
                    "market_p": round(market_p, 4),
                    "raw_model_p": round(base.p, 4),
                },
            )

        return Estimate(
            p=calibrated_p,
            confidence=base.confidence,
            model_name=self.model_name,
            evidence_age_s=base.evidence_age_s,
            p_lower=_lo(calibrated_p, base.p_lower),
            p_upper=_hi(calibrated_p, base.p_upper),
            p_push=base.p_push, p_half_win=base.p_half_win,
            p_half_loss=base.p_half_loss,
            diagnostics={
                **base.diagnostics,
                "shrink_pct": shrink * 100,
                "market_p": round(market_p, 4),
                "raw_model_p": round(base.p, 4),
            },
        )
