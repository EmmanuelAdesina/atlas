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
     multipliers, and a scale-aware multi-parameter scenario envelope across
     14+ SportyBet market families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from atlas.live.math_engine import (
    _smooth_positive_residual,
    build_remaining_score_matrix,
    compute_matrix_marginals,
    compute_posterior_intensities,
    effective_hazard_fraction,
    estimate_remaining_minutes,
    expected_value_detailed,
    infer_prematch_lambdas_from_1x2,
    price_from_joint_matrix,
)
from atlas.live.observe import LiveMatchState


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


def _logit(probability: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, probability))
    return math.log(p) - math.log1p(-p)


@dataclass
class Estimate:
    """One selection's settlement probabilities and uncertainty envelope.

    `p` is the probability of a full win (not the quarter-line effective-win
    equivalent); half wins/losses and pushes are carried separately.
    """

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
    settlement_scenarios: list[tuple[float, float, float, float]] = field(
        default_factory=list, repr=False
    )

    @property
    def abstained(self) -> bool:
        return self.p is None

    @property
    def effective_win_probability(self) -> Optional[float]:
        """Display-equivalent win probability including half-win outcomes."""
        if self.p is None:
            return None
        return self.p + 0.5 * self.p_half_win

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

    def conservative_settlement(
        self, odds: float
    ) -> Optional[tuple[float, float, float, float]]:
        """Return the modeled settlement vector with the lowest EV at `odds`.

        Each vector is (full-win, push, half-win, half-loss). Using a complete
        scenario avoids mixing a lower full-win probability with nominal split
        outcome probabilities when sizing bets.
        """
        if self.p is None:
            return None
        scenarios = self.settlement_scenarios or [
            (
                self.p_lower if self.p_lower is not None else self.p,
                self.p_push,
                self.p_half_win,
                self.p_half_loss,
            )
        ]
        return min(
            scenarios,
            key=lambda item: expected_value_detailed(
                odds,
                p_win=item[0],
                p_push=item[1],
                p_half_win=item[2],
                p_half_loss=item[3],
            ),
        )

    def conservative_expected_value(self, odds: float) -> Optional[float]:
        scenario = self.conservative_settlement(odds)
        if scenario is None:
            return None
        return expected_value_detailed(
            odds,
            p_win=scenario[0],
            p_push=scenario[1],
            p_half_win=scenario[2],
            p_half_loss=scenario[3],
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
      - A parameter-perturbation envelope producing [p_lower, p, p_upper].
    """

    model_name: str = "live_state_space_bivariate_dc_v2"
    prior_home_lambda: float = 1.42
    prior_away_lambda: float = 1.18
    prior_adj_home: float = 0.0
    prior_adj_away: float = 0.0
    dixon_coles_rho: float = -0.06
    shared_tempo_frac: float = 0.05
    intensity_rel_sigma: float = 0.16
    dixon_coles_rho_sigma: float = 0.02
    shared_tempo_abs_sigma: float = 0.03
    max_evidence_age_s: float = 60.0
    now: Optional[datetime] = None
    _surface_cache: dict[tuple, Any] = field(default_factory=dict, repr=False)
    _prematch_prior_cache: dict[tuple, tuple[float, float]] = field(default_factory=dict, repr=False)

    def _resolved_priors(self, state: LiveMatchState) -> tuple[float, float]:
        odds = state.prematch_1x2_odds
        if odds is None:
            return self.prior_home_lambda, self.prior_away_lambda
        key = (
            float(odds["Home"]), float(odds["Draw"]), float(odds["Away"]),
            self.dixon_coles_rho, self.shared_tempo_frac,
        )
        cached = self._prematch_prior_cache.get(key)
        if cached is not None:
            return cached
        priors = infer_prematch_lambdas_from_1x2(
            key[0], key[1], key[2],
            rho=self.dixon_coles_rho,
            shared_tempo_frac=self.shared_tempo_frac,
        )
        self._prematch_prior_cache[key] = priors
        if len(self._prematch_prior_cache) > 64:
            self._prematch_prior_cache.pop(next(iter(self._prematch_prior_cache)))
        return priors

    def prepare_priors(self, state: LiveMatchState) -> tuple[float, float]:
        """Resolve/cache any pre-match 1X2 inversion before the live hot path."""
        return self._resolved_priors(state)

    def _get_surface(self, state: LiveMatchState, period_scope: str) -> tuple:
        eff_scope = "2h" if state.period == "2h" else period_scope
        prior_home, prior_away = self._resolved_priors(state)
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
            state.home.shots_total,
            state.away.shots_total,
            state.home.corners,
            state.away.corners,
            state.home.possession_pct,
            state.away.possession_pct,
            state.substitutions_count,
            state.remaining_unknown,
            state.added_minutes_announced,
            tuple(sorted(state.prematch_1x2_odds.items())) if state.prematch_1x2_odds else None,
            eff_scope,
            prior_home,
            prior_away,
            self.prior_adj_home,
            self.prior_adj_away,
            self.dixon_coles_rho,
            self.shared_tempo_frac,
            self.intensity_rel_sigma,
            self.dixon_coles_rho_sigma,
            self.shared_tempo_abs_sigma,
        )
        cached = self._surface_cache.get(cache_key)
        if cached is not None:
            return cached

        assert state.clock_minute is not None
        stoppage = estimate_remaining_minutes(state, period_scope=eff_scope)
        post = compute_posterior_intensities(
            state,
            prior_home_lambda=prior_home,
            prior_away_lambda=prior_away,
            prior_adj_home=self.prior_adj_home,
            prior_adj_away=self.prior_adj_away,
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
        max_g = 6 if (mu_h + mu_a) < 0.6 else 8

        mat_nom = build_remaining_score_matrix(
            mu_h,
            mu_a,
            rho=self.dixon_coles_rho,
            shared_tempo_frac=self.shared_tempo_frac,
            max_goals=max_g,
        )
        marg_nom = compute_matrix_marginals(mat_nom)
        rel_s = self.intensity_rel_sigma
        rho_low = max(-0.30, min(0.30, self.dixon_coles_rho - self.dixon_coles_rho_sigma))
        rho_high = max(-0.30, min(0.30, self.dixon_coles_rho + self.dixon_coles_rho_sigma))
        shared_low = max(0.0, min(0.25, self.shared_tempo_frac - self.shared_tempo_abs_sigma))
        shared_high = max(0.0, min(0.25, self.shared_tempo_frac + self.shared_tempo_abs_sigma))
        # Keep a 7-surface hot path (nominal + six perturbations), while
        # pairing dependence endpoints with intensity/stoppage perturbations.
        # Thus rho/shared-tempo uncertainty is represented without multiplying
        # the number of full matrix repricings per selection.
        scenarios = [
            (post.lambda_home_90 * haz_lo, post.lambda_away_90 * haz_lo,
             rho_low, shared_low),
            (post.lambda_home_90 * haz_hi, post.lambda_away_90 * haz_hi,
             rho_high, shared_high),
            (mu_h * (1.0 + rel_s), mu_a * (1.0 + rel_s),
             rho_low, shared_high),
            (mu_h * (1.0 - rel_s), mu_a * (1.0 - rel_s),
             rho_high, shared_low),
            (mu_h * (1.0 + rel_s), mu_a * (1.0 - rel_s),
             rho_low, shared_low),
            (mu_h * (1.0 - rel_s), mu_a * (1.0 + rel_s),
             rho_high, shared_high),
        ]

        scenario_mats = []
        for s_mu_h, s_mu_a, s_rho, s_shared in scenarios:
            s_mat = build_remaining_score_matrix(
                s_mu_h,
                s_mu_a,
                rho=s_rho,
                shared_tempo_frac=s_shared,
                max_goals=max_g,
            )
            scenario_mats.append(
                (s_mu_h, s_mu_a, s_mat, compute_matrix_marginals(s_mat))
            )
        bundle = (stoppage, post, mu_h, mu_a, mat_nom, marg_nom, scenario_mats)
        self._surface_cache[cache_key] = bundle
        # A long-running live loop sees a new snapshot every refresh. Retain a
        # small working set for repeated market pricing without unbounded growth.
        if len(self._surface_cache) > 64:
            self._surface_cache.pop(next(iter(self._surface_cache)))
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

        nom_out = price_from_joint_matrix(
            mat_nom, state, market_key, selection, mu_h, mu_a, marginals=marg_nom
        )
        if nom_out is None:
            return Estimate(
                None,
                "low",
                self.model_name,
                evidence_age_s=age,
                abstain_reason=f"market {market_key}/{selection} outside supported state-space families or missing period state",
            )

        win_samples = [nom_out.p_win]
        settlement_samples = [
            (nom_out.p_win, nom_out.p_push, nom_out.p_half_win, nom_out.p_half_loss)
        ]
        for s_mu_h, s_mu_a, s_mat, s_marg in scenario_mats:
            s_out = price_from_joint_matrix(
                s_mat, state, market_key, selection, s_mu_h, s_mu_a, marginals=s_marg
            )
            if s_out is not None:
                win_samples.append(s_out.p_win)
                settlement_samples.append(
                    (s_out.p_win, s_out.p_push, s_out.p_half_win, s_out.p_half_loss)
                )

        p_mean = min(1.0, max(0.0, nom_out.p_win))
        p_lower = min(1.0, max(0.0, min(win_samples)))
        p_upper = min(1.0, max(0.0, max(win_samples)))
        width = p_upper - p_lower
        logit_width = _logit(p_upper) - _logit(p_lower)

        # Determine confidence grade & transparent diagnostics. The 7+ point
        # perturbation envelope has no nominal coverage level; logit width is a
        # scale-aware heuristic, especially near probabilities 0 and 1.
        reasons: list[str] = []
        if stoppage.stoppage_uncertainty_high and width > 0.005:
            reasons.append(
                f"unverified stoppage duration (rem={stoppage.rem_mean:.2f}m [{stoppage.rem_lower:.2f}m..{stoppage.rem_upper:.2f}m])"
            )
        if post.shrinkage_weight == 0.0:
            reasons.append("no live shot/corner stats observed (prior-only intensity)")
        if width > 0.16 or logit_width > 2.0:
            reasons.append(
                f"wide uncertainty envelope (absolute={width:.3f}, logit={logit_width:.3f})"
            )

        if reasons:
            confidence: Confidence = "low"
            abstain_note: Optional[str] = "low confidence: " + "; ".join(reasons)
        elif width <= 0.07 and logit_width <= 0.55 and post.shrinkage_weight >= 0.25:
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
            "logit_uncertainty_width": round(logit_width, 4),
            "effective_win_probability": round(nom_out.effective_win_prob, 4),
            "prior_source": "prematch_1x2" if state.prematch_1x2_odds else "default",
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
            settlement_scenarios=settlement_samples,
            diagnostics=diagnostics,
        )
