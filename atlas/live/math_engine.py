"""Quantitative pricing, de-vigging, and state-space goal engine.

Pure Python stdlib (`math`) implementation with zero C/Fortran binary-wheel
fragility (runs identically on Linux x86_64 and ARMv7/Termux). The full
21-selection regression surface is targeted below 2 ms after setup.

Mathematical pillars:
  1. De-vigging suite: Proportional, Power (constant-elasticity), and Shin (1993)
     insider-trading microstructure de-vigging with adverse-selection parameter z.
  2. Non-linear piecewise football hazard clock Lambda(t) + smooth positive
     mean-residual-life stoppage-time survival estimator (never collapses to 0
     while a match is still live in stoppage time).
  3. Empirical Bayes live-statistics shrinkage filter + Markov red-card manpower
     and score-state hazard multipliers.
  4. Coherent Bivariate Poisson / Dixon-Coles remaining-goal joint distribution
     P(Delta G_H = i, Delta G_A = j) pricing 14+ SportyBet market families from
     a single probability surface.
  5. Parameter-scenario uncertainty envelopes yielding [p_lower, p_mean, p_upper],
     exact push / quarter-line expected value, and settlement-aware fractional Kelly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from atlas.live.observe import LiveMatchState

# Piecewise 15-minute relative goal-scoring hazard rates in regulation time.
# Their regulation-only integral is 90 hazard-minutes; first-half and second-half
# shares are 45% / 55%. Priors are full-match expected goals, so the live hazard
# is normalized by a standard 90' match plus the kickoff expected stoppage
# returned by the same smooth-residual estimator.
_REGULATION_HAZARD_BUCKETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 15.0, 0.76),
    (15.0, 30.0, 0.90),
    (30.0, 45.0, 1.04),
    (45.0, 60.0, 0.98),
    (60.0, 75.0, 1.08),
    (75.0, 90.0, 1.24),
)
_STOPPAGE_1H_HAZARD = 1.10
_STOPPAGE_2H_INITIAL_HAZARD = 1.0
# Expected baseline stoppage from the smooth-residual estimator at kickoff
# (2.0' prior with 0.75' floor; 4.2' prior with 0.85' floor).
_STANDARD_1H_STOPPAGE_MINUTES = 0.5 * (2.0 + math.sqrt(2.0**2 + 4.0 * 0.75**2))
_STANDARD_2H_STOPPAGE_MINUTES = 0.5 * (4.2 + math.sqrt(4.2**2 + 4.0 * 0.85**2))
# The 2H stoppage hazard starts at 1.0 and ramps as stoppage elapses; normalize
# the full-match prior to that same kickoff convention.
_STANDARD_MATCH_HAZARD_MINS = (
    sum((hi - lo) * rate for lo, hi, rate in _REGULATION_HAZARD_BUCKETS)
    + _STANDARD_1H_STOPPAGE_MINUTES * _STOPPAGE_1H_HAZARD
    + _STANDARD_2H_STOPPAGE_MINUTES * _STOPPAGE_2H_INITIAL_HAZARD
)


# ---------------------------------------------------------------------------
# 1. De-Vigging Suite (Proportional, Power, Shin 1993)
# ---------------------------------------------------------------------------


def devig_proportional(selections: dict[str, float]) -> dict[str, float]:
    """Multiplicative (proportional) de-vigging: p_i = (1/o_i) / sum(1/o_j)."""
    if not selections:
        return {}
    implied = {k: 1.0 / float(o) for k, o in selections.items()}
    total = sum(implied.values())
    if total <= 0.0:
        raise ValueError("Sum of implied probabilities must be positive")
    return {k: v / total for k, v in implied.items()}


def devig_power(selections: dict[str, float], tol: float = 1e-8) -> dict[str, float]:
    """Power (constant-elasticity) de-vigging correcting favorite-longshot bias.

    Solves sum_i (1 / o_i)^k = 1 for k > 0 via fast bisection.
    """
    if not selections:
        return {}
    keys = list(selections.keys())
    pis = [1.0 / float(selections[k]) for k in keys]
    total = sum(pis)
    if total <= 1.0 + 1e-12 or any(v >= 1.0 for v in pis):
        return devig_proportional(selections)

    lo, hi = 1.0, 20.0
    while sum(v**hi for v in pis) > 1.0 and hi < 1024.0:
        hi *= 2.0

    for _ in range(32):
        mid = 0.5 * (lo + hi)
        s = sum(v**mid for v in pis)
        if abs(s - 1.0) <= tol:
            lo = hi = mid
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid

    k_star = 0.5 * (lo + hi)
    raw = [v**k_star for v in pis]
    norm = sum(raw)
    return {k: r / norm for k, r in zip(keys, raw)}


def devig_shin(
    selections: dict[str, float], tol: float = 1e-8
) -> tuple[dict[str, float], float]:
    """Shin (1993) insider-trading microstructure de-vigging.

    Solves for insider-trading / adverse-selection fraction z in [0, 1):
      p_i(z) = (sqrt(z^2 + 4*(1 - z)*(pi_i^2 / S)) - z) / (2*(1 - z))
      sum_i p_i(z) = 1

    Returns (fair_probs_dict, shin_z).
    """
    if not selections:
        return {}, 0.0
    keys = list(selections.keys())
    pis = [1.0 / float(selections[k]) for k in keys]
    s_total = sum(pis)
    n = len(pis)
    if s_total <= 1.0 + 1e-9 or n < 2:
        return devig_proportional(selections), 0.0

    # Precompute 4 * pi_i^2 / S
    four_pi2_over_s = [4.0 * (pi * pi) / s_total for pi in pis]
    sqrt = math.sqrt

    lo, hi = 0.0, 0.9999
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        one_minus_z = 1.0 - mid
        z2 = mid * mid
        denom = 2.0 * one_minus_z
        total_p = sum(
            (sqrt(z2 + one_minus_z * c) - mid) / denom for c in four_pi2_over_s
        )
        if abs(total_p - 1.0) <= tol:
            lo = hi = mid
            break
        if total_p > 1.0:
            lo = mid
        else:
            hi = mid

    z_star = 0.5 * (lo + hi)
    one_minus_z = 1.0 - z_star
    z2 = z_star * z_star
    denom = 2.0 * one_minus_z
    raw = [(sqrt(z2 + one_minus_z * c) - z_star) / denom for c in four_pi2_over_s]
    norm = sum(raw)
    return ({k: r / norm for k, r in zip(keys, raw)}, z_star)


# ---------------------------------------------------------------------------
# 2. Non-Linear Clock Hazard & Smooth Stoppage Survival Estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoppageEstimate:
    """Remaining playing time distribution (in minutes) for the active period."""

    rem_mean: float
    rem_lower: float
    rem_upper: float
    in_stoppage: bool
    stoppage_uncertainty_high: bool


def _smooth_positive_residual(delta: float, floor_minutes: float) -> float:
    """Smooth mean-residual-life function psi(delta, floor) > 0 everywhere.

    For delta = S_expected - s_elapsed:
      - When delta >> 0, psi(delta) -> delta.
      - When delta == 0, psi(0) == floor_minutes.
      - When delta < 0 (match in deep stoppage beyond expected), psi(delta)
        decays smoothly toward 0 without ever hitting 0 while live.
    """
    return 0.5 * (delta + math.sqrt(delta * delta + 4.0 * floor_minutes * floor_minutes))


def estimate_remaining_minutes(
    state: LiveMatchState, period_scope: str = "full"
) -> StoppageEstimate:
    """Estimate remaining playing minutes [lower, mean, upper] for `period_scope`.

    `period_scope` is '1h', '2h', or 'full'.
    """
    clock = state.clock_minute if state.clock_minute is not None else 0.0
    reds = state.home_reds + state.away_reds
    subs = state.substitutions_count or 0

    if state.period == "1h":
        goals_1h = state.score_total or 0
        prior_stop_1h = (
            float(state.added_minutes_announced)
            if state.added_minutes_announced is not None
            else 2.0 + 0.40 * goals_1h + 0.50 * reds
        )
        reg_rem_1h = max(0.0, 45.0 - clock)
        elapsed_stop_1h = max(0.0, clock - 45.0)
        in_stop_1h = clock >= 45.0
        stop_rem_1h = _smooth_positive_residual(prior_stop_1h - elapsed_stop_1h, 0.75)
        rem_1h_mean = reg_rem_1h + stop_rem_1h
        rem_1h_lower = reg_rem_1h + max(0.25, stop_rem_1h - 1.25)
        rem_1h_upper = reg_rem_1h + stop_rem_1h + 1.75

        if period_scope == "1h":
            return StoppageEstimate(
                rem_mean=rem_1h_mean,
                rem_lower=rem_1h_lower,
                rem_upper=rem_1h_upper,
                in_stoppage=in_stop_1h,
                stoppage_uncertainty_high=(
                    state.remaining_unknown
                    or (in_stop_1h and state.added_minutes_announced is None)
                ),
            )
        # Full match from 1H includes the complete 2H, using the same smooth
        # expected stoppage duration as the 2H estimator at its kickoff.
        return StoppageEstimate(
            rem_mean=rem_1h_mean + 45.0 + _STANDARD_2H_STOPPAGE_MINUTES,
            rem_lower=rem_1h_lower + 48.0,
            rem_upper=rem_1h_upper + 52.5,
            in_stoppage=in_stop_1h,
            stoppage_uncertainty_high=(
                state.remaining_unknown
                or (in_stop_1h and state.added_minutes_announced is None)
            ),
        )

    # 2nd Half ('2h')
    if (
        state.home.goals is not None
        and state.away.goals is not None
        and state.half_home_goals is not None
        and state.half_away_goals is not None
    ):
        goals_2h = max(
            0,
            (state.home.goals - state.half_home_goals)
            + (state.away.goals - state.half_away_goals),
        )
    else:
        goals_2h = state.score_total or 0

    if state.added_minutes_announced is not None:
        prior_stop_2h = float(state.added_minutes_announced) + 0.35
    else:
        card_minute = state.red_card_minute if state.red_card_minute is not None else 45.0
        card_weight = max(0.0, min(1.0, (90.0 - card_minute) / 90.0))
        prior_stop_2h = (
            4.2
            + 0.45 * goals_2h
            + 0.25 * min(subs, 10)
            + 0.55 * reds * card_weight
        )

    reg_rem_2h = max(0.0, 90.0 - clock)
    elapsed_stop_2h = max(0.0, clock - 90.0)
    in_stop_2h = clock >= 90.0
    floor_min = 0.85 if state.added_minutes_announced is None else 0.45
    stop_rem_2h = _smooth_positive_residual(prior_stop_2h - elapsed_stop_2h, floor_min)

    rem_mean = reg_rem_2h + stop_rem_2h
    spread_down = 1.15 if in_stop_2h else 1.50
    spread_up = 2.10 if (in_stop_2h and state.added_minutes_announced is None) else 1.75
    rem_lower = reg_rem_2h + max(0.20, stop_rem_2h - spread_down)
    rem_upper = reg_rem_2h + stop_rem_2h + spread_up

    return StoppageEstimate(
        rem_mean=rem_mean,
        rem_lower=rem_lower,
        rem_upper=rem_upper,
        in_stoppage=in_stop_2h,
        stoppage_uncertainty_high=(
            state.remaining_unknown
            or (in_stop_2h and state.added_minutes_announced is None)
        ),
    )


def effective_hazard_fraction(
    period: str, clock_minute: float, remaining_minutes: float
) -> float:
    """Integrate piecewise goal-scoring hazard over `remaining_minutes`.

    Returns the fraction of a standard full-match expected-goal prior remaining.
    The reference denominator includes regulation plus canonical 1H/2H stoppage,
    keeping kickoff full-match intensity exactly equal to the prior lambdas.
    """
    if remaining_minutes <= 0.0:
        return 0.0

    if period == "1h":
        reg_rem = max(0.0, 45.0 - clock_minute)
        reg_play = min(remaining_minutes, reg_rem)
        stop_play = max(0.0, remaining_minutes - reg_play)
        hazard_mins = 0.0
        t_start = min(clock_minute, 45.0)
        t_end = t_start + reg_play
        for b_lo, b_hi, rate in _REGULATION_HAZARD_BUCKETS[:3]:
            overlap = max(0.0, min(t_end, b_hi) - max(t_start, b_lo))
            hazard_mins += overlap * rate
        # Check if remaining_minutes includes 2nd half (full match from 1H)
        if stop_play > 15.0:
            # 1H stoppage + 2H regulation + 2H stoppage
            standard_2h_minutes = 45.0 + _STANDARD_2H_STOPPAGE_MINUTES
            stop_1h = max(0.0, stop_play - standard_2h_minutes)
            hazard_mins += stop_1h * _STOPPAGE_1H_HAZARD
            for b_lo, b_hi, rate in _REGULATION_HAZARD_BUCKETS[3:]:
                hazard_mins += (b_hi - b_lo) * rate
            hazard_mins += _STANDARD_2H_STOPPAGE_MINUTES * _STOPPAGE_2H_INITIAL_HAZARD
        else:
            hazard_mins += stop_play * _STOPPAGE_1H_HAZARD
        return hazard_mins / _STANDARD_MATCH_HAZARD_MINS

    # period == '2h'
    reg_rem = max(0.0, 90.0 - clock_minute)
    reg_play = min(remaining_minutes, reg_rem)
    stop_play = max(0.0, remaining_minutes - reg_play)
    hazard_mins = 0.0
    t_start = max(45.0, min(clock_minute, 90.0))
    t_end = t_start + reg_play
    for b_lo, b_hi, rate in _REGULATION_HAZARD_BUCKETS[3:]:
        overlap = max(0.0, min(t_end, b_hi) - max(t_start, b_lo))
        hazard_mins += overlap * rate
    if stop_play > 0.0:
        # Local stoppage hazard rises as elapsed stoppage approaches its total.
        stoppage_elapsed = max(0.0, clock_minute - 90.0)
        stoppage_total = stoppage_elapsed + stop_play
        stoppage_hazard = 1.0 + 0.5 * (stoppage_elapsed / stoppage_total)
        hazard_mins += stop_play * stoppage_hazard
    return hazard_mins / _STANDARD_MATCH_HAZARD_MINS


# ---------------------------------------------------------------------------
# 3. Empirical Bayes Live Shrinkage & Markov State Multipliers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorIntensities:
    """Calibrated per-90 scoring intensities after live stats & state shifts."""

    lambda_home_90: float
    lambda_away_90: float
    shrinkage_weight: float
    red_card_multiplier_home: float
    red_card_multiplier_away: float
    score_state_multiplier_home: float
    score_state_multiplier_away: float


def infer_prematch_lambdas_from_1x2(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    *,
    rho: float = -0.06,
    shared_tempo_frac: float = 0.05,
) -> tuple[float, float]:
    """Invert pre-match 1X2 decimal odds into full-match goal intensities.

    The search uses the same bivariate Poisson / Dixon-Coles surface as the
    live model, a global coarse grid, then local refinement. Its bounds are
    deliberately broad enough to represent strong favourites and weak sides.
    """
    odds = (float(home_odds), float(draw_odds), float(away_odds))
    if any(not math.isfinite(o) or o <= 1.0 for o in odds):
        raise ValueError("1X2 decimal odds must be finite and greater than 1.0")
    fair, _ = devig_shin({"Home": odds[0], "Draw": odds[1], "Away": odds[2]})
    target_h = fair["Home"]
    target_d = fair["Draw"]

    def loss(lh: float, la: float) -> float:
        mat = build_remaining_score_matrix(
            lh,
            la,
            rho=rho,
            shared_tempo_frac=shared_tempo_frac,
            max_goals=12,
        )
        ph = sum(mat[r][c] for r in range(13) for c in range(r))
        pd = sum(mat[r][r] for r in range(13))
        return (ph - target_h) ** 2 + 1.5 * (pd - target_d) ** 2

    lo, hi = 0.05, 8.0
    coarse = sorted({
        lo, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 0.90,
        1.10, 1.30, 1.50, 1.80, 2.10, 2.50, 3.00, 3.50,
        4.00, 4.50, 5.00, 6.00, 7.00, hi,
    })
    best_lh, best_la = 1.42, 1.18
    best_loss = math.inf
    for lh in coarse:
        for la in coarse:
            candidate_loss = loss(lh, la)
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_lh, best_la = lh, la

    for step in (0.10, 0.025, 0.006, 0.0015):
        lh_center, la_center = best_lh, best_la
        for i in range(-6, 7):
            lh = max(lo, min(hi, lh_center + i * step))
            for j in range(-6, 7):
                la = max(lo, min(hi, la_center + j * step))
                candidate_loss = loss(lh, la)
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_lh, best_la = lh, la
    return best_lh, best_la


def compute_posterior_intensities(
    state: LiveMatchState,
    prior_home_lambda: float = 1.42,
    prior_away_lambda: float = 1.18,
    prior_adj_home: float = 0.0,
    prior_adj_away: float = 0.0,
    shrinkage_tau_minutes: float = 65.0,
) -> PosteriorIntensities:
    """Combine baseline priors with live match statistics, red cards, and score state."""
    # Clamp bounded prior adjustments to [-0.30, +0.30]
    base_h = max(0.05, prior_home_lambda + max(-0.30, min(0.30, prior_adj_home)))
    base_a = max(0.05, prior_away_lambda + max(-0.30, min(0.30, prior_adj_away)))

    clock = state.clock_minute if state.clock_minute is not None else 0.0
    if state.period == "1h":
        minutes_played = max(1.0, min(clock, 52.0))
    elif state.period == "2h":
        minutes_played = max(45.0, min(clock, 98.0))
    else:
        minutes_played = max(1.0, clock)

    # Estimated time remaining and empirical-Bayes pressure observations. Possession
    # alone is not attacking-event evidence and must not imply a zero shot rate.
    remaining_minutes = estimate_remaining_minutes(state, period_scope="full").rem_mean

    def has_pressure_stats(team) -> bool:
        return any(
            x is not None
            for x in (team.shots_on_target, team.shots_total, team.corners)
        )

    has_h_stats = has_pressure_stats(state.home)
    has_a_stats = has_pressure_stats(state.away)
    if (has_h_stats or has_a_stats) and minutes_played >= 5.0:
        scale_90 = 90.0 / minutes_played
        h_sot = (state.home.shots_on_target or 0) * scale_90
        a_sot = (state.away.shots_on_target or 0) * scale_90
        h_att = max(h_sot, (state.home.shots_total or state.home.shots_on_target or 0) * scale_90)
        a_att = max(a_sot, (state.away.shots_total or state.away.shots_on_target or 0) * scale_90)
        h_cor = (state.home.corners or 0) * scale_90
        a_cor = (state.away.corners or 0) * scale_90

        # Expected-goal proxy per 90 from live shot/corner creation.
        live_rate_h = 0.24 * h_sot + 0.045 * (h_att - h_sot) + 0.035 * h_cor
        live_rate_a = 0.24 * a_sot + 0.045 * (a_att - a_sot) + 0.035 * a_cor

        # Possession tilt is only meaningful alongside event data.
        if state.home.possession_pct is not None and state.away.possession_pct is not None:
            poss_diff = (state.home.possession_pct - 50.0) / 50.0
            live_rate_h *= 1.0 + 0.08 * poss_diff
            live_rate_a *= 1.0 - 0.08 * poss_diff

        base_weight = minutes_played / (minutes_played + shrinkage_tau_minutes)
        # If SOT is missing, attempts/corners are a noisier proxy; down-weight
        # that team's observations only. A missing side cannot drag the other.
        w_h = (
            base_weight * (1.0 if state.home.shots_on_target is not None else 0.20)
            if has_h_stats else 0.0
        )
        w_a = (
            base_weight * (1.0 if state.away.shots_on_target is not None else 0.20)
            if has_a_stats else 0.0
        )
        post_h = (1.0 - w_h) * base_h + w_h * max(0.10, min(3.80, live_rate_h))
        post_a = (1.0 - w_a) * base_a + w_a * max(0.10, min(3.80, live_rate_a))
        w_stat = 0.5 * (w_h + w_a)
    else:
        w_stat = 0.0
        post_h, post_a = base_h, base_a

    # Markov Red-Card Manpower Multipliers (decayed by remaining playing time)
    h_reds = state.home_reds
    a_reds = state.away_reds
    if h_reds == 0 and a_reds == 0:
        rc_h_full, rc_a_full = 1.0, 1.0
    elif h_reds > 0 and a_reds == 0:
        rc_h_full = 0.68 ** h_reds
        rc_a_full = 1.32 ** min(h_reds, 2)
    elif a_reds > 0 and h_reds == 0:
        rc_h_full = 1.32 ** min(a_reds, 2)
        rc_a_full = 0.68 ** a_reds
    else:
        # Equal dismissals are neutral absent reliable evidence that 10v10
        # raises total scoring. Only the net manpower differential is applied.
        net_adv = a_reds - h_reds
        rc_h_full = (1.28 ** max(0, net_adv)) * (0.72 ** max(0, -net_adv))
        rc_a_full = (1.28 ** max(0, -net_adv)) * (0.72 ** max(0, net_adv))

    # Late cards have little time to affect the remaining-goal expectation.
    decay = max(0.0, min(1.0, remaining_minutes / 30.0))
    rc_h = 1.0 + decay * (rc_h_full - 1.0)
    rc_a = 1.0 + decay * (rc_a_full - 1.0)

    # Score-State Tempo Multipliers (late 2nd half trailing urgency vs leading conservatism)
    ss_h, ss_a = 1.0, 1.0
    if (
        state.home.goals is not None
        and state.away.goals is not None
        and clock >= 60.0
        and remaining_minutes >= 10.0
    ):
        diff = state.home.goals - state.away.goals
        if diff > 0:  # Home leads; away is trailing.
            ss_h, ss_a = 0.94, 1.12
        elif diff < 0:  # Away leads; home is trailing.
            ss_h, ss_a = 1.12, 0.94

    final_h = max(0.05, post_h * rc_h * ss_h)
    final_a = max(0.05, post_a * rc_a * ss_a)

    return PosteriorIntensities(
        lambda_home_90=final_h,
        lambda_away_90=final_a,
        shrinkage_weight=w_stat,
        red_card_multiplier_home=rc_h,
        red_card_multiplier_away=rc_a,
        score_state_multiplier_home=ss_h,
        score_state_multiplier_away=ss_a,
    )


# ---------------------------------------------------------------------------
# 4. Bivariate Poisson / Dixon-Coles Remaining-Score Joint Distribution
# ---------------------------------------------------------------------------


def _poisson_pmf_table(mu: float, max_goals: int) -> list[float]:
    """Compute [P(X=0), ..., P(X=max_goals)] via recurrence in 1 exp + max_goals mults."""
    if mu <= 0.0:
        out = [0.0] * (max_goals + 1)
        out[0] = 1.0
        return out
    p0 = math.exp(-mu)
    out = [p0] * (max_goals + 1)
    cur = p0
    for k in range(1, max_goals + 1):
        cur = cur * mu / k
        out[k] = cur
    return out


def _dixon_coles_tau(i: int, j: int, mu_h: float, mu_a: float, rho: float) -> float:
    """Dixon-Coles (1997) low-scoring correlation adjustment factor tau(i, j)."""
    if i == 0 and j == 0:
        return max(0.1, 1.0 - mu_h * mu_a * rho)
    if i == 0 and j == 1:
        return max(0.1, 1.0 + mu_h * rho)
    if i == 1 and j == 0:
        return max(0.1, 1.0 + mu_a * rho)
    if i == 1 and j == 1:
        return max(0.1, 1.0 - rho)
    return 1.0


def build_remaining_score_matrix(
    mu_home: float,
    mu_away: float,
    rho: float = -0.06,
    shared_tempo_frac: float = 0.05,
    max_goals: int = 10,
) -> list[list[float]]:
    """Build exact (max_goals+1) x (max_goals+1) joint probability matrix P(Delta G_H = i, Delta G_A = j).

    Combines bivariate shared-tempo Poisson shock mu_0 with Dixon-Coles low-score
    dependence adjustment and normalizes strictly to 1.0.
    """
    mu_h = max(1e-7, mu_home)
    mu_a = max(1e-7, mu_away)
    mu_0 = min(mu_h, mu_a) * max(0.0, min(0.25, shared_tempo_frac))
    lam_1 = max(1e-7, mu_h - mu_0)
    lam_2 = max(1e-7, mu_a - mu_0)

    pmf_1 = _poisson_pmf_table(lam_1, max_goals)
    pmf_2 = _poisson_pmf_table(lam_2, max_goals)
    pmf_0 = _poisson_pmf_table(mu_0, max_goals)

    mat: list[list[float]] = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
    total = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m_lim = i if i < j else j
            biv_p = 0.0
            for m in range(m_lim + 1):
                biv_p += pmf_1[i - m] * pmf_2[j - m] * pmf_0[m]
            if i <= 1 and j <= 1:
                val = biv_p * _dixon_coles_tau(i, j, mu_h, mu_a, rho)
            else:
                val = biv_p
            mat[i][j] = val
            total += val

    if total > 0.0:
        inv = 1.0 / total
        for i in range(max_goals + 1):
            row = mat[i]
            for j in range(max_goals + 1):
                row[j] *= inv
    else:
        mat[0][0] = 1.0
    return mat


# ---------------------------------------------------------------------------
# 5. Derivative Surface Pricing Across 14+ SportyBet Market Families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionProbabilityOutcome:
    """Exact settlement probability breakdown for one market selection."""

    p_win: float
    p_push: float = 0.0
    p_half_win: float = 0.0
    p_half_loss: float = 0.0

    @property
    def effective_win_prob(self) -> float:
        """Equivalent win probability for standard display."""
        return self.p_win + 0.5 * self.p_half_win


_HANDICAP_KEY_RE = re.compile(r"^handicap_(\d+)_(\d+)$")
_AH_KEY_RE = re.compile(r"^(?:ah|asian_handicap)_([+-]?\d+(?:[._]\d+)?)$")
_OU_KEY_RE = re.compile(r"^(ou|ou_1h|ou_2h|home_ou|away_ou)_(\d+(?:_\d+)?)$")
_SELECTION_LINE_RE = re.compile(r"\(([+-]?\d+(?:\.\d+)?)\)")
_SELECTION_EURO_HCP_RE = re.compile(r"\((\d+):(\d+)\)")


def _grade_single_asian_line(diff: float, line: float) -> tuple[float, float]:
    """Grade (team_goals - opp_goals + line) on a single whole or half Asian line.

    Returns (win_indicator, push_indicator).
    """
    adj = diff + line
    if abs(adj) < 1e-9:
        return (0.0, 1.0)
    if adj > 0.0:
        return (1.0, 0.0)
    return (0.0, 0.0)


@dataclass(frozen=True)
class MatrixMarginals:
    """1D marginal distributions of P(Delta G_H = i, Delta G_A = j) for O(1)/O(N) derivative pricing."""

    p00: float
    home_pmf: tuple[float, ...]
    away_pmf: tuple[float, ...]
    tot_pmf: tuple[float, ...]
    diff_pmf: dict[int, float]


def compute_matrix_marginals(mat: list[list[float]]) -> MatrixMarginals:
    n = len(mat)
    home_pmf = [0.0] * n
    away_pmf = [0.0] * n
    tot_pmf = [0.0] * (2 * n)
    diff_pmf: dict[int, float] = {}
    for i in range(n):
        row = mat[i]
        h_sum = 0.0
        for j in range(n):
            p = row[j]
            h_sum += p
            away_pmf[j] += p
            tot_pmf[i + j] += p
            d = i - j
            diff_pmf[d] = diff_pmf.get(d, 0.0) + p
        home_pmf[i] = h_sum
    return MatrixMarginals(
        p00=mat[0][0],
        home_pmf=tuple(home_pmf),
        away_pmf=tuple(away_pmf),
        tot_pmf=tuple(tot_pmf),
        diff_pmf=diff_pmf,
    )


def price_from_joint_matrix(
    mat: list[list[float]],
    state: LiveMatchState,
    market_key: str,
    selection: str,
    mu_home: float,
    mu_away: float,
    marginals: Optional[MatrixMarginals] = None,
) -> Optional[SelectionProbabilityOutcome]:
    """Deterministically price `market_key` / `selection` from the remaining-score matrix `mat` in O(N) time."""
    if state.home.goals is None or state.away.goals is None:
        return None

    cur_h = state.home.goals
    cur_a = state.away.goals
    marg = marginals if marginals is not None else compute_matrix_marginals(mat)
    sel = selection.strip()
    mkey = market_key.strip().lower()

    def period_base_scores(scope: str) -> Optional[tuple[int, int]]:
        if scope in ("full", "1h"):
            if scope == "1h" and state.period != "1h":
                return None
            return (cur_h, cur_a)
        if scope == "2h":
            if state.period == "ht":
                # At halftime no second-half goals have occurred yet.
                return (0, 0)
            if state.period != "2h":
                return None
            if state.half_home_goals is None or state.half_away_goals is None:
                return None
            return (cur_h - state.half_home_goals, cur_a - state.half_away_goals)
        if scope == "rest":
            return (0, 0)
        return None

    def outcome_probs_from_base(bh: int, ba: int) -> tuple[float, float, float]:
        base_diff = bh - ba
        p_h = 0.0
        p_d = 0.0
        p_a = 0.0
        for d, p in marg.diff_pmf.items():
            final_d = base_diff + d
            if final_d > 0:
                p_h += p
            elif final_d == 0:
                p_d += p
            else:
                p_a += p
        return p_h, p_d, p_a

    # 1. 1X2 Family (Full Time, 1st Half, 2nd Half, Rest of Match)
    if mkey in (
        "1x2", "1x2_1h", "1x2_2h",
        "rest_of_match", "rest_of_match_1x2", "1x2_rest",
    ):
        scope = (
            "1h"
            if mkey == "1x2_1h"
            else "2h"
            if mkey == "1x2_2h"
            else "rest"
            if "rest" in mkey
            else "full"
        )
        base = period_base_scores(scope)
        if base is None:
            return None
        p_home, p_draw, p_away = outcome_probs_from_base(base[0], base[1])
        if sel in ("Home", "1"):
            return SelectionProbabilityOutcome(p_win=p_home)
        if sel in ("Draw", "X"):
            return SelectionProbabilityOutcome(p_win=p_draw)
        if sel in ("Away", "2"):
            return SelectionProbabilityOutcome(p_win=p_away)
        return None

    # 2. Double Chance Family (Full, 1H, 2H)
    if mkey in (
        "double_chance",
        "dc",
        "1h_double_chance",
        "2h_double_chance",
        "double_chance_1h",
        "double_chance_2h",
    ):
        scope = "1h" if "1h" in mkey else "2h" if "2h" in mkey else "full"
        base = period_base_scores(scope)
        if base is None:
            return None
        p_home, p_draw, p_away = outcome_probs_from_base(base[0], base[1])
        if sel in ("Home or Draw", "1X"):
            return SelectionProbabilityOutcome(p_win=p_home + p_draw)
        if sel in ("Draw or Away", "X2"):
            return SelectionProbabilityOutcome(p_win=p_draw + p_away)
        if sel in ("Home or Away", "12"):
            return SelectionProbabilityOutcome(p_win=p_home + p_away)
        return None

    # 3. Draw No Bet Family (Full, 1H, 2H)
    if mkey in ("dnb", "draw_no_bet", "1h_dnb", "2h_dnb", "dnb_1h", "dnb_2h"):
        scope = "1h" if "1h" in mkey else "2h" if "2h" in mkey else "full"
        base = period_base_scores(scope)
        if base is None:
            return None
        p_home, p_draw, p_away = outcome_probs_from_base(base[0], base[1])
        if sel in ("Home", "1"):
            return SelectionProbabilityOutcome(p_win=p_home, p_push=p_draw)
        if sel in ("Away", "2"):
            return SelectionProbabilityOutcome(p_win=p_away, p_push=p_draw)
        return None

    # 4. Over/Under Goals Family (Full, 1H, 2H, Home Team Total, Away Team Total)
    ou_m = _OU_KEY_RE.match(mkey)
    if ou_m:
        prefix = ou_m.group(1)
        line = float(ou_m.group(2).replace("_", "."))
        scope = "1h" if prefix == "ou_1h" else "2h" if prefix == "ou_2h" else "full"
        base = period_base_scores(scope)
        if base is None:
            return None
        bh, ba = base

        is_over = sel.lower().startswith("over")
        is_under = sel.lower().startswith("under")
        if not (is_over or is_under):
            return None

        if prefix == "home_ou":
            pmf = marg.home_pmf
            base_g = bh
        elif prefix == "away_ou":
            pmf = marg.away_pmf
            base_g = ba
        else:
            pmf = marg.tot_pmf
            base_g = bh + ba

        frac = round(line - math.floor(line), 2)
        lines = [line - 0.25, line + 0.25] if frac in (0.25, 0.75) else [line]

        p_win = 0.0
        p_push = 0.0
        p_hw = 0.0
        p_hl = 0.0

        for k, cell_p in enumerate(pmf):
            tot = float(base_g + k)
            if len(lines) == 1:
                L = lines[0]
                if abs(tot - L) < 1e-9:
                    p_push += cell_p
                elif (is_over and tot > L) or (is_under and tot < L):
                    p_win += cell_p
            else:
                w1 = (
                    1
                    if ((is_over and tot > lines[0]) or (is_under and tot < lines[0]))
                    else (0 if abs(tot - lines[0]) < 1e-9 else -1)
                )
                w2 = (
                    1
                    if ((is_over and tot > lines[1]) or (is_under and tot < lines[1]))
                    else (0 if abs(tot - lines[1]) < 1e-9 else -1)
                )
                if w1 == 1 and w2 == 1:
                    p_win += cell_p
                elif (w1 == 1 and w2 == 0) or (w1 == 0 and w2 == 1):
                    p_hw += cell_p
                elif (w1 == -1 and w2 == 0) or (w1 == 0 and w2 == -1):
                    p_hl += cell_p
                elif w1 == 0 and w2 == 0:
                    p_push += cell_p
        return SelectionProbabilityOutcome(
            p_win=p_win, p_push=p_push, p_half_win=p_hw, p_half_loss=p_hl
        )

    # 5. Both Teams To Score (GG/NG) Family (Full, 1H, 2H) — O(1) Inclusion-Exclusion
    if mkey in ("btts", "gg_ng", "1h_btts", "2h_btts", "btts_1h", "btts_2h"):
        scope = "1h" if "1h" in mkey else "2h" if "2h" in mkey else "full"
        base = period_base_scores(scope)
        if base is None:
            return None
        bh, ba = base
        if bh >= 1 and ba >= 1:
            p_yes = 1.0
        elif bh >= 1 and ba == 0:
            p_yes = max(0.0, 1.0 - marg.away_pmf[0])
        elif bh == 0 and ba >= 1:
            p_yes = max(0.0, 1.0 - marg.home_pmf[0])
        else:
            p_yes = max(
                0.0,
                min(1.0, 1.0 - marg.home_pmf[0] - marg.away_pmf[0] + marg.p00),
            )
        if sel.lower() in ("yes", "gg"):
            return SelectionProbabilityOutcome(p_win=p_yes)
        if sel.lower() in ("no", "ng"):
            return SelectionProbabilityOutcome(p_win=max(0.0, 1.0 - p_yes))
        return None

    # 6. Next Goal Family ('next_goal')
    if mkey == "next_goal":
        p_none = marg.p00
        p_any = max(0.0, 1.0 - p_none)
        tot_mu = mu_home + mu_away
        h_share = (mu_home / tot_mu) if tot_mu > 0.0 else 0.5
        if sel in ("None", "No Goal"):
            return SelectionProbabilityOutcome(p_win=p_none)
        if sel in ("Home", "1"):
            return SelectionProbabilityOutcome(p_win=p_any * h_share)
        if sel in ("Away", "2"):
            return SelectionProbabilityOutcome(p_win=p_any * (1.0 - h_share))
        return None

    # 7. 3-Way European Handicap ('handicap_1_0', 'handicap_0_1', etc.)
    hcp_m = _HANDICAP_KEY_RE.match(mkey)
    if hcp_m:
        h_start = int(hcp_m.group(1))
        a_start = int(hcp_m.group(2))
        p_home, p_draw, p_away = outcome_probs_from_base(
            cur_h + h_start, cur_a + a_start
        )
        if sel.startswith("Home"):
            return SelectionProbabilityOutcome(p_win=p_home)
        if sel.startswith("Draw"):
            return SelectionProbabilityOutcome(p_win=p_draw)
        if sel.startswith("Away"):
            return SelectionProbabilityOutcome(p_win=p_away)
        return None

    # 8. 2-Way Asian Handicap ('ah_0.5', 'ah_-0.25', etc.)
    ah_m = _AH_KEY_RE.match(mkey)
    if ah_m:
        sel_line_m = _SELECTION_LINE_RE.search(sel)
        raw_line = float(ah_m.group(1).replace("_", "."))
        is_home = sel.startswith("Home")
        is_away = sel.startswith("Away")
        if not (is_home or is_away):
            return None
        if sel_line_m:
            team_line = float(sel_line_m.group(1))
        else:
            team_line = raw_line if is_home else -raw_line

        frac = round(abs(team_line) - math.floor(abs(team_line)), 2)
        sub_lines = (
            [team_line - 0.25, team_line + 0.25]
            if frac in (0.25, 0.75)
            else [team_line]
        )

        p_win = 0.0
        p_push = 0.0
        p_hw = 0.0
        p_hl = 0.0
        base_diff = (cur_h - cur_a) if is_home else (cur_a - cur_h)
        for d, cell_p in marg.diff_pmf.items():
            diff = float(base_diff + (d if is_home else -d))
            if len(sub_lines) == 1:
                w, push = _grade_single_asian_line(diff, sub_lines[0])
                p_win += cell_p * w
                p_push += cell_p * push
            else:
                w1, push1 = _grade_single_asian_line(diff, sub_lines[0])
                w2, push2 = _grade_single_asian_line(diff, sub_lines[1])
                if w1 == 1.0 and w2 == 1.0:
                    p_win += cell_p
                elif (w1 == 1.0 and push2 == 1.0) or (push1 == 1.0 and w2 == 1.0):
                    p_hw += cell_p
                elif (w1 == 0.0 and push1 == 0.0 and push2 == 1.0) or (
                    push1 == 1.0 and w2 == 0.0 and push2 == 0.0
                ):
                    p_hl += cell_p
                elif push1 == 1.0 and push2 == 1.0:
                    p_push += cell_p
        return SelectionProbabilityOutcome(
            p_win=p_win, p_push=p_push, p_half_win=p_hw, p_half_loss=p_hl
        )

    return None


# ---------------------------------------------------------------------------
# 6. Expected Value with Push / Quarter-Lines & Uncertainty-Adjusted Kelly
# ---------------------------------------------------------------------------


def expected_value_detailed(
    odds: float,
    p_win: float,
    p_push: float = 0.0,
    p_half_win: float = 0.0,
    p_half_loss: float = 0.0,
) -> float:
    """Exact expected net return per unit staked accounting for pushes and quarter-line splits."""
    if not math.isfinite(odds) or odds < 1.0:
        raise ValueError(f"odds must be finite and >= 1.0, got {odds}")
    probs = (p_win, p_push, p_half_win, p_half_loss)
    if any(not math.isfinite(p) or p < 0.0 for p in probs):
        raise ValueError("outcome probabilities must be finite and non-negative")
    if sum(probs) > 1.0 + 1e-8:
        raise ValueError("outcome probabilities sum to more than one")
    b = odds - 1.0
    p_full_loss = max(0.0, 1.0 - sum(probs))
    return (
        p_win * b
        + p_half_win * (0.5 * b)
        + p_push * 0.0
        - p_half_loss * 0.5
        - p_full_loss * 1.0
    )


def fractional_kelly_stake(
    p_win_conservative: float,
    odds: float,
    p_push: float = 0.0,
    fraction: float = 0.25,
    p_half_win: float = 0.0,
    p_half_loss: float = 0.0,
) -> float:
    """Fractional Kelly bankroll allocation for full/half wins, pushes and losses.

    For ordinary win/push/loss bets this reduces to the closed-form Kelly
    solution. Split Asian outcomes use the exact expected-log-growth optimum,
    found by bisection on the concave derivative.
    """
    if not math.isfinite(odds) or not math.isfinite(fraction):
        raise ValueError("Kelly odds and fraction must be finite")
    if odds <= 1.0 or fraction <= 0.0:
        return 0.0
    probs = (p_win_conservative, p_push, p_half_win, p_half_loss)
    if any(not math.isfinite(p) or p < 0.0 for p in probs):
        raise ValueError("Kelly outcome probabilities must be finite and non-negative")
    if sum(probs) > 1.0 + 1e-8:
        raise ValueError("Kelly outcome probabilities sum to more than one")

    p_full_loss = max(0.0, 1.0 - sum(probs))
    b = odds - 1.0
    returns = (b, 0.0, 0.5 * b, -0.5, -1.0)
    weights = (p_win_conservative, p_push, p_half_win, p_half_loss, p_full_loss)
    expected_return = sum(p * r for p, r in zip(weights, returns))
    if expected_return <= 0.0:
        return 0.0

    # Closed-form solution is exact for a two-outcome win/loss (with pushes
    # conditioned away); use the general optimizer whenever any split outcome
    # is present. The general form also handles pushes transparently.
    if p_half_win == 0.0 and p_half_loss == 0.0:
        denom = b * max(1e-12, 1.0 - p_push)
        full_kelly = (p_win_conservative * b - p_full_loss) / denom
        return max(0.0, min(1.0, fraction * full_kelly))

    def derivative(f: float) -> float:
        total = 0.0
        for p, r in zip(weights, returns):
            if p == 0.0 or r == 0.0:
                continue
            wealth = 1.0 + f * r
            if wealth <= 0.0:
                return -math.inf
            total += p * r / wealth
        return total

    if derivative(1.0) >= 0.0:
        full_kelly = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if derivative(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        full_kelly = 0.5 * (lo + hi)
    return max(0.0, min(1.0, fraction * full_kelly))
