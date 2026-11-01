"""Decision log + SportyBet Nigeria (NGN) betting policy & risk gate.

V1 policy: recommendations are computed, but execution stays OFF by default.
`decide()` returns 'pass' unless ALL of these hold:
  - kill_switch is False and match/market is not blocked,
  - evidence fresh, estimate not abstained, confidence medium or high,
  - EV >= min_ev (and conservative EV_lower >= min_ev_lower when configured),
  - market open with a verified price,
  - selection not already taken (no duplicates),
  - factor-correlation cap per match respected,
  - SportyBet Nigeria (NGN) stake bounds, payout cap, and match exposure caps respected.

Every call appends a JSONL record so calibration (§8) can later ask:
did value survive uncertainty and execution?
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from atlas.live.calibration import CalibrationMap
from atlas.live.estimate import Estimate
from atlas.live.loop import expected_value, selection_correlation_groups
from atlas.live.math_engine import fractional_kelly_stake

BETTABLE_CONFIDENCE = {"medium", "high"}


@dataclass
class PolicyConfig:
    min_ev: float = 0.05
    # The absurd-EV ceiling (2026-11-01 Perth session): see
    # SequentialConfig.max_ev — one-sided live quotes never hand out +25%.
    max_ev: float = 0.25
    min_ev_lower: Optional[float] = None
    max_bets_per_match: int = 3
    max_correlated_in_group: int = 1
    enforce_correlation_groups: bool = True
    currency: str = "NGN"
    min_stake_ngn: float = 100.0
    max_stake_per_bet: float = 1000.0
    max_exposure_per_match: float = 3000.0
    max_payout_per_bet: float = 50_000_000.0
    bankroll_ngn: float = 10_000.0
    kelly_fraction: float = 0.25
    auto_size_kelly: bool = False
    max_evidence_age_s: float = 60.0
    kill_switch: bool = False
    blocked_matches: set[str] = field(default_factory=set)
    blocked_markets: set[str] = field(default_factory=set)
    # Long-odds gate: at high decimal odds the model's probability error
    # dominates the claimed edge (a 1% miss at odds 10 is the whole bet).
    # The NGN 100 loss on 2026-10-28 (Over 1.5 @ 10.0) is the case study.
    # Raise only when the calibration record proves itself at length.
    max_bet_odds: float = 3.0
    # Probability ceiling: no sports outcome is ever 100% certain. p >= cap
    # means the model is miscalibrated or broken — abstain rather than bet.
    # The Perth session saw p=1.0 claims on @13.50 and @10.00 markets that
    # both lost. 0.95 = max 5% residual uncertainty.
    max_p_cap: float = 0.95
    # Market-family stop-loss: freeze a family after N consecutive losses.
    # Prevents the 'ou_1_5 @10.00 stake=100 -> -100' pathology from
    # repeating. 0 = disabled.
    family_stop_loss_n: int = 3
    allow_execution: bool = False  # V1: observe + log only. No bet placement.


def compute_recommended_stake_ngn(
    est: Estimate, odds: float, policy: PolicyConfig, current_match_exposure: float = 0.0
) -> float:
    """Compute whole-Naira fractional Kelly stake on conservative p_lower, subject to SportyBet NG caps."""
    p_cons = est.p_lower if est.p_lower is not None else (est.p or 0.0)
    f_star = fractional_kelly_stake(
        p_win_conservative=p_cons,
        odds=odds,
        p_push=est.p_push,
        fraction=policy.kelly_fraction,
    )
    if f_star <= 0.0:
        return 0.0
    # Quantise the fraction before flooring to the Naira. Kelly arithmetic is
    # float-noisy (e.g. p=0.6,o=2.0 yields edge=0.19999999999999996), so an
    # unguarded floor silently under-stakes by a full unit on exact edges.
    f_star = round(f_star, 6)
    raw_ngn = math.floor(policy.bankroll_ngn * f_star + 1e-9)
    remaining_exp = max(0.0, policy.max_exposure_per_match - current_match_exposure)
    max_by_payout = policy.max_payout_per_bet / max(1.0, odds)

    # FLAT SIZING MODE (operator decision 2026-10-28): when min == max (the
    # NGN 10 floor == the NGN 10 cap), the stake is the operator's flat
    # stake whenever the conservative Kelly edge is positive. Proportional
    # Kelly on a 50-NGN balance computes 1-4 NGN stakes — mathematically
    # honest ('no edge justifies 19% of bankroll') but it refuted 1,133
    # recommendations in one afternoon while the operator explicitly set
    # NGN 10 both as floor AND cap: min==max IS the flat regime. Risk
    # control stays in the pass gates (EV floor, odds cap, exposure caps);
    # balance-proportional sizing resumes the moment max > min.
    if (
        abs(policy.min_stake_ngn - policy.max_stake_per_bet) < 1e-9
        and raw_ngn > 0
    ):
        flat = min(
            float(policy.min_stake_ngn), remaining_exp, max_by_payout
        )
        if flat >= policy.min_stake_ngn:
            return float(math.floor(flat))
        return 0.0

    capped = min(
        float(raw_ngn),
        policy.max_stake_per_bet,
        remaining_exp,
        max_by_payout,
    )
    if capped < policy.min_stake_ngn:
        return 0.0
    return float(math.floor(capped))


@dataclass
class DecisionRecord:
    match_id: str
    market_key: str
    selection: str
    odds: float
    model_p: Optional[float]
    ev: Optional[float]
    confidence: str
    decision: str  # 'pass' | 'recommend'
    reason: str
    stake: float = 0.0
    currency: str = "NGN"
    p_lower: Optional[float] = None
    p_upper: Optional[float] = None
    ev_lower: Optional[float] = None
    calibrated_p: Optional[float] = None
    correlation_groups: list[str] = field(default_factory=list)
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    executed: bool = False  # verified acceptance only; never set by decide()


@dataclass
class DecisionLog:
    path: Path
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    calibration: CalibrationMap = field(default_factory=CalibrationMap.load)
    _taken: set = field(default_factory=set, repr=False)
    _exposure: dict = field(default_factory=dict, repr=False)
    _group_counts: dict = field(default_factory=dict, repr=False)
    _family_loss_count: dict = field(default_factory=dict, repr=False)
    _family_win_count: dict = field(default_factory=dict, repr=False)

    def decide(
        self,
        match_id: str,
        market_key: str,
        selection: str,
        odds: float,
        status: str,
        est: Estimate,
        stake: float = 0.0,
    ) -> DecisionRecord:
        key = (match_id, market_key, selection)
        groups = selection_correlation_groups(market_key, selection)

        if self.policy.kill_switch:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "kill switch engaged by risk controller",
                groups=groups,
            )
        if match_id in self.policy.blocked_matches or market_key in self.policy.blocked_markets:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "match or market blocked by risk controller",
                groups=groups,
            )
        if status != "open":
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"market not open ({status})",
                groups=groups,
            )
        if est.abstained or est.p is None:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"abstained: {est.abstain_reason}",
                groups=groups,
            )
        if (
            est.evidence_age_s is not None
            and est.evidence_age_s > self.policy.max_evidence_age_s
        ):
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "stale evidence",
                groups=groups,
            )

        # Isotonic calibration: raw model_p -> calibrated actual win rate,
        # learned from settled history. Applied before EV computation so the
        # gates see calibrated probabilities, not raw model output.
        if est.p is not None:
            cal_p = self.calibration.calibrate(market_key, est.p)
            est.apply_calibration(cal_p)

        ev = (
            est.expected_value(odds)
            if (est.p_push > 0.0 or est.p_half_win > 0.0 or est.p_half_loss > 0.0)
            else expected_value(est.p_for_ev, odds)
        )
        ev_low = est.conservative_expected_value(odds)

        if est.confidence not in BETTABLE_CONFIDENCE:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"confidence {est.confidence} below betting bar (EV={ev:.3f})",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        if ev is None or ev < self.policy.min_ev:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"EV {(ev or 0.0):.3f} < min {self.policy.min_ev}",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        # The absurd-EV ceiling: symmetric with the portfolio gate. A claim
        # above this is a mislabeled model bug, not a price (2026-11-01
        # Perth: 'Over 3.5' claimed +46% three re-stakes in a row).
        if ev > self.policy.max_ev:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"claimed EV {ev:.3f} > max {self.policy.max_ev:.2f} "
                "(uncalibrated claim, not a price)",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        if (
            self.policy.min_ev_lower is not None
            and ev_low is not None
            and ev_low < self.policy.min_ev_lower
        ):
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"conservative EV_lower {ev_low:.3f} < min {self.policy.min_ev_lower}",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        # --- Long-odds gate: max_bet_odds ---
        # The NGN 100 loss on 2026-10-28 (Over 1.5 @ 10.0) proved that
        # high-odds markets magnify model probability error. A 1% miss at
        # odds 10 is the entire bet. Enforce hard ceiling.
        if odds > self.policy.max_bet_odds:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"odds {odds:.2f} > max_bet_odds {self.policy.max_bet_odds:.2f} "
                "(long-odds gate)",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        # --- Probability ceiling: max_p_cap ---
        # No sports outcome is ever 100% certain. p >= max_p_cap means the
        # model is miscalibrated or broken. The Perth session saw p=1.0
        # claims on @13.50 and @10.00 markets that both lost.
        if est.p is not None and est.p >= self.policy.max_p_cap:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"model_p {est.p:.3f} >= max_p_cap {self.policy.max_p_cap:.2f} "
                "(probability ceiling)",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        # --- Test/debug probe filter ---
        # The 'UNKNOWN chimera probe' entry on 2026-10-28 (@13.50, stake=10,
        # p=1.0) was debug code that leaked to production. Reject any
        # selection whose name looks like a probe or placeholder.
        if selection and (
            "chimera" in selection.lower()
            or "probe" in selection.lower()
            or selection.strip().upper() in {"UNKNOWN", "TEST", "DEBUG", "PLACEHOLDER"}
        ):
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"selection '{selection}' looks like a test/debug probe — refused",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        # --- Market-family stop-loss ---
        # Freeze a market family after N consecutive settled losses. This
        # stops the 'ou_1_5 @10.00 stake=100 -> -100' pathology from
        # repeating. Family = first underscore segment (e.g. 'ou_3_5' -> 'ou').
        family = market_key.split("_")[0] if "_" in market_key else market_key
        family_losses = self._family_loss_count.get(family, 0)
        if family_losses >= self.policy.family_stop_loss_n:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"family '{family}' stop-loss engaged ({family_losses} "
                f"consecutive settled losses, cap={self.policy.family_stop_loss_n})",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        if key in self._taken:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "duplicate selection rejected",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        exposure = self._exposure.get(match_id, 0.0)
        bets = sum(1 for k in self._taken if k[0] == match_id)
        if bets >= self.policy.max_bets_per_match:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "max bets per match reached",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        if self.policy.enforce_correlation_groups and groups:
            match_groups = self._group_counts.get(match_id, {})
            blocked_group = next(
                (
                    g
                    for g in groups
                    if match_groups.get(g, 0) >= self.policy.max_correlated_in_group
                ),
                None,
            )
            if blocked_group is not None:
                return self._record(
                    match_id,
                    market_key,
                    selection,
                    odds,
                    est,
                    "pass",
                    f"correlated with accepted bet in '{blocked_group}'",
                    ev=ev,
                    ev_lower=ev_low,
                    groups=groups,
                )

        effective_stake = stake
        if effective_stake <= 0.0 and self.policy.auto_size_kelly:
            effective_stake = compute_recommended_stake_ngn(
                est, odds, self.policy, current_match_exposure=exposure
            )
            if effective_stake < self.policy.min_stake_ngn:
                return self._record(
                    match_id,
                    market_key,
                    selection,
                    odds,
                    est,
                    "pass",
                    f"Kelly stake below SportyBet NG min {self.policy.min_stake_ngn:.0f} {self.policy.currency}",
                    ev=ev,
                    ev_lower=ev_low,
                    groups=groups,
                )

        if effective_stake > self.policy.max_stake_per_bet:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"stake {effective_stake:.0f} exceeds max_stake_per_bet {self.policy.max_stake_per_bet:.0f}",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        if exposure + effective_stake > self.policy.max_exposure_per_match:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "exposure cap would be exceeded",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        if effective_stake * odds > self.policy.max_payout_per_bet:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                "payout cap per bet would be exceeded",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )
        if not self.policy.allow_execution:
            return self._record(
                match_id,
                market_key,
                selection,
                odds,
                est,
                "pass",
                f"EV {ev:.3f} passes bar but execution disabled (V1 observe-only)",
                ev=ev,
                ev_lower=ev_low,
                groups=groups,
            )

        self._taken.add(key)
        self._exposure[match_id] = exposure + effective_stake
        match_groups = self._group_counts.setdefault(match_id, {})
        for g in groups:
            match_groups[g] = match_groups.get(g, 0) + 1
        return self._record(
            match_id,
            market_key,
            selection,
            odds,
            est,
            "recommend",
            f"EV {ev:.3f} >= {self.policy.min_ev}",
            ev=ev,
            ev_lower=ev_low,
            stake=effective_stake,
            groups=groups,
        )

    def _record(
        self,
        match_id,
        market_key,
        selection,
        odds,
        est,
        decision,
        reason,
        ev=None,
        ev_lower=None,
        stake=0.0,
        groups=None,
    ) -> DecisionRecord:
        rec = DecisionRecord(
            match_id=match_id,
            market_key=market_key,
            selection=selection,
            odds=odds,
            model_p=est.p,
            calibrated_p=est.calibrated_p,
            ev=ev,
            confidence=est.confidence,
            decision=decision,
            reason=reason,
            stake=stake,
            currency=self.policy.currency,
            p_lower=est.p_lower,
            p_upper=est.p_upper,
            ev_lower=ev_lower,
            correlation_groups=groups or [],
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        return rec

    def register_settlement(
        self,
        market_key: str,
        pnl_ngn: float,
        won: bool,
    ) -> None:
        """Record a settled bet outcome for family stop-loss tracking.

        Called by the settlement recon after a bet settles on the book.
        Tracks consecutive wins/losses per market family (first underscore
        segment of market_key, e.g. 'ou_3_5' -> 'ou'). When a family hits
        family_stop_loss_n consecutive losses, all future bets in that
        family are refused until a win resets the counter.

        Family counters reset on a win: a single win breaks the losing
        streak and re-arms the family.
        """
        family = market_key.split("_")[0] if "_" in market_key else market_key
        if won:
            self._family_loss_count[family] = 0
            self._family_win_count[family] = (
                self._family_win_count.get(family, 0) + 1
            )
        else:
            self._family_loss_count[family] = (
                self._family_loss_count.get(family, 0) + 1
            )
            self._family_win_count[family] = 0
