"""Executive Portfolio Manager ("LLM Boss" / CRO Slow-Plane Control Interface).

Architectural Law:
  - Fast Plane (Hot Path): extract -> observe -> math_engine -> estimate -> decisions.
    Executes in < 1 millisecond in pure deterministic math. Zero LLM calls.
  - Slow Plane (This Module): asynchronous portfolio oversight, live operation
    diagnostics, calibration forensics, and asymmetric risk control.

Asymmetric Risk Ratchet ("Ratchet Down, Never Force Up"):
  - The Boss CAN autonomously tighten risk: engage the kill switch, block a match
    or market family, increase `min_ev`, lower `kelly_fraction`, or reduce
    `max_exposure_per_match`.
  - The Boss CAN submit bounded pre-match/halftime prior adjustments
    Delta_lambda in [-0.25, +0.25] with mandatory audit rationale.
  - The Boss CANNOT invent probabilities, bypass the mathematical EV/confidence
    gate, lower `min_ev` below `HARD_MIN_EV_FLOOR`, raise `kelly_fraction` above
    `HARD_MAX_KELLY_FRACTION`, or flip `allow_execution=True`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atlas.live.decisions import DecisionRecord, PolicyConfig
from atlas.live.observe import LiveMatchState
from atlas.models.sport import SportMarket

HARD_MIN_EV_FLOOR = 0.02
HARD_MAX_KELLY_FRACTION = 0.25
HARD_MAX_PRIOR_SHIFT = 0.25
HARD_MAX_EXPOSURE_NGN = 50_000.0


class RiskRatchetViolationError(ValueError):
    """Raised when the Slow-Plane Boss attempts to loosen a hard safety invariant."""


@dataclass
class PriorAdjustment:
    match_id: str
    delta_home_lambda: float
    delta_away_lambda: float
    rationale: str
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class BossControlState:
    kill_switch: bool = False
    kill_reason: str = ""
    blocked_matches: list[str] = field(default_factory=list)
    blocked_markets: list[str] = field(default_factory=list)
    min_ev_override: Optional[float] = None
    min_ev_lower_override: Optional[float] = None
    kelly_fraction_override: Optional[float] = None
    max_exposure_override: Optional[float] = None
    prior_adjustments: dict[str, PriorAdjustment] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)


class BossController:
    """Slow-plane Portfolio Manager & Chief Risk Officer controller."""

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = state_path
        self.state = self._load() if state_path and state_path.exists() else BossControlState()

    def _load(self) -> BossControlState:
        assert self.state_path is not None
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        priors = {
            k: PriorAdjustment(**v)
            for k, v in raw.get("prior_adjustments", {}).items()
        }
        return BossControlState(
            kill_switch=bool(raw.get("kill_switch", False)),
            kill_reason=str(raw.get("kill_reason", "")),
            blocked_matches=list(raw.get("blocked_matches", [])),
            blocked_markets=list(raw.get("blocked_markets", [])),
            min_ev_override=raw.get("min_ev_override"),
            min_ev_lower_override=raw.get("min_ev_lower_override"),
            kelly_fraction_override=raw.get("kelly_fraction_override"),
            max_exposure_override=raw.get("max_exposure_override"),
            prior_adjustments=priors,
            audit_trail=list(raw.get("audit_trail", [])),
        )

    def save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.state)
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        self.state.audit_trail.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                **details,
            }
        )
        self.save()

    def engage_kill_switch(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("kill switch requires an explicit reason")
        self.state.kill_switch = True
        self.state.kill_reason = reason.strip()
        self._log_action("engage_kill_switch", {"reason": self.state.kill_reason})

    def block_market(self, market_key: str, reason: str) -> None:
        mkey = market_key.strip()
        if mkey and mkey not in self.state.blocked_markets:
            self.state.blocked_markets.append(mkey)
        self._log_action("block_market", {"market_key": mkey, "reason": reason})

    def block_match(self, match_id: str, reason: str) -> None:
        mid = match_id.strip()
        if mid and mid not in self.state.blocked_matches:
            self.state.blocked_matches.append(mid)
        self._log_action("block_match", {"match_id": mid, "reason": reason})

    def set_risk_overrides(
        self,
        *,
        min_ev: Optional[float] = None,
        min_ev_lower: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
        max_exposure_ngn: Optional[float] = None,
        allow_execution: Optional[bool] = None,
        rationale: str = "",
    ) -> None:
        """Update risk parameters subject to the Asymmetric Ratchet Law."""
        if allow_execution is True:
            raise RiskRatchetViolationError(
                "Boss cannot autonomously enable live execution (allow_execution=True requires human operator config)"
            )
        if min_ev is not None:
            if min_ev < HARD_MIN_EV_FLOOR:
                raise RiskRatchetViolationError(
                    f"min_ev {min_ev} violates hard safety floor {HARD_MIN_EV_FLOOR}"
                )
            self.state.min_ev_override = float(min_ev)
        if min_ev_lower is not None:
            self.state.min_ev_lower_override = float(min_ev_lower)
        if kelly_fraction is not None:
            if kelly_fraction < 0.0 or kelly_fraction > HARD_MAX_KELLY_FRACTION:
                raise RiskRatchetViolationError(
                    f"kelly_fraction {kelly_fraction} violates [0.0, {HARD_MAX_KELLY_FRACTION}] cap"
                )
            self.state.kelly_fraction_override = float(kelly_fraction)
        if max_exposure_ngn is not None:
            if max_exposure_ngn <= 0.0 or max_exposure_ngn > HARD_MAX_EXPOSURE_NGN:
                raise RiskRatchetViolationError(
                    f"max_exposure_ngn {max_exposure_ngn} exceeds hard ceiling {HARD_MAX_EXPOSURE_NGN}"
                )
            self.state.max_exposure_override = float(max_exposure_ngn)

        self._log_action(
            "set_risk_overrides",
            {
                "min_ev": self.state.min_ev_override,
                "min_ev_lower": self.state.min_ev_lower_override,
                "kelly_fraction": self.state.kelly_fraction_override,
                "max_exposure_ngn": self.state.max_exposure_override,
                "rationale": rationale,
            },
        )

    def set_prior_adjustment(
        self,
        match_id: str,
        delta_home_lambda: float,
        delta_away_lambda: float,
        rationale: str,
    ) -> PriorAdjustment:
        """Register bounded qualitative prior adjustment for a match."""
        if not rationale.strip():
            raise ValueError("Prior adjustment requires an explicit rationale")
        if (
            abs(delta_home_lambda) > HARD_MAX_PRIOR_SHIFT
            or abs(delta_away_lambda) > HARD_MAX_PRIOR_SHIFT
        ):
            raise RiskRatchetViolationError(
                f"Prior shift ({delta_home_lambda}, {delta_away_lambda}) exceeds [-{HARD_MAX_PRIOR_SHIFT}, +{HARD_MAX_PRIOR_SHIFT}]"
            )
        adj = PriorAdjustment(
            match_id=match_id,
            delta_home_lambda=float(delta_home_lambda),
            delta_away_lambda=float(delta_away_lambda),
            rationale=rationale.strip(),
        )
        self.state.prior_adjustments[match_id] = adj
        self._log_action("set_prior_adjustment", asdict(adj))
        return adj

    def apply_to_policy(self, base: PolicyConfig) -> PolicyConfig:
        """Apply Slow-Plane ratchets onto a Fast-Plane PolicyConfig.

        Ratchet guarantee: overrides can only tighten `min_ev` (max), tighten
        `kelly_fraction` (min), and tighten `max_exposure_per_match` (min).
        """
        eff_min_ev = (
            max(base.min_ev, self.state.min_ev_override)
            if self.state.min_ev_override is not None
            else base.min_ev
        )
        eff_min_ev_lower = (
            max(base.min_ev_lower or -1.0, self.state.min_ev_lower_override)
            if self.state.min_ev_lower_override is not None
            else base.min_ev_lower
        )
        eff_kelly = (
            min(base.kelly_fraction, self.state.kelly_fraction_override)
            if self.state.kelly_fraction_override is not None
            else base.kelly_fraction
        )
        eff_exposure = (
            min(base.max_exposure_per_match, self.state.max_exposure_override)
            if self.state.max_exposure_override is not None
            else base.max_exposure_per_match
        )
        return PolicyConfig(
            min_ev=eff_min_ev,
            min_ev_lower=eff_min_ev_lower,
            max_bets_per_match=base.max_bets_per_match,
            max_correlated_in_group=base.max_correlated_in_group,
            enforce_correlation_groups=base.enforce_correlation_groups,
            currency=base.currency,
            min_stake_ngn=base.min_stake_ngn,
            max_stake_per_bet=base.max_stake_per_bet,
            max_exposure_per_match=eff_exposure,
            max_payout_per_bet=base.max_payout_per_bet,
            bankroll_ngn=base.bankroll_ngn,
            kelly_fraction=eff_kelly,
            auto_size_kelly=base.auto_size_kelly,
            max_evidence_age_s=base.max_evidence_age_s,
            kill_switch=base.kill_switch or self.state.kill_switch,
            blocked_matches=set(base.blocked_matches) | set(self.state.blocked_matches),
            blocked_markets=set(base.blocked_markets) | set(self.state.blocked_markets),
            allow_execution=base.allow_execution,
        )


def diagnose_live_operation(
    state: LiveMatchState,
    open_markets: list[SportMarket],
    skipped_markets: list[dict[str, Any]],
    decisions: list[DecisionRecord],
    implications: list[dict[str, Any]],
    fast_plane_latency_us: float,
    as_of: Optional[datetime] = None,
) -> dict[str, Any]:
    """Generate structured JSON executive telemetry for the Portfolio Manager / MCP."""
    age_s = state.age_seconds(as_of)
    violations = [
        {
            "antecedent": c["rule"].antecedent,
            "consequent": c["rule"].consequent,
            "p_antecedent": round(c["p_antecedent"], 4),
            "p_consequent": round(c["p_consequent"], 4),
        }
        for c in implications
        if c.get("status") == "VIOLATION"
    ]
    market_microstructure = [
        {
            "market_key": m.market_key,
            "overround": round(m.overround, 4),
            "shin_adverse_selection_z": round(m.shin_z, 4),
        }
        for m in open_markets
    ]
    recommended = [d for d in decisions if d.decision == "recommend"]
    anomalies: list[str] = []
    if age_s > 60.0:
        anomalies.append(f"stale_feed ({age_s:.1f}s)")
    if state.remaining_unknown:
        anomalies.append("stoppage_duration_unverified")
    if len(open_markets) == 0:
        anomalies.append("all_markets_suspended")
    if violations:
        anomalies.append(f"cross_market_implication_violations ({len(violations)})")
    for mm in market_microstructure:
        if mm["overround"] > 1.12:
            anomalies.append(
                f"high_bookmaker_overround ({mm['market_key']}={mm['overround']:.3f})"
            )

    return {
        "match_id": state.match_id,
        "period": state.period,
        "clock_label": state.clock_label,
        "score": [state.home.goals, state.away.goals],
        "red_cards": [state.home_reds, state.away_reds],
        "evidence_age_s": round(age_s, 2),
        "fast_plane_latency_us": round(fast_plane_latency_us, 1),
        "fast_plane_sub_millisecond": fast_plane_latency_us < 1000.0,
        "markets_open_count": len(open_markets),
        "markets_suspended_count": len(skipped_markets),
        "market_microstructure": market_microstructure,
        "implication_violations": violations,
        "decisions_total": len(decisions),
        "decisions_recommended": len(recommended),
        "operational_anomalies": anomalies,
    }
