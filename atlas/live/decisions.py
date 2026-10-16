"""Decision log + betting policy gate (observation-first).

V1 policy: recommendations are computed, but execution stays OFF.
`decide()` returns 'pass' unless ALL of these hold:
  - evidence fresh, estimate not abstained, confidence medium or high,
  - EV >= min_ev, market open with a verified price,
  - exposure caps respected, selection not already taken (no duplicates).

Every call appends a JSONL record so calibration (§8) can later ask:
did value survive uncertainty and execution?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from atlas.live.estimate import Estimate
from atlas.live.loop import expected_value

BETTABLE_CONFIDENCE = {"medium", "high"}


@dataclass
class PolicyConfig:
    min_ev: float = 0.05
    max_bets_per_match: int = 3
    max_stake_per_bet: float = 1000.0
    max_exposure_per_match: float = 3000.0
    max_evidence_age_s: float = 60.0
    allow_execution: bool = False  # V1: observe + log only. No bet placement.


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
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    executed: bool = False  # verified acceptance only; never set by decide()


@dataclass
class DecisionLog:
    path: Path
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    _taken: set = field(default_factory=set, repr=False)
    _exposure: dict = field(default_factory=dict, repr=False)

    def decide(self, match_id: str, market_key: str, selection: str,
               odds: float, status: str, est: Estimate,
               stake: float = 0.0) -> DecisionRecord:
        key = (match_id, market_key, selection)
        if status != "open":
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", f"market not open ({status})")
        if est.abstained or est.p is None:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", f"abstained: {est.abstain_reason}")
        if (est.evidence_age_s is not None
                and est.evidence_age_s > self.policy.max_evidence_age_s):
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", "stale evidence")
        ev = expected_value(est.p, odds)
        if est.confidence not in BETTABLE_CONFIDENCE:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", f"confidence {est.confidence} below betting bar (EV={ev:.3f})", ev=ev)
        if ev < self.policy.min_ev:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", f"EV {ev:.3f} < min {self.policy.min_ev}", ev=ev)
        if key in self._taken:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", "duplicate selection rejected", ev=ev)
        exposure = self._exposure.get(match_id, 0.0)
        bets = sum(1 for k in self._taken if k[0] == match_id)
        if bets >= self.policy.max_bets_per_match:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", "max bets per match reached", ev=ev)
        if exposure + stake > self.policy.max_exposure_per_match:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", "exposure cap would be exceeded", ev=ev)
        if not self.policy.allow_execution:
            return self._record(match_id, market_key, selection, odds, est,
                                "pass", f"EV {ev:.3f} passes bar but execution disabled (V1 observe-only)", ev=ev)
        self._taken.add(key)
        self._exposure[match_id] = exposure + stake
        return self._record(match_id, market_key, selection, odds, est,
                            "recommend", f"EV {ev:.3f} >= {self.policy.min_ev}", ev=ev, stake=stake)

    def _record(self, match_id, market_key, selection, odds, est,
                decision, reason, ev=None, stake=0.0) -> DecisionRecord:
        rec = DecisionRecord(match_id, market_key, selection, odds,
                             est.p, ev, est.confidence, decision, reason, stake)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        return rec
