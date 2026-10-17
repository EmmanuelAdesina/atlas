"""Post-match calibration, Brier / Log-Loss scoring, and edge attribution (§8).

Evaluates recorded DecisionRecords against verified settled outcomes:
  - Probability calibration (Brier score, log-loss, mean predicted vs realized win rate),
  - Realized returns on recommended/executed stakes vs paper counterfactuals,
  - Performance breakdown by market family and confidence grade,
  - Whether apparent EV survived model uncertainty (EV vs EV_lower vs realized return).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradedDecision:
    match_id: str
    market_key: str
    selection: str
    odds: float
    model_p: float
    p_lower: float
    ev: float
    ev_lower: float
    confidence: str
    decision: str
    stake: float
    won: bool
    brier: float
    log_loss: float
    realized_unit_pnl: float
    realized_stake_pnl: float


@dataclass
class CalibrationSummary:
    total_records: int
    priced_records: int
    settled_records: int
    recommended_count: int
    mean_predicted_p: float
    realized_win_rate: float
    calibration_gap: float
    brier_score: float
    log_loss: float
    mean_apparent_ev: float
    mean_conservative_ev: float
    realized_unit_roi: float
    realized_stake_pnl: float
    by_market_family: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_confidence: dict[str, dict[str, Any]] = field(default_factory=dict)


def grade_decisions(
    records: list[dict[str, Any]],
    settled_outcomes: dict[tuple[str, str], str] | dict[str, str],
    default_match_id: str = "",
) -> tuple[list[GradedDecision], CalibrationSummary]:
    """Grade decision records against settled winning selection names.

    `settled_outcomes` may map either `(match_id, market_key) -> winning_selection`
    or `market_key -> winning_selection` (for a single match).
    """
    graded: list[GradedDecision] = []
    eps = 1e-6

    for rec in records:
        p = rec.get("model_p")
        if p is None:
            continue
        mid = str(rec.get("match_id") or default_match_id)
        mkey = str(rec.get("market_key", ""))
        sel = str(rec.get("selection", ""))

        winner = None
        if (mid, mkey) in settled_outcomes:
            winner = settled_outcomes[(mid, mkey)]  # type: ignore[index]
        elif mkey in settled_outcomes:
            winner = settled_outcomes[mkey]  # type: ignore[index]
        if winner is None:
            continue

        won = sel == winner
        y = 1.0 if won else 0.0
        p_clamped = min(1.0 - eps, max(eps, float(p)))
        p_low = float(rec.get("p_lower")) if rec.get("p_lower") is not None else float(p)
        odds = float(rec.get("odds", 1.0))
        ev = float(rec.get("ev")) if rec.get("ev") is not None else (float(p) * odds - 1.0)
        ev_low = (
            float(rec.get("ev_lower"))
            if rec.get("ev_lower") is not None
            else (p_low * odds - 1.0)
        )
        stake = float(rec.get("stake", 0.0))
        decision = str(rec.get("decision", "pass"))
        conf = str(rec.get("confidence", "low"))

        brier = (float(p) - y) ** 2
        ll = -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
        unit_pnl = (odds - 1.0) if won else -1.0
        stake_pnl = stake * unit_pnl if decision == "recommend" else 0.0

        graded.append(
            GradedDecision(
                match_id=mid,
                market_key=mkey,
                selection=sel,
                odds=odds,
                model_p=float(p),
                p_lower=p_low,
                ev=ev,
                ev_lower=ev_low,
                confidence=conf,
                decision=decision,
                stake=stake,
                won=won,
                brier=brier,
                log_loss=ll,
                realized_unit_pnl=unit_pnl,
                realized_stake_pnl=stake_pnl,
            )
        )

    priced_count = sum(1 for r in records if r.get("model_p") is not None)
    if not graded:
        return [], CalibrationSummary(
            total_records=len(records),
            priced_records=priced_count,
            settled_records=0,
            recommended_count=0,
            mean_predicted_p=0.0,
            realized_win_rate=0.0,
            calibration_gap=0.0,
            brier_score=0.0,
            log_loss=0.0,
            mean_apparent_ev=0.0,
            mean_conservative_ev=0.0,
            realized_unit_roi=0.0,
            realized_stake_pnl=0.0,
        )

    n = len(graded)
    mean_p = sum(g.model_p for g in graded) / n
    win_rate = sum(1.0 if g.won else 0.0 for g in graded) / n
    brier_avg = sum(g.brier for g in graded) / n
    ll_avg = sum(g.log_loss for g in graded) / n
    ev_avg = sum(g.ev for g in graded) / n
    ev_low_avg = sum(g.ev_lower for g in graded) / n
    unit_roi = sum(g.realized_unit_pnl for g in graded) / n
    stake_pnl_total = sum(g.realized_stake_pnl for g in graded)
    rec_count = sum(1 for g in graded if g.decision == "recommend")

    by_mkt: dict[str, list[GradedDecision]] = {}
    by_conf: dict[str, list[GradedDecision]] = {}
    for g in graded:
        by_mkt.setdefault(g.market_key, []).append(g)
        by_conf.setdefault(g.confidence, []).append(g)

    def summarize_bucket(items: list[GradedDecision]) -> dict[str, Any]:
        k = len(items)
        return {
            "count": k,
            "mean_p": round(sum(x.model_p for x in items) / k, 4),
            "win_rate": round(sum(1.0 if x.won else 0.0 for x in items) / k, 4),
            "brier": round(sum(x.brier for x in items) / k, 4),
            "mean_ev": round(sum(x.ev for x in items) / k, 4),
            "mean_ev_lower": round(sum(x.ev_lower for x in items) / k, 4),
            "unit_roi": round(sum(x.realized_unit_pnl for x in items) / k, 4),
        }

    summary = CalibrationSummary(
        total_records=len(records),
        priced_records=priced_count,
        settled_records=n,
        recommended_count=rec_count,
        mean_predicted_p=round(mean_p, 4),
        realized_win_rate=round(win_rate, 4),
        calibration_gap=round(mean_p - win_rate, 4),
        brier_score=round(brier_avg, 4),
        log_loss=round(ll_avg, 4),
        mean_apparent_ev=round(ev_avg, 4),
        mean_conservative_ev=round(ev_low_avg, 4),
        realized_unit_roi=round(unit_roi, 4),
        realized_stake_pnl=round(stake_pnl_total, 2),
        by_market_family={k: summarize_bucket(v) for k, v in sorted(by_mkt.items())},
        by_confidence={k: summarize_bucket(v) for k, v in sorted(by_conf.items())},
    )
    return graded, summary


def load_decision_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
