"""Post-match calibration, settlement grading, and return attribution (§8).

Scores each decision using its full settlement distribution (win, half-win,
push, half-loss, loss). Explicit selection-level settlement statuses are
required to distinguish split Asian outcomes; a simple market winner string
continues to work for ordinary markets. Complete 1X2 sets additionally receive
proper categorical Brier and log-loss scores.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from atlas.live.math_engine import expected_value_detailed


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
    executed: bool
    won: bool
    brier: float
    log_loss: float
    realized_unit_pnl: float
    realized_stake_pnl: float
    settlement: str = "loss"
    p_push: float = 0.0
    p_half_win: float = 0.0
    p_half_loss: float = 0.0
    p_full_loss: float = 0.0
    paper_unit_pnl: float = 0.0
    settled_winner: Optional[str] = None


@dataclass
class CalibrationSummary:
    total_records: int
    priced_records: int
    settled_records: int
    recommended_count: int
    executed_count: int
    mean_predicted_p: float
    realized_win_rate: float
    calibration_gap: float
    brier_score: float
    log_loss: float
    mean_apparent_ev: float
    mean_conservative_ev: float
    realized_unit_roi: float
    realized_stake_pnl: float
    paper_unit_roi: float = 0.0
    recommended_unit_roi: float = 0.0
    recommended_stake_pnl: float = 0.0
    multiclass_market_count: int = 0
    multiclass_brier_score: float = 0.0
    multiclass_log_loss: float = 0.0
    by_market_family: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_confidence: dict[str, dict[str, Any]] = field(default_factory=dict)


_SETTLEMENT_ALIASES = {
    "win": "win",
    "won": "win",
    "fullwin": "win",
    "loss": "loss",
    "lost": "loss",
    "fullloss": "loss",
    "push": "push",
    "void": "push",
    "refunded": "push",
    "halfwin": "half_win",
    "halfwon": "half_win",
    "halfloss": "half_loss",
    "halflost": "half_loss",
}
_SETTLEMENT_CATEGORIES = ("win", "half_win", "push", "half_loss", "loss")
_DNB_KEYS = {"dnb", "draw_no_bet", "1h_dnb", "2h_dnb", "dnb_1h", "dnb_2h"}
_1X2_KEYS = {"1x2", "1x2_1h", "1x2_2h", "rest_of_match", "rest_of_match_1x2"}


def _norm_token(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _normalize_settlement(value: Any) -> str:
    key = _norm_token(value)
    status = _SETTLEMENT_ALIASES.get(key)
    if status is None:
        raise ValueError(
            f"unknown settlement {value!r}; expected win/loss/push/half_win/half_loss"
        )
    return status


def _settlement_for_record(value: Any, market_key: str, selection: str) -> tuple[str, Optional[str]]:
    """Resolve a legacy market winner or an explicit selection settlement."""
    if isinstance(value, Mapping):
        selection_map = value.get("selection_outcomes", value.get("selections"))
        if isinstance(selection_map, Mapping):
            for key, status in selection_map.items():
                if str(key).strip().casefold() == selection.strip().casefold():
                    return _normalize_settlement(status), str(value.get("winner") or "") or None
        # A compact mapping {"Home": "win", ...} is also accepted.
        for key, status in value.items():
            if str(key).strip().casefold() == selection.strip().casefold():
                return _normalize_settlement(status), str(value.get("winner") or "") or None
        winner = value.get("winner", value.get("winning_selection"))
        if winner is None:
            winner = value.get("outcome")
        if winner is None:
            raise ValueError(
                f"settlement mapping for {market_key!r} needs a winner or selection_outcomes"
            )
    else:
        winner = value

    # Also accept a direct selection-level status string for a single record.
    token = _norm_token(winner)
    if token in _SETTLEMENT_ALIASES:
        return _SETTLEMENT_ALIASES[token], None

    winner_text = str(winner)
    if market_key.strip().lower() in _DNB_KEYS and _norm_token(winner_text) in {"draw", "x"}:
        return "push", winner_text
    if selection.strip().casefold() == winner_text.strip().casefold():
        return "win", winner_text
    return "loss", winner_text


def _unit_return(settlement: str, odds: float) -> float:
    if settlement == "win":
        return odds - 1.0
    if settlement == "half_win":
        return 0.5 * (odds - 1.0)
    if settlement == "push":
        return 0.0
    if settlement == "half_loss":
        return -0.5
    return -1.0


def _probability_vector(rec: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    p_win = float(rec.get("model_p", 0.0))
    p_push = float(rec.get("p_push", 0.0) or 0.0)
    p_half_win = float(rec.get("p_half_win", 0.0) or 0.0)
    p_half_loss = float(rec.get("p_half_loss", 0.0) or 0.0)
    vals = (p_win, p_push, p_half_win, p_half_loss)
    if any(not math.isfinite(p) or p < 0.0 for p in vals):
        raise ValueError("decision settlement probabilities must be finite and non-negative")
    total = sum(vals)
    if total > 1.0 + 1e-8:
        raise ValueError(f"decision settlement probabilities sum to {total:.8f} > 1")
    p_loss = max(0.0, 1.0 - total)
    return {
        "win": p_win,
        "half_win": p_half_win,
        "push": p_push,
        "half_loss": p_half_loss,
        "loss": p_loss,
    }, p_loss


def grade_decisions(
    records: list[dict[str, Any]],
    settled_outcomes: dict[tuple[str, str], Any] | dict[str, Any],
    default_match_id: str = "",
) -> tuple[list[GradedDecision], CalibrationSummary]:
    """Grade records against settled outcomes.

    `settled_outcomes` may map `(match_id, market_key)` or (for one match)
    `market_key` to either a legacy winning selection string, or a mapping such
    as `{"selection_outcomes": {"Over 2.25": "half_loss", ...}}`. A draw in
    DNB is recognized as a push. Quarter-line splits must be supplied explicitly
    when the winner name alone does not identify the exact score settlement.
    """
    graded: list[GradedDecision] = []
    eps = 1e-12

    for rec in records:
        p = rec.get("model_p")
        if p is None:
            continue
        mid = str(rec.get("match_id") or default_match_id)
        mkey = str(rec.get("market_key", ""))
        sel = str(rec.get("selection", ""))

        settlement_value = None
        if (mid, mkey) in settled_outcomes:
            settlement_value = settled_outcomes[(mid, mkey)]  # type: ignore[index]
        elif mkey in settled_outcomes:
            settlement_value = settled_outcomes[mkey]  # type: ignore[index]
        if settlement_value is None:
            continue

        settlement, winner = _settlement_for_record(settlement_value, mkey, sel)
        probs, p_loss = _probability_vector(rec)
        outcome_vector = {key: 1.0 if key == settlement else 0.0 for key in _SETTLEMENT_CATEGORIES}
        # Divide the multicategory Brier sum by two so the binary case remains
        # on the familiar [0, 1] squared-error scale.
        brier = sum((probs[key] - outcome_vector[key]) ** 2 for key in _SETTLEMENT_CATEGORIES) / 2.0
        ll = -math.log(max(eps, probs[settlement]))

        p_win = float(p)
        p_low = float(rec.get("p_lower")) if rec.get("p_lower") is not None else p_win
        odds = float(rec.get("odds", 1.0))
        if odds < 1.0 or not math.isfinite(odds):
            raise ValueError(f"invalid decimal odds in decision record: {odds}")
        ev = (
            float(rec.get("ev"))
            if rec.get("ev") is not None
            else expected_value_detailed(
                odds,
                p_win=p_win,
                p_push=probs["push"],
                p_half_win=probs["half_win"],
                p_half_loss=probs["half_loss"],
            )
        )
        ev_low = (
            float(rec.get("ev_lower"))
            if rec.get("ev_lower") is not None
            else expected_value_detailed(
                odds,
                p_win=p_low,
                p_push=probs["push"],
                p_half_win=probs["half_win"],
                p_half_loss=probs["half_loss"],
            )
        )
        stake = float(rec.get("stake", 0.0))
        decision = str(rec.get("decision", "pass"))
        executed = bool(rec.get("executed", False))
        confidence = str(rec.get("confidence", "low"))
        won = settlement == "win"
        paper_unit_pnl = _unit_return(settlement, odds)
        realized_unit_pnl = paper_unit_pnl if executed else 0.0
        stake_pnl = stake * paper_unit_pnl if executed else 0.0

        graded.append(
            GradedDecision(
                match_id=mid,
                market_key=mkey,
                selection=sel,
                odds=odds,
                model_p=p_win,
                p_lower=p_low,
                ev=ev,
                ev_lower=ev_low,
                confidence=confidence,
                decision=decision,
                stake=stake,
                executed=executed,
                won=won,
                brier=brier,
                log_loss=ll,
                realized_unit_pnl=realized_unit_pnl,
                realized_stake_pnl=stake_pnl,
                settlement=settlement,
                p_push=probs["push"],
                p_half_win=probs["half_win"],
                p_half_loss=probs["half_loss"],
                p_full_loss=p_loss,
                paper_unit_pnl=paper_unit_pnl,
                settled_winner=winner,
            )
        )

    priced_count = sum(1 for r in records if r.get("model_p") is not None)
    if not graded:
        return [], CalibrationSummary(
            total_records=len(records),
            priced_records=priced_count,
            settled_records=0,
            recommended_count=0,
            executed_count=0,
            mean_predicted_p=0.0,
            realized_win_rate=0.0,
            calibration_gap=0.0,
            brier_score=0.0,
            log_loss=0.0,
            mean_apparent_ev=0.0,
            mean_conservative_ev=0.0,
            realized_unit_roi=0.0,
            realized_stake_pnl=0.0,
            paper_unit_roi=0.0,
            recommended_unit_roi=0.0,
            recommended_stake_pnl=0.0,
        )

    n = len(graded)
    mean_p = sum(g.model_p for g in graded) / n
    win_rate = sum(1.0 if g.won else 0.0 for g in graded) / n
    brier_avg = sum(g.brier for g in graded) / n
    ll_avg = sum(g.log_loss for g in graded) / n
    ev_avg = sum(g.ev for g in graded) / n
    ev_low_avg = sum(g.ev_lower for g in graded) / n
    recommended = [g for g in graded if g.decision == "recommend"]
    executed = [g for g in graded if g.executed]
    rec_count = len(recommended)
    executed_count = len(executed)
    unit_roi = (
        sum(g.realized_unit_pnl for g in executed) / executed_count
        if executed_count else 0.0
    )
    recommended_roi = (
        sum(g.paper_unit_pnl for g in recommended) / rec_count
        if rec_count else 0.0
    )
    paper_roi = sum(g.paper_unit_pnl for g in graded) / n
    stake_pnl_total = sum(g.realized_stake_pnl for g in graded)
    recommended_stake_pnl = sum(
        g.stake * g.paper_unit_pnl for g in recommended
    )

    by_mkt: dict[str, list[GradedDecision]] = {}
    by_conf: dict[str, list[GradedDecision]] = {}
    for g in graded:
        by_mkt.setdefault(g.market_key, []).append(g)
        by_conf.setdefault(g.confidence, []).append(g)

    def summarize_bucket(items: list[GradedDecision]) -> dict[str, Any]:
        k = len(items)
        bucket_recommended = [x for x in items if x.decision == "recommend"]
        bucket_executed = [x for x in items if x.executed]
        return {
            "count": k,
            "recommended_count": len(bucket_recommended),
            "executed_count": len(bucket_executed),
            "mean_p": round(sum(x.model_p for x in items) / k, 4),
            "win_rate": round(sum(1.0 if x.won else 0.0 for x in items) / k, 4),
            "brier": round(sum(x.brier for x in items) / k, 4),
            "mean_ev": round(sum(x.ev for x in items) / k, 4),
            "mean_ev_lower": round(sum(x.ev_lower for x in items) / k, 4),
            "unit_roi": round(
                sum(x.realized_unit_pnl for x in bucket_executed) / len(bucket_executed), 4
            ) if bucket_executed else 0.0,
            "recommended_unit_roi": round(
                sum(x.paper_unit_pnl for x in bucket_recommended) / len(bucket_recommended), 4
            ) if bucket_recommended else 0.0,
            "paper_unit_roi": round(sum(x.paper_unit_pnl for x in items) / k, 4),
        }

    # Proper categorical scoring for complete 1X2 prediction triplets. The
    # per-selection scores above remain useful for bet-level diagnostics.
    grouped: dict[tuple[str, str], list[GradedDecision]] = {}
    for g in graded:
        if g.market_key.strip().lower() in _1X2_KEYS:
            grouped.setdefault((g.match_id, g.market_key), []).append(g)

    multiclass_scores: list[tuple[float, float]] = []
    selection_aliases = {"home": "home", "1": "home", "draw": "draw", "x": "draw", "away": "away", "2": "away"}
    for (_, mkey), items in grouped.items():
        if len(items) != 3:
            continue
        outcomes: dict[str, GradedDecision] = {}
        for item in items:
            canonical = selection_aliases.get(_norm_token(item.selection))
            if canonical is not None:
                outcomes[canonical] = item
        if set(outcomes) != {"home", "draw", "away"}:
            continue
        winner_names = {(_norm_token(item.settled_winner or "")) for item in items}
        winner = next((selection_aliases[name] for name in winner_names if name in selection_aliases), None)
        if winner is None:
            winners = [name for name, item in outcomes.items() if item.settlement == "win"]
            winner = winners[0] if len(winners) == 1 else None
        if winner is None:
            continue
        pvec = {name: outcomes[name].model_p for name in ("home", "draw", "away")}
        if abs(sum(pvec.values()) - 1.0) > 1e-3:
            continue
        yvec = {name: 1.0 if name == winner else 0.0 for name in pvec}
        mc_brier = sum((pvec[name] - yvec[name]) ** 2 for name in pvec) / 2.0
        mc_log = -math.log(max(eps, pvec[winner]))
        multiclass_scores.append((mc_brier, mc_log))

    multi_n = len(multiclass_scores)
    summary = CalibrationSummary(
        total_records=len(records),
        priced_records=priced_count,
        settled_records=n,
        recommended_count=rec_count,
        executed_count=executed_count,
        mean_predicted_p=round(mean_p, 4),
        realized_win_rate=round(win_rate, 4),
        calibration_gap=round(mean_p - win_rate, 4),
        brier_score=round(brier_avg, 4),
        log_loss=round(ll_avg, 4),
        mean_apparent_ev=round(ev_avg, 4),
        mean_conservative_ev=round(ev_low_avg, 4),
        realized_unit_roi=round(unit_roi, 4),
        realized_stake_pnl=round(stake_pnl_total, 2),
        paper_unit_roi=round(paper_roi, 4),
        recommended_unit_roi=round(recommended_roi, 4),
        recommended_stake_pnl=round(recommended_stake_pnl, 2),
        multiclass_market_count=multi_n,
        multiclass_brier_score=round(
            sum(score[0] for score in multiclass_scores) / multi_n, 4
        ) if multi_n else 0.0,
        multiclass_log_loss=round(
            sum(score[1] for score in multiclass_scores) / multi_n, 4
        ) if multi_n else 0.0,
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
