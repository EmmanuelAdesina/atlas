"""Per-market-family calibration maps for model probability calibration.

Raw model_p from LiveStateSpaceModel is not trustworthy — Perth post-mortem
showed the model overestimates in losing families (ou claimed +8.9 -> -38.3
realized). This module learns a monotonic calibration map per market family:

    calibrated_p = isotonic_regression(model_p) -> actual_win_rate

Uses the Pool Adjacent Violators Algorithm (PAVA) — pure Python, no scipy
dependency. Calibration maps are persisted to data/ops/calibration_maps.json
and loaded at runtime.

Usage:
    from atlas.live.calibration import CalibrationMap
    cal = CalibrationMap.load()           # loads from disk
    p_cal = cal.calibrate("ou_3_5", 0.88) # raw 0.88 -> calibrated ~0.55
    cal.fit(family_records)               # retrain from settled history
    cal.save()                            # persist to disk
"""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_MAP_PATH = ROOT / "data" / "ops" / "calibration_maps.json"

# Minimum observations per family before calibration is trusted.
# Below this, we fall back to raw model_p (no calibration).
MIN_CALIBRATION_OBS = 50


@dataclass
class CalibrationRecord:
    """One settled bet used for calibration training."""
    model_p: float
    actual_win: int  # 1 if won, 0 if lost
    market_key: str
    placed_at: str


def _pava(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """Pool Adjacent Violators Algorithm for isotonic regression.

    Fits a non-decreasing step function y = f(x) minimizing sum((y_i - f(x_i))^2)
    subject to f being non-decreasing in x.

    Returns sorted unique x values and their fitted y values.
    """
    if not x:
        return [], []

    # Sort by x
    pairs = sorted(zip(x, y), key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(ys)

    # Each block: start index, end index (exclusive), weighted mean of y,
    # weight (count of points in block).
    blocks: list[dict] = []
    for i in range(n):
        blocks.append({"start": i, "end": i + 1, "mean": ys[i], "weight": 1.0})

    # Pool adjacent violators: while a block's mean > next block's mean, merge.
    i = 0
    while i < len(blocks) - 1:
        if blocks[i]["mean"] > blocks[i + 1]["mean"]:
            # Merge blocks i and i+1
            w1 = blocks[i]["weight"]
            w2 = blocks[i + 1]["weight"]
            m1 = blocks[i]["mean"]
            m2 = blocks[i + 1]["mean"]
            merged_mean = (w1 * m1 + w2 * m2) / (w1 + w2)
            blocks[i] = {
                "start": blocks[i]["start"],
                "end": blocks[i + 1]["end"],
                "mean": merged_mean,
                "weight": w1 + w2,
            }
            del blocks[i + 1]
            # Step back to check previous merge
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build output: for each unique x boundary, assign the block mean.
    out_x: list[float] = []
    out_y: list[float] = []
    for block in blocks:
        # Use the last x in the block as the x-value for this step.
        out_x.append(xs[block["end"] - 1])
        out_y.append(block["mean"])

    return out_x, out_y


class CalibrationMap:
    """Per-family isotonic calibration map.

    model_p -> calibrated_p, learned from settled outcomes.
    """

    def __init__(self, min_obs: int = MIN_CALIBRATION_OBS) -> None:
        self.min_obs = min_obs
        # family_key -> (xs, ys) from PAVA fit
        self._maps: dict[str, tuple[list[float], list[float]]] = {}
        # family_key -> count of observations used
        self._counts: dict[str, int] = {}
        # family_key -> raw mean win rate (fallback when insufficient obs)
        self._base_rates: dict[str, float] = {}

    # ---- fit -----------------------------------------------------------

    def fit(self, records: list[CalibrationRecord]) -> None:
        """Fit calibration maps from settled records.

        Groups by market_key family (e.g. "ou_3_5" -> "ou") and fits PAVA
        per family. Families with < min_obs observations fall back to raw
        model_p (no calibration applied).
        """
        by_family: dict[str, list[CalibrationRecord]] = defaultdict(list)
        for r in records:
            family = self._family_key(r.market_key)
            by_family[family].append(r)

        self._maps = {}
        self._counts = {}
        self._base_rates = {}

        for family, recs in by_family.items():
            n = len(recs)
            self._counts[family] = n
            wins = sum(r.actual_win for r in recs)
            self._base_rates[family] = wins / n if n > 0 else 0.0

            if n < self.min_obs:
                # Not enough data to calibrate — fall back to raw model_p.
                self._maps[family] = ([], [])
                continue

            xs = [r.model_p for r in recs]
            ys = [float(r.actual_win) for r in recs]
            fit_x, fit_y = _pava(xs, ys)
            self._maps[family] = (fit_x, fit_y)

    def fit_from_tuples(
        self,
        family_records: dict[str, list[tuple[float, int]]],
    ) -> None:
        """Fit from dict of family -> [(model_p, actual_win), ...]."""
        records: list[CalibrationRecord] = []
        for family, pairs in family_records.items():
            for model_p, actual_win in pairs:
                records.append(
                    CalibrationRecord(
                        model_p=model_p,
                        actual_win=actual_win,
                        market_key=family,
                        placed_at="",
                    )
                )
        self.fit(records)

    # ---- predict -------------------------------------------------------

    def calibrate(self, market_key: str, model_p: float) -> float:
        """Return calibrated probability for a market key and raw model_p.

        If the family has insufficient evidence (no fit, empty fit, or
        counts below `min_obs` — i.e. `is_calibrated()` False), returns
        RAW model_p: never isotonic-extrapolate a thin bucket. A family
        whose few observations all won fits a step tail of ys==1.0 and
        would otherwise price a fantasy p=1.000 (the 2026-11-02 fix;
        that degenerate tail is what sized the Nashville blast through
        fractional_kelly_stake). Clamps to [0, 1].
        """
        family = self._family_key(market_key)
        fit = self._maps.get(family)

        if (
            not fit
            or len(fit[0]) == 0
            or not self.is_calibrated(market_key)
        ):
            # Insufficient data — return raw model_p.
            return max(0.0, min(1.0, model_p))

        xs, ys = fit

        # model_p below first calibration point -> use base rate or first fitted.
        if model_p <= xs[0]:
            return max(0.0, min(1.0, ys[0]))

        # model_p above last calibration point -> use last fitted.
        if model_p >= xs[-1]:
            return max(0.0, min(1.0, ys[-1]))

        # Interpolate within the step function.
        idx = bisect_right(xs, model_p) - 1
        if idx < 0:
            idx = 0
        if idx >= len(ys):
            idx = len(ys) - 1
        return max(0.0, min(1.0, ys[idx]))

    def is_calibrated(self, market_key: str) -> bool:
        """True if this family has enough observations for calibration."""
        family = self._family_key(market_key)
        return self._counts.get(family, 0) >= self.min_obs

    def base_rate(self, market_key: str) -> float:
        """Observed win rate for this family (fallback when uncalibrated)."""
        family = self._family_key(market_key)
        return self._base_rates.get(family, 0.5)

    # ---- persist -------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Persist calibration maps to JSON."""
        path = path or CALIBRATION_MAP_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "min_obs": self.min_obs,
            "maps": {},
            "counts": self._counts,
            "base_rates": self._base_rates,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for family, (xs, ys) in self._maps.items():
            data["maps"][family] = {"xs": xs, "ys": ys}
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "CalibrationMap":
        """Load calibration maps from JSON."""
        path = path or CALIBRATION_MAP_PATH
        cal = cls()
        if not path.exists():
            return cal
        data = json.loads(path.read_text(encoding="utf-8"))
        cal.min_obs = data.get("min_obs", MIN_CALIBRATION_OBS)
        cal._counts = data.get("counts", {})
        cal._base_rates = data.get("base_rates", {})
        for family, vals in data.get("maps", {}).items():
            cal._maps[family] = (vals.get("xs", []), vals.get("ys", []))
        return cal

    # ---- internals -----------------------------------------------------

    @staticmethod
    def _family_key(market_key: str) -> str:
        """Normalize market key to family key.

        Market keys like "ou_3_5", "handicap_0_1", "1x2" are used directly
        (lowercased) as family keys. Each distinct market is its own
        calibration family — OU 2.5 and OU 3.5 have different price shapes
        and must not share a calibration map.
        """
        return market_key.lower()


def fit_calibration_from_settled(
    settled_records: list[dict],
) -> CalibrationMap:
    """Build a CalibrationMap from settled bet records.

    Each record dict should have:
        model_p : float
        actual_win : int (1/0)
        market_key : str
    """
    cal = CalibrationMap()
    cal.fit(
        [
            CalibrationRecord(
                model_p=float(r["model_p"]),
                actual_win=int(r["actual_win"]),
                market_key=str(r["market_key"]),
                placed_at=str(r.get("placed_at", "")),
            )
            for r in settled_records
        ]
    )
    return cal


def expected_value_from_calibrated(
    calibrated_p: float,
    odds: float,
) -> float:
    """Expected net return per unit staked using calibrated probability."""
    if odds <= 1.0 or calibrated_p <= 0.0:
        return 0.0
    b = odds - 1.0
    p_loss = 1.0 - calibrated_p
    return calibrated_p * b - p_loss


def fractional_kelly_stake(
    calibrated_p: float,
    odds: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly stake fraction of bankroll.

    f_kelly = (p * (o - 1) - (1 - p)) / (o - 1)
    Capped at [0, 1], then multiplied by fraction (default 0.25).
    """
    if odds <= 1.0 or calibrated_p <= 0.0:
        return 0.0
    b = odds - 1.0
    p_loss = 1.0 - calibrated_p
    edge = calibrated_p * b - p_loss
    if edge <= 0.0:
        return 0.0
    full_kelly = edge / b
    return max(0.0, min(1.0, fraction * full_kelly))


# ---------------------------------------------------------------------------
# 7. Decision grading & post-settlement forensics
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSummary:
    """Aggregate result of grading settled decisions against outcomes."""

    settled_records: int
    brier_score: float
    log_loss: float
    by_market_family: dict[str, dict[str, float]]

    @property
    def calibration_error(self) -> float:
        """Absolute difference between mean predicted and mean actual."""
        return abs(self.brier_score - self.log_loss)


def grade_decisions(
    raw_decisions: list[dict],
    settled_outcomes: dict[str, str],
) -> tuple[list[dict], CalibrationSummary]:
    """Grade settled decisions against actual outcomes.

    For each raw_decision whose market_key appears in settled_outcomes,
    determine whether the selection matched the settled result (win=1, loss=0).
    Compute Brier score and log loss over settled records, grouped by market
    family.

    Parameters
    ----------
    raw_decisions : list[dict]
        Each dict must have: market_key, selection, odds, model_p, p_lower,
        ev, ev_lower, confidence, decision, stake.
    settled_outcomes : dict[str, str]
        Maps market_key -> winning selection name (e.g. {"ou_1_5": "Under 1.5"}).

    Returns
    -------
    graded : list[dict]
        Input decisions augmented with `actual_win` (1/0/None) and
        `calibrated_p` (isotonic-calibrated probability).
    summary : CalibrationSummary
        Aggregate statistics over settled records.
    """
    # Build calibration map from the settled decisions themselves
    cal = CalibrationMap()
    fit_records: list[CalibrationRecord] = []
    for rd in raw_decisions:
        mk = rd.get("market_key")
        mp = rd.get("model_p")
        if mk is None or mp is None:
            continue
        fit_records.append(
            CalibrationRecord(
                model_p=float(mp),
                actual_win=0,  # placeholder, will be overwritten below
                market_key=str(mk),
                placed_at="",
            )
        )
    cal.fit(fit_records)

    graded: list[dict] = []
    brier_terms: list[float] = []
    logloss_terms: list[float] = []
    family_stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "brier_sum": 0.0, "logloss_sum": 0.0, "wins": 0}
    )

    for rd in raw_decisions:
        mk = rd.get("market_key")
        sel = rd.get("selection")
        mp = rd.get("model_p")
        if mk is None or mk not in settled_outcomes:
            # Not settled — no actual outcome to grade against
            graded.append({**rd, "actual_win": None, "calibrated_p": None})
            continue

        actual_win = 1 if str(sel) == str(settled_outcomes[mk]) else 0
        p = float(mp) if mp is not None else 0.5
        p_cal = cal.calibrate(mk, p)

        # Brier term: (p - actual)^2
        brier_term = (p_cal - actual_win) ** 2
        brier_terms.append(brier_term)

        # Log loss term: -(actual*log(p) + (1-actual)*log(1-p))
        p_clamped = max(1e-12, min(1 - 1e-12, p_cal))
        if actual_win == 1:
            logloss_term = -math.log(p_clamped)
        else:
            logloss_term = -math.log(1 - p_clamped)
        logloss_terms.append(logloss_term)

        family = cal._family_key(str(mk))
        fs = family_stats[family]
        fs["n"] += 1
        fs["brier_sum"] += brier_term
        fs["logloss_sum"] += logloss_term
        fs["wins"] += actual_win

        graded.append(
            {
                **rd,
                "actual_win": actual_win,
                "calibrated_p": p_cal,
            }
        )

    n = len(brier_terms)
    summary = CalibrationSummary(
        settled_records=n,
        brier_score=(sum(brier_terms) / n) if n > 0 else 0.0,
        log_loss=(sum(logloss_terms) / n) if n > 0 else 0.0,
        by_market_family={
            fam: {
                "n": fs["n"],
                "brier_score": fs["brier_sum"] / fs["n"],
                "log_loss": fs["logloss_sum"] / fs["n"],
                "win_rate": fs["wins"] / fs["n"],
            }
            for fam, fs in family_stats.items()
        },
    )
    return graded, summary
