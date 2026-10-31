"""Model performance from the ledger's settled bets.

The question this module owns: 'do our priced bets beat the book, or is the
model behind it?' There is one honest metric set — staked NGN, realised P&L,
ROI, hit rate against the break-even implied by the odds actually paid, and
the current consecutive-loss streak — plus ONE decision rule: the sample
gates every verdict. Smaller than MIN_SAMPLE -> the only honest answer is
'keep logging at the minimum stake'.
"""

from __future__ import annotations

from typing import Iterable, Optional

SETTLED = ("won", "lost")

# 150+ NGN at the minimum stake is the smallest stake round whose realised
# ROI can mean anything (see MIN_SAMPLE). Bands by paid decimal odds: the
# model-error blow-up is odds-relative, so verdicts must never mix bands.
MIN_SAMPLE = 12
_BANDS: tuple[tuple[str, float], ...] = (
    ("short <1.5", 1.5),
    ("mid 1.5-3", 3.0),
    ("long >3", float("inf")),
)


def _band(odds: float) -> str:
    for name, hi in _BANDS:
        if odds < hi:
            return name
    return _BANDS[-1][0]


def settled_stats(bets: Iterable[dict]) -> dict:
    """Aggregate ONLY settled bets chronologically (insertion order)."""
    rows = [
        b for b in bets
        if str(b.get("status", "")).lower() in SETTLED and b.get("pnl_ngn") is not None
    ]
    staked = sum(float(b.get("stake_ngn", 0.0)) for b in rows)
    pnl = sum(float(b.get("pnl_ngn", 0.0)) for b in rows)
    won = sum(1 for b in rows if str(b.get("status", "")).lower() == "won")

    by_band: dict[str, dict] = {}
    for b in rows:
        odds = float(b.get("odds") or 0.0)
        band = _band(odds)
        agg = by_band.setdefault(
            band, {"n": 0, "won": 0, "staked": 0.0, "pnl": 0.0, "inv_odds": 0.0}
        )
        agg["n"] += 1
        agg["staked"] += float(b.get("stake_ngn", 0.0))
        agg["pnl"] += float(b.get("pnl_ngn", 0.0))
        agg["inv_odds"] += 1.0 / odds if odds else 0.0
        if str(b.get("status", "")).lower() == "won":
            agg["won"] += 1
    for agg in by_band.values():
        n = agg["n"]
        if not n:
            continue
        agg["hit_rate"] = round(agg["won"] / n, 3)
        agg["roi"] = round(agg["pnl"] / agg["staked"], 4) if agg["staked"] else None
        # the hit rate those odds would need to break even (~sum(1/odds)/n)
        agg["breakeven_hit_rate"] = round(agg["inv_odds"] / n, 3)
        del agg["inv_odds"]

    return {
        "settled": len(rows),
        "won": won,
        "lost": len(rows) - won,
        "hit_rate": round(won / len(rows), 3) if rows else None,
        "staked_ngn": round(staked, 2),
        "pnl_ngn": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else None,
        "streak_losses": _current_loss_streak(rows),
        "by_band": by_band,
        "verdict": verdict(len(rows), pnl / staked if staked else None),
    }


def _current_loss_streak(rows: list[dict]) -> int:
    streak = 0
    for b in reversed(rows):
        if str(b.get("status", "")).lower() == "lost":
            streak += 1
        else:
            break
    return streak


def verdict(settled: int, roi: Optional[float]) -> str:
    """The stake-policy rule the whole machine agrees to be gated by.

    The sample size gates everything: with fewer than MIN_SAMPLE settled
    bets, ANY ROI reading is noise (a couple of 1.11 winners can look like
    mastery; a couple of 10.0 longshots can look like doom).
    """
    if settled < MIN_SAMPLE or roi is None:
        return (
            f"sample too small ({settled}/{MIN_SAMPLE}) — keep logging at the "
            "minimum stake; no evidence to change stakes"
        )
    if roi > 0.03:
        return f"ROI {roi*100:+.1f}% > +3% — pricing is beating the book; a raise can be discussed on evidence"
    if roi < -0.03:
        return f"ROI {roi*100:+.1f}% < -3% — pricing is behind the book; shrink stakes, do not raise"
    return f"ROI {roi*100:+.1f}% within +-3% — inconclusive; no basis to change stakes"
