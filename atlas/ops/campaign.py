"""Campaign verdict — the pre-registered self-terminating real-money rule.

The observation campaign stakes NGN 10 flat on every gate-passing
candidate. The point is MEASUREMENT, not profit, and the measurement has
a pre-registered verdict so the operator never has to watch a losing
account decay one NGN-10 at a time:

  collect  n < MIN_N            — not enough settled evidence yet
  observe  n >= MIN_N and the   95% CI on per-bet ROI excludes a
           positive return      (roi + 1.96*se < STOP_FLOOR) — the engine
                                 loses at its stated edge; real-money
                                 placement FROZEN at BetLedger.place_bet
  proven   n >= MIN_N, roi -    1.96*se > GO_FLOOR and CLV >= CLV_FLOOR —
           edge proven at bet level; stakes resume
  watch    anything else        — keep collecting real outcomes

Rules are fixed here, not tuned after the fact: one-sided 2.5% alpha on
the losing side, MIN_N 100 settled bets, GO threshold +2% per bet (the
realistic going edge after the margin; anything smaller cannot pay for
execution slippage). The verdict is written to
data/ops/campaign_state.json; `BetLedger.is_campaign_frozen()` reads it
and enforces it at the LAST wall before money, exactly like the daily
stop. 2026-11-01: the wall is FAIL-CLOSED — every action other than
'proven' (collect / watch / observe) and a missing/unreadable state
file all freeze real-money placement. Under the old explicitly-open
rule this session placed real NGN bets at collect n=0; the collect-
phase evidence now lives in the paper stream instead.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The verdict file is env-overridable (ATLAS_CAMPAIGN_STATE) so a test
# subprocess, the dashboard process, or a second checkout can point at
# its own state instead of the repo default.
STATE_FILE = (
    Path(os.environ["ATLAS_CAMPAIGN_STATE"])
    if os.environ.get("ATLAS_CAMPAIGN_STATE")
    else ROOT / "data" / "ops" / "campaign_state.json"
)
DECISIONS = ROOT / "data" / "decisions" / "loop_decisions.jsonl"

MIN_N = int(os.environ.get("ATLAS_CAMPAIGN_MIN_N", "100"))
# roi + 1.96*se < STOP_FLOOR  → the CI excludes a positive return: the
# engine is CONFIDENTLY losing money → freeze real-money placement.
STOP_FLOOR = float(os.environ.get("ATLAS_CAMPAIGN_STOP_FLOOR", "-0.02"))
# roi - 1.96*se > GO_FLOOR AND clv >= CLV_FLOOR → the CI excludes zero:
# a proven positive edge at bet level → resuming stakes is statistically
# safe.
GO_FLOOR = float(os.environ.get("ATLAS_CAMPAIGN_GO_FLOOR", "+0.02"))
CLV_FLOOR = float(os.environ.get("ATLAS_CAMPAIGN_CLV_FLOOR", "0.0"))

# The pre-registered FLAT unit of the real-money campaign: this operation
# stakes NGN 10 flat on every gate-passing candidate — a floor AND a cap.
# No sizing engine (Kelly or otherwise) may resize live money; on the
# live path the runner refuses any stake above this unit when the
# operator has not set an explicit cap (2026-11-01 23:13-23:18: a Kelly
# path on the live loop, launched with a fantasy --bankroll 100000,
# staked NGN 21-27 against the NGN-10 pre-registration through exactly
# the operator's silence).
UNIT_STAKE_NGN = float(os.environ.get("ATLAS_CAMPAIGN_UNIT_STAKE", "10.0"))

# Bets placed BEFORE the full gate stack (min_ev + position cap + max_ev
# ceiling) are a different engine; including them would grade today's
# machine on yesterday's pathology. Cutoff: the max_ev ceiling went live
# 2026-11-01 ~12:00 UTC.
CAMPAIGN_START = os.environ.get("ATLAS_CAMPAIGN_START", "2026-11-01T12:00:00")


def campaign_start_ts() -> str:
    return CAMPAIGN_START


def cohort(ledger) -> list:
    """Settled (win/loss) ledger bets placed after the campaign start.

    `ledger` is a BetLedger — its bets are the in-memory truth the runner
    already shares, so the verdict costs no extra file pass. Voids and
    unsettled bets are excluded (a void settles at stake and proves
    nothing about the price).
    """
    start = campaign_start_ts()
    out = []
    for b in ledger._bets.values():
        if b.status not in ("won", "lost"):
            continue
        pa = str(b.placed_at or "")
        if pa and pa < start:
            continue
        out.append(b)
    return out


def _clv_mean(bets: list) -> float | None:
    """Mean CLV over the cohort: placed odds vs the last logged price for
    the same selection before the book settled it (decision-log time is
    UTC; bet_history display time is GMT-08:00, converted in
    data/ops/quant_score.py)."""
    ticks: dict[tuple[str, str], list] = {}
    if not DECISIONS.exists():
        return None
    for line in DECISIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("market_key"):
            ticks.setdefault(
                (d["market_key"], str(d["selection"]).strip().lower()), []
            ).append(d)

    def ts_of(s: str) -> float:
        try:
            dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except ValueError:
            return float("inf")
        return dt.timestamp() if dt.tzinfo else dt.replace(
            tzinfo=timezone.utc
        ).timestamp()

    def settle_epoch(b) -> float:
        sa = str(b.settled_at or "")
        try:
            dt = datetime.fromisoformat(sa.replace("Z", "+00:00"))
        except ValueError:
            return float("nan")
        return dt.timestamp() if dt.tzinfo else dt.replace(
            tzinfo=timezone.utc
        ).timestamp()

    clvs = []
    for b in bets:
        lst = ticks.get(
            (b.market_key or "", str(b.selection or "").strip().lower()), []
        )
        placed_t = ts_of(b.placed_at)
        p_odds = b.priced_odds or b.odds
        last = None
        for d in reversed(lst):  # ticks are appended in time order
            t = ts_of(d.get("at"))
            if t <= settle_epoch(b) and t >= placed_t and d.get("odds"):
                last = float(d["odds"])
                break
        if last and p_odds and last > 0:
            clvs.append(float(p_odds) / last - 1.0)
    return sum(clvs) / len(clvs) if clvs else None


def verdict(bets: list) -> dict:
    """The pre-registered campaign verdict over a settled bet cohort."""
    n = len(bets)
    staked = sum(b.stake_ngn for b in bets)
    pnl = sum(b.pnl_ngn for b in bets)
    roi = pnl / staked if staked > 0 else 0.0
    se = None
    ci_hi = ci_lo = None
    if n >= 2:
        r = [b.pnl_ngn / b.stake_ngn for b in bets if b.stake_ngn > 0]
        m = sum(r) / len(r)
        sd = (sum((x - m) ** 2 for x in r) / (len(r) - 1)) ** 0.5
        # SE of the MEAN per-bet ROI (flat stakes ⇒ roi = mean(r_i)):
        # sd / sqrt(n). The sqrt(n) multiplier would be SE of the SUM —
        # a CI n² times too wide, freezing on noise.
        se = sd / math.sqrt(len(r))
        ci_hi, ci_lo = m + 1.96 * se, m - 1.96 * se
    clv = _clv_mean(bets) if n >= MIN_N else None

    action, why = "collect", f"settled n={n} < MIN_N={MIN_N}"
    if n >= MIN_N:
        if ci_hi is not None and ci_hi < STOP_FLOOR:
            action = "observe"
            why = (
                f"95% CI on per-bet ROI [{ci_lo:+.1%}, {ci_hi:+.1%}] "
                f"excludes a positive return (roi {roi:+.1%}, "
                f"n={n}) — the engine loses at its stated edge"
            )
        elif (
            ci_lo is not None
            and ci_lo > GO_FLOOR
            and clv is not None
            and clv >= CLV_FLOOR
        ):
            action = "proven"
            why = (
                f"95% CI on per-bet ROI [{ci_lo:+.1%}, {ci_hi:+.1%}] "
                f"excludes zero (roi {roi:+.1%}, n={n}), CLV "
                f"{(clv * 100 if clv is not None else 0):+.1f}% — edge "
                "proven at bet level"
            )
        else:
            action = "watch"
            why = (
                f"roi {roi:+.1%}, CI [{ci_lo if ci_lo is not None else 0:+.1%}, "
                f"{ci_hi if ci_hi is not None else 0:+.1%}] over n={n} — "
                "zero not excluded either way; keep collecting"
            )
    return {
        "action": action,
        "why": why,
        "n": n,
        "staked_ngn": round(staked, 2),
        "pnl_ngn": round(pnl, 2),
        "roi_pct": round(roi * 100, 2) if staked else 0.0,
        "ci95_pct": (
            [round(ci_lo * 100, 2), round(ci_hi * 100, 2)]
            if ci_lo is not None
            else None
        ),
        "clv_pct": round(clv * 100, 3) if clv is not None else None,
        "min_n": MIN_N,
        "cutoff": campaign_start_ts(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def verdict_from_ledger(ledger) -> dict:
    return verdict(cohort(ledger))


def refresh_state_file(ledger) -> dict:
    """Recompute the verdict and persist it for the placement gates."""
    v = verdict_from_ledger(ledger)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(v, indent=1), encoding="utf-8")
    tmp.replace(STATE_FILE)
    return v


def frozen() -> bool:
    """FAIL-CLOSED: real money requires the verdict to say 'proven'.

    Missing state, unparsable state, and every action other than 'proven'
    (collect / watch / observe) freeze real-money placement. This is the
    2026-11-01 inversion of the original explicitly-open rule (frozen only
    on 'observe' — open through the entire collection phase by
    construction, which is how this session spent real money at collect
    n=0). Under observe-first, the collect-phase evidence is the paper
    stream; the paper-verdict engine that can honestly produce 'proven'
    from reconciled paper outcomes is the component that re-opens money.
    """
    try:
        v = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return v.get("action") != "proven"
