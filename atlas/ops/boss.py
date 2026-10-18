"""Boss LLM decision protocol — the Atlas operations manager (slow plane).

The Boss does NOT predict. It manages risk and operations:

  1. Risk governor      — tracks daily P&L, kills the loop at -₦500 (hard floor,
                           constant; the Boss can never change it).
  2. Match selector      — when a match ends, scans the live board and picks the
                           next match with the deepest odds menu and highest
                           liquidity (2H preferred over 1H, mid-match preferred
                           over early/late).
  3. Anomaly handler     — stale feed, disappeared odds, expired session, or
                           price mismatch -> decide wait / skip / alert.
  4. Stake sizing        — fractional Kelly from model outputs, within policy caps.
  5. Post-match review   — compares predicted vs realized, flags model drift.

Architectural laws (same asymmetric ratchet as atlas.live.boss.BossController):
  - The Boss NEVER invents probabilities; it only consumes model outputs.
  - The Boss NEVER loosens risk. It can ONLY tighten: raise min_ev, lower
    max_bets_per_match, lower kelly_fraction. Hard-coded floors and caps are
    constants the Boss cannot change.
  - evaluate_operation runs asynchronously and never blocks the pricing loop.
  - Every decision is logged with an explicit reason.

The fast plane (extract -> observe -> math_engine -> estimate -> decisions) is
pure synchronous math. Everything here is the slow plane: async oversight.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from atlas.live.decisions import PolicyConfig
from atlas.live.math_engine import fractional_kelly_stake
from atlas.live.observe import LiveMatchState

DAILY_LOSS_LIMIT_NGN = 500.0
FEED_STALE_AFTER_S = 60.0
PRICE_MISMATCH_TOLERANCE = 0.02
DRIFT_TOLERANCE = 0.25
MIN_BOARD_MARKETS = 2

WEIGHT_MARKET_DEPTH = 0.35
WEIGHT_LIQUIDITY = 0.20
WEIGHT_TOP_EDGE = 0.25
WEIGHT_PHASE = 0.20

_PHASE_SCORE = {"2h": 1.0, "1h": 0.75, "ht": 0.5}
_DEAD_STATUSES = {"suspended", "closed"}


@dataclass
class BossDecision:
    """One logged slow-plane decision."""

    action: str  # continue, pause, switch_match, stop, alert
    reason: str
    target_match_id: Optional[str] = None
    risk_override: Optional[dict] = None


@dataclass
class MatchCandidate:
    """A live-board match scored by the Boss for the next switch."""

    match_id: str
    home_team: str
    away_team: str
    score: str
    minute: float
    period: str
    num_markets: int
    total_volume: float
    top_edges: list[dict]


@dataclass
class BetLedger:
    """Minimal P&L ledger the Boss reads for risk governance.

    Only model-output-backed bets are recorded (each entry carries the model
    probability at placement). daily_pnl_ngn is realized P&L from settled
    bets; open_exposure_ngn is stake currently at risk on pending bets.
    """

    currency: str = "NGN"
    pending: list[dict] = field(default_factory=list)
    settled: list[dict] = field(default_factory=list)

    def record_pending(self, bet: dict) -> None:
        self.pending.append(dict(bet))

    def settle(
        self,
        match_id: str,
        market_key: str,
        selection: str,
        result: str,
        payout_ngn: Optional[float] = None,
    ) -> Optional[dict]:
        """Move a pending bet to settled with realized P&L.

        result: 'win' | 'loss' | 'void'. payout defaults to stake*odds (win),
        stake (void), or 0 (loss).
        """
        for i, bet in enumerate(self.pending):
            if (
                bet.get("match_id") == match_id
                and bet.get("market_key") == market_key
                and bet.get("selection") == selection
            ):
                stake = float(bet.get("stake_ngn", 0.0))
                odds = float(bet.get("odds", 0.0) or 0.0)
                if payout_ngn is None:
                    payout_ngn = stake * odds if result == "win" else (
                        stake if result == "void" else 0.0
                    )
                settled = {
                    **bet,
                    "result": result,
                    "payout_ngn": float(payout_ngn),
                    "pnl_ngn": float(payout_ngn) - stake,
                    "settled_at": datetime.now(timezone.utc).isoformat(),
                }
                self.pending.pop(i)
                self.settled.append(settled)
                return settled
        return None

    def bets_for(self, match_id: str) -> list[dict]:
        return [
            b for b in self.pending + self.settled if b.get("match_id") == match_id
        ]

    @property
    def daily_pnl_ngn(self) -> float:
        return float(sum(b.get("pnl_ngn", 0.0) for b in self.settled))

    @property
    def open_exposure_ngn(self) -> float:
        return float(sum(b.get("stake_ngn", 0.0) for b in self.pending))

    def daily_reset(self) -> None:
        self.pending.clear()
        self.settled.clear()


class Boss:
    """Slow-plane operations manager. Runs every poll_interval_s (default 20s)."""

    def __init__(
        self,
        ledger: BetLedger,
        policy: PolicyConfig,
        poll_interval_s: float = 20.0,
        decision_log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.ledger = ledger
        self.policy = policy
        self.poll_interval_s = float(poll_interval_s)
        self.decision_log_path = (
            Path(decision_log_path) if decision_log_path is not None else None
        )
        self.decisions: list[dict] = []
        self._open_markets: dict[str, str] = {}
        self._match_reviews: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Main decision loop (async, never blocks the fast pricing plane)
    # ------------------------------------------------------------------

    async def evaluate_operation(
        self,
        current_match_id: str,
        current_state: Optional[LiveMatchState],
        placed_bets: list[dict],
        live_board: list[dict],
    ) -> BossDecision:
        """Main decision loop. Runs every 20s asynchronously."""
        await asyncio.sleep(0)
        anomalies = self.check_anomalies(current_state, placed_bets)
        self._track_markets(placed_bets)

        daily_pnl = self.ledger.daily_pnl_ngn

        if daily_pnl <= -DAILY_LOSS_LIMIT_NGN:
            return self._decision(
                "stop",
                f"daily P&L {daily_pnl:.2f} {self.policy.currency} breached hard "
                f"limit -{DAILY_LOSS_LIMIT_NGN:.0f} {self.policy.currency}",
            )

        if self.policy.kill_switch:
            return self._decision(
                "stop", "kill_switch engaged in policy configuration"
            )

        critical = [
            a
            for a in anomalies
            if a.startswith(("session_expired", "price_mismatch"))
        ]
        if critical:
            return self._decision(
                "alert",
                "; ".join(critical),
                risk_override=self._tighten("critical anomaly detected"),
            )

        exposure_lost = [
            a
            for a in anomalies
            if a.startswith("odds_disappeared")
            and self._has_exposure(current_match_id, placed_bets)
        ]
        if exposure_lost:
            return self._decision(
                "alert",
                exposure_lost[0],
                risk_override=self._tighten("open exposure on disappeared odds"),
            )

        # Odds vanished but nothing is at risk: there is no reason to keep
        # watching this match. Move to the deepest-liquidity candidate so the
        # 24/7 cycle does not idle on a dead market.
        dead_market = [a for a in anomalies if a.startswith("odds_disappeared")]
        if dead_market:
            candidate = self.select_next_match(live_board or [])
            if candidate is not None:
                return self._decision(
                    "switch_match",
                    f"{dead_market[0]}; no exposure at risk, "
                    f"switching to deepest-liquidity candidate",
                    target_match_id=candidate.match_id,
                )
            return self._decision("pause", dead_market[0])

        if current_state is None or not current_state.is_live:
            candidate = self.select_next_match(live_board or [])
            if candidate is not None:
                return self._decision(
                    "switch_match",
                    f"current match {current_match_id!r} has no live state; "
                    f"switching to deepest-liquidity candidate",
                    target_match_id=candidate.match_id,
                )
            return self._decision(
                "pause",
                f"current match {current_match_id!r} has no live state and no "
                "suitable candidates on the live board",
            )

        stale = [a for a in anomalies if a.startswith("stale_feed")]
        if stale:
            return self._decision(
                "pause",
                stale[0],
                risk_override=self._tighten("stale feed"),
            )

        risk_override = (
            self._tighten("ratchet after negative daily P&L") if daily_pnl < 0 else None
        )
        return self._decision(
            "continue",
            f"operation nominal; daily P&L {daily_pnl:.2f} {self.policy.currency}, "
            f"open exposure {self.ledger.open_exposure_ngn:.2f} {self.policy.currency}",
            risk_override=risk_override,
        )

    # ------------------------------------------------------------------
    # Match selection
    # ------------------------------------------------------------------

    def select_next_match(self, live_board: list[dict]) -> Optional[MatchCandidate]:
        """Pick the best match from the live board.

        Scores on: market depth (35%), liquidity (20%), top edge EV (25%),
        and match phase (20%; 2H > 1H > HT, mid-match > early/late).
        Passes through board edge data as-is — never invents probabilities.
        """
        parsed = [self._parse_board_entry(e) for e in live_board or []]
        parsed = [p for p in parsed if p["num_markets"] >= MIN_BOARD_MARKETS]
        if not parsed:
            return None

        max_depth = max(p["num_markets"] for p in parsed)
        max_log_vol = max(math.log1p(p["total_volume"]) for p in parsed)
        max_edge = max(
            (self._top_edge_ev(p["edges"]) for p in parsed), default=0.0
        )

        best_key = None
        best_score = -math.inf
        for p in parsed:
            depth_norm = p["num_markets"] / max_depth if max_depth else 0.0
            vol_norm = (
                math.log1p(p["total_volume"]) / max_log_vol if max_log_vol else 0.0
            )
            edge_norm = (
                self._top_edge_ev(p["edges"]) / max_edge if max_edge > 0 else 0.0
            )
            phase = self._phase_score(p["period"], p["minute"])
            score = (
                WEIGHT_MARKET_DEPTH * depth_norm
                + WEIGHT_LIQUIDITY * vol_norm
                + WEIGHT_TOP_EDGE * edge_norm
                + WEIGHT_PHASE * phase
            )
            p["score"] = score
            if score > best_score:
                best_score = score
                best_key = p

        assert best_key is not None
        return MatchCandidate(
            match_id=best_key["match_id"],
            home_team=best_key["home_team"],
            away_team=best_key["away_team"],
            score=best_key["score"],
            minute=best_key["minute"],
            period=best_key["period"],
            num_markets=best_key["num_markets"],
            total_volume=best_key["total_volume"],
            top_edges=self._top_edges(best_key["edges"], limit=3),
        )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def check_anomalies(
        self,
        current_state: Optional[LiveMatchState],
        placed_bets: list[dict],
    ) -> list[str]:
        """Check for operational anomalies. Returns human-readable strings."""
        anomalies: list[str] = []

        if current_state is None:
            anomalies.append("no_state (current match observation missing)")
        else:
            age = current_state.age_seconds()
            if age > FEED_STALE_AFTER_S:
                anomalies.append(
                    f"stale_feed ({age:.0f}s > {FEED_STALE_AFTER_S:.0f}s)"
                )
            if not current_state.is_live:
                anomalies.append(f"match_not_live (period={current_state.period})")

        if any(
            b.get("session_expired") or b.get("login_redirect") for b in placed_bets
        ):
            anomalies.append("session_expired (login redirect detected)")

        for bet in placed_bets:
            clicked = bet.get("clicked_odds")
            accepted = bet.get("accepted_odds")
            if clicked is not None and accepted is not None:
                if clicked <= 0 or abs(clicked - accepted) / clicked > (
                    PRICE_MISMATCH_TOLERANCE
                ):
                    anomalies.append(
                        f"price_mismatch ({bet.get('selection', '?')}: "
                        f"clicked {clicked}, accepted {accepted})"
                    )
            prev = self._open_markets.get(self._bet_key(bet))
            status = bet.get("market_status") or bet.get("status")
            if prev == "open" and status in _DEAD_STATUSES:
                anomalies.append(
                    f"odds_disappeared ({bet.get('market_key', '?')}/"
                    f"{bet.get('selection', '?')} now {status})"
                )

        return anomalies

    # ------------------------------------------------------------------
    # Stake sizing (model output in, policy caps always respected)
    # ------------------------------------------------------------------

    def kelly_stake(
        self,
        model_p: Optional[float],
        odds: float,
        current_match_exposure: float = 0.0,
    ) -> float:
        """Fractional Kelly stake in NGN within hard policy caps.

        `model_p` must come from a model output — the Boss never invents it.
        Returns 0.0 when edge is non-positive or the stake would fall below
        the minimum. Never returns more than max_stake_per_bet, remaining
        match exposure, or the payout cap allow.
        """
        if model_p is None or not 0.0 < model_p < 1.0 or odds <= 1.0:
            return 0.0
        f_star = fractional_kelly_stake(
            model_p, odds, fraction=self.policy.kelly_fraction
        )
        if f_star <= 0.0:
            return 0.0
        raw = math.floor(self.policy.bankroll_ngn * f_star)
        remaining = max(
            0.0, self.policy.max_exposure_per_match - current_match_exposure
        )
        max_by_payout = self.policy.max_payout_per_bet / odds
        capped = min(
            float(raw),
            self.policy.max_stake_per_bet,
            remaining,
            max_by_payout,
        )
        if capped < self.policy.min_stake_ngn:
            return 0.0
        return float(math.floor(capped))

    # ------------------------------------------------------------------
    # Post-match review
    # ------------------------------------------------------------------

    def review_match(self, match_id: str) -> dict:
        """Compare predicted vs realized for one match; flag model drift."""
        bets = self.ledger.bets_for(match_id)
        settled = [b for b in bets if "pnl_ngn" in b]
        open_bets = [b for b in bets if "pnl_ngn" not in b]

        drift_flag = False
        drift_reason = ""
        underperformance_note = ""
        predicted_ev_total = 0.0
        realized_pnl = self.ledger.daily_pnl_ngn if not bets else float(
            sum(b.get("pnl_ngn", 0.0) for b in settled)
        )

        if settled:
            for b in settled:
                ev = b.get("ev")
                if ev is None and b.get("model_p") is not None:
                    ev = b["model_p"] * float(b.get("odds", 0.0)) - 1.0
                if ev is not None:
                    predicted_ev_total += float(ev) * float(b.get("stake_ngn", 0.0))

            total_stake = sum(float(b.get("stake_ngn", 0.0)) for b in settled)
            realized_roi = realized_pnl / total_stake if total_stake else 0.0
            mean_predicted_ev = predicted_ev_total / total_stake if total_stake else 0.0

            if realized_roi < mean_predicted_ev - DRIFT_TOLERANCE:
                drift_flag = True
                drift_reason = (
                    f"realized ROI {realized_roi:.3f} trails predicted EV "
                    f"{mean_predicted_ev:.3f} by more than {DRIFT_TOLERANCE:.2f}"
                )
            elif realized_roi > mean_predicted_ev + DRIFT_TOLERANCE:
                # Outperforming the estimate is NOT drift. Drift means the
                # model's edge is decaying. Beating the estimate is reported
                # separately as conservatism (money left on the table), never
                # as a reason to stop trading.
                underperformance_note = (
                    f"realized ROI {realized_roi:.3f} exceeds predicted EV "
                    f"{mean_predicted_ev:.3f} by more than {DRIFT_TOLERANCE:.2f} "
                    "(model under-predicting — may be leaving value on the table)"
                )

        review = {
            "match_id": match_id,
            "total_bets": len(bets),
            "settled_bets": len(settled),
            "open_bets": len(open_bets),
            "predicted_ev_total": round(predicted_ev_total, 4),
            "realized_pnl_ngn": round(realized_pnl, 2),
            "drift_flag": drift_flag,
            "drift_reason": drift_reason,
            "underperformance_note": underperformance_note,
            "action": "monitor_model" if drift_flag else "ok",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._match_reviews[match_id] = review
        return review

    # ------------------------------------------------------------------
    # Risk ratchet (tighten-only) and policy application
    # ------------------------------------------------------------------

    def _tighten(self, reason: str) -> dict:
        """Build tighten-only risk overrides. Never loosens the base policy."""
        tightened = {
            "min_ev": round(max(self.policy.min_ev, self.policy.min_ev * 1.25), 4),
            "max_bets_per_match": max(1, self.policy.max_bets_per_match - 1),
            "kelly_fraction": round(
                min(self.policy.kelly_fraction, self.policy.kelly_fraction * 0.5), 4
            ),
            "reason": reason,
        }
        assert tightened["min_ev"] >= self.policy.min_ev
        assert tightened["max_bets_per_match"] <= self.policy.max_bets_per_match
        assert tightened["kelly_fraction"] <= self.policy.kelly_fraction
        return tightened

    def apply_to_policy(self, base: PolicyConfig) -> PolicyConfig:
        """Apply the latest tighten-only overrides onto a PolicyConfig."""
        latest = next(
            (
                d["risk_override"]
                for d in reversed(self.decisions)
                if d.get("risk_override")
            ),
            None,
        )
        if latest is None:
            return base
        return PolicyConfig(
            min_ev=max(base.min_ev, latest["min_ev"]),
            min_ev_lower=base.min_ev_lower,
            max_bets_per_match=min(
                base.max_bets_per_match, latest["max_bets_per_match"]
            ),
            max_correlated_in_group=base.max_correlated_in_group,
            enforce_correlation_groups=base.enforce_correlation_groups,
            currency=base.currency,
            min_stake_ngn=base.min_stake_ngn,
            max_stake_per_bet=base.max_stake_per_bet,
            max_exposure_per_match=base.max_exposure_per_match,
            max_payout_per_bet=base.max_payout_per_bet,
            bankroll_ngn=base.bankroll_ngn,
            kelly_fraction=min(base.kelly_fraction, latest["kelly_fraction"]),
            auto_size_kelly=base.auto_size_kelly,
            max_evidence_age_s=base.max_evidence_age_s,
            kill_switch=base.kill_switch,
            blocked_matches=set(base.blocked_matches),
            blocked_markets=set(base.blocked_markets),
            allow_execution=base.allow_execution,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decision(
        self,
        action: str,
        reason: str,
        target_match_id: Optional[str] = None,
        risk_override: Optional[dict] = None,
    ) -> BossDecision:
        decision = BossDecision(
            action=action,
            reason=reason,
            target_match_id=target_match_id,
            risk_override=risk_override,
        )
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "reason": reason,
            "target_match_id": target_match_id,
            "risk_override": risk_override,
        }
        self.decisions.append(entry)
        if self.decision_log_path is not None:
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.decision_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return decision

    @staticmethod
    def _bet_key(bet: dict) -> str:
        return (
            str(bet.get("match_id", "")),
            str(bet.get("market_key", "")),
            str(bet.get("selection", "")),
        )

    def _track_markets(self, placed_bets: list[dict]) -> None:
        for bet in placed_bets:
            status = bet.get("market_status") or bet.get("status") or "open"
            self._open_markets[self._bet_key(bet)] = status

    def _has_exposure(self, match_id: str, placed_bets: list[dict]) -> bool:
        return any(
            b.get("match_id") == match_id and float(b.get("stake_ngn", 0.0)) > 0.0
            for b in placed_bets
        )

    @staticmethod
    def _parse_board_entry(entry: dict) -> dict:
        markets = entry.get("markets") or []
        edges = entry.get("edges") or entry.get("top_edges") or []
        return {
            "match_id": str(entry.get("match_id") or entry.get("id") or ""),
            "home_team": str(entry.get("home_team") or entry.get("home") or ""),
            "away_team": str(entry.get("away_team") or entry.get("away") or ""),
            "score": str(entry.get("score") or ""),
            "minute": float(
                entry.get("minute", entry.get("clock_minute") or 0.0)
            ),
            "period": str(entry.get("period") or "unknown"),
            "num_markets": int(entry.get("num_markets", len(markets))),
            "total_volume": float(
                entry.get("total_volume", entry.get("volume") or 0.0)
            ),
            "edges": edges,
            "score": 0.0,
        }

    @staticmethod
    def _top_edge_ev(edges: list[dict]) -> float:
        if not edges:
            return 0.0
        return max(float(e.get("ev", e.get("edge") or 0.0)) for e in edges)

    @staticmethod
    def _top_edges(edges: list[dict], limit: int = 3) -> list[dict]:
        ranked = sorted(
            edges,
            key=lambda e: float(e.get("ev", e.get("edge") or 0.0)),
            reverse=True,
        )
        return [
            {
                "market_key": str(e.get("market_key", e.get("market", ""))),
                "selection": str(e.get("selection", "")),
                "odds": e.get("odds"),
                "model_p": e.get("model_p"),
                "ev": float(e.get("ev", e.get("edge") or 0.0)),
            }
            for e in ranked[:limit]
        ]

    @staticmethod
    def _phase_score(period: str, minute: float) -> float:
        base = _PHASE_SCORE.get(period, 0.2)
        mid_match = max(0.0, 1.0 - abs(minute - 50.0) / 50.0)
        return base * (0.5 + 0.5 * mid_match)
