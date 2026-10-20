"""Sequential portfolio optimization layer for a single live match.

Decides WHICH opportunities to back and HOW MUCH to stake on each
snapshot. Contains no pricing math of its own: probabilities, EV and
Kelly sizing come from the existing engine
(`atlas.live.loop.Opportunity`, `atlas.live.decisions.compute_recommended_stake_ngn`,
backed by `atlas.live.math_engine`).

Enforced rules (see `SequentialConfig`):
  * Stake sizing — fractional Kelly (0.10x) on the conservative p_lower
    (falling back to model_p when p_lower is None) via the shared
    decisions-layer sizer. Stakes below min_stake_ngn are skipped.
  * Correlation — factor model from selection_correlation_groups(); at
    most max_correlated_in_group bets per risk factor per match. Once a
    factor is at cap, further bets in it require conservative EV
    (ev_lower) above ev_threshold_high_correlation.
  * Exposure — running per-match exposure capped at max_exposure_per_match.
  * Rate limits — max_bets_per_snapshot per evaluate_snapshot() call and
    max_bets_per_minute across placed-bet history plus this instance.
  * Rebalancing — a bet is placed only when the opportunity moved enough
    vs the last bet on the same selection (2pp EV or 6% odds), a match
    event occurred (goal, red card, period change, clock +5min), or
    more than min_minutes_between_same_selection passed.

Every decision (bet or pass) is logged with its reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from atlas.live.decisions import PolicyConfig, compute_recommended_stake_ngn
from atlas.live.estimate import Estimate
from atlas.live.loop import Opportunity, selection_correlation_groups

logger = logging.getLogger("atlas.portfolio.sequential")

_BANKROLL_NGN = 10_000.0

_TS_KEYS = ("timestamp", "ts", "placed_at", "time", "decided_at", "created_at")
_RATE_WINDOW_S = 60.0
_PRUNE_WINDOW_S = 300.0
_MATCH_CLOCK_EVENT_MIN = 5.0


@dataclass
class SequentialConfig:
    kelly_fraction: float = 0.10
    # The bankroll the stake is sized against. It must be the REAL balance:
    # with a 10,000 bankroll, a 0.10 Kelly fraction and the 5pp model-market
    # gap guard, the largest stake the guard can justify is about NGN 50 —
    # below the NGN 100 minimum, so the system would never place a bet.
    bankroll_ngn: float = 10_000.0
    min_stake_ngn: float = 100.0
    max_stake_per_bet: float = 1000.0
    max_exposure_per_match: float = 3000.0
    max_bets_per_match: int = 100
    max_bets_per_snapshot: int = 2
    max_bets_per_minute: int = 5
    max_correlated_in_group: int = 12
    ev_threshold_high_correlation: float = 0.12
    rebalance_ev_delta: float = 0.02
    rebalance_odds_delta: float = 0.06
    min_minutes_between_same_selection: float = 8.0


@dataclass
class SequentialDecision:
    opportunity: Opportunity
    stake_ngn: float
    should_bet: bool
    reason: str
    factor_counts: dict[str, int]
    # The same vocabulary the DecisionLog uses, so one decision log can hold
    # both the sequential and the single-shot decisions.
    decision: str = "pass"

    def __post_init__(self) -> None:
        if not self.decision or self.decision == "pass":
            self.decision = "recommend" if self.should_bet else "pass"


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _first_ts(bet: dict) -> Optional[datetime]:
    for key in _TS_KEYS:
        if bet.get(key) is not None:
            parsed = _parse_ts(bet[key])
            if parsed is not None:
                return parsed
    return None


def _norm_score(score: Any) -> Optional[tuple[int, int]]:
    if isinstance(score, dict):
        home, away = score.get("home"), score.get("away")
        if isinstance(home, (int, float)) and isinstance(away, (int, float)):
            return (int(home), int(away))
        return None
    if isinstance(score, (list, tuple)) and len(score) == 2:
        home, away = score
        if isinstance(home, (int, float)) and isinstance(away, (int, float)):
            return (int(home), int(away))
        return None
    if isinstance(score, str):
        parts = score.split("-") if "-" in score else score.split(":")
        if len(parts) == 2:
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                return None
    return None


def _event_count(events: Any) -> Optional[int]:
    if isinstance(events, bool):
        return None
    if isinstance(events, int):
        return events
    if isinstance(events, list):
        return len(events)
    if isinstance(events, dict) and events:
        total = 0
        for v in events.values():
            if not isinstance(v, int) or isinstance(v, bool):
                return None
            total += v
        return total
    return None


def _estimate_for(opp: Opportunity) -> Estimate:
    return Estimate(
        p=opp.model_p,
        confidence="high",
        model_name=opp.model_name,
        p_lower=opp.p_lower,
        p_upper=opp.model_p,
        p_push=opp.p_push,
    )


class SequentialPortfolio:
    """Per-snapshot bet selector with per-match risk state.

    Instance state tracks bet timestamps (with match_id) so the
    per-minute rate limit survives across evaluate_snapshot() calls.
    All other state (exposure, factor counts, bet counts) is derived
    from the placed-bets history passed into each call.
    """

    def __init__(self, config: SequentialConfig | None = None) -> None:
        self.config = config or SequentialConfig()
        self._bet_timestamps: list[tuple[float, str]] = []

    def evaluate_snapshot(
        self,
        opportunities: list[Opportunity],
        match_state: dict,
        placed_bets: list[dict],
    ) -> list[SequentialDecision]:
        """Evaluate all opportunities for this snapshot, return bets to place."""
        cfg = self.config
        match_ids = {o.match_id for o in opportunities}
        prior = [b for b in placed_bets if b.get("match_id") in match_ids]
        now = datetime.now(timezone.utc)
        now_epoch = now.timestamp()

        exposure: dict[str, float] = {}
        bet_counts: dict[str, int] = {}
        factor_counts: dict[str, dict[str, int]] = {}
        recent_prior = 0
        for bet in prior:
            mid = bet.get("match_id")
            if mid is None:
                continue
            try:
                # Ledger records carry stake_ngn (BetRecord.to_dict); Boss/boss
                # test dicts carry stake. Accept both so prior-bet accounting
                # never silently reads zero exposure.
                stake_val = float(bet.get("stake_ngn") or bet.get("stake") or 0.0)
            except (TypeError, ValueError):
                stake_val = 0.0
            exposure[mid] = exposure.get(mid, 0.0) + stake_val
            bet_counts[mid] = bet_counts.get(mid, 0) + 1
            groups = selection_correlation_groups(
                str(bet.get("market_key", "")), str(bet.get("selection", ""))
            )
            counts = factor_counts.setdefault(mid, {})
            for g in groups:
                counts[g] = counts.get(g, 0) + 1
            ts = _first_ts(bet)
            if ts is not None and (now - ts).total_seconds() <= _RATE_WINDOW_S:
                recent_prior += 1

        self._bet_timestamps = [
            (ts, mid)
            for ts, mid in self._bet_timestamps
            if now_epoch - ts <= _PRUNE_WINDOW_S
        ]
        instance_recent = sum(
            1
            for ts, mid in self._bet_timestamps
            if mid in match_ids and now_epoch - ts <= _RATE_WINDOW_S
        )

        policy = PolicyConfig(
            min_stake_ngn=cfg.min_stake_ngn,
            max_stake_per_bet=cfg.max_stake_per_bet,
            max_exposure_per_match=cfg.max_exposure_per_match,
            bankroll_ngn=cfg.bankroll_ngn,
            kelly_fraction=cfg.kelly_fraction,
        )

        decisions: list[SequentialDecision] = []
        snapshot_placed = 0
        snapshot_exposure: dict[str, float] = {}
        snapshot_factors: dict[str, dict[str, int]] = {}
        snapshot_counts: dict[str, int] = {}
        snapshot_keys: set[tuple[str, str, str]] = set()

        for opp in sorted(opportunities, key=lambda o: o.ev, reverse=True):
            mid = opp.match_id
            opp_factors = selection_correlation_groups(opp.market_key, opp.selection)
            snap_factors = snapshot_factors.setdefault(mid, {})
            hist_counts = factor_counts.get(mid, {})
            # Report every factor that is at/over cap, not just the opportunity's
            # own factors: the gating factor is what the operator must audit.
            capped_factors = [
                g for g, c in hist_counts.items() if c >= cfg.max_correlated_in_group
            ]
            current_counts = {
                g: hist_counts.get(g, 0) + snap_factors.get(g, 0)
                for g in set(opp_factors) | set(capped_factors)
            }
            # Only factors this opportunity actually belongs to may gate it.
            # Capped factors from other families are reported for audit but
            # must never block: that would let one saturated factor freeze the
            # entire portfolio.
            own_counts = {g: current_counts.get(g, 0) for g in opp_factors}
            cur_exposure = exposure.get(mid, 0.0) + snapshot_exposure.get(mid, 0.0)
            cur_bets = bet_counts.get(mid, 0) + snapshot_counts.get(mid, 0)

            def reject(reason: str) -> SequentialDecision:
                logger.info(
                    "PASS %s %s/%s — %s", mid, opp.market_key, opp.selection, reason
                )
                return SequentialDecision(
                    opportunity=opp,
                    stake_ngn=0.0,
                    should_bet=False,
                    reason=reason,
                    factor_counts=dict(current_counts),
                )

            blocked = next(
                (
                    g
                    for g, c in own_counts.items()
                    if c >= cfg.max_correlated_in_group
                ),
                None,
            )
            if blocked is not None:
                cons_ev = opp.ev_lower if opp.ev_lower is not None else opp.ev
                if cons_ev <= cfg.ev_threshold_high_correlation:
                    decisions.append(
                        reject(
                            f"factor '{blocked}' at {cfg.max_correlated_in_group}-bet cap; "
                            f"conservative EV {cons_ev:.3f} <= "
                            f"{cfg.ev_threshold_high_correlation}"
                        )
                    )
                    continue

            key = (mid, opp.market_key, opp.selection.strip().lower())
            if key in snapshot_keys:
                decisions.append(
                    reject("duplicate selection already accepted this snapshot")
                )
                continue

            move_reason = self._rebalance_reason(opp, prior, match_state)
            if move_reason is None:
                decisions.append(
                    reject(
                        f"no rebalance trigger (|dEV| <= {cfg.rebalance_ev_delta:.2f}, "
                        f"|dOdds| <= {cfg.rebalance_odds_delta:.0%}, no match event, "
                        f"<= {cfg.min_minutes_between_same_selection:.0f}min since last bet)"
                    )
                )
                continue

            if snapshot_placed >= cfg.max_bets_per_snapshot:
                decisions.append(
                    reject(f"max bets per snapshot ({cfg.max_bets_per_snapshot}) reached")
                )
                continue

            headroom = (
                cfg.max_bets_per_minute - recent_prior - instance_recent - snapshot_placed
            )
            if headroom <= 0:
                decisions.append(
                    reject(f"max bets per minute ({cfg.max_bets_per_minute}) reached")
                )
                continue

            if cur_bets >= cfg.max_bets_per_match:
                decisions.append(
                    reject(f"max bets per match ({cfg.max_bets_per_match}) reached")
                )
                continue

            if cur_exposure >= cfg.max_exposure_per_match:
                decisions.append(
                    reject(
                        f"match exposure cap reached "
                        f"({cur_exposure:.0f}/{cfg.max_exposure_per_match:.0f} NGN)"
                    )
                )
                continue

            stake = compute_recommended_stake_ngn(
                _estimate_for(opp), opp.odds, policy, cur_exposure
            )
            if stake < cfg.min_stake_ngn:
                decisions.append(
                    reject(
                        f"stake {stake:.0f} NGN below min {cfg.min_stake_ngn:.0f} NGN"
                    )
                )
                continue

            snapshot_placed += 1
            snapshot_exposure[mid] = snapshot_exposure.get(mid, 0.0) + stake
            snapshot_counts[mid] = snapshot_counts.get(mid, 0) + 1
            for g in opp_factors:
                snap_factors[g] = snap_factors.get(g, 0) + 1
            snapshot_keys.add(key)
            self._bet_timestamps.append((now_epoch, mid))

            p_cons = opp.p_lower if opp.p_lower is not None else opp.model_p
            reason = (
                f"{move_reason}; stake NGN {stake:.0f} "
                f"({cfg.kelly_fraction:.2f}x Kelly on p_lower={p_cons:.3f}, "
                f"EV={opp.ev:.3f})"
            )
            logger.info(
                "BET %s %s/%s odds=%.2f stake=NGN %.0f — %s",
                mid,
                opp.market_key,
                opp.selection,
                opp.odds,
                stake,
                reason,
            )
            decisions.append(
                SequentialDecision(
                    opportunity=opp,
                    stake_ngn=float(stake),
                    should_bet=True,
                    reason=reason,
                    factor_counts={
                        # Own factors increment; other reported factors
                        # (e.g. ones already at cap) stay as counted.
                        g: (current_counts.get(g, 0) + 1)
                        if g in opp_factors
                        else current_counts.get(g, 0)
                        for g in current_counts
                    },
                )
            )

        return decisions

    def should_rebalance(
        self,
        opp: Opportunity,
        placed_bets: list[dict],
        match_state: dict,
    ) -> bool:
        """Check if rebalancing threshold is met for this opportunity."""
        return self._rebalance_reason(opp, placed_bets, match_state) is not None

    def record_bet(self, decision: SequentialDecision, match_state: dict) -> dict:
        """Build a placed-bets history entry from an accepted decision."""
        opp = decision.opportunity
        return {
            "match_id": opp.match_id,
            "market_key": opp.market_key,
            "selection": opp.selection,
            "odds": opp.odds,
            "ev": opp.ev,
            "stake": decision.stake_ngn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "period": match_state.get("period"),
            "clock_minute": match_state.get("clock_minute"),
            "score": _norm_score(match_state.get("score")),
            "events_count": _event_count(match_state.get("events")),
        }

    def _rebalance_reason(
        self,
        opp: Opportunity,
        placed_bets: list[dict],
        match_state: dict,
    ) -> Optional[str]:
        cfg = self.config
        sel = opp.selection.strip().lower()
        same = [
            b
            for b in placed_bets
            if b.get("match_id") == opp.match_id
            and b.get("market_key") == opp.market_key
            and str(b.get("selection", "")).strip().lower() == sel
        ]
        if not same:
            return "new selection (no prior bet on this market/selection)"

        dated = [(b, ts) for b in same if (ts := _first_ts(b)) is not None]
        last = max(dated, key=lambda x: x[1])[0] if dated else same[-1]

        if dated:
            last_ts = max(ts for _, ts in dated)
            mins = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0
            if mins > cfg.min_minutes_between_same_selection:
                return (
                    f"{mins:.1f}min since last bet on this selection "
                    f"(> {cfg.min_minutes_between_same_selection:.0f}min)"
                )

        old_ev = last.get("ev")
        if old_ev is not None:
            try:
                d_ev = abs(opp.ev - float(old_ev))
            except (TypeError, ValueError):
                d_ev = None
            if d_ev is not None and d_ev > cfg.rebalance_ev_delta:
                return (
                    f"|dEV|={d_ev:.3f} > {cfg.rebalance_ev_delta:.2f} "
                    f"(EV {float(old_ev):.3f} -> {opp.ev:.3f})"
                )

        old_odds = last.get("odds")
        if old_odds is not None:
            try:
                d_odds = abs(opp.odds / float(old_odds) - 1.0)
            except (TypeError, ValueError, ZeroDivisionError):
                d_odds = None
            if d_odds is not None and d_odds > cfg.rebalance_odds_delta:
                return (
                    f"|dOdds|={d_odds:.1%} > {cfg.rebalance_odds_delta:.0%} "
                    f"({float(old_odds):.2f} -> {opp.odds:.2f})"
                )

        event = self._match_event(last, match_state)
        if event is not None:
            return f"match event: {event}"
        return None

    @staticmethod
    def _match_event(last: dict, match_state: dict) -> Optional[str]:
        period = match_state.get("period")
        if (
            period is not None
            and last.get("period") is not None
            and last.get("period") != period
        ):
            return f"period change ({last.get('period')} -> {period})"

        new_score = _norm_score(match_state.get("score"))
        old_score = _norm_score(last.get("score"))
        if new_score is not None and old_score is not None and new_score != old_score:
            return (
                f"score change ({old_score[0]}-{old_score[1]} -> "
                f"{new_score[0]}-{new_score[1]})"
            )

        clock = match_state.get("clock_minute")
        last_clock = last.get("clock_minute")
        if (
            isinstance(clock, (int, float))
            and not isinstance(clock, bool)
            and isinstance(last_clock, (int, float))
            and not isinstance(last_clock, bool)
            and float(clock) - float(last_clock) >= _MATCH_CLOCK_EVENT_MIN
        ):
            return (
                f"clock advanced {float(last_clock):.0f}' -> {float(clock):.0f}' "
                f"(>= +{_MATCH_CLOCK_EVENT_MIN:.0f}min)"
            )

        n_events = _event_count(match_state.get("events"))
        if n_events is not None and last.get("events_count") is not None:
            try:
                prev_events = int(last.get("events_count"))
            except (TypeError, ValueError):
                prev_events = None
            if prev_events is not None and n_events > prev_events:
                return f"{n_events - prev_events} new match event(s)"
        return None
