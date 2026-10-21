"""The autonomous live loop.

    capture -> readiness -> price -> posture -> plan -> place -> record -> settle

The loop is deterministic end to end. The only non-deterministic part is the
browser read, and every gate in the chain is a gate that can stop the
cycle:

  - the panel readiness gate stops a half-rendered board;
  - the daily loss stop stops a losing day;
  - the calibration layer's 5pp gap stops a fake edge;
  - the portfolio's exposure, correlation and rate limits stop overtrading;
  - the posture decides the appetite for the whole cycle;
  - the betslip driver refuses to submit unless it was built with
    `allow_submit=True`.

Money moves in exactly one place, `place_bets`, and it is off by default.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atlas.live.arb import Selection, best_arbs
from atlas.live.decisions import PolicyConfig
from atlas.live.capture import CaptureResult, LiveCapture
from atlas.live.estimate import CalibratedModel, LiveStateSpaceModel
from atlas.live.loop import Opportunity
from atlas.live.observe import LiveMatchState
from atlas.live.settle import UNKNOWN, WON, settle_from_final_score
from atlas.ops.cashout import Position, value_cashout
from atlas.ops.ledger import BetLedger
from atlas.ops.notify import Notifier
from atlas.ops.posture import Posture, posture_for_mode
from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

# The maximum relative difference between the odds that were priced and
# the odds the book accepted. A live price can move between the click and
# the acceptance, and a 2% drift turns a +EV bet into a -EV one silently.
PRICE_MISMATCH_TOLERANCE = 0.02

# Markers that mean a match is not live and must never be selected.
FINISHED_MARKERS = (
    "FT",
    "ENDED",
    "FINISHED",
    "POSTPONED",
    "CANCELED",
    "CANCELLED",
    "AWARDED",
    "ABANDONED",
    "SUSPENDED",
    "HALF TIME",
)


@dataclass
class CycleReport:
    """What one loop cycle did."""

    cycle: int
    at: str
    ready: bool
    reason: str
    match_id: str = ""
    priced_selections: int = 0
    opportunities: int = 0
    planned: int = 0
    placed: int = 0
    arbs: int = 0
    settled: int = 0
    cashouts: int = 0
    cashout_taken: int = 0
    daily_pnl: float = 0.0
    stop_active: bool = False
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"cycle {self.cycle} [{self.at}] {self.reason}: priced "
            f"{self.priced_selections}, opps {self.opportunities}, planned "
            f"{self.planned}, placed {self.placed}, settled {self.settled}, "
            f"daily P&L {self.daily_pnl:+.0f}, cashouts {self.cashouts}"
            + (" [STOP]" if self.stop_active else "")
        )


class LiveRunner:
    """The 24/7 loop."""

    def __init__(
        self,
        capture: LiveCapture,
        ledger: BetLedger,
        notifier: Optional[Notifier] = None,
        betslip: Any = None,
        cashout_driver: Any = None,
        posture: Optional[Posture] = None,
        policy: Optional[PolicyConfig] = None,
        cycle_s: float = 30.0,
        model: Any = None,
        allow_submission: bool = False,
        state_path: Optional[str] = None,
        decision_log_path: Optional[str] = None,
    ) -> None:
        self.capture = capture
        self.ledger = ledger
        self.notifier = notifier
        self.betslip = betslip
        self.cashout_driver = cashout_driver
        self.policy = policy
        self.posture = posture or posture_for_mode("balanced")
        self.cycle_s = cycle_s
        self.model = model or LiveStateSpaceModel(now=datetime.now(timezone.utc))
        self.allow_submission = allow_submission
        self.state_path = Path(state_path) if state_path else None
        self.decision_log_path = (
            Path(decision_log_path) if decision_log_path else None
        )
        self.cycle = 0
        self.reports: list[CycleReport] = []
        self.last_error: Optional[str] = None
        self._last_notify_at: Optional[float] = None
        self._finished_match_ids: set[str] = set()

    # --- state -----------------------------------------------------------

    def save_state(self) -> None:
        """Persist the operator's posture so it survives a restart."""
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(
                {"posture": self.posture.to_dict(), "at": datetime.now(timezone.utc).isoformat()},
                f,
                ensure_ascii=False,
                indent=1,
            )

    def load_state(self) -> Optional[Posture]:
        """Restore a posture saved by the dashboard."""
        if not self.state_path or not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = data.get("posture") or {}
        try:
            from atlas.ops.posture import PostureMode

            posture = Posture(
                mode=PostureMode(raw.get("mode", "balanced")),
                min_ev=float(raw.get("min_ev", 0.05)),
                kelly_fraction=float(raw.get("kelly_fraction", 0.10)),
                max_exposure_per_match=float(raw.get("max_exposure_per_match", 3000.0)),
                max_bets_per_snapshot=int(raw.get("max_bets_per_snapshot", 2)),
                max_stake_per_bet=float(raw.get("max_stake_per_bet", 1000.0)),
                min_stake_ngn=float(raw.get("min_stake_ngn", 100.0)),
                cashout_min_discount=float(raw.get("cashout_min_discount", 0.0)),
                require_guaranteed_profit=bool(raw.get("require_guaranteed_profit", False)),
                rationale=str(raw.get("rationale", "restored")),
                source=str(raw.get("source", "restored")),
            )
        except (KeyError, ValueError, TypeError):
            return None
        self.posture = posture
        return posture

    def set_posture(self, posture: Posture) -> Posture:
        self.posture = posture.clamped()
        self.save_state()
        if self.notifier is not None:
            self.notifier.send(
                "Posture updated",
                self.posture.describe(),
                **self.posture.to_dict(),
            )
        return self.posture

    # --- pricing ---------------------------------------------------------

    def price_match(
        self, state: LiveMatchState, markets: list[dict]
    ) -> tuple[list[Opportunity], int, list[Selection]]:
        """Price every open market and return the opportunities.

        The calibration layer needs the book's own de-vigged probability for
        each selection, so the market's odds are de-vigged with Shin (the
        same method the mapper uses) before the model is consulted.
        """
        from atlas.live.math_engine import devig_shin

        priced = 0
        opportunities: list[Opportunity] = []
        selections: list[Selection] = []
        market_fair: dict[tuple[str, str], float] = {}
        for m in markets:
            odds_map: dict[str, float] = {}
            for s in m.get("selections", []):
                name = s.get("name") if isinstance(s, dict) else s
                odds = s.get("odds") if isinstance(s, dict) else None
                if name and odds:
                    odds_map[name] = float(odds)
            if len(odds_map) >= 2:
                try:
                    fair, _ = devig_shin(odds_map)
                except Exception:
                    fair = {}
                for name, p in fair.items():
                    market_fair[(m["market_key"], name)] = p
        calibrated = CalibratedModel(base_model=self.model, market_fair=market_fair)
        for m in markets:
            for s in m.get("selections", []):
                name = s.get("name") if isinstance(s, dict) else s
                odds = s.get("odds") if isinstance(s, dict) else None
                if not name or not odds:
                    continue
                est = calibrated.estimate(state, m["market_key"], name)
                if est.abstained or est.p is None:
                    continue
                priced += 1
                opp = Opportunity(
                    match_id=m.get("match_id", state.match_id),
                    market_key=m["market_key"],
                    selection=name,
                    odds=odds,
                    model_p=est.p,
                    model_name=est.model_name,
                    p_lower=est.p_lower,
                    p_push=est.p_push,
                )
                opportunities.append(opp)
                selections.append(
                    Selection(m["market_key"], name, odds, est.p)
                )
        return opportunities, priced, selections

    def _log_decisions(self, decisions: list, match_id: str) -> None:
        """Append this cycle's decisions to the audit log."""
        if self.decision_log_path is None:
            return
        try:
            path = Path(self.decision_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for d in decisions:
                    f.write(
                        json.dumps(
                            {
                                "at": datetime.now(timezone.utc).isoformat(),
                                "cycle": self.cycle,
                                "match_id": match_id,
                                "market_key": d.opportunity.market_key,
                                "selection": d.opportunity.selection,
                                "odds": d.opportunity.odds,
                                "model_p": d.opportunity.model_p,
                                "ev": d.opportunity.ev,
                                "ev_lower": d.opportunity.ev_lower,
                                "stake_ngn": d.stake_ngn,
                                "decision": d.decision,
                                "reason": d.reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception as e:
            self.last_error = f"decision log failed: {e}"

    # --- placing ---------------------------------------------------------

    def place_bets(
        self, decisions: list, state: LiveMatchState, cycle: int
    ) -> int:
        """Submit the accepted bets. The only place money moves.

        Three gates stand between a decision and the book:
          1. the policy must allow execution (an explicit operator opt-in);
          2. the daily stop and the kill switch are re-checked HERE, not
             only at the start of the cycle, so a stop that engaged while
             the cycle was pricing cannot be raced past;
          3. the odds the book ACCEPTED must match the odds that were
             priced, within PRICE_MISMATCH_TOLERANCE. The book can move a
             live price between the click and the acceptance, and a 2%
             drift turns a +EV bet into a -EV one silently.
        """
        if self.betslip is None or not self.allow_submission:
            return 0
        if self.policy is not None and not self.policy.allow_execution:
            return 0
        # Defence in depth: the stop may have engaged since the cycle began.
        if self.ledger.is_daily_stop_active():
            if self.notifier is not None:
                self.notifier.warn(
                    "Daily stop active — placement refused",
                    f"daily P&L {self.ledger.daily_pnl():.2f}",
                    cycle=cycle,
                )
            return 0
        if self.policy is not None and self.policy.kill_switch:
            return 0

        placed = 0
        for d in decisions:
            if not d.should_bet:
                continue
            # Re-check the stop before EVERY bet, not just once per cycle.
            if self.ledger.is_daily_stop_active():
                break
            label = d.opportunity.market_label
            try:
                # A leftover selection would silently turn this bet into a
                # multiple, so the slip starts empty, every time.
                if not self.betslip.slip_is_empty():
                    self.betslip.clear()
                if not self.betslip.slip_is_empty():
                    self.last_error = "the betslip would not clear"
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Betslip would not clear — bet skipped",
                            f"{label} / {d.opportunity.selection}",
                            cycle=cycle,
                        )
                    continue
                # Find the selection by NAME: the price may have moved since
                # it was priced, and that must be caught, not clicked.
                added = {}
                if label:
                    added = self.betslip.add_selection(label, d.opportunity.selection)
                if not added or not added.get("ok"):
                    added = self.betslip.add_odds(d.opportunity.odds)
                if not added or not added.get("ok"):
                    self.last_error = f"selection not on the board: {label} {d.opportunity.selection}"
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Selection not on the board — bet skipped",
                            f"{label} / {d.opportunity.selection}",
                            cycle=cycle,
                        )
                    continue
                self.betslip.set_stake(d.stake_ngn)
                slip = self.betslip.read_slip()
                accepted = slip.accepted_odds
                if accepted is None:
                    # The slip did not report a price, so it cannot be
                    # verified against the priced odds. Do not guess.
                    self.betslip.clear()
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Slip odds unreadable — bet refused",
                            f"{label} {d.opportunity.selection}",
                            cycle=cycle,
                        )
                    continue
                # The drift check comes BEFORE the placeable check: when the
                # price moved, the book shows 'Accept Changes' instead of
                # 'Place Bet', and the slip is not placeable until the new
                # price is acknowledged. Refusing on drift is correct, and
                # the slip must be cleared either way.
                drift = abs(accepted - d.opportunity.odds) / d.opportunity.odds
                if drift > PRICE_MISMATCH_TOLERANCE:
                    self.betslip.clear()
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Odds moved — bet refused",
                            f"priced {d.opportunity.odds:.2f}, accepted {accepted:.2f} "
                            f"({drift:.1%} > {PRICE_MISMATCH_TOLERANCE:.0%})",
                            cycle=cycle,
                            market_key=d.opportunity.market_key,
                            selection=d.opportunity.selection,
                        )
                    continue
                if slip.needs_accept:
                    # The book moved the price and wants the new one
                    # acknowledged. The drift is inside the tolerance, so
                    # accept it and let the slip settle.
                    self.betslip.accept_changes()
                    slip = self.betslip.read_slip()
                if not slip.can_place_bet:
                    self.betslip.clear()
                    self.last_error = "the slip is not placeable (the book refused it)"
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Slip not placeable — bet skipped",
                            f"{label} / {d.opportunity.selection}",
                            cycle=cycle,
                        )
                    continue
                self.betslip.place_bet()
                placed += 1
                self.ledger.record_pending(
                    {
                        "match_id": state.match_id,
                        "market_key": d.opportunity.market_key,
                        "market_label": label,
                        "selection": d.opportunity.selection,
                        "odds": accepted,
                        "priced_odds": d.opportunity.odds,
                        "stake_ngn": d.stake_ngn,
                        "model_p": d.opportunity.model_p,
                        "p_lower": d.opportunity.p_lower,
                        "ev": d.opportunity.ev,
                        "confidence": "unknown",
                    }
                )
            except Exception as e:
                self.last_error = f"place failed: {e}"
                if self.notifier is not None:
                    self.notifier.warn("Bet placement failed", str(e))
                try:
                    self.betslip.clear()
                except Exception:
                    pass
                continue
        return placed

    # --- settling --------------------------------------------------------

    def settle_finished(
        self, result: CaptureResult
    ) -> int:
        """Settle the ledger's bets for a match that has finished."""
        if not result.state or result.state.period != "ft":
            return 0
        if result.state.home.goals is None or result.state.away.goals is None:
            return 0
        final_h = result.state.home.goals
        final_a = result.state.away.goals
        settled = 0
        for bet in self.ledger.bets_for(result.match_id):
            if bet.get("status") in ("won", "lost", "void"):
                continue
            r = settle_from_final_score(
                bet.get("market_key", ""),
                bet.get("selection", ""),
                final_h,
                final_a,
                float(bet.get("stake_ngn", 0.0)),
                float(bet.get("odds", 0.0)),
            )
            if r.status == UNKNOWN:
                continue
            self.ledger.settle_bet(
                bet.get("bet_id", ""), r.status, r.settled_odds
            )
            settled += 1
        return settled

    # --- choosing a match -------------------------------------------------

    @staticmethod
    def choose_match(candidates: list) -> Optional[object]:
        """Pick which live match to work.

        The board offers dozens of live matches and the loop can only hold
        one position at a time, so the choice matters. The policy is
        explicit rather than incidental:

          depth  45% - how many markets the match carries (a richer match
                  has more priced edges to work with, and more liquidity to
                  absorb a stake)
          phase  35% - the second half is worth more than the first
                  (faster odds, clearer state)
          clock  20% - prefer the live middle of the match, where the odds
                  are moving and the state is informative

        Finished, postponed and cancelled matches are excluded outright.
        """
        live = []
        for c in candidates:
            # The status can be in either field: a finished card shows 'ENDED'
            # in its result text and the HALF-TIME score in its period, so
            # both have to be checked or a dead match gets chosen.
            haystack = (
                f"{getattr(c, 'period', '')} {getattr(c, 'score', '')}"
            ).upper()
            if any(k in haystack for k in FINISHED_MARKERS):
                continue
            live.append(c)
        if not live:
            return None
        max_depth = max((c.num_markets for c in live), default=0) or 1

        def phase_score(c) -> float:
            p = (getattr(c, "period", "") or "").upper()
            # The board labels a half as 'H1'/'H2' in Multi View and as
            # '1st Half'/'2nd Half' in Single View; both must be recognised or
            # the phase factor is inert.
            if "2ND" in p or "2H" in p or p.startswith("H2"):
                return 1.0
            if "1ST" in p or "1H" in p or p.startswith("H1"):
                return 0.7
            if "HT" in p or "HALF" in p:
                return 0.5
            return 0.6

        def clock_score(c) -> float:
            m = getattr(c, "minute", None)
            if m is None:
                return 0.5
            if 20.0 <= m <= 75.0:
                return 1.0
            if 75.0 < m <= 88.0:
                return 0.7
            if 10.0 <= m < 20.0:
                return 0.5
            return 0.3

        best = None
        best_score = -1.0
        for c in live:
            score = (
                0.45 * (c.num_markets / max_depth)
                + 0.35 * phase_score(c)
                + 0.20 * clock_score(c)
            )
            if score > best_score:
                best, best_score = c, score
        return best

    def switch_match(self) -> Optional[object]:
        """Pick and open the best live match on the board."""
        try:
            candidates = self.capture.board_candidates()
        except Exception:
            return None
        chosen = self.choose_match(candidates)
        if chosen is None:
            return None
        try:
            self.capture.select_match(chosen.index)
        except Exception:
            return None
        return chosen

    # --- the cycle -------------------------------------------------------

    def run_cycle(self) -> CycleReport:
        self.cycle += 1
        report = CycleReport(
            cycle=self.cycle,
            at=datetime.now(timezone.utc).isoformat(),
            ready=False,
            reason="",
        )
        try:
            result = self.capture.capture()
        except Exception as e:
            report.reason = f"capture error: {e}"
            self.last_error = report.reason
            self.reports.append(report)
            return report

        # P0: an expired session is a distinct failure from a busy board.
        # The operator must be told, because no bet can be placed until the
        # session is restored.
        if "session expired" in (result.reason or ""):
            if self.notifier is not None:
                self.notifier.critical(
                    "SportyBet session expired",
                    "the board is showing the login form — re-login is needed; "
                    "the loop will keep running and will not place bets",
                    cycle=self.cycle,
                )
            report.reason = result.reason
            self.reports.append(report)
            return report

        # If the focused match is not live (finished, or the board parked on
        # one), pick a live match from the board and open it.
        if not result.ready or (
            result.state is not None and result.state.period not in ("1h", "2h", "ht")
        ):
            chosen = self.switch_match()
            if chosen is not None:
                report.notes.append(
                    f"switched to {chosen.home_team} v {chosen.away_team} "
                    f"({chosen.period})"
                )
                result = self.capture.capture()

        report.ready = result.ready
        report.reason = result.reason
        report.match_id = result.match_id
        report.daily_pnl = self.ledger.daily_pnl()
        report.stop_active = self.ledger.is_daily_stop_active()

        if not result.ready or result.state is None:
            # Rate-limit the noise: a half-rendered board is normal.
            now = time.monotonic()
            if (
                self.notifier is not None
                and (self._last_notify_at is None or now - self._last_notify_at > 300)
            ):
                self.notifier.warn(
                    "Panel not ready", result.reason, cycle=self.cycle
                )
                self._last_notify_at = now
            self.reports.append(report)
            return report

        # A finished match: settle and stop.
        if result.state.period == "ft":
            report.settled = self.settle_finished(result)
            report.reason = f"match finished, settled {report.settled} bet(s)"
            self.reports.append(report)
            return report

        if report.stop_active:
            report.reason = (
                f"daily stop active (P&L {report.daily_pnl:+.0f}); no planning"
            )
            self.reports.append(report)
            return report

        opportunities, priced, selections = self.price_match(
            result.state, result.markets
        )
        report.priced_selections = priced
        report.opportunities = len(opportunities)

        arbs = best_arbs(selections, min_edge=0.005)
        report.arbs = len(arbs)
        if arbs and self.notifier is not None:
            self.notifier.good(
                "Sure-thing arb found",
                arbs[0].describe(),
                cycle=self.cycle,
                match_id=result.match_id,
            )

        base_config = SequentialConfig()
        if self.policy is not None:
            base_config = replace(
                base_config, bankroll_ngn=self.policy.bankroll_ngn
            )
        portfolio = SequentialPortfolio(
            self.posture.apply_to(base_config)
        )
        placed_bets = self.ledger.bets_for(result.match_id)
        decisions = portfolio.evaluate_snapshot(
            opportunities,
            {
                "period": result.state.period,
                "clock_minute": result.state.clock_minute,
                "score": (result.state.home.goals, result.state.away.goals),
                "events": [],
            },
            placed_bets,
        )
        accepted = [d for d in decisions if d.should_bet]
        report.planned = len(accepted)

        # The policy is the last word. With execution disabled the decisions
        # are still planned and logged — they are just not recommendations.
        if self.policy is not None and not self.policy.allow_execution:
            for d in accepted:
                d.decision = "pass"
                d.should_bet = False
                d.reason = (
                    f"{d.reason} | execution disabled (observe-only)"
                )
            report.planned = 0
        self._log_decisions(decisions, result.match_id)

        if self.posture.require_guaranteed_profit:
            arb_names = {name for arb in arbs for name, _ in arb.legs}
            accepted = [
                d for d in accepted if d.opportunity.selection in arb_names
            ]
            report.planned = len(accepted)

        if accepted:
            report.placed = self.place_bets(accepted, result.state, self.cycle)

        # P1: an open losing position gets a cashout decision every cycle.
        # Without this the position just sits there losing while the loop
        # only ever considers new bets.
        try:
            cashouts = self.handle_cashout_offers()
        except Exception as e:
            cashouts = []
            self.last_error = f"cashout check failed: {e}"
        report.cashouts = len(cashouts)
        report.cashout_taken = sum(1 for c in cashouts if c.get("recommended"))

        self.reports.append(report)
        return report

    def run_forever(self, max_cycles: Optional[int] = None) -> list[CycleReport]:
        """Loop until stopped (or `max_cycles` is reached)."""
        self.load_state()
        n = 0
        while max_cycles is None or n < max_cycles:
            report = self.run_cycle()
            n += 1
            if max_cycles is None:
                time.sleep(self.cycle_s)
        return self.reports

    # --- cashout ---------------------------------------------------------

    def evaluate_cashouts(
        self, cashout_offers: dict[str, float],
        min_discount: float = 0.0,
    ) -> list[dict]:
        """Value any cashout offers against the open positions.

        `cashout_offers` maps a bet id to the amount the book is offering.
        """
        out: list[dict] = []
        open_bets = {
            b.get("bet_id"): b
            for b in self.ledger.all_bets()
            if b.get("status") not in ("won", "lost", "void")
        }
        for bet_id, offered in (cashout_offers or {}).items():
            match = open_bets.get(bet_id)
            if match is None:
                continue
            position = Position(
                bet_id=bet_id,
                match_id=match.get("match_id", ""),
                market_key=match.get("market_key", ""),
                selection=match.get("selection", ""),
                stake_ngn=float(match.get("stake_ngn", 0.0)),
                odds=float(match.get("odds", 0.0)),
                model_p=float(match.get("model_p", 0.0)),
            )
            valuation = value_cashout(position, offered, min_discount=min_discount)
            out.append(
                {
                    "bet_id": bet_id,
                    "offered_ngn": offered,
                    "recommended": valuation.recommended,
                    "discount": valuation.cashout_discount,
                    "reason": valuation.reason,
                }
            )
        return out

    def handle_cashout_offers(self) -> list[dict]:
        """Read the book's cashout offers, value them, and act on the good ones.

        A cashout is only taken when it beats the position's model value by
        the posture's required margin, so selling a good bet cheaply never
        happens by default.
        """
        if self.cashout_driver is None:
            return []
        try:
            offers = self.cashout_driver.read_offers()
        except Exception:
            return []
        if not offers:
            return []
        by_stake = {round(o.stake_ngn, 2): o for o in offers}
        priced: dict[str, float] = {}
        open_bets = self.ledger.open_bets()
        # A cashout offer is matched to a position by stake. That is only
        # safe when the stake is UNIQUE among the open bets: with two open
        # bets of the same stake there is no way to tell which position the
        # book is offering to buy back, and guessing would sell the wrong
        # position.
        stake_counts: dict[float, int] = {}
        for bet in open_bets:
            stake = round(float(bet.get("stake_ngn", 0.0)), 2)
            stake_counts[stake] = stake_counts.get(stake, 0) + 1
        for bet in open_bets:
            stake = round(float(bet.get("stake_ngn", 0.0)), 2)
            if stake_counts.get(stake, 0) != 1:
                continue
            offer = by_stake.get(stake)
            if offer is not None:
                priced[bet.get("bet_id", "")] = offer.offered_ngn
        valuations = self.evaluate_cashouts(
            priced, min_discount=self.posture.cashout_min_discount
        )
        by_bet_id = {b.get("bet_id", ""): b for b in open_bets}
        for v in valuations:
            if not v["recommended"]:
                continue
            if self.notifier is not None:
                self.notifier.info(
                    "Cashout taken",
                    f"NGN {v['offered_ngn']:.2f} for bet {v['bet_id']}",
                    **v,
                )
            if self.cashout_driver.allow_execute:
                bet = by_bet_id.get(v["bet_id"])
                stake = (
                    round(float(bet.get("stake_ngn", 0.0)), 2) if bet else None
                )
                offer = by_stake.get(stake) if stake is not None else None
                if offer is not None:
                    self.cashout_driver.execute(offer.index)
        return valuations
