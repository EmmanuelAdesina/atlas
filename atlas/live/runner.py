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
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atlas.live.arb import Selection, best_arbs, normalized_identity_set
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

# How long a priced decision may wait in the placement queue before it is
# stale. The book's price gate would usually catch a moved price, but a
# decision priced before a goal is stale even at the same price.
PLACEMENT_MAX_AGE_S = 90.0
# The placement worker must never block the cycle forever on a page lock
# held by a wedged browser call.
PAGE_LOCK_TIMEOUT_S = 60.0
# Non-critical page work (a match switch, a cashout read) SKIPS when the
# placement worker owns the page. It is retried next cycle, so a short
# wait is enough — waiting the full placement window would stall the loop.
PAGE_BUSY_TIMEOUT_S = 5.0

# How long a market the book refused stays off the placement queue. In-play
# markets are SUSPENDED through half-time and injury checks; a suspended
# selection makes the slip unplaceable, and re-adding it next cycle only
# burns the placement window (2026-10-29: 8 of 10 attempts on two O/U
# lines at HT). ~2 minutes of cycles is enough to move on without
# blacklisting a market that comes back when play resumes.
MARKET_FAILURE_COOLDOWN_S = 120.0

# After a placement comes back WITHOUT a receipt, a second balance read at
# this much later detects a phantom position (book took money, ledger did
# not). 30s: long enough for the book to deduct the stake, short enough
# that the alert surfaces the same session (2026-10-31).
BALANCE_PROBE_DELAY_S = 30.0
# If no balance reading lands within this long after the attempt, the
# probe gives up loudly instead of staying armed forever.
BALANCE_PROBE_MAX_WAIT_S = 300.0

# Degenerate-certainty thresholds. A model probability this close to 0 or 1
# is only credible when the book has also stopped pricing the outcome
# (odds ~1.00). At a live price above 1.05 the two are contradicting each
# other by more than the book's entire margin, and the model has mis-read
# the market family, not found free money. 2026-10-28: 'ou_1_5 Over 1.5'
# priced at p=1.0 while 'Under 1.5' traded at 1.01 — NGN 100 lost.
DEGENERATE_CERTAINTY_P = 0.995
DEGENERATE_CERTAINTY_MIN_ODDS = 1.05
# The betslip is a singleton; a queue deeper than this means placements are
# failing faster than they complete. Refuse new work instead of piling up.
MAX_PLACE_QUEUE = 8

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
        supervisor: Any = None,
        posture: Optional[Posture] = None,
        policy: Optional[PolicyConfig] = None,
        cycle_s: float = 30.0,
        model: Any = None,
        allow_submission: bool = False,
        state_path: Optional[str] = None,
        decision_log_path: Optional[str] = None,
        operator_min_stake: Optional[float] = None,
        operator_max_stake: Optional[float] = None,
    ) -> None:
        self.capture = capture
        self.ledger = ledger
        self.notifier = notifier
        self.betslip = betslip
        self.cashout_driver = cashout_driver
        self.supervisor = supervisor
        self.policy = policy
        self.posture = posture or posture_for_mode("balanced")
        # Operator stake bounds (from the CLI: --min-stake/--max-stake).
        # These are money-safety bounds the operator TYPED — they outrank
        # the dashboard's saved posture, which is an appetite at best. The
        # saved state file once held max_stake_per_bet=100 against an
        # explicit NGN 10 command line and silently re-applied it every
        # cycle (2026-10-28: NGN 35/100 stakes reached recommendations); so
        # the bounds are re-applied after EVERY posture load, not just at
        # construction.
        self.operator_min_stake = operator_min_stake
        self.operator_max_stake = operator_max_stake
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
        # Per-market failure cooldown: a market the book refuses (slip not
        # placeable — the signature of a SUSPENDED in-play market at
        # half-time) must not be re-tried every cycle. 2026-10-29: 8 of 10
        # placement attempts burned on the same two O/U lines at HT.
        self._avoid: dict[tuple[str, str, str], float] = {}
        self._last_avoided: Optional[str] = None
        # Balance-vs-ledger guard: the capture reads the account balance each
        # ready cycle. A balance DROP right after an unconfirmed placement is
        # the signature of a live bet the ledger does not hold — that must
        # surface the same cycle, not at settlement.
        self._balance_observed: Optional[float] = None
        self._unconfirmed_at: Optional[float] = None
        self._balance_alert_fired: bool = False
        # The unconfirmed-placement probe: 'no receipt seen' cannot tell
        # 'never went through' from 'receipt missed', so a SECOND balance
        # read is scheduled ~30s later. A drop against the pre-attempt
        # balance is a phantom position with real money on it; no drop
        # closes the case. One-shot and expiring — it must never stay
        # armed through later, legitimate placements.
        self._probe_at: Optional[float] = None
        self._probe_baseline: Optional[float] = None
        # --- page ownership ------------------------------------------------
        # ONE re-entrant lock for every PAGE MUTATION: the cycle's board
        # switches/recovery and the placement worker's betslip flow must
        # never interleave on the shared tab. Reads do not take it, so the
        # cycle keeps pricing while a placement is in flight.
        self._page_lock = threading.RLock()
        if capture is not None and hasattr(capture, "page_lock"):
            capture.page_lock = self._page_lock
        # --- asynchronous placement worker --------------------------------
        # Placement is the slowest step per bet (add -> Confirm -> receipt
        # poll ≈ 2-15s), and the cycle used to BLOCK on it: a wedged slip
        # (Unavailable selection) froze the whole loop for hours
        # (2026-10-28, cycle 15). A dedicated worker thread drains a queue
        # of placement requests while the cycle keeps scanning and pricing.
        # Hard limit of the book: the betslip is a SINGLETON on the page —
        # two bets can never be placed truly in parallel; the queue only
        # prevents the cycle from freezing behind one placement.
        self._place_queue: "queue.Queue" = queue.Queue()
        self._place_stop = threading.Event()
        self._place_stats = {
            "queued": 0,
            "done": 0,
            "failed": 0,
            "expired": 0,
            "duplicate": 0,
            "dropped": 0,
        }
        self._place_last: Optional[dict] = None
        self._place_started_at: Optional[float] = None
        # In-flight reservations: a decision handed to the worker is not in
        # the ledger yet, so the next cycle must still see it — otherwise it
        # re-plans the same selection as "new" and places it twice.
        self._inflight: dict[tuple[str, str, str], dict] = {}
        self._inflight_lock = threading.Lock()
        self._place_thread: Optional[threading.Thread] = None
        if self.betslip is not None and self.allow_submission:
            self._ensure_worker()

    def _ensure_worker(self) -> None:
        """Start the placement thread once, on first real need.

        Only a runner that can actually place bets gets a worker: a
        plan-only or test runner must not leave a daemon thread behind.
        """
        if self._place_thread is not None and self._place_thread.is_alive():
            return
        self._place_thread = threading.Thread(
            target=self._placement_worker,
            daemon=True,
            name="atlas-placement",
        )
        self._place_thread.start()

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
        """Restore a posture saved by the dashboard.

        The dashboard's posture is an APPETITE; the operator's CLI stake
        bounds are money safety and re-apply to every loaded posture —
        the saved file once carried max_stake_per_bet=100 against an
        explicit NGN 10 command line and re-applied it every cycle.
        """
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
        # Operator bounds re-applied: money safety outranks saved appetite.
        posture = self._apply_operator_bounds(posture)
        self.posture = posture
        return posture

    def _apply_operator_bounds(self, posture):
        """Clamp a posture to the operator's CLI stake bounds (if given)."""
        from dataclasses import replace

        if self.operator_max_stake is not None:
            posture = replace(
                posture, max_stake_per_bet=float(self.operator_max_stake)
            )
        if self.operator_min_stake is not None:
            posture = replace(
                posture, min_stake_ngn=float(self.operator_min_stake)
            )
        return posture.clamped()

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
                # A selection the market itself does not imply a probability
                # for cannot be calibrated, and an uncalibrated model must
                # not produce a bet (yesterday's phantom: a cross-wired
                # exact-score selection priced on "calibration skipped").
                # Log-blocked, never priced.
                if (m["market_key"], name) not in market_fair:
                    self.last_error = (
                        f"no market-implied prob for {m['market_key']}/{name} — not priced"
                    )
                    continue
                priced += 1
                opp = Opportunity(
                    match_id=m.get("match_id", state.match_id),
                    market_key=m["market_key"],
                    market_label=m.get("market_label", ""),
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

    def apply_policy_gates(self, decisions: list) -> list:
        """Sanity gates on the accepted set: DEGENERATE CERTAINTY first,
        then the long-odds cap. Runs in run_cycle AND again inside
        place_bets — the money path re-checks everything.

        Degenerate certainty (case study 2026-10-28): the model priced an
        overtime-family market with p = 1.0 / EV 900% ('Over 1.5' when the
        full-match condition was already satisfied) at odds 10.0 — NGN 100,
        lost. When the model claims near-certainty while the book still
        sells the outcome at live odds, either the book is giving away
        money every invocation or the model has mis-read the market family;
        assume the latter and refuse.
        """
        accepted = [d for d in decisions if d.should_bet]
        if self.policy is None:
            return accepted
        for d in list(accepted):
            p = d.opportunity.model_p
            if p is None:
                continue
            p = float(p)
            # (a) Total certainty at a live price. p == 1.0 (or 0.0) is a
            # terminal-state reading: the model believes the outcome is
            # already decided. The book disagreeing by >5% of price means
            # one of the two is wrong, and it is not the book — a market
            # that is genuinely settled is suspended, not sold at 1.24.
            contradictory = (
                (p >= DEGENERATE_CERTAINTY_P or p <= 1.0 - DEGENERATE_CERTAINTY_P)
                and d.opportunity.odds > DEGENERATE_CERTAINTY_MIN_ODDS
            )
            # (b) The older, looser gate: 90%+ certainty at 1.50+ is the
            # same class of error with a less extreme reading.
            long_odds_certainty = p >= 0.90 and d.opportunity.odds >= 1.5
            if contradictory or long_odds_certainty:
                d.decision = "pass"
                d.should_bet = False
                self.last_error = (
                    f"degenerate certainty: model_p {p:.3f} "
                    f"at odds {d.opportunity.odds:.2f} — refused"
                )
                d.reason = f"{d.reason} | {self.last_error} — market-family mispricing"
                if self.notifier is not None:
                    self.notifier.info(
                        "Bet refused (degenerate certainty)",
                        f"{d.opportunity.market_key} {d.opportunity.selection}: "
                        f"model_p {p:.3f} @ {d.opportunity.odds:.2f}",
                        cycle=self.cycle,
                    )
        accepted = [d for d in accepted if d.should_bet]
        for d in list(accepted):
            if d.opportunity.odds > self.policy.max_bet_odds:
                d.decision = "pass"
                d.should_bet = False
                d.reason = (
                    f"{d.reason} | odds {d.opportunity.odds:.2f} > "
                    f"{self.policy.max_bet_odds:.2f} cap (model error risk)"
                )
                if self.notifier is not None:
                    self.notifier.info(
                        "Bet refused (long-odds gate)",
                        f"{d.opportunity.market_key} {d.opportunity.selection} "
                        f"@ {d.opportunity.odds:.2f} — cap "
                        f"{self.policy.max_bet_odds:.2f}",
                        cycle=self.cycle,
                    )
        return [d for d in accepted if d.should_bet]

    # --- asynchronous placement --------------------------------------------

    @staticmethod
    def _decision_key(match_id: str, decision: Any) -> tuple[str, str, str]:
        """Identity of a bet: match + market + selection."""
        return (
            str(match_id or ""),
            str(getattr(decision.opportunity, "market_key", "")),
            str(getattr(decision.opportunity, "selection", "")).strip().lower(),
        )

    def _release_inflight(self, keys) -> None:
        """Drop reservations once the worker is done with a request."""
        with self._inflight_lock:
            for key in keys or ():
                self._inflight.pop(key, None)

    def _inflight_bets(self, match_id: str = "") -> list[dict]:
        """Queued bets as ledger-shaped rows, so the portfolio counts them.

        A queued bet is money already committed even though the ledger has
        not seen it yet (the ledger only records CONFIRMED tickets). Without
        this the next cycle re-plans the same selection as "new" and places
        it a second time.
        """
        with self._inflight_lock:
            rows = [dict(row) for row in self._inflight.values()]
        if not match_id:
            return rows
        return [row for row in rows if row.get("match_id") == match_id]

    def placement_status(self) -> dict:
        """Worker state for the heartbeat and the operations panel."""
        with self._inflight_lock:
            inflight = [dict(row) for row in self._inflight.values()]
        started = self._place_started_at
        return {
            "queue_depth": self._place_queue.qsize(),
            "stats": dict(self._place_stats),
            "inflight": inflight,
            "inflight_s": (
                round(time.monotonic() - started, 1)
                if started is not None
                else None
            ),
            "last": dict(self._place_last) if self._place_last else None,
        }

    def enqueue_place(self, decisions: list, state, cycle: int) -> int:
        """Hand accepted bets to the placement worker without blocking.

        The worker drains the queue one request at a time — the betslip is
        a singleton, so ordering is enforced — while the cycle continues
        capturing and pricing. The heartbeat queues/done/failed counters
        report placement progress independently of the cycle.

        Every queued bet is RESERVED in `_inflight` before it is handed
        over: the ledger only learns about a bet after the book confirms
        it, and without the reservation the next cycle would see the same
        selection as unbacked and place it again.
        """
        if not decisions or self.betslip is None or not self.allow_submission:
            return 0
        if self.policy is not None and not self.policy.allow_execution:
            return 0
        if self._place_stop.is_set():
            # The runner is shutting down; nothing would drain the queue.
            return 0
        if self._place_thread is None or not self._place_thread.is_alive():
            self._ensure_worker()
        match_id = getattr(state, "match_id", "") or ""
        fresh: list = []
        keys: list[tuple[str, str, str]] = []
        with self._inflight_lock:
            for d in decisions:
                if not getattr(d, "should_bet", False):
                    continue
                # The opportunity carries the match id the LEDGER and the
                # portfolio key on; state.match_id is only the fallback.
                mid = getattr(d.opportunity, "match_id", "") or match_id
                key = self._decision_key(mid, d)
                if key in self._inflight:
                    # Already queued or being placed; placing it twice is
                    # double exposure on the same outcome.
                    self._place_stats["duplicate"] += 1
                    continue
                self._inflight[key] = {
                    "match_id": mid,
                    "market_key": getattr(d.opportunity, "market_key", ""),
                    "selection": getattr(d.opportunity, "selection", ""),
                    "stake_ngn": float(getattr(d, "stake_ngn", 0.0) or 0.0),
                    "odds": float(getattr(d.opportunity, "odds", 0.0) or 0.0),
                    "ev": float(getattr(d.opportunity, "ev", 0.0) or 0.0),
                    "status": "pending",
                    # `placed_at` is the key the portfolio's accounting reads
                    # for its per-minute window; a reservation is committed
                    # money and must count against the rate limit now, not
                    # only once the ledger records the confirmed ticket.
                    "placed_at": datetime.now(timezone.utc).isoformat(),
                    "queued_at": datetime.now(timezone.utc).isoformat(),
                }
                fresh.append(d)
                keys.append(key)
        if not fresh:
            return 0
        if self._place_queue.qsize() >= MAX_PLACE_QUEUE:
            # A deep queue means placements are failing faster than they
            # complete; stacking more stale prices on it only makes the
            # backlog worse. Refuse and let the next cycle re-price.
            self._release_inflight(keys)
            self._place_stats["dropped"] += len(fresh)
            self.last_error = (
                f"placement queue full ({MAX_PLACE_QUEUE}) — "
                f"{len(fresh)} bet(s) dropped"
            )
            return 0
        self._place_stats["queued"] += len(fresh)
        self._place_queue.put(
            {
                "decisions": fresh,
                "state": state,
                "cycle": cycle,
                "enqueued_at": time.monotonic(),
                "keys": keys,
            }
        )
        return len(fresh)

    def _placement_worker(self) -> None:
        """Drain the placement queue one request at a time, forever.

        Never raises past the loop's lifetime: an exception inside a
        placement is contained per request, counters reflect reality, and
        the notifier receives the failure like a synchronous refusal did.
        """
        while not self._place_stop.is_set():
            try:
                req = self._place_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            decisions = req.get("decisions") or []
            keys = req.get("keys") or []
            enqueued_at = req.get("enqueued_at")
            age = (
                time.monotonic() - float(enqueued_at)
                if enqueued_at is not None
                else 0.0
            )
            self._place_started_at = time.monotonic()
            try:
                if age > PLACEMENT_MAX_AGE_S:
                    # The price was read before the queue backed up. The
                    # book's own price gate may not catch it: a decision
                    # priced before a goal is stale at the same odds.
                    self._place_stats["expired"] += len(decisions)
                    self._place_last = {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "cycle": req.get("cycle"),
                        "queued": len(decisions),
                        "placed": 0,
                        "error": f"request expired after {age:.0f}s",
                    }
                    continue
                placed = self.place_bets(
                    decisions, req["state"], req["cycle"]
                )
                self._place_stats["done"] += placed
                if placed < len(decisions):
                    self._place_stats["failed"] += len(decisions) - placed
                self._place_last = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "cycle": req.get("cycle"),
                    "queued": len(decisions),
                    "placed": placed,
                    "error": (
                        None
                        if placed == len(decisions)
                        else self.last_error
                    ),
                }
            except Exception as e:  # noqa: BLE001
                self._place_stats["failed"] += len(decisions)
                self.last_error = f"placement worker: {e!r}"
                self._place_last = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "cycle": req.get("cycle"),
                    "queued": len(decisions),
                    "placed": 0,
                    "error": repr(e)[:300],
                }
                if self.notifier is not None:
                    try:
                        self.notifier.warn(
                            "Placement worker error",
                            repr(e)[:300],
                            cycle=req["cycle"],
                        )
                    except Exception:
                        pass
            finally:
                self._release_inflight(keys)
                self._place_started_at = None
                self._place_queue.task_done()

    def _drop_queued_placements(self, reason: str) -> int:
        """Throw away queued placements WITHOUT touching the page.

        Used when the session expires: the queue was priced against a board
        the account can no longer bet on, and the login modal reuses the
        betslip's own component classes. Draining it would drive the login
        form (2026-10-30: stakes typed into the phone field).
        """
        dropped = 0
        while True:
            try:
                req = self._place_queue.get_nowait()
            except queue.Empty:
                break
            decisions = req.get("decisions") or []
            try:
                self._release_inflight(req.get("keys") or [])
                self._place_stats["dropped"] += len(decisions)
                dropped += len(decisions)
            finally:
                self._place_queue.task_done()
        if dropped:
            self.last_error = f"{reason} — {dropped} queued bet(s) dropped"
            self._place_last = {
                "at": datetime.now(timezone.utc).isoformat(),
                "cycle": self.cycle,
                "queued": dropped,
                "placed": 0,
                "error": f"{reason} — queued bet(s) dropped",
            }
        return dropped

    def drain(self, timeout: float = 30.0) -> bool:
        """Wait for queued placements to finish. True if the queue emptied."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._place_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.1)
        return self._place_queue.unfinished_tasks == 0

    def shutdown(self, timeout: float = 30.0) -> None:
        """Let queued placements finish, then stop the worker.

        Stopping the process with a bet in flight would leave a ticket the
        ledger never records — a phantom position with real money on it.
        """
        self.drain(timeout=timeout)
        self._place_stop.set()
        thread = self._place_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._place_thread = None

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
        # The session is the operator's LOGIN. When it has expired the page is
        # a login modal built from the same components as the betslip — the
        # stake box and the login phone field are both `input.m-input`, and the
        # modal's Login button is the same `.af-button--primary` — so nothing
        # may be typed or clicked. On 2026-10-30 a queue filled BEFORE the
        # logout kept draining against that modal and typed a stake into the
        # phone field ('+234 10.00'). Check the PAGE, not the queue's age.
        if self.capture is not None and hasattr(self.capture, "session_expired"):
            try:
                if self.capture.session_expired().get("expired"):
                    self.last_error = (
                        "session expired (login form showing) — placement refused"
                    )
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Session expired — placement refused",
                            f"{len(decisions)} bet(s) not submitted",
                            cycle=cycle,
                        )
                    return 0
            except Exception:
                # An unreadable page is not a licence to click: the guards in
                # the betslip payloads refuse on a login form independently.
                pass
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
        # The WHOLE submission window (slip clear -> add -> Confirm ->
        # receipt poll) holds the page lock: the cycle's navigation dance
        # (load_board/switch_match) must never interleave with an
        # in-flight placement on the shared tab. The lock is taken with a
        # TIMEOUT: a wedged browser call must skip this placement, not
        # freeze the worker forever.
        if not self._page_lock.acquire(timeout=PAGE_LOCK_TIMEOUT_S):
            self.last_error = (
                f"page busy for >{PAGE_LOCK_TIMEOUT_S:.0f}s — "
                f"{len(decisions)} bet(s) not submitted"
            )
            if self.notifier is not None:
                self.notifier.warn(
                    "Page lock timeout — placement skipped",
                    self.last_error,
                    cycle=cycle,
                )
            return 0
        try:
            placed = self._place_bets_locked(decisions, state, cycle)
        finally:
            self._page_lock.release()
        return placed

    def _avoid_key(self, state, d) -> tuple[str, str, str]:
        """Identity of a refusable bet: match + market + selection."""
        return (
            str(getattr(state, "match_id", "") or ""),
            str(d.opportunity.market_key or ""),
            str(d.opportunity.selection or ""),
        )

    def _cool_off(self, key: tuple[str, str, str]) -> None:
        """Put a refused market on cooldown, expiring itself."""
        self._avoid[key] = time.monotonic() + MARKET_FAILURE_COOLDOWN_S

    def _is_avoided(self, key: tuple[str, str, str]) -> bool:
        exp = self._avoid.get(key)
        return exp is not None and exp > time.monotonic()

    def _avoid_prune(self) -> None:
        now = time.monotonic()
        for k in [k for k, exp in self._avoid.items() if exp <= now]:
            self._avoid.pop(k, None)

    def _arm_unconfirmed_probe(self) -> None:
        """Schedule the post-unconfirmed balance probe.

        The baseline is the balance OBSERVED before this write is possible —
        or, on a fresh process that has never read it, the first reading
        after arming (the probe then compares the FOLLOWING reading to it).
        """
        self._unconfirmed_at = time.monotonic()
        self._probe_at = time.monotonic() + BALANCE_PROBE_DELAY_S
        if self._probe_baseline is None:
            self._probe_baseline = self._balance_observed

    def _check_balance_probe(self, bal: Optional[float]) -> None:
        """Run the pending probe on a fresh balance reading. ONE-SHOT.

        A drop past the baseline within the probe window means the book
        took money the ledger does not hold — a critical alert, once. No
        drop closes the case. No reading inside the window leaves the
        probe pending; waiting past 5 minutes without any reading gives up
        with a warn (the tab may not be reading a header at all).
        """
        now = time.monotonic()
        if self._probe_at is None:
            return
        # A fresh reading before the deadline arrives BEFORE the probe
        # fires: from it the probe learns the safe baseline (the phantom
        # may have already dropped the account, so 'first read after
        # arm' is only a baseline when nothing else confirms it — the
        # honest check is the NEXT reading against THIS one).
        if self._probe_baseline is None and bal is not None:
            self._probe_baseline = bal
            return
        if bal is None:
            if now - self._unconfirmed_at > BALANCE_PROBE_MAX_WAIT_S:
                self._probe_at = None
                self._probe_baseline = None
                if self.notifier is not None:
                    self.notifier.warn(
                        "Phantom probe never got a balance reading",
                        f"price-unconfirmed attempt at {self.cycle}; verify "
                        "Bet History against the ledger manually",
                        cycle=self.cycle,
                    )
            return
        if now < self._probe_at:
            return
        baseline = self._probe_baseline
        self._probe_at = None
        self._probe_baseline = None
        if baseline is not None and bal < baseline - 0.5:
            self._balance_alert_fired = True
            if self.notifier is not None:
                self.notifier.critical(
                    "Balance dropped after an unconfirmed placement",
                    f"account went {baseline:.2f} -> {bal:.2f} NGN ~30s after "
                    "a placement that never showed a receipt — a live bet may "
                    "be missing from the ledger; check Bet History and "
                    "reconcile",
                    cycle=self.cycle,
                )

    def _place_bets_locked(self, decisions: list, state, cycle: int) -> int:
        placed = 0
        self._avoid_prune()
        for d in decisions:
            if not d.should_bet:
                continue
            # A market the book refused recently is skipped WITHOUT an
            # alert: the refusal already sounded this cycle's email, and
            # retrying a suspended market every cycle is spam (2026-10-29:
            # 8 failures on two O/U lines at HT).
            key = self._avoid_key(state, d)
            if self._is_avoided(key):
                self._last_avoided = (
                    f"{d.opportunity.market_key}/{d.opportunity.selection}"
                )
                continue
            # Re-check the stop before EVERY bet, not just once per cycle.
            if self.ledger.is_daily_stop_active():
                break
            # Re-check the sanity gates here as well: a decision can be
            # handed to place_bets un-gated (tests, manual calls, future
            # board-mode paths), and this is the last point before money.
            if not self.apply_policy_gates([d]) or not d.should_bet:
                continue
            # The operator's printed stake cap is the LAST wall. Whatever
            # posture, saved state or sizing math produced, a stake above
            # it never reaches the book (2026-10-28: stale state held
            # max_stake_per_bet=100 against an explicit NGN 10 CLI).
            if (
                self.operator_max_stake is not None
                and float(d.stake_ngn or 0.0) > self.operator_max_stake + 0.01
            ):
                self.last_error = (
                    f"stake {float(d.stake_ngn):.2f} exceeds the operator "
                    f"cap {self.operator_max_stake:.2f} — refused"
                )
                if self.notifier is not None:
                    self.notifier.warn(
                        "Stake over operator cap — bet refused",
                        f"{d.opportunity.market_key} {d.opportunity.selection} "
                        f"stake {d.stake_ngn:.0f} > {self.operator_max_stake:.0f}",
                        cycle=cycle,
                    )
                continue
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
                # SIM-mode is enforced behaviorally AFTER the population
                # step (the slip itself reports whether it is a virtual
                # simulation slip), and place_bet re-judges before the
                # click. The visual REAL|SIM toggle proved unreliable on
                # this UI build (its clicks do not register and its class
                # semantics lie), so it must not be a pre-condition here —
                # that only produced false refusals of real bets.
                # Find the selection by NAME: the price may have moved since
                # it was priced, and that must be caught, not clicked.
                # There is deliberately NO fallback that clicks by price: a
                # price string can match a cell in a DIFFERENT market table,
                # and the slip would then hold a bet nobody priced. If the
                # named selection is not on the board, the bet is refused.
                added = self.betslip.add_selection(label, d.opportunity.selection)
                if not added or not added.get("ok"):
                    self.last_error = f"selection not on the board: {label} {d.opportunity.selection}"
                    # A selection that is not on the board is usually a
                    # suspended/renamed market column: do not re-click it
                    # every cycle.
                    self._cool_off(key)
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Selection not on the board — bet skipped",
                            f"{label} / {d.opportunity.selection}",
                            cycle=cycle,
                        )
                    continue
                self.betslip.set_stake(d.stake_ngn)
                slip = self.betslip.read_slip()
                if slip.session_expired:
                    # The page turned into the login modal under the request.
                    # Nothing else in this batch may be attempted.
                    self.last_error = (
                        "session expired mid-placement — batch abandoned"
                    )
                    break
                # A simulation slip is virtual money: its bets never touch
                # the account. Detected from the slip itself, not from the
                # (unreliable) REAL|SIM toggle widget.
                if slip.is_sim:
                    self.betslip.clear()
                    self.last_error = "the slip is a SIMULATION slip — refused"
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Simulation slip — bet refused",
                            f"{label} / {d.opportunity.selection}",
                            cycle=cycle,
                        )
                    continue
                # Identity check: the slip's own text must carry the market
                # label AND the selection that was priced. Yesterday's
                # phantom came from a click whose slip nobody identified;
                # a slip that cannot name what it holds never moves money.
                slip_text_l = (slip.raw_text or "").lower()
                label_ok = (
                    not label
                    or str(label).lower() in slip_text_l
                )
                sel_ok = str(d.opportunity.selection).lower() in slip_text_l
                # A slip rendered by the simulator says so ('Simulate',
                # 'virtually simulated'); its bets never hit the real book.
                looks_sim = "simulat" in slip_text_l
                if not (label_ok and sel_ok) or looks_sim:
                    self.betslip.clear()
                    self.last_error = (
                        "slip does not contain the priced market/selection — refused"
                        if not looks_sim
                        else "the slip is a SIMULATION slip — refused"
                    )
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Slip identity mismatch — bet refused",
                            f"expected {label} / {d.opportunity.selection}; "
                            f"slip text: {slip.raw_text[:120]}",
                            cycle=cycle,
                        )
                    continue
                accepted = slip.accepted_odds
                # Stake fidelity: the slip's OWN total must be the planned
                # stake. The balance-insufficient dialog after Confirm
                # (2026-10-28, Overtime U2.5) proved a slip can reach the
                # book with a stake nobody planned — leftover selections
                # and defaults blend into a bigger total. The identity and
                # odds gates above cannot see that; only the total can.
                slip_total = getattr(slip, "total_stake_ngn", None) or 0.0
                planned = float(d.stake_ngn or 0.0)
                if planned > 0 and abs(slip_total - planned) > 1.0:
                    self.betslip.clear()
                    self.last_error = (
                        f"slip stake {slip_total:.2f} != planned {planned:.2f} — refused"
                    )
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Slip stake mismatch — bet refused",
                            f"{label} / {d.opportunity.selection}: slip total "
                            f"{slip_total:.2f}, planned {planned:.2f}",
                            cycle=cycle,
                        )
                    continue

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
                    # WHY the book would not take the slip, from the slip's
                    # own signals: no button at all means the selection
                    # never landed; a present-but-disabled button means the
                    # book is validating or the market is suspended; the
                    # slip's 'Unavailable' text means suspension outright.
                    diag = getattr(slip, "diagnostic", None) or {}
                    why = (
                        f"unavailable={bool(slip.unavailable)} "
                        f"needsAccept={bool(slip.needs_accept)} "
                        f"button={'present' if getattr(slip, 'has_place_button', False) else 'absent'} "
                        f"info={json.dumps(diag)[:120]}"
                    )
                    self.last_error = (
                        "the slip is not placeable (the book refused it) — "
                        f"{why}"
                    )
                    # A refused slip is not re-tried blind: cooldown keeps
                    # the next cycles on other markets instead of burning
                    # the placement window on a suspended line.
                    self._cool_off(key)
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Slip not placeable — bet skipped",
                            f"{label} / {d.opportunity.selection}: {why}",
                            cycle=cycle,
                        )
                    continue
                result = self.betslip.place_bet()
                # The book must CONFIRM the ticket. A failed or unconfirmed
                # submission is NOT a placed bet: recording it would put a
                # phantom position into the ledger's exposure and P&L, and
                # settlement would settle a bet that never existed. The
                # verified case is the only case that becomes a position.
                confirmed = (
                    isinstance(result, dict)
                    and bool(result.get("ok"))
                )
                if not confirmed:
                    error = (
                        result.get("error") if isinstance(result, dict) else "no result"
                    )
                    self.last_error = f"placement not confirmed by the book: {error}"
                    self._arm_unconfirmed_probe()
                    if self.notifier is not None:
                        self.notifier.warn(
                            "Placement unconfirmed — NOT recorded",
                            f"{label} / {d.opportunity.selection}: {error}",
                            cycle=cycle,
                        )
                    try:
                        self.betslip.clear()
                    except Exception:
                        pass
                    continue
                placed += 1
                self._unconfirmed_at = None
                # A CONFIRMED bet owns the balance drop: the probe is
                # superseded, otherwise the next legitimate placement would
                # trip a phantom-position alert (2026-10-31).
                self._probe_at = None
                self._probe_baseline = None
                self._balance_alert_fired = False
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
            # A scheduled/pre-match row carries no live period. Opening it
            # trades live pricing for a pre-match panel, so it is excluded
            # even when its kick-off time looks like a clock.
            if not getattr(c, "live", True):
                continue
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
        """Pick and open the best live match on the board.

        Navigation runs under the page lock: an in-flight placement (the
        worker thread) must never see the board reload mid-flow. If the
        page is busy the switch is SKIPPED, not queued — the next cycle
        retries, and navigating now would yank the tab out from under a
        live bet.
        """
        if not self._page_lock.acquire(timeout=PAGE_BUSY_TIMEOUT_S):
            self.last_error = "page busy — match switch skipped"
            return None
        try:
            candidates = []
            try:
                candidates = self.capture.board_candidates()
            except Exception:
                return None
            if not candidates:
                # The board may be parked on one match's detail view with the
                # list not rendered at all (single view), or on a match at
                # 120' AET whose markets are all suspended — zero rows exist to
                # click. Re-render the board the way the capture initialises
                # it (the direct live URL is never navigated to), then scan
                # once more before giving up.
                try:
                    self.capture.load_board(force=True)
                except Exception:
                    return None
                try:
                    candidates = self.capture.board_candidates()
                except Exception:
                    return None
            if not candidates:
                transport = getattr(
                    self.capture, "last_transport_error", ""
                )
                where = getattr(self.capture, "last_href", "") or "unknown page"
                self.last_error = (
                    "board scan failed — browser automation unreachable: "
                    f"{transport}"
                    if transport
                    else f"no live board rows on {where}"
                )
            chosen = self.choose_match(candidates)
            if chosen is None:
                return None
            try:
                self.capture.select_match(chosen.index)
            except Exception:
                return None
            return chosen
        finally:
            self._page_lock.release()

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
        # The operator must be told, and the supervisor tries to restore it:
        # the known case is handled deterministically (no model needed), and
        # the model is the fallback for the cases nobody has seen.
        if "session expired" in (result.reason or ""):
            # A queue filled BEFORE the logout must never be drained against
            # the login modal: its controls share the betslip's selectors, so
            # the worker would type stakes into the phone field (2026-10-30).
            # Drop the queued bets; the next cycle re-prices from scratch.
            self._drop_queued_placements("session expired")
            if self.notifier is not None:
                self.notifier.critical(
                    "SportyBet session expired",
                    "the board is showing the login form — restoring the session; "
                    "no bet can be placed until it is back",
                    cycle=self.cycle,
                )
            if self.supervisor is not None:
                outcome = self.supervisor.handle(result)
                report.reason = f"{result.reason} | {outcome.describe()}"
                if outcome.handled:
                    # The session is back: retry the cycle immediately.
                    retry = self.capture.capture()
                    if retry.ready and retry.state is not None:
                        result = retry
                        report.reason = "session restored, cycle continued"
                    else:
                        report.reason = f"session restored: {result.reason}"
                else:
                    report.reason = result.reason
            else:
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

        # Balance observation runs even on NOT-READY cycles: the header is
        # on every page, so the guard must not go blind just because no
        # panel is open (2026-10-28: account_balance_ngn stayed None all
        # night while the board header showed NGN 52.68). The panel's own
        # reading wins when present; the board header is the fallback.
        bal = (
            result.state.account_balance_ngn
            if result.state is not None
            else None
        )
        if bal is None and self.capture is not None:
            try:
                bal = self.capture.read_balance_ngn()
            except Exception:
                bal = None
        # The probe runs even on a failed read (bal None): that is exactly
        # when it must be able to give up loudly instead of staying armed.
        self._check_balance_probe(bal)
        if bal is not None:
            now = time.monotonic()
            # Safety net INSIDE the probe window: a reading that arrives
            # before the scheduled probe can still catch the drop (the
            # ledger-vs-book check uses the pre-attempt balance).
            if (
                self._probe_at is not None
                and self._balance_observed is not None
                and bal < self._balance_observed - 0.5
                and self.notifier is not None
                and not self._balance_alert_fired
            ):
                self._balance_alert_fired = True
                self.notifier.critical(
                    "Balance dropped after an unconfirmed placement",
                    f"account went {self._balance_observed:.2f} -> {bal:.2f} NGN "
                    "while an unconfirmed placement was un-resolved — a live "
                    "bet may be missing from the ledger; check Bet History "
                    "and reconcile",
                    cycle=self.cycle,
                )
            self._balance_observed = bal

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

        # (Balance observation happens above, before the readiness gate:
        # the header reading must be available even on an empty board.)

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
            # Size against the REAL observed account balance, not the static
            # --bankroll figure: 'the higher the amount available, the higher
            # the stake; the lower, the lower' — the drop to NGN 49 after
            # losses must shrink the stake math the same way growth raises it.
            bankroll = self.policy.bankroll_ngn
            if isinstance(self._balance_observed, (int, float)) and self._balance_observed > 0:
                bankroll = min(bankroll, self._balance_observed)
            base_config = replace(base_config, bankroll_ngn=bankroll)
        portfolio = SequentialPortfolio(
            self.posture.apply_to(base_config)
        )
        placed_bets = self.ledger.bets_for(result.match_id)
        # A bet handed to the placement worker is committed money that the
        # ledger does not hold yet. Counting it here keeps exposure,
        # correlation and rate limits honest across the queue window. The
        # portfolio filters to the opportunities' match ids itself, so ALL
        # reservations are passed rather than a possibly-mismatched subset.
        placed_bets = placed_bets + self._inflight_bets()
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
        accepted = self.apply_policy_gates(decisions)
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
            # Key by (market_key, selection): matching on the selection text
            # alone let a leg named "No" admit every unrelated "No" on the
            # board (btts No, clean sheet No, win-to-nil No, ...), so the
            # "superedge-only" posture would have staked bets that are not
            # part of the covering set at all.
            arb_ids = normalized_identity_set(arbs)
            accepted = [
                d
                for d in accepted
                if (
                    d.opportunity.market_key.strip().lower(),
                    d.opportunity.selection.strip().lower(),
                )
                in arb_ids
            ]
            report.planned = len(accepted)

        if accepted:
            # ASYNC placement: hand the accepted set to the worker and keep
            # the cycle moving. report.placed stays 0 here — real counts
            # surface through the heartbeat/ledger as the worker lands
            # them; blocking here would re-create the cycle-freeze wedge
            # this worker exists to prevent.
            self.enqueue_place(accepted, result.state, self.cycle)

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
        """Loop until stopped (or `max_cycles` is reached).

        One crashed cycle must never kill the loop. The watchdog restarts
        the PROCESS, but a restart costs minutes and loses the board; the
        exception is recorded, the heartbeat is still written (so the
        panel never reads STALE while the process is alive), and the next
        cycle runs.
        """
        self.load_state()
        n = 0
        try:
            while max_cycles is None or n < max_cycles:
                # Re-read the persisted posture each cycle: the dashboard
                # writes state.json when the operator steers, so a running
                # loop picks the change up on the next cycle without a
                # restart.
                self.load_state()
                # Liveness truth at the CYCLE START: a heartbeat is otherwise
                # only written at cycle end, so one slow step (a wedged page
                # read timing out at 45s + its CDP fallback) made the panel
                # show STALE while the process was alive and working
                # (2026-10-31). The ping says 'scanning' with the cycle
                # number, and the real report overwrites it at cycle end.
                self._heartbeat_scan_ping()
                try:
                    report = self.run_cycle()
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"cycle {self.cycle} crashed: {e!r}"
                    report = CycleReport(
                        cycle=self.cycle,
                        at=datetime.now(timezone.utc).isoformat(),
                        ready=False,
                        reason=self.last_error,
                    )
                    self.reports.append(report)
                    if self.notifier is not None:
                        try:
                            self.notifier.critical(
                                "Loop cycle crashed",
                                self.last_error[:300],
                                cycle=self.cycle,
                            )
                        except Exception:
                            pass
                self._write_heartbeat(report)
                n += 1
                if max_cycles is None:
                    time.sleep(self.cycle_s)
        finally:
            # Never abandon a queued bet on Ctrl-C: a submitted ticket the
            # ledger never records is a phantom position with real money
            # on it.
            self.drain(timeout=30.0)
        return self.reports

    def _heartbeat_scan_ping(self) -> None:
        """A cycle-start heartbeat: the loop is known-alive and scanning.

        `run_cycle` increments self.cycle on entry, so the ping announces
        the cycle number the END-report will carry (same file, same figure).
        """
        report = CycleReport(
            cycle=self.cycle + 1,
            at=datetime.now(timezone.utc).isoformat(),
            ready=False,
            reason=f"scanning (cycle {self.cycle + 1} starting)",
        )
        self._write_heartbeat(report)

    def _write_heartbeat(self, report: CycleReport) -> None:
        """Write the per-cycle heartbeat file.

        The operations panel reads this to know how FRESH the loop actually
        is: an old heartbeat is the unmissable sign the loop died while
        nobody was watching. The file lives next to state.json so both
        processes agree on where it is.
        """
        if not self.state_path:
            return
        path = self.state_path.parent / "heartbeat.json"
        try:
            status = self.placement_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "pid": os.getpid(),
                        "cycle_s": self.cycle_s,
                        "submission_enabled": self.allow_submission,
                        # 'balanced', not 'PostureMode.BALANCED': the panel
                        # renders this verbatim and the enum repr leaked
                        # into the operator display (2026-10-28).
                        "posture_mode": str(
                            getattr(
                                getattr(self.posture, "mode", ""),
                                "name",
                                getattr(self.posture, "mode", ""),
                            )
                        ).lower(),
                        "account_balance_ngn": self._balance_observed,
                        # WHAT PAGE the automation is actually on. Without
                        # this, "board empty" could not be told apart from
                        # "the tab is parked somewhere else".
                        "page_url": getattr(self.capture, "last_href", None),
                        "page_error": (
                            getattr(self.capture, "last_transport_error", "")
                            or None
                        ),
                        # True means the session check could not read the
                        # page: 'not expired' is then UNKNOWN, not proof of a
                        # live login. Betting stays blocked by the betslip's
                        # own login guard either way.
                        "session_probe_failed": getattr(
                            self.capture, "session_probe_failed", None
                        ),
                        "last_error": self.last_error,
                        # Markets the book refused recently and their
                        # cooldown — the panel must show WHY a planned
                        # bet is not being re-tried.
                        "avoid_markets": len(
                            [k for k, exp in self._avoid.items() if exp > time.monotonic()]
                        ),
                        "avoided_last_error": self._last_avoided,
                        # The post-unconfirmed balance probe: 'pending'
                        # means a receipt-missed bet is still un-resolved.
                        "balance_probe": "pending" if self._probe_at else None,
                        # Placement worker progress: the cycle no longer
                        # blocks on placement, so 'placed' in the report
                        # can lag real completions. The worker counters are
                        # the live truth; the watchdog and panel read both.
                        "place_queue_depth": status["queue_depth"],
                        "place_stats": status["stats"],
                        "place_inflight": status["inflight"],
                        "place_inflight_s": status["inflight_s"],
                        "place_last": status["last"],
                        **asdict(report),
                    },
                    f,
                    ensure_ascii=False,
                    indent=1,
                )
        except (OSError, TypeError):
            pass

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
        # The cashout tab is a PAGE MUTATION on the same tab the placement
        # worker drives. Before placement was asynchronous the two were
        # serialised inside one cycle; now the read must not interleave
        # with a betslip flow. A busy page is skipped, not waited on —
        # offers are re-read every cycle.
        if not self._page_lock.acquire(timeout=PAGE_BUSY_TIMEOUT_S):
            return []
        try:
            offers = self.cashout_driver.read_offers()
        except Exception:
            return []
        finally:
            self._page_lock.release()
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
                if offer is None:
                    continue
                # A cashout click is a page mutation too: never interleave
                # it with the placement worker's betslip flow.
                if not self._page_lock.acquire(timeout=PAGE_BUSY_TIMEOUT_S):
                    continue
                try:
                    self.cashout_driver.execute(offer.index)
                finally:
                    self._page_lock.release()
        return valuations
