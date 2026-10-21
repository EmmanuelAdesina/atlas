"""Atlas V1 CLI — SportyBet Nigeria live single-match intelligence & Boss control plane.

Commands:
  discover   Ingest a browser-exported SportyBet snapshot JSON file, archive
             a raw snapshot, map to canonical records, print a summary
             (including Proportional & Shin de-vigged fair probs).

  live-demo  Replay a recorded live observation (fixture) through the full
             sub-millisecond Fast-Plane loop:
             normalize -> state-space / baseline estimate -> Shin de-vig ->
             implication check -> risk-gated decision -> calibration & Boss diagnostics.

  boss       Slow-plane Executive Portfolio Manager / CRO diagnostics & calibration
             report in structured JSON (MCP/agent-ready).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from atlas.collectors.sportybet import SportyBetCollector
from atlas.live.boss import BossController, diagnose_live_operation
from atlas.live.calibration import grade_decisions
from atlas.live.decisions import DecisionLog, DecisionRecord, PolicyConfig
from atlas.live.estimate import (
    CalibratedModel,
    LiveStateSpaceModel,
    MarketModel,
    NoMoreGoalsBaseline,
    check_implications,
)
from atlas.live.observe import LiveMatchState
from atlas.mappers.sportybet_mapper import map_sport_markets

console = Console()


def cmd_discover(args: argparse.Namespace) -> int:
    async def run() -> int:
        raw_path = Path(args.snapshot)
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[red]Cannot read snapshot file: {e}[/red]")
            return 1
        collector = SportyBetCollector()
        result = collector.ingest_snapshot(payload, source_url=args.source_url)
        if not result.is_success:
            console.print(
                f"[red]Collection failed ({result.status}): {result.error}[/red]"
            )
            return 1
        raws = result.raw_data or []
        records, skipped = map_sport_markets(
            raws,
            source_url=result.metadata.get("source_url", ""),
            collected_at=result.collected_at,
        )
        console.print(
            f"[green]Collected {len(raws)} markets -> {len(records)} canonical records "
            f"({len(skipped)} skipped).[/green]"
        )
        console.print(f"Snapshot: {result.metadata.get('snapshot_path')}")
        table = Table(
            "market", "selection", "odds", "implied", "fair (prop)", "fair (shin)", "shin_z"
        )
        for r in records[: args.limit]:
            fair_prop = r.fair_probs("proportional")
            fair_shin = r.fair_probs("shin")
            z = r.shin_z
            for s in r.selections:
                table.add_row(
                    r.market_key,
                    s.name,
                    f"{s.odds:.2f}",
                    f"{s.implied:.3f}",
                    f"{fair_prop[s.name]:.3f}",
                    f"{fair_shin[s.name]:.3f}",
                    f"{z:.4f}",
                )
        console.print(table)
        if skipped:
            console.print(f"[yellow]Skipped: {skipped}[/yellow]")
        return 0

    return asyncio.run(run())


def _build_state_from_fixture(fixture: dict, observed_at: datetime) -> LiveMatchState:
    match_id = fixture["match_id"]
    meta = fixture.get("meta", {})
    st = fixture["state"]
    return LiveMatchState(
        match_id=match_id,
        competition=st.get("competition"),
        home=st["home"],
        away=st["away"],
        period=st.get("period", "unknown"),
        clock_minute=st.get("clock_minute"),
        clock_label=st.get("clock_label"),
        half_home_goals=st.get("half_home_goals"),
        half_away_goals=st.get("half_away_goals"),
        substitutions_count=st.get("substitutions_count"),
        added_minutes_announced=st.get("added_minutes_announced"),
        remaining_unknown=st.get("remaining_unknown", False),
        observed_at=observed_at,
        source=meta.get("source", ""),
        source_url=meta.get("source_url", ""),
    )


def cmd_live_demo(args: argparse.Namespace) -> int:
    """Replay a recorded fixture through the sub-millisecond Fast-Plane intelligence loop."""
    try:
        fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[red]Cannot read fixture: {e}[/red]")
        return 1
    match_id = fixture["match_id"]
    meta = fixture.get("meta", {})
    observed_at = datetime.fromisoformat(meta.get("observed_at", ""))

    # The operator's posture decides the risk appetite for this snapshot.
    posture = propose_posture(
        args.instruction, use_llm=args.llm
    ) if args.instruction else posture_for_mode(DEFAULT_POSTURE)
    console.print(f"[bold]Posture:[/bold] {posture.describe()}")
    as_of = observed_at  # evaluate freshness as-of observation, not wall-clock

    for m in fixture.get("markets", []):
        m.setdefault("match_id", match_id)
    collector = SportyBetCollector()
    result = collector.ingest_snapshot(
        fixture["markets"],
        source_url=meta.get("source_url", ""),
        collected_at=observed_at,
    )
    records, skipped = map_sport_markets(
        result.raw_data or [],
        source_url=result.metadata.get("source_url", ""),
        collected_at=observed_at,
    )
    open_markets = [r for r in records if r.status == "open"]
    state = _build_state_from_fixture(fixture, observed_at)

    boss = BossController()
    policy = boss.apply_to_policy(PolicyConfig())
    adj = boss.state.prior_adjustments.get(match_id)

    model: MarketModel
    if getattr(args, "model", "state-space") == "baseline":
        model = NoMoreGoalsBaseline(now=as_of)
    else:
        model = LiveStateSpaceModel(
            now=as_of,
            prior_adj_home=adj.delta_home_lambda if adj else 0.0,
            prior_adj_away=adj.delta_away_lambda if adj else 0.0,
        )

    log = DecisionLog(Path(args.log), policy)
    fair: dict[tuple[str, str], float] = {}
    decisions_made: list[DecisionRecord] = []

    # Time the pure Fast-Plane mathematical evaluation loop (zero disk I/O)
    t0 = time.perf_counter_ns()
    staged_rows: list[tuple] = []
    for r in open_markets:
        fair.update({(r.market_key, k): v for k, v in r.fair_probs("shin").items()})
        for s in r.selections:
            est = model.estimate(state, r.market_key, s.name)
            ev = est.expected_value(s.odds)
            ev_low = est.conservative_expected_value(s.odds)
            staged_rows.append((r, s, est, ev, ev_low))
    implications = check_implications(state, fair)
    fast_plane_us = (time.perf_counter_ns() - t0) / 1000.0

    eval_rows: list[tuple] = []
    for r, s, est, ev, ev_low in staged_rows:
        rec = log.decide(match_id, r.market_key, s.name, s.odds, r.status, est)
        decisions_made.append(rec)
        eval_rows.append((r, s, est, ev, ev_low, rec))

    console.print(
        f"[bold]{meta.get('fixture', match_id)}[/bold]  "
        f"{state.home.name} {state.home.goals}:{state.away.goals} {state.away.name}  "
        f"({state.period}, {state.clock_label}) | "
        f"[cyan]Fast-Plane Math: {fast_plane_us:.1f} us[/cyan]"
    )
    console.print(
        f"Observed: {observed_at.isoformat()} | Model: {model.model_name} | "
        f"{len(records)} mapped, {len(open_markets)} open, {len(skipped)} suspended/skipped"
    )

    est_table = Table(
        "market",
        "selection",
        "odds",
        "model_p",
        "[p_low..p_high]",
        "conf",
        "EV",
        "EV_low",
        "decision",
        "reason",
    )
    for r, s, est, ev, ev_low, rec in eval_rows:
        bounds_str = (
            f"[{est.p_lower:.3f}..{est.p_upper:.3f}]"
            if (est.p_lower is not None and est.p_upper is not None)
            else "-"
        )
        est_table.add_row(
            r.market_key,
            s.name,
            f"{s.odds:.2f}",
            f"{est.p:.3f}" if est.p is not None else "-",
            bounds_str,
            est.confidence,
            f"{ev:+.3f}" if ev is not None else "-",
            f"{ev_low:+.3f}" if ev_low is not None else "-",
            rec.decision,
            rec.reason[:65],
        )
    console.print(est_table)

    console.print("[bold]Cross-market implications (Shin de-vigged fair probs):[/bold]")
    for chk in implications:
        rule = chk["rule"]
        if chk["status"] in ("consistent", "VIOLATION"):
            console.print(
                f"  {rule.antecedent} -> {rule.consequent}: "
                f"{chk['p_antecedent']:.3f} vs {chk['p_consequent']:.3f} "
                f"[{'red' if chk['status'] == 'VIOLATION' else 'green'}]{chk['status']}[/]"
            )
        elif chk["status"] == "skipped-suspended":
            console.print(
                f"  {rule.antecedent} -> {rule.consequent}: {chk['status']} ({rule.when})"
            )

    outcome = fixture.get("outcome", {})
    settled = outcome.get("settled", {})
    if settled:
        console.print(
            "[bold]Recorded outcome (observed later, never used as input):[/bold] "
            f"FT {outcome.get('home_goals')}:{outcome.get('away_goals')} — {settled}"
        )
        _, cal = grade_decisions(
            [d.__dict__ for d in decisions_made],
            settled_outcomes=settled,
            default_match_id=match_id,
        )
        console.print(
            f"[bold]Post-Settlement Calibration (§8):[/bold] "
            f"settled={cal.settled_records}, Brier={cal.brier_score:.4f}, "
            f"LogLoss={cal.log_loss:.4f}, mean_p={cal.mean_predicted_p:.3f}, "
            f"win_rate={cal.realized_win_rate:.3f}"
        )

    console.print(f"Decision log: {args.log} (execution disabled in V1)")
    if skipped:
        reasons = sorted({s.get("reason", "?") for s in skipped})
        console.print(
            f"[yellow]{len(skipped)} suspended/skipped, e.g.: {reasons[:3]}[/yellow]"
        )
    return 0


def cmd_boss(args: argparse.Namespace) -> int:
    """Slow-plane Executive Portfolio Manager JSON diagnostics & calibration output."""
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    match_id = fixture["match_id"]
    meta = fixture.get("meta", {})
    observed_at = datetime.fromisoformat(meta.get("observed_at", ""))

    # The operator's posture decides the risk appetite for this snapshot.
    posture = propose_posture(
        args.instruction, use_llm=args.llm
    ) if args.instruction else posture_for_mode(DEFAULT_POSTURE)
    console.print(f"[bold]Posture:[/bold] {posture.describe()}")
    for m in fixture.get("markets", []):
        m.setdefault("match_id", match_id)
    records, skipped = map_sport_markets(
        fixture.get("markets", []),
        source_url=meta.get("source_url", ""),
        collected_at=observed_at,
    )
    open_markets = [r for r in records if r.status == "open"]
    state = _build_state_from_fixture(fixture, observed_at)
    model = LiveStateSpaceModel(now=observed_at)
    log = DecisionLog(Path(args.log), PolicyConfig())

    t0 = time.perf_counter_ns()
    fair: dict[tuple[str, str], float] = {}
    staged: list[tuple] = []
    for r in open_markets:
        fair.update({(r.market_key, k): v for k, v in r.fair_probs("shin").items()})
        for s in r.selections:
            est = model.estimate(state, r.market_key, s.name)
            staged.append((r, s, est))
    implications = check_implications(state, fair)
    fast_plane_us = (time.perf_counter_ns() - t0) / 1000.0

    decisions: list[DecisionRecord] = [
        log.decide(match_id, r.market_key, s.name, s.odds, r.status, est)
        for r, s, est in staged
    ]

    diag = diagnose_live_operation(
        state=state,
        open_markets=open_markets,
        skipped_markets=skipped,
        decisions=decisions,
        implications=implications,
        fast_plane_latency_us=fast_plane_us,
        as_of=observed_at,
    )
    settled = fixture.get("outcome", {}).get("settled", {})
    if settled:
        _, cal = grade_decisions(
            [d.__dict__ for d in decisions],
            settled_outcomes=settled,
            default_match_id=match_id,
        )
        diag["calibration"] = cal.__dict__
    print(json.dumps(diag, indent=2, ensure_ascii=False))
    return 0


def cmd_posture(args: argparse.Namespace) -> int:
    """Steer the operating posture from an operator instruction.

    The instruction is interpreted into a posture, the posture is clamped to
    the hard invariants, and the result is what the portfolio will execute.
    With --llm and a configured API key the interpretation is done by a
    model; without one, the deterministic rule-based parser handles it.
    """
    from atlas.ops.notify import Notifier
    from atlas.ops.posture import DEFAULT_POSTURE, posture_for_mode, propose_posture

    notifier = Notifier(
        log_path=args.audit_log,
        webhook_url=args.webhook,
        console=True,
    )
    context = {}
    if args.fixture:
        try:
            fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
            context = {
                "match_id": fixture.get("match_id"),
                "markets": len(fixture.get("markets", [])),
                "observed_at": fixture.get("meta", {}).get("observed_at"),
            }
        except (OSError, json.JSONDecodeError):
            context = {}

    posture = propose_posture(args.instruction, context=context, use_llm=args.llm)
    console.print(f"[bold]Instruction:[/bold] {args.instruction!r}")
    console.print(f"[bold]Posture:[/bold] {posture.describe()}")
    console.print(f"[dim]source: {posture.source} | rationale: {posture.rationale}[/dim]")
    if posture.clamps:
        console.print(
            f"[yellow]Clamped to hard invariants: {'; '.join(posture.clamps)}[/yellow]"
        )

    from atlas.portfolio.sequential import SequentialConfig

    cfg = posture.apply_to(SequentialConfig())
    console.print(
        f"[bold]Portfolio now:[/bold] kelly {cfg.kelly_fraction:.2f}, "
        f"exposure cap NGN {cfg.max_exposure_per_match:.0f}, "
        f"{cfg.max_bets_per_snapshot} bet(s)/snapshot, "
        f"EV floor {posture.min_ev:.3f}, "
        f"max stake NGN {cfg.max_stake_per_bet:.0f}"
    )
    notifier.send(
        "Posture set",
        posture.describe(),
        **posture.to_dict(),
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the autonomous live loop.

    capture -> readiness -> price -> posture -> plan -> place -> record -> settle

    Money moves only when --submit is given; the loop is observe-only by
    default, which is how it should be run until it has been trusted.
    """
    from atlas.browser.betslip import BetslipDriver
    from atlas.browser.cashout import CashoutDriver
    from atlas.browser.webbridge import WebBridgeAdapter
    from atlas.live.decisions import PolicyConfig
    from atlas.live.capture import LiveCapture
    from atlas.live.runner import LiveRunner
    from atlas.ops.ledger import BetLedger, LedgerConfig
    from atlas.ops.notify import Notifier
    from atlas.ops.posture import DEFAULT_POSTURE, posture_for_mode, propose_posture

    adapter = WebBridgeAdapter(session=args.session)
    capture = LiveCapture(
        adapter,
        min_markets=args.min_markets,
        max_wait_s=args.max_wait,
    )
    ledger = BetLedger(
        LedgerConfig(daily_loss_limit=args.daily_loss_limit), log_path=args.ledger
    )
    notifier = Notifier(
        log_path=args.notifications,
        webhook_url=args.webhook,
        console=True,
    )
    # A posture is always built, so the run-level overrides (min_ev,
    # min_stake) apply whether or not an instruction was given.
    posture = (
        propose_posture(args.instruction, use_llm=args.llm)
        if args.instruction
        else posture_for_mode(DEFAULT_POSTURE)
    )
    if args.min_ev is not None:
        posture = replace(posture, min_ev=args.min_ev)
    if args.min_stake is not None:
        posture = replace(posture, min_stake_ngn=args.min_stake)
    if args.max_stake is not None:
        posture = replace(posture, max_stake_per_bet=args.max_stake)
    betslip = BetslipDriver(adapter, allow_submit=args.submit)
    # The cashout driver is only armed with --submit: a cashout moves money
    # exactly like a bet does.
    cashout_driver = CashoutDriver(adapter, allow_execute=args.submit)
    # The policy gate is the LAST word on execution. --submit is the only
    # thing that turns it on, and it stays off by default.
    policy = PolicyConfig(
        allow_execution=args.submit,
        bankroll_ngn=args.bankroll,
    )
    runner = LiveRunner(
        capture=capture,
        ledger=ledger,
        notifier=notifier,
        betslip=betslip,
        cashout_driver=cashout_driver,
        posture=posture,
        policy=policy,
        cycle_s=args.cycle_seconds,
        allow_submission=args.submit,
        state_path=args.state,
        decision_log_path=args.decision_log,
    )
    runner.load_state()
    console.print(
        f"[bold]Atlas live loop[/bold] | session {args.session} | "
        f"ledger {args.ledger} | submission "
        f"[{'ENABLED' if args.submit else 'OFF (observe only)'}]"
    )
    console.print(f"Posture: {runner.posture.describe()}")
    if args.dashboard:
        from atlas.live.control import serve

        console.print(
            f"[cyan]Control dashboard on http://{args.host}:{args.port}[/cyan]"
        )
        serve(runner, host=args.host, port=args.port, notifier=notifier)
        return 0
    if not args.submit:
        console.print(
            "[yellow]Submission is OFF. The loop will plan but not place. "
            "Use --submit to arm execution.[/yellow]"
        )
    try:
        runner.run_forever(max_cycles=args.cycles)
    except KeyboardInterrupt:
        console.print("[yellow]loop interrupted[/yellow]")
    for report in runner.reports[-10:]:
        console.print(report.describe())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Full snapshot: price every open market, run the sequential portfolio,
    record accepted bets in the ledger (observe-only).

    The ledger is the single source of truth for exposure, P&L and the hard
    daily stop, so re-running this command against the same ledger reflects
    real operational state.
    """
    from atlas.live.extract import panel_is_ready
    from atlas.live.loop import Opportunity
    from atlas.ops.ledger import BetLedger, LedgerConfig
    from atlas.ops.posture import DEFAULT_POSTURE, posture_for_mode, propose_posture
    from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

    try:
        fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[red]Cannot read fixture: {e}[/red]")
        return 1
    match_id = fixture["match_id"]
    meta = fixture.get("meta", {})
    observed_at = datetime.fromisoformat(meta.get("observed_at", ""))

    # The operator's posture decides the risk appetite for this snapshot.
    posture = propose_posture(
        args.instruction, use_llm=args.llm
    ) if args.instruction else posture_for_mode(DEFAULT_POSTURE)
    console.print(f"[bold]Posture:[/bold] {posture.describe()}")

    ledger = BetLedger(
        LedgerConfig(daily_loss_limit=args.daily_loss_limit), log_path=args.ledger
    )
    stop_active = ledger.is_daily_stop_active()
    prior_bets = ledger.bets_for(match_id)
    console.print(
        f"[bold]{meta.get('fixture', match_id)}[/bold] | ledger "
        f"{args.ledger}: {len(prior_bets)} prior bet(s) on this match, "
        f"daily P&L NGN {ledger.daily_pnl():.0f}"
    )
    if stop_active:
        console.print(
            f"[red]HARD STOP active — daily P&L NGN {ledger.daily_pnl():.0f} "
            f"breached {args.daily_loss_limit:.0f}. No new bets will be planned.[/red]"
        )

    for m in fixture.get("markets", []):
        m.setdefault("match_id", match_id)
    collector = SportyBetCollector()
    result = collector.ingest_snapshot(
        fixture["markets"],
        source_url=meta.get("source_url", ""),
        collected_at=observed_at,
    )
    records, skipped = map_sport_markets(
        result.raw_data or [],
        source_url=result.metadata.get("source_url", ""),
        collected_at=observed_at,
    )
    open_records = [r for r in records if r.status == "open"]
    state = _build_state_from_fixture(fixture, observed_at)
    console.print(
        f"{state.home.name} {state.home.goals}:{state.away.goals} "
        f"{state.away.name} ({state.period}, {state.clock_label}) | "
        f"{len(records)} mapped, {len(open_records)} open, "
        f"{len(skipped)} suspended/skipped"
    )

    # Readiness gate: never act on a half-rendered board.
    raw_markets = [
        {
            "status": "open" if len(r.selections) >= 2 else "suspended",
        }
        for r in records
    ]
    ready, why = panel_is_ready(state, raw_markets, min_markets=args.min_markets)
    if not ready and not args.force:
        console.print(
            f"[yellow]Panel not ready ({why}). Nothing is planned. "
            f"Use --force to plan anyway.[/yellow]"
        )
        return 0
    if not ready:
        console.print(f"[yellow]Panel not ready ({why}) but --force was given.[/yellow]")

    model = LiveStateSpaceModel(now=observed_at)
    # Market-implied (Shin de-vigged) probabilities, used for calibration and
    # for the 5pp model-market gap guard that blocks fake edges.
    market_fair: dict[tuple[str, str], float] = {}
    for r in open_records:
        market_fair.update({(r.market_key, k): v for k, v in r.fair_probs("shin").items()})
    calibrated = CalibratedModel(base_model=model, market_fair=market_fair)

    opportunities: list[Opportunity] = []
    blocked = 0
    for r in open_records:
        for s in r.selections:
            est = calibrated.estimate(state, r.market_key, s.name)
            if est.abstained or est.p is None:
                blocked += 1
                continue
            opp = Opportunity(
                match_id=match_id,
                market_key=r.market_key,
                selection=s.name,
                odds=s.odds,
                model_p=est.p,
                model_name=est.model_name,
                p_lower=est.p_lower,
                p_push=est.p_push,
            )
            if (opp.ev_lower or opp.ev) >= max(args.min_ev, posture.min_ev):
                opportunities.append(opp)
    console.print(
        f"{len(opportunities)} selections clear the conservative EV floor "
        f"{max(args.min_ev, posture.min_ev):.2f} (of "
        f"{sum(len(r.selections) for r in open_records)} open odds; "
        f"{blocked} abstained/blocked)"
    )

    # Sure-thing scan: a covering set whose inverse prices sum below one is a
    # guaranteed profit, independent of how the match ends.
    from atlas.live.arb import Selection, best_arbs

    arb_selections = [
        Selection(
            market_key=o.market_key,
            selection=o.selection,
            odds=o.odds,
            model_p=o.model_p,
        )
        for o in opportunities
    ]
    arbs = best_arbs(arb_selections, min_edge=args.min_arb_edge)
    if arbs:
        console.print(
            f"[bold green]SURE-THING ARBS:[/bold green] {len(arbs)} covering set(s) "
            f"profit regardless of the result"
        )
        for arb in arbs[:8]:
            console.print(f"  {arb.describe()}")
            for name, stake, payout in arb.stakes():
                console.print(
                    f"      {name}: stake {stake:.4f} -> pays {payout:.4f} if it lands"
                )
    else:
        console.print("[dim]No sure-thing arb on this board.[/dim]")

    match_state = {
        "period": state.period,
        "clock_minute": state.clock_minute,
        "score": (state.home.goals, state.away.goals),
        "events": fixture.get("events", []),
    }
    portfolio = SequentialPortfolio(posture.apply_to(SequentialConfig()))
    decisions = portfolio.evaluate_snapshot(opportunities, match_state, prior_bets)

    table = Table("market", "selection", "odds", "EV", "stake", "action", "reason")
    for d in decisions[:25]:
        table.add_row(
            d.opportunity.market_key,
            d.opportunity.selection,
            f"{d.opportunity.odds:.2f}",
            f"{d.opportunity.ev:+.3f}",
            f"{d.stake_ngn:.0f}",
            "BET" if d.should_bet else "pass",
            d.reason[:52],
        )
    console.print(table)

    accepted = [d for d in decisions if d.should_bet and not stop_active]

    if posture.require_guaranteed_profit:
        # Superedge-only: only legs of a sure-thing covering set may be staked.
        arb_legs = {
            (d.opportunity.market_key, d.opportunity.selection)
            for arb in arbs
            for d in decisions
            if d.opportunity.market_key == arb.family
        }
        arb_names = {
            (d.opportunity.market_key, d.opportunity.selection)
            for arb in arbs
            for name, _ in arb.legs
            for d in decisions
            if d.opportunity.selection == name
        }
        allowed = arb_legs | arb_names
        before = len(accepted)
        accepted = [
            d for d in accepted
            if (d.opportunity.market_key, d.opportunity.selection) in allowed
        ]
        if before != len(accepted):
            console.print(
                f"[dim]Superedge-only posture: dropped {before - len(accepted)} "
                f"bet(s) that are not part of a guaranteed-profit set.[/dim]"
            )
    for d in accepted:
        bet_id = f"{match_id}:{d.opportunity.market_key}:{d.opportunity.selection}"
        ok = ledger.record_pending(
            {
                "bet_id": bet_id,
                "match_id": match_id,
                "market_key": d.opportunity.market_key,
                "selection": d.opportunity.selection,
                "odds": d.opportunity.odds,
                "stake_ngn": d.stake_ngn,
                "model_p": d.opportunity.model_p,
                "p_lower": d.opportunity.p_lower,
                "ev": d.opportunity.ev,
                "confidence": "unknown",
            }
        )
        status = (
            "[green]recorded[/green]" if ok else "[yellow]rejected by policy gate[/yellow]"
        )
        console.print(
            f"BET {d.opportunity.market_key} '{d.opportunity.selection}' "
            f"NGN {d.stake_ngn:.0f} @ {d.opportunity.odds:.2f} -> {status}"
        )

    summary = ledger.summary()
    console.print(
        f"[bold]Plan:[/bold] {len(accepted)} bet(s) in this snapshot | "
        f"match exposure now NGN {ledger.match_exposure(match_id):.0f} "
        f"(cap {SequentialConfig().max_exposure_per_match:.0f}) | "
        f"ledger: {summary['total_bets']} bets, daily P&L NGN "
        f"{summary['daily_pnl_ngn']:.0f}, open exposure NGN "
        f"{summary['unrealized_exposure_ngn']:.0f}, stop active={summary['daily_stop_active']}"
    )
    console.print(f"Recorded in {args.ledger} (observe-only — nothing submitted)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atlas", description="Atlas V1 SportyBet Nigeria intelligence pipeline"
    )
    sub = p.add_subparsers(dest="command")
    d = sub.add_parser("discover", help="Ingest a SportyBet snapshot file (read-only)")
    d.add_argument(
        "--snapshot", required=True, help="Path to browser-exported snapshot JSON"
    )
    d.add_argument(
        "--source-url",
        default="browser://sportybet",
        help="Page URL the snapshot came from",
    )
    d.add_argument(
        "--limit", type=int, default=10, help="Markets to display (default 10)"
    )
    d.set_defaults(func=cmd_discover)

    demo = sub.add_parser(
        "live-demo", help="Replay a recorded live observation (observe-only)"
    )
    demo.add_argument(
        "--fixture", required=True, help="Path to recorded observation JSON"
    )
    demo.add_argument(
        "--log",
        default="data/decisions/demo.jsonl",
        help="Decision log output path",
    )
    demo.add_argument(
        "--model",
        choices=["state-space", "baseline"],
        default="state-space",
        help="Pricing model family (default: state-space)",
    )
    demo.set_defaults(func=cmd_live_demo)

    b = sub.add_parser(
        "boss",
        help="Slow-plane Executive Portfolio Manager JSON diagnostics & calibration",
    )
    b.add_argument("--fixture", required=True, help="Path to recorded observation JSON")
    b.add_argument(
        "--log",
        default="data/decisions/boss_audit.jsonl",
        help="Decision log output path",
    )
    b.set_defaults(func=cmd_boss)

    plan = sub.add_parser(
        "plan",
        help="Full snapshot: price every open market, run the sequential "
             "portfolio, and record bets in the ledger (observe-only).",
    )
    plan.add_argument("--fixture", required=True, help="Recorded observation JSON")
    plan.add_argument("--ledger", default="data/ops/ledger.jsonl", help="Ledger JSONL path")
    plan.add_argument("--min-ev", type=float, default=0.05, help="EV floor")
    plan.add_argument(
        "--force",
        action="store_true",
        help="Plan even if the panel readiness gate fails",
    )
    plan.add_argument(
        "--min-markets",
        type=int,
        default=1,
        help="Minimum open markets for the panel to count as ready "
             "(a live board shows dozens)",
    )
    plan.add_argument(
        "--instruction",
        default=None,
        help="Operator instruction that sets the posture for this snapshot "
             "(e.g. 'low risk and more revenue')",
    )
    plan.add_argument(
        "--llm",
        action="store_true",
        help="Use the LLM proposer for --instruction (needs ATLAS_BOSS_API_KEY)",
    )
    plan.add_argument(
        "--min-arb-edge",
        type=float,
        default=0.005,
        help="Minimum sure-thing arb edge over the implied margin (default: 0.005)",
    )
    plan.add_argument(
        "--daily-loss-limit",
        type=float,
        default=-500.0,
        help="Hard daily stop in NGN (default: -500)",
    )
    plan.set_defaults(func=cmd_plan)

    posture_cmd = sub.add_parser(
        "posture",
        help="Interpret an operator instruction into a validated operating posture",
    )
    posture_cmd.add_argument("instruction", help="e.g. 'more revenue and low risk'")
    posture_cmd.add_argument("--fixture", help="Optional fixture for context")
    posture_cmd.add_argument(
        "--llm",
        action="store_true",
        help="Use the LLM proposer (needs ATLAS_BOSS_API_KEY); falls back to rules",
    )
    posture_cmd.add_argument(
        "--audit-log",
        default="data/decisions/notifications.jsonl",
        help="Notification audit log path",
    )
    posture_cmd.add_argument(
        "--webhook",
        default=None,
        help="Optional webhook for custom notifications (or ATLAS_WEBHOOK_URL)",
    )
    posture_cmd.set_defaults(func=cmd_posture)

    run = sub.add_parser(
        "run",
        help="Run the autonomous live loop (capture -> price -> plan -> place -> settle)",
    )
    run.add_argument("--session", default="atlas-live", help="WebBridge session")
    run.add_argument("--ledger", default="data/ops/ledger.jsonl", help="Ledger JSONL path")
    run.add_argument(
        "--daily-loss-limit", type=float, default=-500.0, help="Hard daily stop (NGN)"
    )
    run.add_argument(
        "--instruction", default=None, help="Initial posture instruction"
    )
    run.add_argument(
        "--llm", action="store_true", help="Use the LLM proposer for --instruction"
    )
    run.add_argument(
        "--submit",
        action="store_true",
        help="ACTUALLY PLACE BETS (off by default: the loop is observe-only)",
    )
    run.add_argument(
        "--cycles", type=int, default=None, help="Stop after N cycles (default: run forever)"
    )
    run.add_argument(
        "--cycle-seconds", type=float, default=30.0, help="Seconds between cycles"
    )
    run.add_argument(
        "--min-markets", type=int, default=10, help="Minimum open markets for a ready panel"
    )
    run.add_argument(
        "--max-wait", type=float, default=90.0, help="Seconds to wait for a ready panel"
    )
    run.add_argument(
        "--notifications",
        default="data/decisions/notifications.jsonl",
        help="Notification audit log path",
    )
    run.add_argument("--webhook", default=None, help="Optional notification webhook")
    run.add_argument(
        "--state", default="data/ops/state.json", help="Persisted posture path"
    )
    run.add_argument(
        "--min-ev",
        type=float,
        default=None,
        help="Override the EV floor for this run. 0.0 forces a placement "
             "MECHANICS TEST with no edge requirement — use only to verify "
             "the submission path, never in normal operation.",
    )
    run.add_argument(
        "--max-stake",
        type=float,
        default=None,
        help="Maximum stake per bet in NGN (default: the posture's cap)",
    )
    run.add_argument(
        "--min-stake",
        type=float,
        default=None,
        help="Minimum stake in NGN. SportyBet NG's floor is 10; the "
             "portfolio default is 100.",
    )
    run.add_argument(
        "--bankroll",
        type=float,
        default=100_000.0,
        help="Bankroll the stakes are sized against (NGN). Set this to the "
             "real account balance: the gap guard plus a 0.10 Kelly fraction "
             "cannot justify a NGN 100 minimum bet on a small bankroll.",
    )
    run.add_argument(
        "--decision-log",
        default="data/decisions/loop_decisions.jsonl",
        help="Where every cycle's decisions are logged",
    )
    run.add_argument(
        "--dashboard", action="store_true", help="Serve the control dashboard instead of looping"
    )
    run.add_argument(
        "--host",
        default=None,
        help="Dashboard bind address (default: ATLAS_HOST, else 127.0.0.1)",
    )
    run.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard port (default: ATLAS_PORT, else 8077)",
    )
    run.set_defaults(func=cmd_run)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        console.print("[green]Atlas V1 initialized...[/green]")
        console.print("Try: python -m atlas.main discover --limit 5")
        return
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
