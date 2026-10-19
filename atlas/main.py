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


def cmd_plan(args: argparse.Namespace) -> int:
    """Full snapshot: price every open market, run the sequential portfolio,
    record accepted bets in the ledger (observe-only).

    The ledger is the single source of truth for exposure, P&L and the hard
    daily stop, so re-running this command against the same ledger reflects
    real operational state.
    """
    from atlas.live.loop import Opportunity
    from atlas.ops.ledger import BetLedger, LedgerConfig
    from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

    try:
        fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[red]Cannot read fixture: {e}[/red]")
        return 1
    match_id = fixture["match_id"]
    meta = fixture.get("meta", {})
    observed_at = datetime.fromisoformat(meta.get("observed_at", ""))

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
            if (opp.ev_lower or opp.ev) >= args.min_ev:
                opportunities.append(opp)
    console.print(
        f"{len(opportunities)} selections clear the conservative EV floor "
        f"{args.min_ev:.2f} (of {sum(len(r.selections) for r in open_records)} open odds; "
        f"{blocked} abstained/blocked)"
    )

    match_state = {
        "period": state.period,
        "clock_minute": state.clock_minute,
        "score": (state.home.goals, state.away.goals),
        "events": fixture.get("events", []),
    }
    portfolio = SequentialPortfolio(SequentialConfig())
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
        "--daily-loss-limit",
        type=float,
        default=-500.0,
        help="Hard daily stop in NGN (default: -500)",
    )
    plan.set_defaults(func=cmd_plan)
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
