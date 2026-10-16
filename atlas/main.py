"""Atlas V1 CLI — minimal command surface over the SportyBet research pipeline.

Commands:
  discover   Ingest a browser-exported SportyBet snapshot JSON file, archive
             a raw snapshot, map to canonical records, print a summary.
             Read-only; no keys needed. No automated bet placement in V1.

  live-demo  Replay a recorded live observation (fixture) through the full
             loop: normalize -> estimate -> implication check -> decide.
             Observe-only; execution is disabled. No live data is fabricated:
             the fixture carries its own observation timestamps and outcome.

  Snapshot JSON format: a list of markets, each
  {"match_id": ..., "market_key": ..., "market_label": ...,
   "selections": [{"name": ..., "odds": ...}], "status": "open"}.

`python -m atlas.main` with no args keeps the legacy hello output.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from atlas.collectors.sportybet import SportyBetCollector
from atlas.live.decisions import DecisionLog, PolicyConfig
from atlas.live.estimate import NoMoreGoalsBaseline, check_implications
from atlas.live.loop import expected_value
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
            console.print(f"[red]Collection failed ({result.status}): {result.error}[/red]")
            return 1
        raws = result.raw_data or []
        records, skipped = map_sport_markets(
            raws,
            source_url=result.metadata.get("source_url", ""),
            collected_at=result.collected_at,
        )
        console.print(f"[green]Collected {len(raws)} markets -> {len(records)} canonical records "
                      f"({len(skipped)} skipped).[/green]")
        console.print(f"Snapshot: {result.metadata.get('snapshot_path')}")
        table = Table("market", "selection", "odds", "implied", "fair")
        for r in records[: args.limit]:
            fair = r.fair_probs()
            for s in r.selections:
                table.add_row(
                    r.market_key, s.name, f"{s.odds:.2f}",
                    f"{s.implied:.3f}", f"{fair[s.name]:.3f}",
                )
        console.print(table)
        if skipped:
            console.print(f"[yellow]Skipped: {skipped}[/yellow]")
        return 0

    return asyncio.run(run())


def cmd_live_demo(args: argparse.Namespace) -> int:
    """Replay a recorded fixture: the loop as it would have run live."""
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
        fixture["markets"], source_url=meta.get("source_url", ""),
        collected_at=observed_at,
    )
    records, skipped = map_sport_markets(
        result.raw_data or [], source_url=result.metadata.get("source_url", ""),
        collected_at=observed_at,
    )
    open_markets = [r for r in records if r.status == "open"]

    st = fixture["state"]
    state = LiveMatchState(
        match_id=match_id, competition=st.get("competition"),
        home=st["home"], away=st["away"], period=st.get("period", "unknown"),
        clock_minute=st.get("clock_minute"), clock_label=st.get("clock_label"),
        half_home_goals=st.get("half_home_goals"),
        half_away_goals=st.get("half_away_goals"),
        remaining_unknown=st.get("remaining_unknown", False),
        observed_at=observed_at, source=meta.get("source", ""),
        source_url=meta.get("source_url", ""),
    )
    console.print(f"[bold]{meta.get('fixture', match_id)}[/bold]  "
                  f"{state.home.name} {state.home.goals}:{state.away.goals} {state.away.name}  "
                  f"({state.period}, {state.clock_label})")
    console.print(f"Observed: {observed_at.isoformat()} | "
                  f"{len(records)} mapped, {len(open_markets)} open, {len(skipped)} suspended/skipped")

    model = NoMoreGoalsBaseline(now=as_of)
    log = DecisionLog(Path(args.log), PolicyConfig())
    fair: dict[tuple[str, str], float] = {}
    est_table = Table("market", "selection", "odds", "model_p", "conf", "EV", "decision", "reason")
    for r in open_markets:
        fair.update({(r.market_key, k): v for k, v in r.fair_probs().items()})
        for s in r.selections:
            est = model.estimate(state, r.market_key, s.name)
            ev = expected_value(est.p, s.odds) if est.p is not None else None
            rec = log.decide(match_id, r.market_key, s.name, s.odds, r.status, est)
            est_table.add_row(
                r.market_key, s.name, f"{s.odds:.2f}",
                f"{est.p:.3f}" if est.p is not None else "-",
                est.confidence, f"{ev:+.3f}" if ev is not None else "-",
                rec.decision, (rec.reason[:70]),
            )
    # Markets outside the baseline family still get an explicit abstention record.
    console.print(est_table)

    console.print("[bold]Cross-market implications (de-vigged fair probs):[/bold]")
    for chk in check_implications(state, fair):
        rule = chk["rule"]
        if chk["status"] in ("consistent", "VIOLATION"):
            console.print(f"  {rule.antecedent} -> {rule.consequent}: "
                          f"{chk['p_antecedent']:.3f} vs {chk['p_consequent']:.3f} "
                          f"[{'red' if chk['status'] == 'VIOLATION' else 'green'}]{chk['status']}[/]")
        else:
            console.print(f"  {rule.antecedent} -> {rule.consequent}: {chk['status']} ({rule.when})")

    outcome = fixture.get("outcome", {})
    if outcome.get("settled"):
        console.print("[bold]Recorded outcome (observed later, not used for decisions):[/bold] "
                      f"FT {outcome.get('home_goals')}:{outcome.get('away_goals')} — {outcome['settled']}")
    console.print(f"Decision log: {args.log} (execution disabled in V1)")
    if skipped:
        reasons = sorted({s.get("reason", "?") for s in skipped})
        console.print(f"[yellow]{len(skipped)} suspended/skipped, e.g.: {reasons[:3]}[/yellow]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas", description="Atlas V1 SportyBet research pipeline")
    sub = p.add_subparsers(dest="command")
    d = sub.add_parser("discover", help="Ingest a SportyBet snapshot file (read-only)")
    d.add_argument("--snapshot", required=True, help="Path to browser-exported snapshot JSON")
    d.add_argument("--source-url", default="browser://sportybet", help="Page URL the snapshot came from")
    d.add_argument("--limit", type=int, default=10, help="Markets to display (default 10)")
    d.set_defaults(func=cmd_discover)
    demo = sub.add_parser("live-demo", help="Replay a recorded live observation (observe-only)")
    demo.add_argument("--fixture", required=True, help="Path to recorded observation JSON")
    demo.add_argument("--log", default="data/decisions/demo.jsonl", help="Decision log output path")
    demo.set_defaults(func=cmd_live_demo)
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
