"""The control dashboard: a small HTTP API to steer ATLAS and read its state.

The operator needs to be able to say "more revenue, low risk" while the
system runs, and to see what it is doing. This exposes exactly that:

    POST /posture   {"instruction": "more revenue, low risk"}
    GET  /status    posture, daily P&L, stop state, open positions, last cycle
    GET  /notifications   the recent notification feed
    GET  /health

The API can set a posture but it CANNOT place a bet, widen a limit, or
invent a probability. The posture it accepts is clamped by the same
hard bounds as every other path, and the loop reads the posture from the
same file the runner persists.

FastAPI is already a dependency, so this adds no new weight.
"""

from __future__ import annotations

import html
import json
import os
import pathlib
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from atlas.live.runner import LiveRunner
from atlas.ops.notify import Notifier
from atlas.ops.posture import propose_posture

DECISIONS_LOG = pathlib.Path("data/decisions/loop_decisions.jsonl")
NOTIFICATIONS_LOG = pathlib.Path("data/decisions/notifications.jsonl")


def _tail_lines(path: pathlib.Path, n: int) -> list[str]:
    """The last n lines of a JSONL file, oldest last. Missing file = []"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def _table(headers: list[str], rows: list[list[str]], title: str = "") -> str:
    """A plain, borderless work table. Boring on purpose."""
    out = []
    if title:
        out.append(f"<h3>{html.escape(title)}</h3>")
    out.append("<table border='1' cellspacing='0' cellpadding='3'>")
    out.append("<tr>" + "".join(f"<th align='left'>{html.escape(h)}</th>" for h in headers) + "</tr>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _read_heartbeat(runner: LiveRunner) -> Optional[dict]:
    """The loop's heartbeat as this process sees it on disk."""
    if not runner.state_path:
        return None
    try:
        return json.loads(
            (runner.state_path.parent / "heartbeat.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return None


def _placement_snapshot(
    runner: LiveRunner, heartbeat: Optional[dict]
) -> dict:
    """Placement-worker progress.

    The dashboard is a SEPARATE process from the loop, so the heartbeat on
    disk is the source of truth; the in-process runner is the fallback for
    tests and single-process use.
    """
    if heartbeat and "place_stats" in heartbeat:
        return {
            "queue_depth": heartbeat.get("place_queue_depth"),
            "stats": heartbeat.get("place_stats") or {},
            "inflight": heartbeat.get("place_inflight") or [],
            "inflight_s": heartbeat.get("place_inflight_s"),
            "last": heartbeat.get("place_last"),
        }
    return runner.placement_status()


def render_panel(runner: LiveRunner, now: Optional[datetime] = None) -> str:
    """The operations panel: every table the operator needs, no design."""
    now = now or datetime.now(timezone.utc)
    runner.ledger.refresh()
    runner.load_state()
    parts: list[str] = [
        "<h1>Atlas operations</h1>",
        f"<p>page generated at {now.isoformat()}</p>",
    ]

    # --- heartbeat -------------------------------------------------------
    hb_rows = []
    heartbeat = _read_heartbeat(runner)
    if heartbeat:
        try:
            hb_at = datetime.fromisoformat(heartbeat.get("at", ""))
            age_s = (now - hb_at).total_seconds()
        except ValueError:
            hb_at, age_s = None, None
        fresh = age_s is not None and age_s <= max(
            4.0 * heartbeat.get("cycle_s", 30), 180
        )
        hb_rows.append(
            [
                "heartbeat age (s)",
                f"{age_s:.0f}" if age_s is not None else "?",
                "FRESH" if fresh else "STALE - IS THE LOOP ALIVE?",
            ]
        )
        for k in (
            "at", "pid", "cycle", "cycle_s", "ready", "reason", "match_id",
            "priced_selections", "opportunities", "planned", "placed",
            "settled", "arbs", "cashout_taken", "daily_pnl", "stop_active",
            "submission_enabled", "posture_mode",
            "account_balance_ngn", "page_url", "page_error", "last_error",
        ):
            v = heartbeat.get(k)
            if k in ("at", "pid", "cycle_s"):
                continue
            hb_rows.append([k.replace("_", " "), "" if v is None else str(v)])
    else:
        hb_rows.append(
            ["heartbeat", "no file", "the loop has not run a cycle yet, or is dead"]
        )
    parts.append(_table(["item", "value", "verdict"], hb_rows, "loop heartbeat"))

    # --- placement worker ------------------------------------------------
    # The cycle no longer blocks on placement, so placement progress has to
    # be visible on its own: a queue that keeps growing means the worker is
    # failing faster than it completes, not that the loop is idle.
    place = _placement_snapshot(runner, heartbeat)
    place_stats = place.get("stats") or {}
    place_inflight = place.get("inflight") or []
    place_last = place.get("last") or None
    if place_last:
        last_text = (
            f"cycle {place_last.get('cycle')}: "
            f"{place_last.get('placed')}/{place_last.get('queued')} placed"
        )
        if place_last.get("error"):
            last_text += f" — {place_last['error']}"
    else:
        last_text = "no placement yet"
    parts.append(
        _table(
            ["item", "value"],
            [
                ["queue depth", str(place.get("queue_depth") or 0)],
                ["queued", str(place_stats.get("queued", 0))],
                ["done", str(place_stats.get("done", 0))],
                ["failed", str(place_stats.get("failed", 0))],
                ["expired (stale)", str(place_stats.get("expired", 0))],
                ["duplicate (suppressed)", str(place_stats.get("duplicate", 0))],
                ["dropped (queue full)", str(place_stats.get("dropped", 0))],
                ["in flight", str(len(place_inflight))],
                [
                    "in flight (s)",
                    str(place.get("inflight_s"))
                    if place.get("inflight_s") is not None
                    else "",
                ],
                ["last", last_text],
            ],
            "placement worker (async)",
        )
    )
    if place_inflight:
        parts.append(
            _table(
                ["match", "market", "selection", "stake", "odds", "queued at"],
                [
                    [
                        str(r.get("match_id", "")),
                        str(r.get("market_key", "")),
                        str(r.get("selection", "")),
                        str(r.get("stake_ngn", "")),
                        str(r.get("odds", "")),
                        str(r.get("queued_at", ""))[:19],
                    ]
                    for r in place_inflight
                ],
                "bets reserved, awaiting confirmation",
            )
        )

    # --- posture ---------------------------------------------------------
    p = runner.posture.to_dict()
    parts.append(
        _table(
            ["setting", "value"],
            [[k, str(v)] for k, v in p.items()],
            "posture (read from disk)",
        )
    )

    # --- money: daily state and the ledger -------------------------------
    ledger = runner.ledger
    parts.append(
        _table(
            ["item", "value"],
            [
                ["submission (execution)", "ENABLED" if runner.allow_submission else "OFF (observe only)"],
                ["daily P&L (NGN)", f"{ledger.daily_pnl():+.2f}"],
                ["daily stop active", str(ledger.is_daily_stop_active())],
                ["open exposure (NGN)", f"{sum(b.get('stake_ngn', 0.0) for b in ledger.open_bets()):.2f}"],
                ["total bets in ledger", str(len(ledger.all_bets()))],
            ],
            "money state",
        )
    )
    bet_rows = []
    for b in reversed(ledger.all_bets()[-15:]):
        bet_rows.append(
            [
                str(b.get("placed_at", ""))[:19],
                f"{b.get('market_key', '')}",
                f"{b.get('selection', '')}",
                f"{b.get('odds', '')}",
                f"{b.get('stake_ngn', '')}",
                f"{b.get('status', '')}",
                f"{b.get('pnl_ngn', '')}",
            ]
        )
    parts.append(
        _table(
            ["placed_at", "market", "selection", "odds", "stake", "status", "pnl"],
            bet_rows,
            "ledger, latest last-15 first",
        )
    )

    # --- model performance over ALL settled bets -------------------------
    from atlas.ops.perf import settled_stats

    stats = settled_stats(runner.ledger.all_bets())
    if stats["settled"]:
        perf_rows = [
            ["settled bets", f"{stats['settled']} (won {stats['won']} / lost {stats['lost']})"],
            ["hit rate", str(stats["hit_rate"])],
            ["staked (NGN)", f"{stats['staked_ngn']:.2f}"],
            ["P&L (NGN)", f"{stats['pnl_ngn']:+.2f}"],
            ["ROI", f"{100 * stats['roi']:+.1f}%" if stats["roi"] is not None else "?"],
            ["current loss streak", str(stats["streak_losses"])],
            ["verdict", stats["verdict"]],
        ]
        band_rows = []
        for band, agg in stats["by_band"].items():
            band_rows.append(
                [
                    band,
                    str(agg["n"]),
                    str(agg["hit_rate"]),
                    str(agg["breakeven_hit_rate"]),
                    f"{agg['pnl']:+.2f}",
                ]
            )
        parts.append(_table(["item", "value"], perf_rows, "model performance (settled bets)"))
        parts.append(
            _table(
                ["band", "n", "hit rate", "break-even", "pnl"],
                band_rows,
                "performance by odds band",
            )
        )

    # --- alerts ----------------------------------------------------------
    notif_rows = []
    for line in reversed(_tail_lines(NOTIFICATIONS_LOG, 15)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        notif_rows.append(
            [
                str(r.get("created_at", ""))[:19],
                str(r.get("severity", "")),
                str(r.get("title", "")),
                str(r.get("body", "")).replace("\n", " "),
            ]
        )
    parts.append(
        _table(["at", "sev", "title", "body"], notif_rows, "alerts (latest first)")
    )

    # --- decision log tail ----------------------------------------------
    parts.append("<h3>decision log, last 12 entries (raw)</h3><pre>")
    parts.append(
        html.escape("".join(_tail_lines(DECISIONS_LOG, 12))[-6000:])
    )
    parts.append("</pre>")

    runner_err = runner.last_error or ""
    if runner_err:
        parts.append(f"<p>last_error: {html.escape(runner_err)}</p>")

    return (
        "<!DOCTYPE html>\n<html><head>"
        # Boring realism aid: the panel is only server-rendered on load, so
        # the browser reloads it on a fixed cadence instead of showing a
        # frozen page. 15 s < half the heartbeat freshness verdict.
        '<meta http-equiv="refresh" content="15">\n'
        "<title>Atlas operations</title></head>\n<body>\n"
        + "\n".join(parts)
        + "\n</body></html>"
    )


class PostureRequest(BaseModel):
    instruction: str = Field(..., description="Plain-language instruction")
    use_llm: bool = Field(
        False, description="Use the LLM proposer (needs ATLAS_BOSS_API_KEY)"
    )


def build_app(
    runner: LiveRunner,
    notifier: Optional[Notifier] = None,
    *,
    manual_cycles: bool = True,
) -> FastAPI:
    app = FastAPI(title="Atlas Control", version="1.0")
    state: dict[str, Any] = {"notifier": notifier}

    @app.get("/")
    @app.get("/panel")
    def panel() -> HTMLResponse:
        return HTMLResponse(render_panel(runner, datetime.now(timezone.utc)))

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "at": datetime.now(timezone.utc).isoformat(),
            "cycle": runner.cycle,
        }

    @app.get("/status")
    def status() -> dict:
        # The dashboard is a separate process from the live loop, so mirror
        # the on-disk truth: the loop's ledger and its persisted posture.
        runner.ledger.refresh()
        runner.load_state()
        ledger = runner.ledger
        last = runner.reports[-1] if runner.reports else None
        heartbeat = _read_heartbeat(runner)
        return {
            "at": datetime.now(timezone.utc).isoformat(),
            "cycle": runner.cycle,
            "posture": runner.posture.to_dict(),
            "daily_pnl_ngn": ledger.daily_pnl(),
            "daily_stop_active": ledger.is_daily_stop_active(),
            "open_exposure_ngn": sum(
                b.get("stake_ngn", 0.0)
                for b in ledger.open_bets()
            ),
            "bets": len(ledger.all_bets()),
            "submission_enabled": runner.allow_submission,
            # Async placement: the cycle no longer waits for the book, so
            # this object is how a client sees queued/reserved/in-flight
            # bets without reading the heartbeat file itself.
            "placement": _placement_snapshot(runner, heartbeat),
            "last_cycle": asdict(last) if last is not None else None,
            "last_error": runner.last_error,
        }

    @app.post("/posture")
    def set_posture(req: PostureRequest) -> dict:
        if not req.instruction.strip():
            raise HTTPException(status_code=400, detail="instruction is empty")
        posture = propose_posture(req.instruction, use_llm=req.use_llm)
        applied = runner.set_posture(posture)
        return {
            "applied": applied.to_dict(),
            "describe": applied.describe(),
        }

    @app.get("/notifications")
    def notifications(limit: int = 20) -> dict:
        notifier = state.get("notifier")
        if notifier is None:
            return {"notifications": []}
        sent = notifier.sent[-limit:]
        return {"notifications": [n.to_dict() for n in sent]}

    @app.post("/cycle")
    def run_cycle() -> dict:
        # A dashboard-only process shares the WebBridge tab with the trading
        # loop: a cycle HERE would navigate the board while the loop may be
        # mid-submission on its own page lock (which only protects within
        # one process). The loop owns the page; this server observes.
        if not manual_cycles:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This dashboard is an observer (separate loop process "
                    "owns the browser tab) — manual cycles are disabled. "
                    "It can still steer posture via POST /posture."
                ),
            )
        report = runner.run_cycle()
        return asdict(report)

    return app


def serve(
    runner: LiveRunner,
    host: Optional[str] = None,
    port: Optional[int] = None,
    notifier: Optional[Notifier] = None,
    *,
    manual_cycles: bool = True,
) -> None:
    """Run the control API.

    The bind address comes from ATLAS_HOST / ATLAS_PORT, because on a VPS
    127.0.0.1 is unreachable from outside the box. The default stays on the
    loopback: an open dashboard on a public interface should be a deliberate
    choice, not the default.
    """
    import uvicorn

    host = host or os.environ.get("ATLAS_HOST", "127.0.0.1")
    port = int(port or os.environ.get("ATLAS_PORT", "8077"))
    app = build_app(runner, notifier, manual_cycles=manual_cycles)
    uvicorn.run(app, host=host, port=port, log_level="warning")
