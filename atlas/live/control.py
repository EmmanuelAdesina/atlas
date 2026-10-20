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

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from atlas.live.runner import LiveRunner
from atlas.ops.notify import Notifier
from atlas.ops.posture import propose_posture


class PostureRequest(BaseModel):
    instruction: str = Field(..., description="Plain-language instruction")
    use_llm: bool = Field(
        False, description="Use the LLM proposer (needs ATLAS_BOSS_API_KEY)"
    )


def build_app(runner: LiveRunner, notifier: Optional[Notifier] = None) -> FastAPI:
    app = FastAPI(title="Atlas Control", version="1.0")
    state: dict[str, Any] = {"notifier": notifier}

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "at": datetime.now(timezone.utc).isoformat(),
            "cycle": runner.cycle,
        }

    @app.get("/status")
    def status() -> dict:
        ledger = runner.ledger
        last = runner.reports[-1] if runner.reports else None
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
        report = runner.run_cycle()
        return asdict(report)

    return app


def serve(
    runner: LiveRunner,
    host: Optional[str] = None,
    port: Optional[int] = None,
    notifier: Optional[Notifier] = None,
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
    app = build_app(runner, notifier)
    uvicorn.run(app, host=host, port=port, log_level="warning")
