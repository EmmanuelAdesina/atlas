"""Minimal raw-JSON status endpoint — replaces the deleted HTML dashboard.

Serves /status and /ledger as pure JSON (no HTML, no FastAPI). Uses only
Python stdlib (http.server) so it needs no new dependencies and exposes
zero security-sensitive surfaces beyond the read-only data files.

Reads directly from the loop's persisted files on disk — the source of
the same truth the telemetry facade uses — so the record never diverges.
No listening sockets exist unless this file is started explicitly.
"""

from __future__ import annotations

import json
import os
import pathlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Paths match the server's /opt/atlas layout (same as loop/deploy).
DECISIONS_PATH = pathlib.Path(
    os.environ.get("ATLAS_DECISIONS_LOG", "/opt/atlas/data/decisions/loop_decisions.jsonl")
)
LEDGER_PATH = pathlib.Path(
    os.environ.get("ATLAS_LEDGER_LOG", "/opt/atlas/data/ops/ledger.jsonl")
)
HEARTBEAT_PATH = pathlib.Path(
    "/opt/atlas/data/heartbeat.json"
)

# The endpoint binds to loopback by default: an open public port is a
# deliberate choice, not an accident. Change ATLAS_STATUS_HOST to expose.
HOST = os.environ.get("ATLAS_STATUS_HOST", "127.0.0.1")
PORT = int(os.environ.get("ATLAS_STATUS_PORT", "8077") or 8077)


class _Handler(BaseHTTPRequestHandler):
    # Suppress default log noise; this is an observer endpoint, not a
    # production web server.
    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/status" or path == "/":
            # Mirror the dashboard's /status contract: cycle, posture,
            # daily P&L, open exposure, heartbeat freshness, latest cycles.
            try:
                heartbeat = {}
                if HEARTBEAT_PATH.exists():
                    heartbeat = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
                decisions = []
                if DECISIONS_PATH.exists():
                    for line in DECISIONS_PATH.read_text(encoding="utf-8").splitlines()[-12:]:
                        try:
                            decisions.append(json.loads(line))
                        except Exception:
                            pass
                # Most recent cycle only.
                last_decision = decisions[-1] if decisions else {}
                # Read latest open bets from ledger.
                open_bets = []
                open_exposure = 0.0
                if LEDGER_PATH.exists():
                    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()[-20:]:
                        try:
                            r = json.loads(line)
                            if r.get("event") == "bet_placed" and r.get("record", {}).get("status") == "placed":
                                rec = r.get("record", {})
                                open_bets.append({
                                    "bet_id": rec.get("bet_id"),
                                    "match_id": rec.get("match_id"),
                                    "market_key": rec.get("market_key"),
                                    "selection": rec.get("selection"),
                                    "stake_ngn": rec.get("stake_ngn"),
                                    "status": rec.get("status"),
                                    "placed_at": rec.get("placed_at"),
                                })
                                open_exposure += float(rec.get("stake_ngn", 0.0) or 0)
                        except Exception:
                            pass
                self._json(200, {
                    "ok": True,
                    "service": "atlas-status-endpoint",
                    "version": "1.0",
                    "at": heartbeat.get("at"),
                    "heartbeat_age_s": (heartbeat.get("at") is not None),  # caller computes
                    "loop_cycle": heartbeat.get("cycle"),
                    "loop_ready": heartbeat.get("ready"),
                    "loop_reason": heartbeat.get("reason"),
                    "last_decision": last_decision,
                    "open_bets": open_bets,
                    "open_exposure_ngn": open_exposure,
                    # Note: posture/state file mirror is read by the loop; the
                    # endpoint does not load it directly to avoid file-lock races.
                    "telemetry_enabled": bool(os.environ.get("ATLAS_OTLP_ENDPOINT", "")),
                    "limited_to_read_only": True,
                })
                return
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
                return

        if path == "/ledger" or path.startswith("/ledger"):
            # Raw ledger tail (last 10 entries), same evidence format.
            entries = []
            if LEDGER_PATH.exists():
                for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()[-10:]:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
            self._json(200, {"ok": True, "count": len(entries), "entries": entries})
            return

        # Unknown paths return 404 with a short explanation.
        self._json(404, {
            "ok": False,
            "detail": f"unknown path: {path}. Available: /status, /ledger",
            "service": "atlas-status-endpoint",
            "version": "1.0",
        })


def serve(host: str = HOST, port: int = PORT) -> None:
    srv = HTTPServer((host, port), _Handler)
    # Best-effort graceful shutdown: the endpoint is read-only and the loop
    # does not depend on it; a hard interrupt is safe.
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    serve()
