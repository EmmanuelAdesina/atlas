"""Append-only action log for the agent plane.

The ledger (ops/ledger.py) records POSITIONS. This log records what the
agent DID: every tab action it attempted in the venue page, every AI cycle
it reasoned through, and every failure its supervision surfaced. One
JSON-per-line file, fsynced, replay-safe by construction — no reader is
allowed to reinterpret these lines INTO BetRecord objects, so the schemas
here stay free of the ledger's constraints.

This log is the forensic surface for "the AI did what, when": without it,
an agent controlling the tab live would operate a real session with no
audit trail at all.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = "data/ops/agent_actions.jsonl"

# One field's ceiling in a log line. Snapshot trees and evaluated JS are
# multi-kilobyte blobs that the AI reads from the page; the audit trail
# needs the fact of the call, not its full payload.
MAX_FIELD_CHARS = 600


def _compact(value: Any) -> Any:
    """Truncate long strings/lists so one snapshot cannot bloat the log."""
    if isinstance(value, str):
        if len(value) > MAX_FIELD_CHARS:
            return value[:MAX_FIELD_CHARS] + f"...[{len(value)} chars]"
        return value
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 40:
            return [_compact(v) for v in list(value)[:40]] + [f"...[{len(value)} items]"]
        return [_compact(v) for v in value]
    return value


class AgentActionLog:
    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **fields: Any) -> None:
        entry = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record": _compact(fields),
        }
        line = json.dumps(entry, ensure_ascii=False)
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def tail(self, n: int = 20) -> list[dict]:
        """Last n entries, newest last. For the agent's own context."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: list[dict] = []
        for line in lines[-n:]:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
