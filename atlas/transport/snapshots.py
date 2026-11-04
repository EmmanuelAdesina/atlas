"""Archive raw responses BEFORE parsing them, and quarantine drift.

This is the transport layer's memory of what the book actually said. It is
not optional: without the raw archive, endpoint drift is undetectable and
post-mortems are fiction.

Layout:

  data/snapshots/<kind>/<utc>-<nonce>.json   — accepted raw responses
  data/quarantine/<kind>/<utc>-<nonce>.bin   — payloads that failed schema/JSON

Files never contain the Cookie header (the archive is the response body and
non-sensitive metadata only).
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive_raw(kind: str, body: bytes | str, meta: Optional[dict[str, Any]] = None,
                root: Optional[Path] = None) -> Path:
    base = (root or ROOT / "data" / "snapshots") / kind
    base.mkdir(parents=True, exist_ok=True)
    payload = body.encode("utf-8") if isinstance(body, str) else body
    env = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "body_bytes": len(payload),
        "meta": meta or {},
    }
    name = f"{_stamp()}-{secrets.token_hex(4)}"
    (base / f"{name}.meta.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    out = base / f"{name}.json"
    out.write_bytes(payload)
    return out


def quarantine_raw(kind: str, body: bytes | str, reason: str,
                   root: Optional[Path] = None) -> Path:
    base = (root or ROOT / "data" / "quarantine") / kind
    base.mkdir(parents=True, exist_ok=True)
    payload = body.encode("utf-8", errors="replace") if isinstance(body, str) else body
    name = f"{_stamp()}-{secrets.token_hex(4)}"
    out = base / f"{name}.bin"
    out.write_bytes(payload)
    reason_doc = {
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "reason": reason,
        "body_bytes": len(payload),
    }
    (base / f"{name}.reason.json").write_text(json.dumps(reason_doc, indent=1), encoding="utf-8")
    return out
