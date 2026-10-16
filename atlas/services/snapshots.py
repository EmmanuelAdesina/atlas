"""Immutable raw-snapshot archival.

Every collector persists the exact provider payload BEFORE any parsing, in an
envelope carrying provenance (source, endpoint, collection time, SHA-256).
History is append-only: snapshots are never overwritten or mutated, so later
corrections can be recorded as new snapshots and past decisions can be
re-evaluated without lookahead leakage (brief section 9).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_raw_snapshot(
    source: str,
    endpoint: str,
    payload: Any,
    raw_data_path: str | Path,
    collected_at: datetime | None = None,
) -> Path:
    """Write one immutable snapshot envelope. Returns the file path."""
    collected_at = collected_at or datetime.now(timezone.utc)
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    envelope = {
        "source": source,
        "endpoint": endpoint,
        "collected_at": collected_at.isoformat(),
        "payload_sha256": digest,
        "payload": payload,
    }
    directory = Path(raw_data_path) / source
    directory.mkdir(parents=True, exist_ok=True)
    stamp = collected_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}-{digest[:12]}.json"
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
