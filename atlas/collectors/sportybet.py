"""Read-only SportyBet snapshot collector (V1).

SportyBet exposes no public odds API, so V1 does NOT guess HTTP endpoints.
The browser (Kimi WebBridge snapshot/evaluate, or a saved page export)
supplies the raw market payload; this collector archives it immutably
BEFORE any mapping and returns a CollectionResult.

Raw snapshot format (one dict per market):
  {"match_id": "sr:match:73265408", "market_key": "1x2",
   "market_label": "1X2",
   "selections": [{"name": "Home", "odds": 1.61}, ...],
   "status": "open"}
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atlas.config.settings import settings
from atlas.schemas.collection import CollectionResult, CollectionStatus
from atlas.services.snapshots import save_raw_snapshot

SOURCE = "sportybet_browser"


class SportyBetCollector:
    """Archive browser-supplied SportyBet snapshots (read-only, no placing)."""

    def __init__(self, raw_data_path: Optional[str] = None):
        self.raw_data_path = raw_data_path or settings.raw_data_path

    def ingest_snapshot(
        self,
        payload: list[dict[str, Any]],
        source_url: str,
        collected_at: Optional[datetime] = None,
    ) -> CollectionResult:
        collected_at = collected_at or datetime.now(timezone.utc)
        if not isinstance(payload, list) or not payload:
            return CollectionResult(
                source=SOURCE,
                collected_at=collected_at,
                status=CollectionStatus.EMPTY,
                raw_data=payload,
                metadata={"source_url": source_url},
                error="empty snapshot payload",
            )
        snapshot_path = save_raw_snapshot(
            source=SOURCE,
            endpoint=source_url,
            payload=payload,
            raw_data_path=Path(self.raw_data_path),
            collected_at=collected_at,
        )
        return CollectionResult(
            source=SOURCE,
            collected_at=collected_at,
            status=CollectionStatus.SUCCESS,
            raw_data=payload,
            metadata={
                "source_url": source_url,
                "snapshot_path": str(snapshot_path),
                "count": len(payload),
            },
        )
