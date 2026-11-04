"""Cashout source — reads only. Execution stays with the browser driver.

Confirmed observed (netlist_temp.json, 2026-10-29):
  GET  /api/ng/realSportsGame/cashAbleBets/count?_t=…
  POST /api/ng/orders/config/cutbet
Bodies were never archived, so offer details are NOT modelled yet. What
this source does today:

  * `cashable_count()` — badge count probe; tolerant JSON sniffing like the
    balance source; raises SchemaDriftError rather than guessing.
  * `list_offers()` — explicit BoardNotConfigured-style refusal until Phase A
    archives the cashable-bets list fixture. It does not fabricate offers.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from atlas.transport import snapshots
from atlas.transport.client import TransportClient
from atlas.transport.errors import SchemaDriftError, TransportError

CASHABLE_COUNT_PATH = "/api/ng/realSportsGame/cashAbleBets/count"


class CashoutNotConfiguredError(TransportError):
    """The cashable-offers list endpoint has no archived schema yet."""


def cashable_count(client: TransportClient) -> int:
    resp = client.get(CASHABLE_COUNT_PATH, referer="https://www.sportybet.com/ng/", retries=1)
    snapshots.archive_raw("cashout_count", resp.body, meta={"status_code": resp.status_code})
    try:
        payload = json.loads(resp.body)
    except json.JSONDecodeError as exc:
        snapshots.quarantine_raw("cashout_count", resp.body, f"json: {exc}")
        raise SchemaDriftError("cashable count payload is not JSON") from exc
    if isinstance(payload, dict) and "bizCode" in payload and payload.get("bizCode") != 10000:
        raise SchemaDriftError(
            f"cashable count bizCode={payload.get('bizCode')} message={payload.get('message')!r}"
        )
    count: Optional[int] = None

    def _find_count(node: Any, depth: int = 0) -> Optional[int]:
        if depth > 4 or node is None:
            return None
        if isinstance(node, dict):
            for key in ("count", "total", "num", "cashableCount", "totalNum"):
                value = node.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            for value in node.values():
                found = _find_count(value, depth + 1)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find_count(item, depth + 1)
                if found is not None:
                    return found
        return None

    count = _find_count(payload)
    if count is None:
        raise SchemaDriftError("cashable count payload has no recognizable count field")
    return count


def list_offers():  # pragma: no cover - intentionally unimplemented
    raise CashoutNotConfiguredError(
        "cashout offer listing requires the Phase A capture of the cashable-bets "
        "endpoint; the DOM driver remains the read path until then"
    )
