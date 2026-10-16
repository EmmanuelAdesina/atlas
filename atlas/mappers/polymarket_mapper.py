"""Map Polymarket Gamma API payloads to canonical MarketRecord.

Observed wire format (gamma-api.polymarket.com, verified 2026-10-12):
  - `outcomes` / `outcomePrices` arrive as JSON-encoded STRINGS, e.g.
    "[\\"Yes\\", \\"No\\"]" / "[\\"0.0275\\", \\"0.9725\\"]". Some responses
    already carry real lists — both are accepted.
  - Numeric microstructure fields (volume, liquidity, spread, bestBid, ...)
    arrive as numbers or numeric strings; missing values stay None.
  - `endDateIso` is the ISO-8601 expiry; absent on some markets.

Parser version is recorded on every record so schema drift stays traceable.
"""

import json
from datetime import datetime
from typing import Any, Optional

from atlas.models.market import MarketRecord

PARSER_VERSION = "polymarket_gamma_v1"


def _as_list(value: Any) -> Optional[list]:
    """Accept a real list or a JSON-encoded string of a list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _derive_status(raw: dict) -> str:
    if raw.get("archived"):
        return "archived"
    if raw.get("closed"):
        return "closed"
    # Gamma exposes umaResolutionStatuses; a resolved market reports done states.
    statuses = _as_list(raw.get("umaResolutionStatuses")) or []
    if any(str(s).lower() in {"resolved", "resolved_yes", "resolved_no"} for s in statuses):
        return "resolved"
    return "open"


def map_market(
    raw: dict,
    *,
    source_url: str,
    collected_at: datetime,
    source: str = "polymarket_gamma",
) -> MarketRecord:
    """Map one Gamma market dict to a canonical MarketRecord.

    Raises ValueError if required fields (id, question, outcomes, prices)
    are missing or inconsistent — callers decide skip-vs-fail policy.
    """
    market_id = str(raw.get("id", "")).strip()
    question = str(raw.get("question", "")).strip()
    outcomes = _as_list(raw.get("outcomes"))
    prices_raw = _as_list(raw.get("outcomePrices"))

    if not market_id:
        raise ValueError("Gamma market missing 'id'")
    if not question:
        raise ValueError(f"Gamma market {market_id} missing 'question'")
    if not outcomes or len(outcomes) < 2:
        raise ValueError(f"Gamma market {market_id} has no usable 'outcomes'")
    if not prices_raw or len(prices_raw) != len(outcomes):
        raise ValueError(
            f"Gamma market {market_id}: outcomes/prices length mismatch "
            f"({outcomes} vs {prices_raw})"
        )
    try:
        prices = [float(p) for p in prices_raw]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Gamma market {market_id} has non-numeric prices: {prices_raw}") from e

    events = raw.get("events") or []
    event_id = None
    if isinstance(events, list) and events and isinstance(events[0], dict):
        event_id = events[0].get("id")

    return MarketRecord(
        platform="polymarket",
        market_id=market_id,
        event_id=str(event_id) if event_id is not None else None,
        slug=raw.get("slug"),
        question=question,
        description=raw.get("description"),
        outcomes=[str(o) for o in outcomes],
        outcome_prices=prices,
        resolution_source=raw.get("resolutionSource"),
        status=_derive_status(raw),
        end_time=_as_datetime(raw.get("endDateIso")),
        volume=_as_float(raw.get("volumeNum") or raw.get("volume")),
        liquidity=_as_float(raw.get("liquidityNum") or raw.get("liquidity")),
        best_bid=_as_float(raw.get("bestBid")),
        best_ask=_as_float(raw.get("bestAsk")),
        spread=_as_float(raw.get("spread")),
        last_trade_price=_as_float(raw.get("lastTradePrice")),
        fee_bps=_as_float(raw.get("takerBaseFee")),
        source=source,
        source_url=source_url,
        collected_at=collected_at,
        parser_version=PARSER_VERSION,
    )


def map_markets(
    raws: list[dict],
    *,
    source_url: str,
    collected_at: datetime,
    source: str = "polymarket_gamma",
) -> tuple[list[MarketRecord], list[dict]]:
    """Map a batch. Returns (records, skipped) — skipped entries carry
    {'index', 'market_id', 'reason'} so data-quality reports stay exact."""
    records: list[MarketRecord] = []
    skipped: list[dict] = []
    for i, raw in enumerate(raws):
        try:
            records.append(map_market(raw, source_url=source_url, collected_at=collected_at, source=source))
        except (ValueError, TypeError) as e:
            skipped.append({"index": i, "market_id": raw.get("id") if isinstance(raw, dict) else None, "reason": str(e)})
    return records, skipped
