"""Map SportyBet snapshot payloads to canonical SportMarket records.

Accepts the browser-supplied snapshot format documented in
atlas/collectors/sportybet.py. Decimal odds must be >= 1.0.
Suspended/locked selections (padlock in the PDF) arrive as status
'suspended' markets or are omitted by the extractor — never invented.
"""

from datetime import datetime
from typing import Any

from atlas.models.sport import PARSER_VERSION, SportMarket

SOURCE = "sportybet_browser"


def map_sport_market(
    raw: dict[str, Any],
    *,
    source_url: str,
    collected_at: datetime,
    source: str = SOURCE,
) -> SportMarket:
    match_id = str(raw.get("match_id", "")).strip()
    market_key = str(raw.get("market_key", "")).strip()
    market_label = str(raw.get("market_label", market_key)).strip()
    selections = raw.get("selections")
    if not match_id:
        raise ValueError("SportyBet market missing 'match_id'")
    if not market_key:
        raise ValueError(f"SportyBet market in {match_id} missing 'market_key'")
    if not isinstance(selections, list) or len(selections) < 2:
        if str(raw.get("status", "open")) != "open":
            raise ValueError(
                f"SportyBet market {market_key} is {raw.get('status')} "
                f"(no live prices) — not evaluable"
            )
        raise ValueError(f"SportyBet market {market_key} needs >= 2 selections")
    parsed = []
    for s in selections:
        if not isinstance(s, dict) or "name" not in s or "odds" not in s:
            raise ValueError(f"SportyBet market {market_key} has malformed selection {s!r}")
        odds = float(s["odds"])
        if odds < 1.0:
            raise ValueError(f"SportyBet market {market_key} odds must be >= 1.0, got {odds}")
        parsed.append({"name": str(s["name"]), "odds": odds})
    status = str(raw.get("status", "open"))
    line = raw.get("line")
    period = raw.get("period")
    return SportMarket(
        match_id=match_id,
        market_key=market_key,
        market_label=market_label,
        selections=parsed,
        status=status,
        line=float(line) if line is not None else None,
        period=str(period) if period is not None else None,
        settled_outcome=raw.get("settled_outcome"),
        source=source,
        source_url=source_url,
        collected_at=collected_at,
        parser_version=PARSER_VERSION,
    )


def map_sport_markets(
    raws: list[dict[str, Any]],
    *,
    source_url: str,
    collected_at: datetime,
    source: str = SOURCE,
) -> tuple[list[SportMarket], list[dict]]:
    records: list[SportMarket] = []
    skipped: list[dict] = []
    for i, raw in enumerate(raws):
        try:
            records.append(
                map_sport_market(raw, source_url=source_url, collected_at=collected_at, source=source)
            )
        except (ValueError, TypeError) as e:
            mid = raw.get("match_id") if isinstance(raw, dict) else None
            skipped.append({"index": i, "market_id": mid, "reason": str(e)})
    return records, skipped
