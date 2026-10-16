"""Platform-agnostic canonical market record.

One MarketRecord describes a single resolvable prediction market on any
platform (Polymarket, etc.). It deliberately avoids sport-specific fields:
binary YES/NO markets are the initial case, but outcomes/prices are lists so
multi-outcome markets fit without a schema change.

Market price is stored as observed data — it is NOT an Atlas probability
estimate. Estimates live in a separate layer (brief section 5).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketRecord(BaseModel):
    """Canonical representation of one prediction market."""

    # Identity
    platform: str = Field(description="Platform name, e.g. 'polymarket'")
    market_id: str = Field(description="Platform-native market identifier")
    event_id: Optional[str] = Field(None, description="Parent event identifier, if any")
    slug: Optional[str] = None
    question: str
    description: Optional[str] = None

    # Outcomes: parallel lists, outcomes[i] trades at outcome_prices[i].
    outcomes: list[str] = Field(min_length=2)
    outcome_prices: list[float] = Field(min_length=2)

    # Resolution
    resolution_criteria: Optional[str] = None
    resolution_source: Optional[str] = None
    status: str = Field(description="'open', 'closed', 'resolved', or 'archived'")
    end_time: Optional[datetime] = None

    # Market microstructure as observed (optional — not always published)
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    last_trade_price: Optional[float] = None
    fee_bps: Optional[float] = Field(None, description="Taker fee in basis points, if known")

    # Provenance (brief section 9: never lose what was known and when)
    source: str = Field(description="Collector name, e.g. 'polymarket_gamma'")
    source_url: str = Field(description="Exact request URL the payload came from")
    collected_at: datetime = Field(description="When Atlas observed this (UTC)")
    parser_version: str = Field(description="Mapper name + version, e.g. 'polymarket_gamma_v1'")

    @field_validator("outcome_prices")
    @classmethod
    def validate_prices(cls, v: list[float]) -> list[float]:
        for p in v:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Outcome price must be in [0, 1], got {p}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"open", "closed", "resolved", "archived"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v!r}")
        return v

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        if len(self.outcomes) != len(self.outcome_prices):
            raise ValueError(
                f"outcomes ({len(self.outcomes)}) and outcome_prices "
                f"({len(self.outcome_prices)}) must have equal length"
            )
