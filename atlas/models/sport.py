"""SportyBet canonical market model (V1).

SportyBet prices are decimal odds (>= 1.0), not 0-1 probabilities.
This model stores observed odds verbatim and derives implied/fair
probabilities explicitly so Atlas estimates never get confused with
bookmaker prices (mission: explainable, traceable).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

PARSER_VERSION = "sportybet_v1"


class SportSelection(BaseModel):
    name: str
    odds: float = Field(ge=1.0)

    @property
    def implied(self) -> float:
        return 1.0 / self.odds


class SportMarket(BaseModel):
    """One SportyBet market (e.g. 1X2, Over/Under 2.5, Home Total Corners)."""

    platform: str = Field(default="sportybet")
    match_id: str = Field(description="Match identifier used by Atlas for this observation")
    market_key: str = Field(description="Stable key, e.g. '1x2', 'ou_2_5', 'home_corners_5_5'")
    market_label: str = Field(description="Human label as shown, e.g. 'Over/Under 2.5'")
    selections: list[SportSelection] = Field(min_length=2)
    status: str = Field(default="open", description="'open', 'suspended', or 'closed'")
    line: Optional[float] = Field(default=None, description="Betting line, e.g. 2.5 for O/U 2.5")
    period: Optional[str] = Field(default=None, description="'full', '1h', '2h', or minute-window label")
    settled_outcome: Optional[str] = Field(default=None, description="Winning selection name once known")

    source: str = Field(description="Collector name, e.g. 'sportybet_browser'")
    source_url: str = Field(description="Exact page URL the snapshot came from")
    collected_at: datetime = Field(description="When Atlas observed this (UTC)")
    parser_version: str = Field(default=PARSER_VERSION)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"open", "suspended", "closed"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v!r}")
        return v

    @property
    def overround(self) -> float:
        """Sum of implied probs; 1.0 = no margin. ~1.07-1.09 typical here."""
        return sum(s.implied for s in self.selections)

    def fair_probs(self) -> dict[str, float]:
        """De-vigged probabilities: implied / overround."""
        total = self.overround
        return {s.name: s.implied / total for s in self.selections}
