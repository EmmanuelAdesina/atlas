"""Live match observation schema — what Atlas actually saw, and when.

Every field records an observation, never an inference. Unknown values
stay None: a missing clock or stale feed must produce abstention downstream,
not a guessed probability (directive sections 2 and 6).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TeamStats(BaseModel):
    name: str
    goals: Optional[int] = None
    possession_pct: Optional[float] = None
    shots_on_target: Optional[int] = None
    shots_total: Optional[int] = None
    saves: Optional[int] = None
    corners: Optional[int] = None
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None
    fouls: Optional[int] = None
    free_kicks: Optional[int] = None
    offsides: Optional[int] = None


class LiveMatchState(BaseModel):
    """One timestamped observation of a single live match."""

    match_id: str
    competition: Optional[str] = None
    home: TeamStats
    away: TeamStats
    period: str = Field(description="'1h', '2h', 'ht', 'ft', or 'unknown'")
    clock_minute: Optional[float] = Field(
        default=None, description="Elapsed minute incl. stoppage already played, e.g. 93.7"
    )
    clock_label: Optional[str] = Field(
        default=None, description="Raw clock text as shown, e.g. '90:00 +03:42'"
    )
    half_home_goals: Optional[int] = None
    half_away_goals: Optional[int] = None
    substitutions_count: Optional[int] = None
    added_minutes_announced: Optional[int] = None
    remaining_unknown: bool = Field(
        default=False, description="True when stoppage length / whistle timing is unknown"
    )
    account_balance_ngn: Optional[float] = None
    prematch_1x2_odds: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Optional observed pre-match decimal odds {Home, Draw, Away}; "
            "never populate this with current live prices"
        ),
    )
    observed_at: datetime = Field(description="When Atlas observed this (UTC)")
    source: str = Field(description="e.g. 'sportybet_browser'")
    source_url: str = ""

    @field_validator("prematch_1x2_odds")
    @classmethod
    def validate_prematch_1x2_odds(
        cls, value: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if value is None:
            return None
        aliases = {"home": "Home", "1": "Home", "draw": "Draw", "x": "Draw", "away": "Away", "2": "Away"}
        normalized: dict[str, float] = {}
        for key, raw_odds in value.items():
            canonical = aliases.get(str(key).strip().lower())
            if canonical is None:
                raise ValueError(f"unknown pre-match 1X2 outcome key {key!r}")
            odds = float(raw_odds)
            if not math.isfinite(odds) or odds <= 1.0:
                raise ValueError("pre-match 1X2 decimal odds must be finite and greater than 1.0")
            normalized[canonical] = odds
        if set(normalized) != {"Home", "Draw", "Away"}:
            raise ValueError("pre-match 1X2 odds must include Home, Draw, and Away")
        return normalized

    @property
    def score_total(self) -> Optional[int]:
        if self.home.goals is None or self.away.goals is None:
            return None
        return self.home.goals + self.away.goals

    @property
    def is_live(self) -> bool:
        return self.period in {"1h", "2h"}

    @property
    def home_reds(self) -> int:
        return self.home.red_cards or 0

    @property
    def away_reds(self) -> int:
        return self.away.red_cards or 0

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.observed_at).total_seconds()

    def is_fresh(self, max_age_s: float = 60.0,
                 now: Optional[datetime] = None) -> bool:
        return self.age_seconds(now) <= max_age_s

    def evidence_summary(self) -> dict:
        """Compact provenance record for decision logs."""
        return {
            "match_id": self.match_id,
            "period": self.period,
            "clock_label": self.clock_label,
            "score": [self.home.goals, self.away.goals],
            "home_reds": self.home_reds,
            "away_reds": self.away_reds,
            "observed_at": self.observed_at.isoformat(),
            "source": self.source,
            "prematch_1x2_odds": self.prematch_1x2_odds,
        }
