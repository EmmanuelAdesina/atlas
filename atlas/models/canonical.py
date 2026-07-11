from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum

class Competition(str, Enum):
    PREMIER_LEAGUE = "premier_league"
    LA_LIGA = "la_liga"
    BUNDESLIGA = "bundesliga"
    SERIE_A = "serie_a"
    LIGUE_1 = "ligue_1"
    CHAMPIONS_LEAGUE = "champions_league"

class Team(BaseModel):
    id: str
    name: str
    short_name: Optional[str] = None
    league: Competition
    founded_year: Optional[int] = None
    stadium: Optional[str] = None
    city: Optional[str] = None
    country: str
    
class Player(BaseModel):
    id: str
    name: str
    position: str  # GK, DF, MF, FW
    team_id: str
    nationality: str
    jersey_number: Optional[int] = None
    
class Match(BaseModel):
    id: str
    competition: Competition
    season: str
    matchday: int
    home_team_id: str
    away_team_id: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    status: str  # scheduled, live, finished, cancelled
    date: datetime
    venue: Optional[str] = None
    
    @validator('home_goals', 'away_goals')
    def validate_goals(cls, v):
        if v is not None and v < 0:
            raise ValueError("Goals cannot be negative")
        return v

class Odds(BaseModel):
    match_id: str
    bookmaker: str
    timestamp: datetime
    home_odds: float = Field(ge=1.0)
    draw_odds: float = Field(ge=1.0)
    away_odds: float = Field(ge=1.0)
    opening_odds: bool = False
    
    @validator('home_odds', 'draw_odds', 'away_odds')
    def validate_odds(cls, v):
        if v < 1.0:
            raise ValueError("Odds must be >= 1.0")
        return v

class Event(BaseModel):
    match_id: str
    minute: int
    event_type: str  # goal, card, substitution, penalty
    player_id: Optional[str] = None
    team_id: Optional[str] = None
    description: Optional[str] = None
    extra_data: Optional[dict] = None
