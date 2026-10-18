"""Bet tracking and P&L ledger for Atlas.

Tracks all bets through their lifecycle (pending -> placed -> won/lost/void),
computes P&L, enforces a daily loss stop, and aggregates per match.
All records are persisted to an append-only JSONL file.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

VALID_STATUSES = {"pending", "placed", "won", "lost", "void"}
SETTLED_STATUSES = {"won", "lost", "void"}


@dataclass
class BetRecord:
    bet_id: str
    match_id: str
    market_key: str
    selection: str
    odds: float
    stake_ngn: float
    model_p: float
    p_lower: Optional[float]
    ev: float
    confidence: str
    correlation_groups: list[str]
    status: str = "pending"
    pnl_ngn: float = 0.0
    placed_at: Optional[str] = None
    settled_at: Optional[str] = None
    settled_odds: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BetRecord:
        return cls(**d)


@dataclass
class LedgerConfig:
    currency: str = "NGN"
    daily_loss_limit: float = -500.0
    min_stake: float = 100.0
    max_stake: float = 1000.0


class BetLedger:
    def __init__(self, config: LedgerConfig | None = None, log_path: str = "data/ops/ledger.jsonl"):
        self.config = config or LedgerConfig()
        self.log_path = Path(log_path)
        # Reentrant: place_bet/settle_bet hold this lock while calling
        # is_daily_stop_active()/daily_pnl()/summary(), which re-acquire it.
        # A plain Lock self-deadlocks there.
        self._lock = threading.RLock()
        self._bets: dict[str, BetRecord] = {}

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.log_path.exists():
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                rec = BetRecord.from_dict(entry["record"])
                self._bets[rec.bet_id] = rec

    def _append(self, record: BetRecord, event: str) -> None:
        entry = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record": record.to_dict(),
        }
        line = json.dumps(entry, ensure_ascii=False)
        fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def place_bet(self, bet: BetRecord) -> bool:
        with self._lock:
            if self.is_daily_stop_active():
                return False
            if not self._stake_in_range(bet.stake_ngn):
                return False
            bet.status = "placed"
            bet.placed_at = datetime.now(timezone.utc).isoformat()
            self._bets[bet.bet_id] = bet
            self._append(bet, "bet_placed")
            return True

    def settle_bet(self, bet_id: str, status: str, settled_odds: float | None = None) -> None:
        if status not in SETTLED_STATUSES:
            raise ValueError(f"Invalid settle status: {status!r}. Must be one of {SETTLED_STATUSES}")
        with self._lock:
            bet = self._bets.get(bet_id)
            if bet is None:
                raise KeyError(f"Unknown bet_id: {bet_id}")
            bet.status = status
            bet.settled_at = datetime.now(timezone.utc).isoformat()
            bet.settled_odds = settled_odds
            bet.pnl_ngn = self._compute_pnl(bet, status, settled_odds)
            self._append(bet, "bet_settled")

    def daily_pnl(self, date: str | None = None) -> float:
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            return sum(
                b.pnl_ngn
                for b in self._bets.values()
                if b.settled_at and b.settled_at[:10] == date and b.status in SETTLED_STATUSES
            )

    def match_exposure(self, match_id: str) -> float:
        with self._lock:
            return sum(
                b.stake_ngn
                for b in self._bets.values()
                if b.match_id == match_id and b.status not in SETTLED_STATUSES
            )

    def match_pnl(self, match_id: str) -> float:
        with self._lock:
            return sum(
                b.pnl_ngn
                for b in self._bets.values()
                if b.match_id == match_id and b.status in SETTLED_STATUSES
            )

    def is_daily_stop_active(self) -> bool:
        return self.daily_pnl() <= self.config.daily_loss_limit

    def can_place_bet(self, stake_ngn: float) -> bool:
        return not self.is_daily_stop_active() and self._stake_in_range(stake_ngn)

    def summary(self) -> dict:
        with self._lock:
            total_bets = len(self._bets)
            settled = [b for b in self._bets.values() if b.status in SETTLED_STATUSES]
            won = [b for b in settled if b.status == "won"]
            lost = [b for b in settled if b.status == "lost"]
            void = [b for b in settled if b.status == "void"]
            return {
                "total_bets": total_bets,
                "pending": sum(1 for b in self._bets.values() if b.status == "pending"),
                "placed": sum(1 for b in self._bets.values() if b.status == "placed"),
                "won": len(won),
                "lost": len(lost),
                "void": len(void),
                "total_staked_ngn": sum(b.stake_ngn for b in settled),
                "total_pnl_ngn": sum(b.pnl_ngn for b in settled),
                "daily_pnl_ngn": self.daily_pnl(),
                "daily_stop_active": self.is_daily_stop_active(),
                "unrealized_exposure_ngn": sum(
                    b.stake_ngn for b in self._bets.values() if b.status not in SETTLED_STATUSES
                ),
                "currency": self.config.currency,
            }

    def _compute_pnl(self, bet: BetRecord, status: str, settled_odds: float | None) -> float:
        odds = settled_odds if settled_odds is not None else bet.odds
        if status == "won":
            return round(bet.stake_ngn * (odds - 1.0), 2)
        if status == "lost":
            return -bet.stake_ngn
        return 0.0

    def _stake_in_range(self, stake_ngn: float) -> bool:
        return self.config.min_stake <= stake_ngn <= self.config.max_stake
