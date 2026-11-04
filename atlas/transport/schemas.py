"""Pydantic models for the SportyBet envelope and confirmed payloads.

Discipline (mirrors the rest of Atlas):

  * Archive raw bytes BEFORE parsing — these models describe the archived
    fixtures, they do not replace them.
  * ``extra="ignore"`` — the book may add fields; absence of a known field is
    a schema error, addition is not.
  * Unknown enum values are preserved (``winningStatus`` may grow; the settle
    path reports ``unknown:<raw>`` and never settles on it).
  * Quantity-shaped fields that the book ships as strings stay strings here;
    parsing to float happens in normalize/settle code with explicit guards.

Only schemas with archived evidence are modelled. The board (Outcomes)
schemas land in Phase A against sanitized fixtures — never invented.
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

DataT = TypeVar("DataT")


class Envelope(BaseModel, Generic[DataT]):
    """The book's standard response envelope.

    bizCode 10000 = success. Anything else is REJECTED — callers must branch
    on it via :meth:`require_ok`, which also surfaces the book's message.
    """

    model_config = ConfigDict(extra="ignore")

    bizCode: int
    isAvailable: Optional[bool] = None
    message: Optional[str] = None
    data: Optional[DataT] = None

    def require_ok(self) -> "Envelope[DataT]":
        from atlas.transport.errors import RejectedError

        if self.bizCode != 10000:
            raise RejectedError(
                f"bizCode={self.bizCode} message={self.message!r}"
            )
        if self.isAvailable is False:
            raise RejectedError(f"isAvailable=false message={self.message!r}")
        if self.data is None:
            raise RejectedError("envelope carries no data payload")
        return self


class TicketSelection(BaseModel):
    """One selection inside a settled ticket (subset of observed keys)."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    eventId: Optional[str] = None
    jointId: Optional[str] = None
    marketId: Optional[str] = None
    outcomeId: Optional[str] = None
    status: Optional[int] = None
    matchStatus: Optional[str] = None
    eventStatus: Optional[int] = None
    correctOutcome: Optional[str] = None
    specifier: Optional[str] = None
    odds: Optional[str] = None
    marketDesc: Optional[str] = None
    outcomeDesc: Optional[str] = None
    home: Optional[str] = None
    away: Optional[str] = None
    categoryName: Optional[str] = None
    tournamentName: Optional[str] = None
    startTime: Optional[int] = None
    setScore: Optional[Any] = None
    gameScore: Optional[Any] = None


class Ticket(BaseModel):
    """One settled order row from realbetlist entityList."""

    model_config = ConfigDict(extra="ignore")

    orderId: Optional[str] = None
    shortId: Optional[str] = None
    orderType: Optional[int] = None
    totalStake: Optional[str] = None
    totalWinnings: Optional[str] = None
    winningStatus: Optional[int] = None
    selectionSize: Optional[int] = None
    combinationSize: Optional[int] = None
    createTime: Optional[int] = Field(default=None, description="epoch ms")
    currency: Optional[str] = None
    shareCode: Optional[str] = None
    isPaymentInProgress: Optional[bool] = None
    selections: list[TicketSelection] = Field(default_factory=list)


class RealbetlistData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pageSize: Optional[int] = None
    pageNo: Optional[int] = None
    totalNum: int
    entityList: list[Ticket] = Field(default_factory=list)


RealbetlistEnvelope = Envelope[RealbetlistData]
