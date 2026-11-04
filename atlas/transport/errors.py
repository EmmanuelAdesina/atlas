"""Transport error taxonomy.

These classes are the contract between the transport layer and the rest of
Atlas. The distinction between them is not cosmetic: the runner, the campaign
wall and the ledger make different decisions depending on which one fires.

Classes
-------
PreSendError
    The request provably never left the client (DNS failure, connect refused,
    TLS error, local validation). Safe to retry reads. Unsafe to assume
    anything about POSTs beyond "nothing was sent".
OutcomeUnknownError
    The request was sent and the outcome is ambiguous (read timeout, reset
    mid-body, 5xx after the server accepted). NEVER retried automatically.
    Only an independent second read (e.g. Bet History) may resolve it.
RejectedError
    The server affirmatively refused: HTTP 4xx, an error envelope
    (bizCode != 10000), or the site-level error page. Deterministic — no
    retry, the session/config is wrong or the book is refusing.
SessionExpiredError
    Authentication is invalid or refused. Raised on 401/403, on auth-shaped
    bizCode values, or on two consecutive failed session probes. Callers must
    treat the account as unreadable — never as "empty".
SchemaDriftError
    The payload arrived but did not match the archived schema. The raw body is
    quarantined to data/quarantine/ before this is raised. Deterministic.
StaleDataError
    The payload age exceeds its TTL. Raised by sources, not the client.
UnreadableBoardError
    The board payload parsed but carries no usable markets/fixtures. An
    unreadable board is NOT an empty board — this error exists precisely to
    keep that invariant machine-checkable.
BoardNotConfiguredError
    The board source has no archived request recipe (see
    ATLAS_BOARD_OUTCOMES_BODY). Explicit refusal — the transport layer must
    never invent an undocumented request body.
PartialCoverageError / warning-class
    Coverage of priced cells below the acceptance bar. Surfaced as a typed
    result flag rather than an exception; see sources/board.py.
"""

from __future__ import annotations


class TransportError(Exception):
    """Base class for all transport-layer failures."""


class PreSendError(TransportError):
    """The request never reached the server."""


class OutcomeUnknownError(TransportError):
    """The request was sent; the outcome cannot be asserted."""


class RejectedError(TransportError):
    """The server affirmatively refused the request."""


class SessionExpiredError(TransportError):
    """Authentication failed; the account must be treated as unreadable."""


class SchemaDriftError(TransportError):
    """Payload violated the archived schema; raw body quarantined."""


class StaleDataError(TransportError):
    """Payload is older than its freshness TTL."""


class UnreadableBoardError(TransportError):
    """Board parsed but is unusable (e.g. totalNum=0, zero markets)."""


class BoardNotConfiguredError(TransportError):
    """No archived request recipe for the board endpoint (pre-recon)."""


class OrderNotConfiguredError(TransportError):
    """No archived recipe/verification path for direct order submission.

    The place-order endpoint's request/response contract has never been
    captured (deliberately — Atlas never called it). Until Phase A archives
    it with explicit operator approval, direct submission refuses.
    """

