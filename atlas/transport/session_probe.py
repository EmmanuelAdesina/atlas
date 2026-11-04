"""Session health probe CLI.

  python -m atlas.transport.session_probe

Exit codes: 0 session alive (and balance read, if parseable),
2 vault missing/invalid, 3 session rejected/expired, 4 unreadable payload.
Never prints cookie material — only the verdict and the balance.
"""

from __future__ import annotations

import sys

from atlas.transport.client import TransportClient
from atlas.transport.errors import (
    PreSendError,
    RejectedError,
    SchemaDriftError,
    SessionExpiredError,
    OutcomeUnknownError,
)
from atlas.transport.session import load_vault
from atlas.transport.sources import balance as balance_source


def main() -> int:
    try:
        session = load_vault()
    except FileNotFoundError:
        print("probe: vault missing — seed it with python -m atlas.transport.seed")
        return 2
    except ValueError as exc:
        print(f"probe: vault invalid: {exc}")
        return 2

    try:
        with TransportClient(session) as client:
            reading = balance_source.read_balance_ngn(client)
            print(f"probe: session ALIVE — balance NGN {reading.balance_ngn:.2f} (via {reading.source_key})")
            return 0
    except SessionExpiredError as exc:
        print(f"probe: session EXPIRED/REJECTED — {exc}")
        return 3
    except RejectedError as exc:
        print(f"probe: rejected by the book — {exc}")
        return 3
    except (PreSendError, OutcomeUnknownError) as exc:
        print(f"probe: transport failure (fail closed) — {exc}")
        return 4
    except SchemaDriftError as exc:
        print(f"probe: balance payload unparsable (raw quarantined) — {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
