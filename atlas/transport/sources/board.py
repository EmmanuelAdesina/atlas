"""Board source — built to the archive, not to imagination.

The live board is served by `POST /api/ng/factsCenter/Outcomes` (observed
200 in recon, net1.json, requestId 13352.22733). The request body and the
response schema have NEVER been captured, and this module does not invent
them. What exists today:

  * the transport client, error taxonomy and raw-archive pipeline (Phase B);
  * `dump_raw()` — replay an operator-supplied request recipe against the
    endpoint and archive the verbatim response, for Phase A recon;
  * `BoardNotConfiguredError` — until a sanitized Outcomes fixture exists,
    callers get an explicit refusal, never fabricated odds.

Phase A work (see docs/browserless-transport-design.md §3) turns
`dump_raw()` output into `tests/fixtures/api/outcomes_*.json`; Phase B2 then
adds the pydantic schema and a `read_markets()` implementation against that
fixture. Anything written here before that moment would be fiction wearing a
schema's clothes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from atlas.transport import snapshots
from atlas.transport.client import TransportClient
from atlas.transport.errors import BoardNotConfiguredError

OUTCOMES_PATH = "/api/ng/factsCenter/Outcomes"
RECIPE_ENV = "ATLAS_BOARD_OUTCOMES_BODY"  # path to a JSON file with the request body


def _load_recipe(recipe_path: Optional[str]) -> Any:
    path = recipe_path or os.environ.get(RECIPE_ENV)
    if not path:
        raise BoardNotConfiguredError(
            f"board recon incomplete: set {RECIPE_ENV} to the archived Outcomes "
            "request body (Phase A, docs/browserless-transport-design.md §3 U1)"
        )
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, (dict, list)):
        raise BoardNotConfiguredError(f"{RECIPE_ENV} must decode to a JSON object/array")
    return body


def dump_raw(client: TransportClient, recipe_path: Optional[str] = None) -> Path:
    """Replay the archived Outcomes recipe and archive the raw response.

    Read-only recon tool. The recipe file is produced by Phase A; responses
    land in data/snapshots/outcomes/ for schema extraction. Never returns
    parsed state — parsing does not exist yet, by design.
    """
    body = _load_recipe(recipe_path)
    resp = client.post(OUTCOMES_PATH, body=body, referer="https://www.sportybet.com/ng/sport/football/live_list/")
    out = snapshots.archive_raw(
        "outcomes", resp.body, meta={"status_code": resp.status_code, "recon": True}
    )
    return out


def read_markets():  # pragma: no cover - intentionally unimplemented
    raise BoardNotConfiguredError(
        "board parsing is gated on Phase A: archive sanitized Outcomes fixtures, "
        "then land schemas.py::OutcomesBoard and normalize.py — in that order"
    )
