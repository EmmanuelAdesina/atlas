"""Balance + session-health probe source.

Confirmed endpoints observed in recon (netlist_temp.json, 2026-10-29):

  GET /api/ng/pocket/v1/finAccs/finAcc/userBal/NGN?_t=…   — user balance (NGN)
  GET /api/ng/pocket/v1/wallet/assetsInfo?_t=…            — wallet assets
  GET /api/ng/patron/account/info?_t=…                     — account session probe

Response bodies were NOT yet archived (Phase A U5). So this source is
deliberately tolerant: it accepts any JSON, searches a small set of known
balance-shaped keys recursively, and returns the raw payload alongside the
best-effort parse. If no plausible number is found it raises
SchemaDriftError — it NEVER reports a fabricated balance, and an unreadable
account is an error, not NGN 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from atlas.transport import snapshots
from atlas.transport.client import TransportClient
from atlas.transport.errors import (
    RejectedError,
    SchemaDriftError,
)

USER_BAL_PATH = "/api/ng/pocket/v1/finAccs/finAcc/userBal/NGN"
ACCOUNT_INFO_PATH = "/api/ng/patron/account/info"

# Recursive search candidates, in priority order. Observed wallet APIs nest
# balances under assorted keys; we take the first exact-match numeric hit and
# report which key produced it — provenance beats cleverness.
#
# "avlBal" observed live on the NGN pocket endpoint (bizCode 10000 envelope,
# data.avlBal, 2026-11-03 snapshot). It is the *available* balance; it is not
# to be interpreted as winnings, bonus, or total coins — those ride siblings
# (avlCoins / coinsStatus) and are deliberately NOT candidates.
BALANCE_KEY_CANDIDATES = (
    "balance", "availableBalance", "availableAmount", "userBal",
    "amount", "totalAmount", "walletBalance", "accBalance", "avlBal",
)


@dataclass
class BalanceReading:
    balance_ngn: float
    source_key: str
    raw: dict

    def as_dict(self) -> dict:
        return {
            "balance_ngn": self.balance_ngn,
            "source_key": self.source_key,
        }


def _find_balance(node: Any, key_candidates: tuple[str, ...]) -> Optional[tuple[str, float]]:
    if isinstance(node, dict):
        for key in key_candidates:
            value = node.get(key)
            if value is not None:
                try:
                    return key, round(float(str(value).replace(",", "")), 2)
                except (TypeError, ValueError):
                    pass
        for value in node.values():
            found = _find_balance(value, key_candidates)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_balance(item, key_candidates)
            if found:
                return found
    return None


def read_balance_ngn(client: TransportClient, archive: bool = True) -> BalanceReading:
    resp = client.get(USER_BAL_PATH, referer="https://www.sportybet.com/ng/", retries=1)
    if archive:
        snapshots.archive_raw(
            "balance", resp.body, meta={"status_code": resp.status_code}
        )
    try:
        payload = json.loads(resp.body)
    except json.JSONDecodeError as exc:
        snapshots.quarantine_raw("balance", resp.body, f"json: {exc}")
        raise SchemaDriftError("balance payload is not JSON") from exc

    # Envelope discipline where present — but never trust it blindly: some
    # wallet endpoints answer with bare objects.
    if isinstance(payload, dict) and "bizCode" in payload and payload.get("bizCode") != 10000:
        raise RejectedError(
            f"balance endpoint bizCode={payload.get('bizCode')} message={payload.get('message')!r}"
        )

    found = _find_balance(payload, BALANCE_KEY_CANDIDATES)
    if not found:
        snapshots.quarantine_raw("balance", resp.body, "no recognizable balance key")
        raise SchemaDriftError(
            "balance payload has no recognizable balance field — refusing to guess"
        )
    key, value = found
    return BalanceReading(balance_ngn=value, source_key=key, raw=payload)


def probe_session(client: TransportClient) -> bool:
    """Cheap authenticated probe. True = alive. Raises on hard failure."""
    resp = client.get(ACCOUNT_INFO_PATH, referer="https://www.sportybet.com/ng/", retries=0)
    try:
        payload = json.loads(resp.body)
    except json.JSONDecodeError:
        return False
    if isinstance(payload, dict) and "bizCode" in payload:
        return payload.get("bizCode") == 10000
    return bool(payload)
