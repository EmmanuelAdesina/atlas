"""DirectOrderExecutor — the Option B boundary. DISABLED BY DEFAULT.

Read this before enabling anything:

  * This module exists so the transport redesign does not paint Atlas into
    "one browser click per bet" forever. It is NOT wired into the runner,
    NOT reachable from any CLI flag today, and `allow_submit` defaults to
    False — the constructor itself raises unless the caller passes
    `allow_submit=True` explicitly.
  * The place-order endpoint contract has NEVER been captured (Atlas has
    never called it — POST /api/ng/orders/order returns 200 even on refusal,
    which is why receipts are verified, never trusted). Enabling requires:
      1. Phase A capture of the order request/response (operator approval),
      2. a recipe file (ATLAS_ORDER_RECIPE) built from that capture,
      3. an injected `verifier` — an independent read (pending-orders or
         Bet History) confirming the order by its identifiers,
      4. all existing walls intact: campaign wall, daily stop, stake caps,
         `--submit`, `PolicyConfig(allow_execution=True)`.
  * No retries on POST, ever. An ambiguous outcome stays ambiguous; the
    verifier (or the operator) resolves it. The ledger records the attempt
    with the exact time and the real verdict — see mark_verified_placed().

Ledger semantics on every path:
  refused at gates   -> no bet recorded
  rejected by book   -> attempt appended as 'bet_refused_http' (no position)
  accepted+verified  -> place_bet() then mark_verified_placed(): the book's
                        createTime (epoch ms) becomes the stake time, which
                        is the exact moment the money was staked
  accepted+unproven  -> place_bet() with receipt='unverified:<id>'; the
                        position IS real exposure and counts immediately;
                        settlement reconciles by identifiers
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from atlas.transport import snapshots
from atlas.transport.client import TransportClient
from atlas.transport.errors import (
    OrderNotConfiguredError,
    OutcomeUnknownError,
    RejectedError,
)

ORDER_PATH = "/api/ng/orders/order"
RECIPE_ENV = "ATLAS_ORDER_RECIPE"  # JSON file: {"body_template": {...}, "note": ...}


class SubmissionNotAllowed(Exception):
    """Raised before anything is sent when allow_submit is not explicitly True."""


@dataclass
class OrderRequest:
    event_id: str
    market_id: str
    outcome_id: str
    odds: float
    stake_ngn: float
    bet_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    match_label: Optional[str] = None
    selection: Optional[str] = None
    market_key: Optional[str] = None
    model_p: Optional[float] = None
    p_lower: Optional[float] = None
    ev: Optional[float] = None
    confidence: Optional[str] = None
    correlation_groups: list[str] = field(default_factory=list)
    match_id: str = "live"


Verifier = Callable[[OrderRequest, dict], Optional[dict]]
# verifier(order, posted_payload) -> None if unproven, else
# {"order_id": str, "short_id": str|None, "create_time_ms": int, "odds": float|None}


def _load_recipe(recipe_path: Optional[str]) -> dict:
    path = recipe_path or os.environ.get(RECIPE_ENV)
    if not path:
        raise OrderNotConfiguredError(
            f"direct order submission is not configured: set {RECIPE_ENV} to the "
            "archived order-recipe capture (Phase A + operator approval required)"
        )
    recipe = json.loads(open(path, encoding="utf-8").read())
    template = recipe.get("body_template")
    if not isinstance(template, dict):
        raise OrderNotConfiguredError(f"{RECIPE_ENV} recipe lacks body_template")
    return recipe


def _render(template, mapping: dict):
    """Recursive {{placeholder}} substitution. Identity for absent keys."""
    if isinstance(template, dict):
        return {k: _render(v, mapping) for k, v in template.items()}
    if isinstance(template, list):
        return [_render(v, mapping) for v in template]
    if isinstance(template, str):
        out = template
        for key, value in mapping.items():
            out = out.replace("{{" + key + "}}", str(value))
        return out
    return template


@dataclass
class OrderOutcome:
    state: str  # 'rejected' | 'accepted_unverified' | 'verified'
    bet_id: str
    detail: str
    order_id: Optional[str] = None
    create_time_ms: Optional[int] = None


class DirectOrderExecutor:
    def __init__(self, client: TransportClient, ledger, *, allow_submit: bool = False,
                 recipe_path: Optional[str] = None, verifier: Optional[Verifier] = None):
        if not allow_submit:
            raise SubmissionNotAllowed(
                "DirectOrderExecutor requires allow_submit=True — pass it only "
                "behind --submit, PolicyConfig(allow_execution=True) and the campaign wall"
            )
        self._client = client
        self._ledger = ledger
        self._recipe_path = recipe_path
        self._verifier = verifier

    # ------------------------------------------------------------------
    def place(self, order: OrderRequest) -> OrderOutcome:
        recipe = _load_recipe(self._recipe_path)
        mapping = {
            "eventId": order.event_id,
            "marketId": order.market_id,
            "outcomeId": order.outcome_id,
            "odds": f"{order.odds:.2f}",
            "stake": f"{order.stake_ngn:.2f}",
            "betId": order.bet_id,
        }
        body = _render(recipe["body_template"], mapping)

        resp = self._client.post(ORDER_PATH, body=body,
                                 referer="https://www.sportybet.com/ng/")
        snapshots.archive_raw(
            "order", resp.body,
            meta={"bet_id": order.bet_id, "status_code": resp.status_code},
        )
        if resp.status_code in (401, 403):
            from atlas.transport.errors import SessionExpiredError
            raise SessionExpiredError(f"order POST HTTP {resp.status_code}")

        try:
            payload = json.loads(resp.body)
        except json.JSONDecodeError as exc:
            # 200 with a non-JSON (or error-page) body after an order POST is
            # the AMBIGUOUS case: the book may have placed the bet. Never
            # call it a refusal. The caller must reconcile via Bet History.
            snapshots.quarantine_raw(
                "order", resp.body, f"unparseable order response: {exc}"
            )
            raise OutcomeUnknownError(
                "order POST returned an unreadable body — outcome unknown; "
                "reconcile via Bet History before retrying"
            ) from exc

        biz_code = payload.get("bizCode") if isinstance(payload, dict) else None
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(biz_code, int):
            raise OutcomeUnknownError(
                f"order POST 200 with no bizCode (message={message!r}) — "
                "outcome unknown; reconcile via Bet History"
            )
        if biz_code != 10000:
            # Affirmative refusal is recorded but never as a position.
            self._record_refusal(order, f"bizCode={biz_code} message={message!r}")
            raise RejectedError(
                f"order refused: bizCode={biz_code} message={message!r}"
            )

        # Accepted by envelope — but trust nothing yet.
        verified = self._verify(order, payload if isinstance(payload, dict) else {})
        self._record_acceptance(order, verified, payload)
        if verified:
            self._ledger.mark_verified_placed(
                order.bet_id,
                order_id=str(verified.get("order_id") or ""),
                short_id=verified.get("short_id"),
                create_time_ms=int(verified["create_time_ms"]),
                odds=verified.get("odds"),
            )
            return OrderOutcome("verified", order.bet_id,
                                "order verified against the book",
                                order_id=str(verified.get("order_id")),
                                create_time_ms=int(verified["create_time_ms"]))
        return OrderOutcome("accepted_unverified", order.bet_id,
                            "envelope accepted; independent verification pending/unavailable",
                            order_id=self._extract_order_id(payload))

    # ------------------------------------------------------------------
    def _verify(self, order: OrderRequest, posted_payload: dict) -> Optional[dict]:
        if self._verifier is None:
            return None
        try:
            result = self._verifier(order, posted_payload)
        except Exception:
            return None  # verification failure must not lose a real acceptance
        if not result:
            return None
        try:
            ctm = int(result["create_time_ms"])
        except (KeyError, TypeError, ValueError):
            return None
        if ctm <= 0:
            return None
        return result

    def _extract_order_id(self, payload: Optional[dict]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("orderId", "id"):
                if data.get(key):
                    return str(data[key])
        return None

    def _record_acceptance(self, order: OrderRequest, verified: Optional[dict],
                           payload) -> None:
        receipt = "unverified"
        if verified:
            receipt = f"verified:{verified.get('order_id')}"
        self._ledger.record_pending({
            "bet_id": order.bet_id,
            "match_id": order.match_id,
            "match_label": order.match_label,
            "event_id": order.event_id,
            "market_key": order.market_key or "",
            "selection": order.selection or "",
            "odds": order.odds,
            "stake_ngn": order.stake_ngn,
            "model_p": order.model_p or 0.0,
            "p_lower": order.p_lower,
            "ev": order.ev or 0.0,
            "confidence": order.confidence or "unknown",
            "correlation_groups": list(order.correlation_groups),
            "receipt": receipt,
            "odds_source": "book" if verified else "priced",
        })

    def _record_refusal(self, order: OrderRequest, why: str) -> None:
        # Refusals never become positions and never touch exposure math —
        # they are audited to their own append-only file, not the ledger.
        try:
            entry = {
                "event": "order_refused_http",
                "at": datetime.now(timezone.utc).isoformat(),
                "bet_id": order.bet_id,
                "match_label": order.match_label,
                "selection": order.selection,
                "odds": order.odds,
                "stake_ngn": order.stake_ngn,
                "why": why[:300],
            }
            path = Path(os.environ.get("ATLAS_REFUSAL_LOG", "data/ops/order_refusals.jsonl"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # audit failure must not mask the refusal itself

