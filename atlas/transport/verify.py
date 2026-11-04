"""Independent order verification against the book's own Bet History.

Shadows the DOM settle oracle the same way :mod:`atlas.transport.sources.settle`
does for settlement: after the book *accepts* an order (envelope says
bizCode 10000), the ONLY trustworthy confirmation is the book's own
record of the bet. This module builds that verifier from the realbetlist
feed the settle path already archives and schema-validates — never from
anything the POST response claims about itself.

Discipline (identical to the settle path):

  * Archive raw bytes BEFORE parsing (``fetch_page`` does) — never parse
    an unarchived response.
  * Fail-closed: any mismatch, ambiguity (zero or many matches), schema
    drift, or transport error returns ``None`` → the outcome stays
    ``accepted_unverified`` and the ledger records exposure without a
    forged placement time. Never guess.
  * No retries at the verifier level beyond the single ``retries=2``
    transport read in ``fetch_page`` — verification is not placement.

The verifier matches on identities the book cannot lie about for our
own bet: eventId + marketId + outcomeId (where the click landed),
totalStake (exactly, to the kobo, as the book prints it), odds, and
createTime (a plausible recency window — a re-placed identical old bet
does not match a fresh order). One and exactly one match → verified.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from atlas.transport.executor import OrderRequest

FetchPage = Callable[[int], list[dict]]

_DEFAULT_LOOKBACK_MS = 15 * 60 * 1000  # a placement is verified within minutes
_FUTURE_SKEW_MS = 60 * 1000  # tolerate book clock slightly ahead of ours


def _float_or_none(value) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def make_realbetlist_verifier(
    fetch_page: FetchPage,
    *,
    now_ms: Optional[Callable[[], int]] = None,
    lookback_ms: int = _DEFAULT_LOOKBACK_MS,
    page_no: int = 1,
) -> Callable[[OrderRequest, dict], Optional[dict]]:
    """Build a ``Verifier`` from a paginated realbetlist fetch.

    ``fetch_page(page_no)`` must return the raw ``entityList`` dicts (the
    shape ``sources.settle.fetch_page`` archives and validates). Newest
    bets are first; one page is enough for a fresh order — the caller may
    re-invoke the verifier later rather than page through history at the
    point of acceptance.
    """

    def verifier(order: OrderRequest, posted_payload: dict) -> Optional[dict]:
        now = now_ms() if now_ms else _now_ms()
        try:
            rows = fetch_page(page_no)
        except Exception:
            # Transport/schema failure must not fabricate a verification,
            # and must not mask a real acceptance — caller keeps the
            # exposure recorded as accepted_unverified.
            return None

        wanted_stake = f"{order.stake_ngn:.2f}"
        wanted_odds = round(float(order.odds), 2)
        oldest = now - lookback_ms

        matches: list[dict] = []
        for ticket in rows:
            if not isinstance(ticket, dict):
                continue
            create_time = ticket.get("createTime")
            if not isinstance(create_time, int) or create_time <= 0:
                continue
            if create_time > now + _FUTURE_SKEW_MS:
                continue  # book clock implausibly ahead — not ours (never ours)
            if create_time < oldest:
                continue
            if str(ticket.get("totalStake") or "").strip() != wanted_stake:
                continue
            selections = ticket.get("selections") or []
            if len(selections) != 1:
                continue  # Atlas places singles; combos are out of scope
            sel = selections[0]
            if str(sel.get("eventId") or "") != str(order.event_id):
                continue
            if str(sel.get("marketId") or "") != str(order.market_id):
                continue
            if str(sel.get("outcomeId") or "") != str(order.outcome_id):
                continue
            odds = _float_or_none(sel.get("odds"))
            if odds is None or abs(odds - wanted_odds) > 0.005:
                continue
            matches.append(ticket)

        if len(matches) != 1:
            return None  # ambiguity is a no-answer, never a coin toss

        hit = matches[0]
        sel = (hit.get("selections") or [{}])[0]
        order_id = str(hit.get("orderId") or "")
        if not order_id:
            return None  # a match without an orderId is not verifiable
        return {
            "order_id": order_id,
            "short_id": str(hit["shortId"]) if hit.get("shortId") else None,
            "create_time_ms": int(hit["createTime"]),
            "odds": _float_or_none(sel.get("odds")),
        }

    return verifier


def make_client_verifier(client) -> Callable[[OrderRequest, dict], Optional[dict]]:
    """Wire the verifier to a live TransportClient via the settle path."""
    from atlas.transport.sources import settle

    def fetch_page(page_no: int) -> list[dict]:
        rows, _total, _hint = settle.fetch_page(client, page_no)
        return rows

    return make_realbetlist_verifier(fetch_page)
