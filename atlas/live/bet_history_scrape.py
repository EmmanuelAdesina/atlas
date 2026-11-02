"""Bet History scraper — read the book's own settled ticket list.

This is the independent truth source for settlement: SportyBet's account
API `GET /api/ng/orders/order/v2/realbetlist?isSettled=10` reports, per
account, every settled ticket with its stake, its winnings and a
`winningStatus` enum. ATLAS reads it through the operator's own logged-in
browser session — the same proven path the live board capture uses — and
writes a normalized snapshot to `data/ops/settle_state.json` for the
reconcile step (`atlas.ops.settle_history`).

How the reading works (every point below is probe-proven on the live
account, 2026-11-01):

  - The Bet History page is an SPA like the live board: `navigate()` may
    raise a daemon-side "page load timeout (30s)" even though the tab was
    created and finishes loading afterwards. The scraper creates the tab,
    tolerates the timeout, and verifies the tab through the session
    registry + readiness polls instead of trusting navigate's return.
  - The API only answers same-origin XHR/credential requests fired FROM
    the site; an async `fetch` through the daemon gets the site-level
    "Sorry, something went wrong" error page. Synchronous XHR from the
    loaded page returns the real JSON (200, ~71KB for 50 tickets).
  - `evaluate_safe` returns an envelope `{'type': 'string', 'value':
    '<json string>'}`; every value is unwrapped before parsing.
  - The daemon does NOT await Promises, so nothing async is used.

The scraper is strictly read-only: it opens NO betslips, clicks NO bet
buttons, and never transfers money — it only navigates the account's own
Bet History page and reads its (already authenticated) API. Money walls
are unaffected: placement remains frozen elsewhere, this module cannot
place anything.

Schema truth (from the live response, totalNum=264 on 2026-11-01):
  top-level     `{"bizCode":10000, "data": {"pageSize","pageNo",
                "totalNum","entityList": [...]}}`
  ticket keys   orderId, shortId, totalStake, totalWinnings,
                winningStatus, createTime (epoch ms), currency,
                selectionSize, selections[...]
  winningStatus 20 = won (totalWinnings > 0), 30 = lost (0), 40 = void /
                push (stake returned, totalWinnings == stake). Anything
                else is reported as 'unknown:<raw>' and NEVER settled.

Usage (also usable standalone):
  python -m atlas.live.bet_history_scrape [--max-pages N] [--out PATH]
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atlas.browser.webbridge import DEFAULT_DAEMON_URL, WebBridgeAdapter, WebBridgeError

ROOT = Path(__file__).resolve().parents[2]

HISTORY_URL = "https://www.sportybet.com/ng/my_accounts/bet_history/sport_bets?isSettled=10"
HISTORY_PATH_HINT = "my_accounts/bet_history"
API_PATH = "/api/ng/orders/order/v2/realbetlist"
PAGE_SIZE = 50

WINNING_STATUS = {
    20: "won",
    30: "lost",
    40: "void",
}


def unwrap(value: Any) -> Any:
    """Peel the daemon's evaluate envelope: {'type': 'string', 'value': json-str}."""
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


def _iso_from_ms(ms: int) -> str:
    """Epoch milliseconds -> ISO-8601 UTC (the ledger's placed_at clock)."""
    secs, rem = divmod(int(ms), 1000)
    dt = datetime.fromtimestamp(secs, tz=timezone.utc)
    if rem:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{rem:03d}+00:00"
    return dt.isoformat()


def parse_ticket(row: dict) -> dict:
    """Normalize one raw `entityList` row into the settle_state ticket shape.

    Defensive: missing keys become None/absent rather than guesses — the
    reconcile is fail-closed and a malformed ticket must never look like a
    settled one.
    """
    raw_status = row.get("winningStatus")
    status = WINNING_STATUS.get(raw_status, f"unknown:{raw_status}")
    try:
        stake = round(float(row.get("totalStake") or 0.0), 2)
        winnings = round(float(row.get("totalWinnings") or 0.0), 2)
    except (TypeError, ValueError):
        stake = None
        winnings = None

    create_ms = row.get("createTime")
    selections = []
    for sel in row.get("selections") or []:
        odds_raw = sel.get("odds")
        try:
            odds = float(odds_raw) if odds_raw is not None else None
        except (TypeError, ValueError):
            odds = None
        score = None
        for key in ("setScore", "gameScore"):
            if sel.get(key):
                score = sel.get(key)
                break
        selections.append(
            {
                "event_id": sel.get("eventId"),
                "market_desc": sel.get("marketDesc"),
                "outcome_desc": sel.get("outcomeDesc"),
                "correct_outcome": sel.get("correctOutcome"),
                "odds": odds,
                "home": sel.get("home"),
                "away": sel.get("away"),
                "leg_status": sel.get("status"),
                "match_status": sel.get("matchStatus"),
                "event_status": sel.get("eventStatus"),
                "score": score,
                "specifier": sel.get("specifier"),
                "tournament": sel.get("tournamentName"),
            }
        )

    first = (row.get("selections") or [{}])[0]
    try:
        first_odds = float(first.get("odds")) if first.get("odds") is not None else None
    except (TypeError, ValueError):
        first_odds = None

    ticket = {
        "order_id": row.get("orderId"),
        "short_id": row.get("shortId"),
        "create_time_ms": create_ms,
        "create_time": _iso_from_ms(create_ms) if create_ms else None,
        "winning_status": raw_status,
        "status": status,
        "stake_ngn": stake,
        "winnings_ngn": winnings,
        "currency": row.get("currency"),
        "selection_size": row.get("selectionSize"),
        "odds": first_odds,
        "selections": selections,
    }
    return ticket


def _build_page_js(page_no: int) -> str:
    """Synchronous same-origin XHR for one page, returning the raw body.

    Sync XHR is the only request shape the site accepts here (probe-proven;
    async fetch through the daemon returns the site's error page). The
    status and body travel together as small JSON for exact failures.
    """
    return (
        "(function(){"
        "  try {"
        "    var x = new XMLHttpRequest();"
        f"    x.open('GET', '{API_PATH}?isSettled=10&pageSize={PAGE_SIZE}&pageNo={page_no}', false);"
        "    x.withCredentials = true;"
        "    x.setRequestHeader('Accept', 'application/json, text/plain, */*');"
        "    x.send(null);"
        "    return JSON.stringify({s: x.status, b: x.responseText || ''});"
        "  } catch (e) {"
        "    return JSON.stringify({s: 0, b: '', e: String(e)});"
        "  }"
        "})()"
    )


def _fetch_page(adapter: WebBridgeAdapter, page_no: int) -> dict:
    """One API page as a parsed dict. Raises on any non-bizCode-10000 reply."""
    raw = unwrap(adapter.evaluate_safe(_build_page_js(page_no)))
    if not (isinstance(raw, str) and raw.strip().startswith("{")):
        raise WebBridgeError(
            f"bet history page {page_no}: unexpected evaluate result "
            f"{str(raw)[:120]!r}",
            transient=True,
        )
    wrapper = json.loads(raw)
    if wrapper.get("s") != 200 or wrapper.get("e"):
        raise WebBridgeError(
            f"bet history page {page_no}: HTTP {wrapper.get('s')} "
            f"{str(wrapper.get('e') or '')[:120]}",
            transient=True,
        )
    body = wrapper.get("b") or ""
    try:
        payload = json.loads(body)
    except Exception as exc:
        raise WebBridgeError(
            f"bet history page {page_no}: unparseable body head {body[:160]!r}",
            transient=True,
        ) from exc
    if payload.get("bizCode") != 10000:
        raise WebBridgeError(
            f"bet history page {page_no}: bizCode={payload.get('bizCode')} "
            f"message={str(payload.get('message'))[:120]!r}",
            transient=True,
        )
    data = payload.get("data") or {}
    if not isinstance(data.get("entityList"), list):
        raise WebBridgeError(
            f"bet history page {page_no}: missing data.entityList "
            f"(keys: {sorted(data)[:12]})",
            transient=True,
        )
    return data


def _open_history_tab(adapter: WebBridgeAdapter) -> None:
    """Create the Bet History tab in the scraper's own session.

    navigate() may time out (history SPA never fires 'load' quickly) —
    tolerated, because the extension still creates the tab. Verification
    is by the session registry and readiness polls, not by navigate's
    return value.
    """
    created = False
    last_err = "navigate returned success but the tab never appeared in the session registry"

    # An earlier run may already hold a Bet History tab in this session —
    # reuse it instead of opening another one every hourly pass (the box is
    # memory-constrained; every idle tab is real RAM).
    found = _find_history_tab(adapter)
    if found is not None:
        return

    for hint_try in (1, 2):
        try:
            adapter.navigate(HISTORY_URL, new_tab=True)
            created = True
        except WebBridgeError as exc:
            # Timeout is expected behaviour on this SPA; check the tab anyway.
            found = _find_history_tab(adapter)
            if found is not None:
                return
            last_err = str(exc)
        if created and _find_history_tab(adapter) is not None:
            return
        time.sleep(5)
    found = _find_history_tab(adapter)
    if found is not None:
        return
    raise WebBridgeError(
        "bet history tab could not be opened in session "
        f"{adapter.session!r} (last navigate error: {last_err[:160]})",
        transient=True,
    )


def _find_history_tab(adapter: WebBridgeAdapter) -> Optional[dict]:
    try:
        tab = adapter.find_tab(HISTORY_PATH_HINT)
        return tab if isinstance(tab, dict) and tab.get("url") else None
    except Exception:
        return None


READY_BODY_MIN_CHARS = 500


def _wait_ready(adapter: WebBridgeAdapter, max_wait_s: float = 75.0) -> None:
    """Readiness gate: the history app has rendered and is on the right URL.

    A body shorter than READY_BODY_MIN_CHARS means the SPA shell has not
    rendered its account chrome yet; polling continues instead of storing
    an error-page snapshot.
    """
    deadline = time.time() + max_wait_s
    last = None
    while time.time() < deadline:
        try:
            state = unwrap(
                adapter.evaluate_safe(
                    "JSON.stringify({rs: document.readyState,"
                    " url: location.href,"
                    " len: (document.body ? document.body.innerText.length : 0)})"
                )
            )
            if isinstance(state, str):
                state = json.loads(state)
            if isinstance(state, dict):
                last = state
                if (
                    HISTORY_PATH_HINT in str(state.get("url") or "")
                    and int(state.get("len") or 0) >= READY_BODY_MIN_CHARS
                    and state.get("rs") in ("complete", "interactive")
                ):
                    return
        except Exception:
            pass
        time.sleep(2.0)
    raise WebBridgeError(
        "bet history page never reached readiness; last state "
        f"{json.dumps(last)[:200]}",
        transient=True,
    )


# ---------------------------------------------------------------------------
# CDP transport — probe-proven 2026-11-02 (264 tickets, one session cdp-9222)
# ---------------------------------------------------------------------------

CDP_DEBUG_HOST = "http://127.0.0.1:9222"
CDP_READY_BODY_MIN_CHARS = 500


class _CDPTab:
    """Minimal CDP driver for tab control + Runtime.evaluate.

    Chrome 156 ignores `/json/new?url=` — a blank target is created and
    navigated over the websocket (probe-proven; URL-carrying variants
    receive about:blank).
    """

    def __init__(self, debug_host: str):
        import websocket  # VPS venv: pip install --no-deps websocket-client

        self._ws_mod = websocket
        targets = _http_json(debug_host + "/json/new", method="PUT")
        self.id = targets.get("id") or ""
        self._ws = websocket.create_connection(
            targets["webSocketDebuggerUrl"], timeout=90
        )
        self._mid = 0

    def call(self, method: str, params: Optional[dict] = None, timeout: Optional[float] = None):
        """One CDP round-trip, deadline-protected.

        The recv loop is open-ended against event storms; without a
        wall-clock deadline a silenced socket burns the oneshot unit. Any
        socket-layer failure (timeout, closed, parse) surfaces as
        WebBridgeError(transient) — the unit's loud-skip path — never a
        raw crash (the 10:23 job hard-failed on exactly this).
        """
        self._mid += 1
        tid = self._mid
        deadline = time.monotonic() + (timeout if timeout else 90.0)
        try:
            self._ws.send(
                json.dumps({"id": tid, "method": method, "params": params or {}})
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WebBridgeError(
                        f"cdp call {method} stalled >{timeout or 90}s "
                        "(no matching response)",
                        transient=True,
                    )
                self._ws.settimeout(min(15.0, max(1.0, remaining)))
                msg = json.loads(self._ws.recv())
                if msg.get("id") == tid:
                    if "error" in msg:
                        raise WebBridgeError(
                            f"cdp {method}: {msg['error']}", transient=True
                        )
                    return msg.get("result") or {}
                # non-matching ids are events; keep draining to the id
        except WebBridgeError:
            raise
        except self._ws_mod.WebSocketTimeoutException as exc:
            raise WebBridgeError(
                f"cdp call {method}: socket timeout ({str(exc)[:100]})",
                transient=True,
            ) from exc
        except self._ws_mod.WebSocketException as exc:
            raise WebBridgeError(
                f"cdp call {method}: transport gone ({str(exc)[:100]})",
                transient=True,
            ) from exc
        except (json.JSONDecodeError, OSError) as exc:
            raise WebBridgeError(
                f"cdp call {method}: transport/parse failure ({str(exc)[:100]})",
                transient=True,
            ) from exc

    def evaluate(self, expression: str):
        res = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "userGesture": True},
        )
        return res.get("result") or {}

    def close(self) -> None:
        try:
            self.call("Page.close")
        except Exception:
            pass
        try:
            self._ws.close()
        except Exception:
            pass


def _http_json(url: str, method: str = "GET") -> dict:
    from urllib.request import Request, urlopen

    req = Request(url, method=method)  # noqa: S310 — local debug endpoint only
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode() or "{}")


def _wait_ready_cdp(tab: _CDPTab, max_wait_s: float = 120.0) -> None:
    """CDP readiness gate (probe-proven: history SPA body >= 500 chars,
    the same floor the extension transport proves; 900 was an overfit to
    one poll moment)."""
    deadline = time.time() + max_wait_s
    last: Any = None
    while time.time() < deadline:
        try:
            v = json.loads(tab.evaluate(
                "JSON.stringify({rs: document.readyState,"
                " url: location.href,"
                " len: (document.body ? document.body.innerText.length : 0)})"
            ).get("value") or "{}")
            last = v
            if (
                HISTORY_PATH_HINT in str(v.get("url") or "")
                and int(v.get("len") or 0) >= CDP_READY_BODY_MIN_CHARS
                and v.get("rs") in ("complete", "interactive")
            ):
                return
        except WebBridgeError as exc:
            # Mid-poll transport blips (socket timeout / closed / event
            # storm — the 10:23 raw-WebSocketTimeoutException class) keep
            # polling; the deadline below converts persistence into ONE
            # loud readiness error. Never crash the pass from a blip.
            if not exc.transient:
                raise
            last = f"poll blip: {str(exc)[:120]}"
        except OSError as exc:
            last = f"poll blip: connection-level error {str(exc)[:120]}"
        except Exception:
            pass
        time.sleep(3.0)
    raise WebBridgeError(
        f"cdp: bet history page never reached readiness; last {json.dumps(last)[:200]}",
        transient=True,
    )


def _fetch_page_cdp(tab: _CDPTab, page_no: int) -> dict:
    """One API page through the SAME sync-XHR shape the extension uses."""
    raw = tab.evaluate(_build_page_js(page_no))
    if not (isinstance(raw, dict) and raw.get("value")):
        raise WebBridgeError(
            f"cdp bet history page {page_no}: unexpected evaluate result "
            f"{str(raw)[:120]!r}",
            transient=True,
        )
    wrapper = json.loads(raw["value"])
    if wrapper.get("s") != 200 or wrapper.get("e"):
        raise WebBridgeError(
            f"cdp bet history page {page_no}: HTTP {wrapper.get('s')} "
            f"{str(wrapper.get('e') or '')[:120]}",
            transient=True,
        )
    body = wrapper.get("b") or ""
    try:
        payload = json.loads(body)
    except Exception as exc:
        raise WebBridgeError(
            f"cdp bet history page {page_no}: unparseable body head {body[:160]!r}",
            transient=True,
        ) from exc
    if payload.get("bizCode") != 10000:
        raise WebBridgeError(
            f"cdp bet history page {page_no}: bizCode={payload.get('bizCode')} "
            f"message={str(payload.get('message'))[:120]!r}",
            transient=True,
        )
    data = payload.get("data") or {}
    if not isinstance(data.get("entityList"), list):
        raise WebBridgeError(
            f"cdp bet history page {page_no}: missing data.entityList",
            transient=True,
        )
    return data


def _build_state(
    raw_rows: list[dict],
    pages_fetched: int,
    total_num: int,
    session: str,
    transport: str,
) -> dict:
    """Normalize + dedupe + assemble — the exact same settle_state.json either way."""
    tickets = [parse_ticket(row) for row in raw_rows]
    seen: set[str] = set()
    unique: list[dict] = []
    for t in tickets:
        oid = t.get("order_id") or ""
        if oid and oid in seen:
            continue
        if oid:
            seen.add(oid)
        unique.append(t)

    enum_counts: dict[str, int] = {}
    for t in unique:
        enum_counts[t["status"]] = enum_counts.get(t["status"], 0) + 1

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "api": API_PATH,
            "host": "www.sportybet.com",
            "filter": "isSettled=10",
            "transport": transport,
            "session": session,
            "pages_fetched": pages_fetched,
            "page_size": PAGE_SIZE,
        },
        "total_num": total_num,
        "count": len(unique),
        "ticket_status_enum": enum_counts,
        "tickets": unique,
    }


def _scrape_cdp(out: Path, max_pages: int) -> dict:
    """CDP transport: navigate the history SPA, same-origin sync XHRs.

    Spin-off failure is WebBridgeError(too) so the hourly unit's skip
    logic is transport-agnostic. Read-only: no betslips, no clicks, no
    money; the target is closed afterwards (memory hygiene).
    """
    try:
        import websocket  # noqa: F401 — lazily: only present in the VPS venv
    except ImportError as exc:
        raise WebBridgeError(
            "cdp transport needs 'websocket-client' in the venv "
            "(pip install --no-deps websocket-client)",
            transient=True,
        ) from exc

    debug_host = os.environ.get("ATLAS_CDP_HOST", CDP_DEBUG_HOST)
    try:
        tab = _CDPTab(debug_host)
    except Exception as exc:  # chrome down / port closed: skip, not crash
        raise WebBridgeError(
            f"cdp: no debug target on {debug_host}: {str(exc)[:160]}",
            transient=True,
        ) from exc

    try:
        tab.call("Page.navigate", {"url": HISTORY_URL}, timeout=30)
        _wait_ready_cdp(tab)
        data = _fetch_page_cdp(tab, 1)
        try:
            total_num = int(data.get("totalNum") or len(data.get("entityList") or []))
        except (TypeError, ValueError) as exc:
            raise WebBridgeError(
                f"cdp bet history: malformed totalNum "
                f"{data.get('totalNum')!r} (entityList len "
                f"{len(data.get('entityList') or [])})",
                transient=True,
            ) from exc
        pages_needed = max(1, (total_num + PAGE_SIZE - 1) // PAGE_SIZE)
        if max_pages > 0:
            pages_needed = min(pages_needed, max_pages)

        raw_rows: list[dict] = []
        for page_no in range(1, pages_needed + 1):
            if page_no > 1:
                time.sleep(0.6)
                data = _fetch_page_cdp(tab, page_no)
            raw_rows.extend(data.get("entityList") or [])
    finally:
        tab.close()

    state = _build_state(raw_rows, pages_needed, total_num, "cdp-9222", "cdp")
    _write_atomic(out, state)
    return state


def scrape(
    session: Optional[str] = None,
    daemon_url: Optional[str] = None,
    out_path: Optional[str] = None,
    max_pages: Optional[int] = None,
    transport: Optional[str] = None,
) -> dict:
    """Read the book's full settled Bet History and write settle_state.json.

    Returns the state dict that was written. Raises WebBridgeError when
    both transports are unavailable — callers (e.g. the hourly oneshot
    unit) may treat that as a skip.

    Transports (`ATLAS_BET_HISTORY_TRANSPORT` = extension | cdp | auto):

      - extension: the daemon/extension path (the live-board proven path).
      - cdp: Chrome DevTools protocol directly on the logged-in profile
        (probes proved the daemon wedge, not the site — this bypasses it;
        probe-proven 2026-11-02, 264 tickets).
      - auto (default): extension first, CDP on extension failure — the
        durable path on the memory-squeezed box.
    """
    session = session or os.environ.get(
        "ATLAS_SETTLE_SESSION", "atlas-recon-history"
    )
    # Dedicated recon session, NEVER the money loop's own browser session.
    if session == "atlas-live" or session.startswith("atlas-live-"):
        raise ValueError(
            f"refusing to run the Bet History scrape in the money loop's "
            f"browser session {session!r}"
        )

    out = Path(out_path) if out_path else ROOT / "data" / "ops" / "settle_state.json"

    cap = max_pages
    if cap is None:
        cap = int(os.environ.get("ATLAS_BET_HISTORY_MAX_PAGES", "0") or 0)

    transport = (
        transport
        or os.environ.get("ATLAS_BET_HISTORY_TRANSPORT", "auto")
        or "auto"
    ).strip().lower()

    if transport == "cdp":
        return _scrape_cdp(out=out, max_pages=cap)

    try:
        return _scrape_extension(session, daemon_url, out, cap)
    except WebBridgeError:
        if transport != "auto":
            raise
        # auto: fall through to the probe-proven CDP path
        return _scrape_cdp(out=out, max_pages=cap)


def _scrape_extension(
    session: str,
    daemon_url: Optional[str],
    out: Path,
    cap: int,
) -> dict:
    """The original daemon/extension transport (see module docstring)."""
    daemon_url = daemon_url or os.environ.get(
        "ATLAS_WEBBRIDGE_DAEMON_URL", DEFAULT_DAEMON_URL
    )
    adapter = WebBridgeAdapter(session=session, daemon_url=daemon_url)

    _open_history_tab(adapter)
    _wait_ready(adapter)

    try:
        # Page 1 fixes the total; later pages follow it.
        data = _fetch_page(adapter, 1)
        try:
            total_num = int(data.get("totalNum") or len(data.get("entityList") or []))
        except (TypeError, ValueError) as exc:
            raise WebBridgeError(
                f"bet history page 1: malformed totalNum {data.get('totalNum')!r}",
                transient=True,
            ) from exc
        pages_needed = max(1, (total_num + PAGE_SIZE - 1) // PAGE_SIZE)
        if cap > 0:
            pages_needed = min(pages_needed, cap)

        raw_rows: list[dict] = []
        for page_no in range(1, pages_needed + 1):
            if page_no > 1:
                time.sleep(0.6)
                data = _fetch_page(adapter, page_no)
            raw_rows.extend(data.get("entityList") or [])
    finally:
        # Recon hygiene: release the tab and its RAM; the account page state
        # (cookies/session) lives in the browser profile, not the tab.
        try:
            adapter.close_session()
        except Exception:
            pass

    tickets = [parse_ticket(row) for row in raw_rows]

    # Deduplicate (a re-listed order must not double-count).
    seen: set[str] = set()
    unique: list[dict] = []
    for t in tickets:
        oid = t.get("order_id") or ""
        if oid and oid in seen:
            continue
        if oid:
            seen.add(oid)
        unique.append(t)

    enum_counts: dict[str, int] = {}
    for t in unique:
        enum_counts[t["status"]] = enum_counts.get(t["status"], 0) + 1

    state = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "api": API_PATH,
            "host": "www.sportybet.com",
            "filter": "isSettled=10",
            "session": session,
            "pages_fetched": pages_needed,
            "page_size": PAGE_SIZE,
        },
        "total_num": total_num,
        "count": len(unique),
        "ticket_status_enum": enum_counts,
        "tickets": unique,
    }

    _write_atomic(out, state)
    return state


def _write_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = None
    max_pages = None
    strict = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]
            i += 2
        elif arg == "--max-pages" and i + 1 < len(argv):
            max_pages = int(argv[i + 1])
            i += 2
        elif arg == "--strict":
            strict = True
            i += 1
        else:
            print(f"bet_history_scrape: unknown argument {arg!r}")
            return 2

    try:
        state = scrape(out_path=out_path, max_pages=max_pages)
    except WebBridgeError as exc:
        # Browser/extension stack down or page unreachable: the hourly
        # oneshot must not page the operator for an environmental skip.
        # --strict re-raises for interactive diagnosis.
        if strict:
            raise
        print(f"bet_history_scrape: SKIPPED (browser unavailable): {str(exc)[:300]}")
        return 0

    src = state.get("source") or {}
    print(
        "bet_history_scrape: "
        f"count={state['count']} total_num={state['total_num']} "
        f"pages={src.get('pages_fetched')} session={src.get('session')} "
        f"enum={json.dumps(state['ticket_status_enum'], sort_keys=True)}"
    )
    print(f"  out={Path(out_path or (ROOT / 'data' / 'ops' / 'settle_state.json')).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
