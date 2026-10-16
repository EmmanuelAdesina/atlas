"""Extraction contract: WebBridge snapshot tree -> snapshot payload.

Input is the parsed `tree` from a WebBridge `snapshot` call (a list of
accessibility nodes with role/name/children). Output is the raw market list
the SportyBet collector ingests, plus best-effort match-state fields.

Lock semantics (observed on the live page): a selection row shows either a
decimal price (open) or no price at all (locked/suspended, padlock glyph).
Selections without a price are NOT given invented odds — the market is
emitted with status 'suspended' and empty selections, and the mapper records
it in `skipped` with reason 'suspended' so availability stays auditable.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_ODDS_RE = re.compile(r"^\d+\.\d{2}$")
_GLYPH_RE = re.compile(r"^[\ue000-\uf8ff]$")
_CLOCK_RE = re.compile(r"^(\d+):(\d{2})(?:\s*\+(\d+):(\d{2}))?$")
_SCORE_RE = re.compile(r"^(\d+)\s*:\s*(\d+)$")

_TABS_ROW = {"All", "Main", "Goals", "Half", "Bookings", "Corners", "Specials", "Players"}
_END_MARKERS = {"Betslip", "Cashout"}

# Header text -> (market_key, line, period). Unknown headers fall back to a
# slug key so observed markets are never silently dropped.
_HEADER_PATTERNS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"^1X2 - 1UP$"), ("1x2_1up", None, "full")),
    (re.compile(r"^1X2 - 2UP$"), ("1x2_2up", None, "full")),
    (re.compile(r"^1X2 - Never Down$"), ("1x2_never_down", None, "full")),
    (re.compile(r"^1X2$"), ("1x2", None, "full")),
    (re.compile(r"^Over/Under$"), (None, None, "full")),  # line from selections
    (re.compile(r"^2nd Goal$"), ("next_goal", None, "full")),
    (re.compile(r"^Handicap (\d+:\d+)$"), ("handicap", None, "full")),
    (re.compile(r"^Asian Handicap ([+-]?\d+(?:\.\d+)?)$"), ("asian_handicap", None, "full")),
    (re.compile(r"^2nd Half - 1X2$"), ("1x2_2h", None, "2h")),
    (re.compile(r"^1st Half - 1X2$"), ("1x2_1h", None, "1h")),
    (re.compile(r"^2nd Half - Over/Under$"), ((None, None, "2h"))),
    (re.compile(r"^1st Half - Over/Under$"), ((None, None, "1h"))),
]

_OU_SELECTION_RE = re.compile(r"^(Over|Under)\s+(\d+(?:\.\d+)?)$")

_SELECTION_STARTERS = ("Home", "Draw", "Away", "Over ", "Under ", "Yes", "No", "None")


def _looks_like_selection(tok: str) -> bool:
    t = tok.strip()
    if t in {"Home", "Draw", "Away", "Yes", "No", "None"}:
        return True
    return t.startswith(_SELECTION_STARTERS) or "(" in t


def iter_texts(tree: Any) -> list[str]:
    """DFS over snapshot nodes, returning StaticText/link names in order."""
    out: list[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("role") in {"StaticText", "link", "button"} and n.get("name"):
                out.append(str(n["name"]))
            children = n.get("children")
            if isinstance(children, list):
                for x in children:
                    walk(x)
        elif isinstance(n, list):
            for x in n:
                walk(x)

    walk(tree)
    return out


def _match_header(label: str) -> Optional[tuple[Optional[str], Optional[float], Optional[str]]]:
    for pat, spec in _HEADER_PATTERNS:
        m = pat.match(label)
        if m:
            key, line, period = spec
            if label.startswith("Handicap"):
                return (f"handicap_{m.group(1).replace(':', '_')}", None, "full")
            if label.startswith("Asian Handicap"):
                return (f"ah_{m.group(1)}", float(m.group(1)), "full")
            return (key, line, period)
    return None


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"other_{s}" if s else "other_unknown"


def extract_markets(texts: list[str], match_id: str) -> list[dict]:
    """Extract raw market dicts from ordered snapshot texts.

    Only the region between the market-tabs row and the Betslip footer is
    scanned, so scoreboard numbers (scores, stats) can never be mistaken
    for odds.
    """
    try:
        start = next(i for i, t in enumerate(texts)
                     if t in _TABS_ROW and texts[i:i + 2] == ["Teams", "Minutes"] or False)
    except StopIteration:
        start = -1
    # Robust start: first index where a tabs-row run appears.
    if start == -1:
        for i in range(len(texts) - 3):
            if texts[i:i + 4] == ["All", "Main", "Goals", "Half"]:
                start = i
                break
    if start == -1:
        return []
    end = next((i for i in range(start, len(texts)) if texts[i] in _END_MARKERS), len(texts))
    # Strip icon glyphs up front (locks included): a lock is encoded purely
    # as the absence of a price after a selection name.
    region = [t for t in texts[start:end] if t.strip() and not _GLYPH_RE.match(t)]

    markets: list[dict] = []
    current: Optional[dict] = None
    pending_name: Optional[str] = None

    def flush() -> None:
        nonlocal current, pending_name
        if current is None:
            return
        if pending_name is not None:
            current.setdefault("unpriced", []).append(pending_name)
            pending_name = None
        sels = current.pop("selections")
        unpriced = current.pop("unpriced")
        if len(sels) >= 2 and not unpriced:
            current["selections"] = sels
            current["status"] = "open"
        elif len(sels) >= 2:
            current["selections"] = sels
            current["status"] = "suspended"
        else:
            current["selections"] = []
            current["status"] = "suspended"
        markets.append(current)
        current = None

    i = 0
    while i < len(region):
        tok = region[i]
        hdr = _match_header(tok)
        if hdr is not None:
            flush()
            key, line, period = hdr if hdr else (None, None, None)
            current = {"match_id": match_id, "market_key": key or _slug(tok),
                       "market_label": tok, "selections": [], "unpriced": [],
                       "line": line, "period": period}
            pending_name = None
            i += 1
            continue
        # A token right after a price is usually the next selection name
        # (e.g. 'Under 1.5' after '3.75'), not a new market header. Only an
        # unknown header when the token itself is not followed by a price.
        if (current is not None and pending_name is None and tok
                and not _ODDS_RE.match(tok)
                and i > 0 and _ODDS_RE.match(region[i - 1] or "")
                and not (i + 1 < len(region) and _ODDS_RE.match(region[i + 1]))
                and not _looks_like_selection(tok)):
            flush()
            current = {"match_id": match_id, "market_key": _slug(tok),
                       "market_label": tok, "selections": [], "unpriced": [],
                       "line": None, "period": None}
            i += 1
            continue
        if current is None:
            # Stray header-like label with no pattern: open an 'other' block
            # only if it looks like a market title (else skip, e.g. Teams/Minutes).
            i += 1
            continue
        if _ODDS_RE.match(tok):
            if pending_name is not None:
                try:
                    odds = float(tok)
                except ValueError:
                    odds = None
                if odds is not None and odds >= 1.0:
                    current["selections"].append({"name": pending_name, "odds": odds})
                    _maybe_set_ou_line(current, pending_name, odds)
                else:
                    current["unpriced"].append(pending_name)
                pending_name = None
            i += 1
            continue
        # A name token: previous pending name (if any) had no price -> unpriced.
        if pending_name is not None:
            current["unpriced"].append(pending_name)
        pending_name = tok
        i += 1
    flush()

    # Drop non-market blocks (Teams/Minutes/Match sub-tabs row etc.)
    cleaned = [m for m in markets if m["selections"] or m["market_key"].startswith(("1x2", "ou_", "ah_", "handicap", "next_goal", "other_"))]
    for m in cleaned:
        m.pop("unpriced", None)
    # Post-process O/U keys: line known only after reading selections.
    for m in cleaned:
        if m["market_key"] in ("ou_pending",) or (m["market_label"] == "Over/Under" and m["market_key"].startswith("other_")):
            pass
    for m in cleaned:
        if m["market_label"] in ("Over/Under", "2nd Half - Over/Under", "1st Half - Over/Under"):
            if m.get("line") is not None:
                prefix = {"full": "ou", "2h": "ou_2h", "1h": "ou_1h"}[m.get("period") or "full"]
                line_txt = ("%g" % m["line"]).replace(".", "_")
                m["market_key"] = f"{prefix}_{line_txt}"
    return cleaned


def _maybe_set_ou_line(market: dict, selection: str, odds: float) -> None:
    if market["market_label"] in ("Over/Under", "2nd Half - Over/Under", "1st Half - Over/Under"):
        m = _OU_SELECTION_RE.match(selection)
        if m and market.get("line") is None:
            market["line"] = float(m.group(2))


def extract_clock_score(texts: list[str]) -> dict:
    """Best-effort clock/score/period read from the scoreboard region.

    Returns Nones for anything not found — callers abstain on None.
    """
    info: dict = {"period": None, "clock_label": None, "clock_minute": None,
                  "home_goals": None, "away_goals": None}
    if "2nd Half" in texts:
        info["period"] = "2h"
    elif "1st Half" in texts:
        info["period"] = "1h"
    for i, t in enumerate(texts):
        cm = _CLOCK_RE.match(t)
        if cm and info["clock_label"] is None:
            base_m, base_s = int(cm.group(1)), int(cm.group(2))
            if cm.group(3) is not None:
                info["clock_minute"] = base_m + int(cm.group(3)) + int(cm.group(4)) / 60.0
                info["clock_label"] = f"{base_m}:{base_s:02d} +{cm.group(3)}:{cm.group(4)}"
            else:
                info["clock_minute"] = base_m + base_s / 60.0
                info["clock_label"] = t
            break
    # Score: 'N : M' either as one token or split across three tokens
    # ('0', ':', '1') in the scoreboard area (before market tabs).
    try:
        tabs_at = next(i for i, t in enumerate(texts) if t == "All")
    except StopIteration:
        tabs_at = len(texts)
    board = texts[:tabs_at]
    for t in board:
        sm = _SCORE_RE.match(t)
        if sm:
            info["home_goals"] = int(sm.group(1))
            info["away_goals"] = int(sm.group(2))
            break
    else:
        for a, b, c in zip(board, board[1:], board[2:]):
            if b == ":" and a.strip().isdigit() and c.strip().isdigit():
                info["home_goals"] = int(a.strip())
                info["away_goals"] = int(c.strip())
                break
    return info
