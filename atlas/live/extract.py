"""Extraction contract: WebBridge snapshot tree -> snapshot payload & LiveMatchState.

Input is the parsed `tree` from a WebBridge `snapshot` call (a list of
accessibility nodes with role/name/children). Output is the raw market list
the SportyBet collector ingests, plus match-state and live statistics fields.

Lock semantics (observed on the live page): a selection row shows either a
decimal price (open) or no price at all (locked/suspended, padlock glyph).
Selections without a price are NOT given invented odds — the market is
emitted with status 'suspended' and empty selections, and the mapper records
it in `skipped` with reason 'suspended' so availability stays auditable.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from atlas.live.observe import LiveMatchState, TeamStats

_ODDS_RE = re.compile(r"^\d+\.\d{2}$")
_GLYPH_RE = re.compile(r"^[\ue000-\uf8ff]$")
_CLOCK_RE = re.compile(r"^(\d+):(\d{2})(?:\s*\+(\d+):(\d{2}))?$")
_SCORE_RE = re.compile(r"^(\d+)\s*:\s*(\d+)$")
_NGN_BALANCE_RE = re.compile(r"^NGN\s+([\d,]+(?:\.\d+)?)$", re.IGNORECASE)
_SUB_MINUTE_RE = re.compile(r"^\d+(?:\+\d+)?'$")

_TABS_ROW = {"All", "Main", "Goals", "Half", "Bookings", "Corners", "Specials", "Players"}
_END_MARKERS = {"Betslip", "Cashout"}

# Header text -> (market_key, line, period). Unknown headers fall back to a
# slug key so observed markets are never silently dropped.
_HEADER_PATTERNS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"^1X2 - 1UP$"), ("1x2_1up", None, "full")),
    (re.compile(r"^1X2 - 2UP$"), ("1x2_2up", None, "full")),
    (re.compile(r"^1X2 - Never Down$"), ("1x2_never_down", None, "full")),
    (re.compile(r"^1X2$"), ("1x2", None, "full")),
    (re.compile(r"^Double Chance$"), ("double_chance", None, "full")),
    (re.compile(r"^1st Half - Double Chance$"), ("1h_double_chance", None, "1h")),
    (re.compile(r"^2nd Half - Double Chance$"), ("2h_double_chance", None, "2h")),
    (re.compile(r"^Draw No Bet$"), ("dnb", None, "full")),
    (re.compile(r"^1st Half - Draw No Bet$"), ("1h_dnb", None, "1h")),
    (re.compile(r"^2nd Half - Draw No Bet$"), ("2h_dnb", None, "2h")),
    (re.compile(r"^(?:GG/NG|Both Teams To Score)$", re.IGNORECASE), ("btts", None, "full")),
    (re.compile(r"^1st Half - (?:GG/NG|Both Teams To Score)$", re.IGNORECASE), ("1h_btts", None, "1h")),
    (re.compile(r"^2nd Half - (?:GG/NG|Both Teams To Score)$", re.IGNORECASE), ("2h_btts", None, "2h")),
    (re.compile(r"^Rest of the Match$"), ("rest_of_match_1x2", None, "full")),
    (re.compile(r"^Over/Under$"), (None, None, "full")),  # line from selections
    (re.compile(r"^Home Team Over/Under$"), ("home_ou_pending", None, "full")),
    (re.compile(r"^Away Team Over/Under$"), ("away_ou_pending", None, "full")),
    (re.compile(r"^(?:\d+(?:st|nd|rd|th)|Next) Goal$"), ("next_goal", None, "full")),
    (re.compile(r"^Handicap (\d+:\d+)$"), ("handicap", None, "full")),
    (re.compile(r"^Asian Handicap ([+-]?\d+(?:\.\d+)?)$"), ("asian_handicap", None, "full")),
    (re.compile(r"^2nd Half - 1X2$"), ("1x2_2h", None, "2h")),
    (re.compile(r"^1st Half - 1X2$"), ("1x2_1h", None, "1h")),
    (re.compile(r"^2nd Half - Over/Under$"), (None, None, "2h")),
    (re.compile(r"^1st Half - Over/Under$"), (None, None, "1h")),
]

_OU_SELECTION_RE = re.compile(r"^(Over|Under)\s+(\d+(?:\.\d+)?)$")

_SELECTION_STARTERS = (
    "Home",
    "Draw",
    "Away",
    "Over ",
    "Under ",
    "Yes",
    "No",
    "None",
    "1X",
    "12",
    "X2",
)


def _looks_like_selection(tok: str) -> bool:
    t = tok.strip()
    if t in {"Home", "Draw", "Away", "Yes", "No", "None", "1X", "12", "X2", "yes", "no"}:
        return True
    return t.startswith(_SELECTION_STARTERS) or "(" in t


def iter_texts(tree: Any) -> list[str]:
    """DFS over snapshot nodes, returning StaticText/link/button names in order."""
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
    start = -1
    for i in range(len(texts) - 3):
        if texts[i : i + 4] == ["All", "Main", "Goals", "Half"]:
            start = i
            break
    if start == -1:
        return []
    end = next(
        (i for i in range(start, len(texts)) if texts[i] in _END_MARKERS),
        len(texts),
    )
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
            _maybe_set_ou_line(current, pending_name)
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
            current = {
                "match_id": match_id,
                "market_key": key or _slug(tok),
                "market_label": tok,
                "selections": [],
                "unpriced": [],
                "line": line,
                "period": period,
            }
            pending_name = None
            i += 1
            continue
        # A token right after a price is usually the next selection name
        # (e.g. 'Under 1.5' after '3.75'), not a new market header. Only an
        # unknown header when the token itself is not followed by a price.
        if (
            current is not None
            and pending_name is None
            and tok
            and not _ODDS_RE.match(tok)
            and i > 0
            and _ODDS_RE.match(region[i - 1] or "")
            and not (i + 1 < len(region) and _ODDS_RE.match(region[i + 1]))
            and not _looks_like_selection(tok)
        ):
            flush()
            current = {
                "match_id": match_id,
                "market_key": _slug(tok),
                "market_label": tok,
                "selections": [],
                "unpriced": [],
                "line": None,
                "period": None,
            }
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if _ODDS_RE.match(tok):
            if pending_name is not None:
                try:
                    odds = float(tok)
                except ValueError:
                    odds = None
                _maybe_set_ou_line(current, pending_name)
                if odds is not None and odds >= 1.0:
                    current["selections"].append({"name": pending_name, "odds": odds})
                else:
                    current["unpriced"].append(pending_name)
                pending_name = None
            i += 1
            continue
        # A name token: previous pending name (if any) had no price -> unpriced.
        if pending_name is not None:
            _maybe_set_ou_line(current, pending_name)
            current["unpriced"].append(pending_name)
        pending_name = tok
        _maybe_set_ou_line(current, pending_name)
        i += 1
    flush()

    # Drop non-market blocks (Teams/Minutes/Match sub-tabs row etc.)
    cleaned = [
        m
        for m in markets
        if m["selections"]
        or m["market_key"].startswith(
            (
                "1x2",
                "ou_",
                "home_ou_",
                "away_ou_",
                "ah_",
                "handicap",
                "next_goal",
                "double_chance",
                "1h_",
                "2h_",
                "dnb",
                "btts",
                "rest_of_match",
                "other_",
            )
        )
    ]
    for m in cleaned:
        m.pop("unpriced", None)
    # Post-process O/U keys: line known after reading selection names (even if locked).
    for m in cleaned:
        label = m["market_label"]
        if label in (
            "Over/Under",
            "2nd Half - Over/Under",
            "1st Half - Over/Under",
            "Home Team Over/Under",
            "Away Team Over/Under",
        ):
            if m.get("line") is not None:
                if label == "Home Team Over/Under":
                    prefix = "home_ou"
                elif label == "Away Team Over/Under":
                    prefix = "away_ou"
                else:
                    prefix = {"full": "ou", "2h": "ou_2h", "1h": "ou_1h"}[
                        m.get("period") or "full"
                    ]
                line_txt = ("%g" % m["line"]).replace(".", "_")
                m["market_key"] = f"{prefix}_{line_txt}"
    return cleaned


def _maybe_set_ou_line(market: dict, selection: str, odds: Optional[float] = None) -> None:
    if market["market_label"] in (
        "Over/Under",
        "2nd Half - Over/Under",
        "1st Half - Over/Under",
        "Home Team Over/Under",
        "Away Team Over/Under",
    ):
        m = _OU_SELECTION_RE.match(selection)
        if m and market.get("line") is None:
            market["line"] = float(m.group(2))


def extract_clock_score(texts: list[str]) -> dict:
    """Best-effort clock/score/period read from the scoreboard region.

    Returns Nones for anything not found — callers abstain on None.
    """
    info: dict = {
        "period": None,
        "clock_label": None,
        "clock_minute": None,
        "home_goals": None,
        "away_goals": None,
    }
    if "2nd Half" in texts:
        info["period"] = "2h"
    elif "1st Half" in texts:
        info["period"] = "1h"
    elif "Half Time" in texts or "HT" in texts:
        info["period"] = "ht"
    elif "Ended" in texts or "Full Time" in texts:
        info["period"] = "ft"
    for t in texts:
        cm = _CLOCK_RE.match(t)
        if cm and info["clock_label"] is None:
            base_m, base_s = int(cm.group(1)), int(cm.group(2))
            if cm.group(3) is not None:
                info["clock_minute"] = (
                    base_m + int(cm.group(3)) + int(cm.group(4)) / 60.0
                )
                info["clock_label"] = (
                    f"{base_m}:{base_s:02d} +{cm.group(3)}:{cm.group(4)}"
                )
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


def extract_live_match_state(
    texts: list[str],
    match_id: str,
    observed_at: Optional[datetime] = None,
    source_url: str = "",
    source: str = "sportybet_browser",
) -> LiveMatchState:
    """Extract full LiveMatchState (including team names, HT score, cards, corners, shots, possession, subs) from snapshot texts."""
    observed_at = observed_at or datetime.now(timezone.utc)
    basic = extract_clock_score(texts)

    try:
        tabs_at = next(i for i, t in enumerate(texts) if t == "All")
    except StopIteration:
        tabs_at = len(texts)
    board = [t.strip() for t in texts[:tabs_at] if t.strip()]

    # Account balance (e.g. 'NGN 166.11')
    balance_ngn: Optional[float] = None
    for t in board[:15]:
        bm = _NGN_BALANCE_RE.match(t)
        if bm:
            try:
                balance_ngn = float(bm.group(1).replace(",", ""))
            except ValueError:
                pass
            break

    # Team names from '<Home> vs <Away>'
    home_name = "Home"
    away_name = "Away"
    vs_idx = -1
    for i, t in enumerate(board):
        if " vs " in t and len(t) < 80:
            parts = t.split(" vs ", 1)
            home_name, away_name = parts[0].strip(), parts[1].strip()
            vs_idx = i
            break

    # Competition label: '<League> | <Matchday>' after 'Statistics'
    competition: Optional[str] = None
    if vs_idx != -1:
        for i in range(vs_idx + 1, min(vs_idx + 12, len(board) - 2)):
            if board[i + 1] == "|" and board[i] not in ("1st Half", "2nd Half"):
                competition = f"{board[i]} | {board[i + 2]}"
                break

    # Full-time and Halftime split scores ('0', ':', '1' then '0', ':', '0')
    score_pairs: list[tuple[int, int]] = []
    search_start = vs_idx if vs_idx != -1 else 0
    for i in range(search_start, len(board) - 2):
        if board[i + 1] == ":" and board[i].isdigit() and board[i + 2].isdigit():
            score_pairs.append((int(board[i]), int(board[i + 2])))
    half_home_goals: Optional[int] = None
    half_away_goals: Optional[int] = None
    if len(score_pairs) >= 2:
        half_home_goals, half_away_goals = score_pairs[1]

    # Helper for post-label pair: ['LABEL', home_int, away_int]
    def pair_after(label: str) -> tuple[Optional[int], Optional[int]]:
        for i in range(search_start, len(board) - 2):
            if board[i] == label and board[i + 1].isdigit() and board[i + 2].isdigit():
                return int(board[i + 1]), int(board[i + 2])
        return None, None

    # Helper for pre-label pair: [home_int, away_int, 'LABEL']
    def pair_before(label: str) -> tuple[Optional[int], Optional[int]]:
        for i in range(search_start + 2, len(board)):
            if board[i] == label and board[i - 2].isdigit() and board[i - 1].isdigit():
                return int(board[i - 2]), int(board[i - 1])
        return None, None

    # Possession: ['BALL POSSESSION', '44', '%', '56', '%']
    home_poss: Optional[float] = None
    away_poss: Optional[float] = None
    for i in range(search_start, len(board) - 4):
        if board[i] == "BALL POSSESSION" and board[i + 1].isdigit() and board[i + 3].isdigit():
            home_poss = float(board[i + 1])
            away_poss = float(board[i + 3])
            break

    # Cards: [home_red, home_yellow, 'CARDS', away_yellow, away_red]
    home_yellow: Optional[int] = None
    home_red: Optional[int] = None
    away_yellow: Optional[int] = None
    away_red: Optional[int] = None
    for i in range(search_start + 2, len(board) - 2):
        if (
            board[i] == "CARDS"
            and board[i - 2].isdigit()
            and board[i - 1].isdigit()
            and board[i + 1].isdigit()
            and board[i + 2].isdigit()
        ):
            home_red = int(board[i - 2])
            home_yellow = int(board[i - 1])
            away_yellow = int(board[i + 1])
            away_red = int(board[i + 2])
            break

    h_corners, a_corners = pair_after("CORNERS")
    h_sot, a_sot = pair_after("ON TARGET")
    h_saves, a_saves = pair_after("SAVES")
    h_attempts, a_attempts = pair_after("ATTEMPTS")
    h_fouls, a_fouls = pair_before("FOULS")
    h_fk, a_fk = pair_before("FREE KICKS")
    h_offsides, a_offsides = pair_before("OFFSIDES")

    # Substitutions count in the scoreboard
    subs_count: Optional[int] = None
    if "Substitutions" in board:
        sub_idx = board.index("Substitutions")
        subs_count = sum(1 for t in board[sub_idx + 1 :] if _SUB_MINUTE_RE.match(t))

    clock_label = basic["clock_label"] or ""
    in_stoppage_unknown = "+" in clock_label

    home_stats = TeamStats(
        name=home_name,
        goals=basic["home_goals"],
        possession_pct=home_poss,
        shots_on_target=h_sot,
        shots_total=h_attempts,
        saves=h_saves,
        corners=h_corners,
        yellow_cards=home_yellow,
        red_cards=home_red,
        fouls=h_fouls,
        free_kicks=h_fk,
        offsides=h_offsides,
    )
    away_stats = TeamStats(
        name=away_name,
        goals=basic["away_goals"],
        possession_pct=away_poss,
        shots_on_target=a_sot,
        shots_total=a_attempts,
        saves=a_saves,
        corners=a_corners,
        yellow_cards=away_yellow,
        red_cards=away_red,
        fouls=a_fouls,
        free_kicks=a_fk,
        offsides=a_offsides,
    )

    return LiveMatchState(
        match_id=match_id,
        competition=competition,
        home=home_stats,
        away=away_stats,
        period=basic["period"] or "unknown",
        clock_minute=basic["clock_minute"],
        clock_label=basic["clock_label"],
        half_home_goals=half_home_goals,
        half_away_goals=half_away_goals,
        substitutions_count=subs_count,
        remaining_unknown=in_stoppage_unknown,
        account_balance_ngn=balance_ngn,
        observed_at=observed_at,
        source=source,
        source_url=source_url,
    )
