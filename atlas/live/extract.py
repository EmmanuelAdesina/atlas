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
# Stoppage-only clock, e.g. '45+3' (base minute plus stoppage).
_CLOCK_PLUS_RE = re.compile(r"^(\d+)\+(\d+)$")
_SCORE_RE = re.compile(r"^(\d+)\s*:\s*(\d+)$")
_NGN_BALANCE_RE = re.compile(r"^NGN\s+([\d,]+(?:\.\d+)?)$", re.IGNORECASE)
_SUB_MINUTE_RE = re.compile(r"^\d+(?:\+\d+)?'$")

# Scoreboard period token -> canonical period. Order is the priority: a
# finished match wins over a live half, and HT wins over a half label.
_PERIOD_TOKENS: tuple[tuple[str, str], ...] = (
    ("Ended", "ft"),
    ("FT", "ft"),
    ("Full Time", "ft"),
    ("HT", "ht"),
    ("Half Time", "ht"),
    ("2nd Half", "2h"),
    ("1st Half", "1h"),
)

# The live match header renders a bare period label ('2nd', '1st', 'HT')
# immediately before the clock, without the word 'Half'. These are only read
# inside the header window around the clock, never from the whole page.
_PERIOD_ALIASES: dict[str, str] = {
    "1st": "1h",
    "1st half": "1h",
    "first half": "1h",
    "2nd": "2h",
    "2nd half": "2h",
    "second half": "2h",
    "ht": "ht",
    "half time": "ht",
    "halftime": "ht",
    "ft": "ft",
    "ended": "ft",
    "full time": "ft",
}
_PERIOD_WINDOW = 3

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
    (re.compile(r"^Rest of the Match"), ("rest_of_match_1x2", None, "full")),
    (re.compile(r"^Over/Under$"), (None, None, "full")),  # line from selections
    (re.compile(r"^Home Team Over/Under$"), ("home_ou_pending", None, "full")),
    (re.compile(r"^Away Team Over/Under$"), ("away_ou_pending", None, "full")),
    (re.compile(r"^(?:\d+(?:st|nd|rd|th)|Next) Goal$"), ("next_goal", None, "full")),
    (re.compile(r"^Handicap (\d+:\d+)$"), ("handicap", None, "full")),
    (re.compile(r"^Asian Handicap ([+-]?\d+(?:\.\d+)?)$"), ("asian_handicap", None, "full")),
    (re.compile(r"^2nd Half - Asian Handicap$"), ("ah_2h_pending", None, "2h")),
    (re.compile(r"^2nd Half - 1X2$"), ("1x2_2h", None, "2h")),
    (re.compile(r"^1st Half - 1X2$"), ("1x2_1h", None, "1h")),
    (re.compile(r"^2nd Half - Over/Under$"), (None, None, "2h")),
    (re.compile(r"^1st Half - Over/Under$"), (None, None, "1h")),
    # --- Path-based 'at any time' families (priced from the score path) ---
    (re.compile(r"^Any Team to lead by (\d+) Goals? at any time$", re.I),
     ("lead_by_{n}_any", None, "full")),
    (re.compile(r"^Home Team to lead by (\d+) Goals? at any time$", re.I),
     ("lead_by_{n}_home", None, "full")),
    (re.compile(r"^Away Team to lead by (\d+) Goals? at any time$", re.I),
     ("lead_by_{n}_away", None, "full")),
    (re.compile(r"^Any Team To Score (\d+) or More Goals in a Row$", re.I),
     ("goals_in_row_{n}_any", None, "full")),
    (re.compile(r"^Home Team To Score (\d+) or More Goals in a Row$", re.I),
     ("goals_in_row_{n}_home", None, "full")),
    (re.compile(r"^Away Team To Score (\d+) or More Goals in a Row$", re.I),
     ("goals_in_row_{n}_away", None, "full")),
    # --- Final-state families priced from the joint matrix ---
    (re.compile(r"^Odd/Even$"), ("odd_even", None, "full")),
    (re.compile(r"^Home Team Odd/Even$"), ("home_odd_even", None, "full")),
    (re.compile(r"^Away Team Odd/Even$"), ("away_odd_even", None, "full")),
    (re.compile(r"^Home Team Clean Sheet$"), ("home_clean_sheet", None, "full")),
    (re.compile(r"^Away Team Clean Sheet$"), ("away_clean_sheet", None, "full")),
    (re.compile(r"^Home Team to Win to Nil$"), ("home_win_to_nil", None, "full")),
    (re.compile(r"^Away Team to Win to Nil$"), ("away_win_to_nil", None, "full")),
    (re.compile(r"^Home Team Goals$"), ("home_exact_goals", None, "full")),
    (re.compile(r"^Away Team Goals$"), ("away_exact_goals", None, "full")),
    (re.compile(r"^Correct Score"), ("correct_score", None, "full")),
    (re.compile(r"^2nd Half - Odd/Even$"), ("odd_even_2h", None, "2h")),
    (re.compile(r"^2nd Half Home Team Total Goals Odd or Even$"),
     ("home_odd_even_2h", None, "2h")),
    (re.compile(r"^2nd Half Away Team Total Goals Odd or Even$"),
     ("away_odd_even_2h", None, "2h")),
    (re.compile(r"^(?:GG/NG|No Draw Both Teams To Score) 2\+$"), ("btts_2plus", None, "full")),
    (re.compile(r"^No Draw Both Teams To Score$"), ("btts_no_draw", None, "full")),
    (re.compile(r"^2nd Half - GG/NG 2\+$"), ("btts_2plus_2h", None, "2h")),
    (re.compile(r"^2nd Half - Home Team Clean Sheet$"), ("home_clean_sheet", None, "2h")),
    (re.compile(r"^2nd Half - Away Team Clean Sheet$"), ("away_clean_sheet", None, "2h")),
    (re.compile(r"^2nd Half Home Team to Win to Nil$"), ("home_win_to_nil", None, "2h")),
    (re.compile(r"^2nd Half Away Team to Win to Nil$"), ("away_win_to_nil", None, "2h")),
    (re.compile(r"^2nd Half - Correct Score$"), ("correct_score", None, "2h")),
    # --- Combination markets (joint outcomes, priced off the joint matrix) ---
    (re.compile(r"^1X2 & Over/Under (\d+(?:\.\d+)?)$"), ("combo_1x2__ou_{line}", None, "full")),
    (re.compile(r"^1X2 & GG/NG$"), ("combo_1x2__btts", None, "full")),
    (re.compile(r"^Over/Under & GG/NG$"), ("combo_ou_2_5__btts", None, "full")),
    (re.compile(r"^Double Chance & Over/Under (\d+(?:\.\d+)?)$"),
     ("combo_double_chance__ou_{line}", None, "full")),
    (re.compile(r"^Double Chance & GG/NG$"), ("combo_double_chance__btts", None, "full")),
    (re.compile(r"^2nd Half - 1X2 & Over/Under (\d+(?:\.\d+)?)$"),
     ("combo_1x2_2h__ou_{line}", None, "2h")),
    (re.compile(r"^2nd Half - 1X2 & GG/NG$"), ("combo_1x2_2h__btts", None, "2h")),
    (re.compile(r"^2nd Half - Double Chance & Total Goals$"),
     ("combo_2h_double_chance__ou_1_5", None, "2h")),
    (re.compile(r"^2nd Half Double Chance & Total Goals$"),
     ("combo_2h_double_chance__ou_1_5", None, "2h")),
    (re.compile(r"^2nd Half - Double Chance & GG/NG$"),
     ("combo_2h_double_chance__btts", None, "2h")),
    (re.compile(r"^2nd Half - Double Chance & Over/Under (\d+(?:\.\d+)?)$"),
     ("combo_2h_double_chance__ou_{line}", None, "2h")),
    # --- Rest-of-match variants (the current score is in the title) ---
    (re.compile(r"^Rest of Match Total Goals \(current score \d+:\d+\)$"),
     ("rest_of_match_total_goals", None, "full")),
    (re.compile(r"^Rest of Match Asian Handicap \(current score \d+:\d+\)$"),
     ("rest_of_match_asian_handicap", None, "full")),
    # --- Time-windowed markets (the state of the match at a given minute) ---
    (re.compile(r"^1X2 from 1 to (\d+) minute$"), ("1x2_to_{n}", None, "full")),
    (re.compile(r"^Total Goals Over/Under from 1 to (\d+) minute$"),
     ("ou_to_{n}", None, "full")),
    # --- No-draw variants ---
    (re.compile(r"^No Draw Both Teams To Score(?: Yes/No)?$"),
     ("btts_no_draw", None, "full")),
    # --- Corner markets (a separate counting process) ---
    # The exact titles on the book's Corners tab are not yet confirmed
    # against a live board; the mapping lives here so adding one is a
    # one-line change. The line is read from the selection text.
    (re.compile(r"^Corners Over/Under$"), ("corners_ou_pending", None, "full")),
    (re.compile(r"^Total Corners$"), ("corners_ou_pending", None, "full")),
    (re.compile(r"^Corners Handicap$"), ("corners_ah_pending", None, "full")),
    (re.compile(r"^Asian Handicap Corners$"), ("corners_ah_pending", None, "full")),
    (re.compile(r"^Handicap Corners$"), ("corners_ah_pending", None, "full")),
    (re.compile(r"^Corners$"), ("corners_ou_pending", None, "full")),
    # --- Excluded number of goals: the bet wins on any OTHER goal count ---
    (re.compile(r"^Excluded Number of Goals - Home$"),
     ("excluded_goals_home", None, "full")),
    (re.compile(r"^Excluded Number of Goals - Away$"),
     ("excluded_goals_away", None, "full")),
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
    """DFS over snapshot nodes, returning StaticText/link/button/div names in order.

    The SportyBet scoreboard renders its score in divs that wrap children, so
    'div' is included alongside the text roles or the score is never seen.
    """
    out: list[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("role") in {"StaticText", "link", "button", "div"} and n.get("name"):
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
            if label.startswith("Handicap") and m.re.groups:
                return (f"handicap_{m.group(1).replace(':', '_')}", None, "full")
            if label.startswith("Asian Handicap") and m.re.groups:
                return (f"ah_{m.group(1)}", float(m.group(1)), "full")
            if key and "{n}" in key:
                # 'Any Team to lead by 2 Goals at any time' -> lead_by_2_any
                key = key.format(n=m.group(1))
            if key and "{line}" in key:
                # '1X2 & Over/Under 1.5' -> combo_1x2__ou_1_5
                line_txt = m.group(1).replace(".", "_")
                key = key.format(line=line_txt)
            return (key, line, period)
    return None


_TEAM_OU_RE = re.compile(r"^(?P<team>.+?) Over/Under$")


def _team_over_under_key(label: str, home_team: str, away_team: str) -> Optional[str]:
    """Map a team-specific Over/Under title to home_ou / away_ou.

    The live board titles these markets with the team name ('Beitar
    Jerusalem FC Over/Under'), so the side can only be identified against
    the match's own team names. The second-half variants keep their scope.
    """
    scope = "full"
    title = label.strip()
    if title.lower().startswith("2nd half"):
        scope = "2h"
        title = title[len("2nd Half") :].lstrip(" -")
    m = _TEAM_OU_RE.match(title)
    if not m:
        return None
    team = m.group("team").strip()
    suffix = "_2h" if scope == "2h" else ""
    if home_team and team.lower() == home_team.strip().lower():
        return f"home_ou{suffix}_pending"
    if away_team and team.lower() == away_team.strip().lower():
        return f"away_ou{suffix}_pending"
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
    region = [t.strip() for t in texts[start:end] if t.strip() and not _GLYPH_RE.match(t)]

    # The viewed match's team names, needed to identify team-specific markets
    # such as 'Beitar Jerusalem FC Over/Under'. The match header sits ABOVE
    # the market-tabs row, so it has to be read from the whole snapshot, not
    # from the market region.
    home_team = ""
    away_team = ""
    for t in texts:
        if " vs " in t and len(t) < 80:
            parts = t.split(" vs ", 1)
            home_team, away_team = parts[0].strip(), parts[1].strip()
            break
    if not home_team:
        for t in region:
            if " vs " in t and len(t) < 80:
                parts = t.split(" vs ", 1)
                home_team, away_team = parts[0].strip(), parts[1].strip()
                break

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
        if hdr is None:
            team_ou = _team_over_under_key(tok, home_team, away_team)
            if team_ou is not None:
                hdr = (team_ou, None, "full")
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

    return _post_process_markets(markets)


def _post_process_markets(markets: list[dict]) -> list[dict]:
    """Shared post-processing for the snapshot and DOM market paths."""
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
                "corners_",
                "combo_",
                "excluded_",
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
        ) or label.endswith("Over/Under"):
            if m.get("line") is not None:
                if label == "Home Team Over/Under":
                    prefix = "home_ou"
                elif label == "Away Team Over/Under":
                    prefix = "away_ou"
                elif "2nd Half" in label:
                    prefix = "ou_2h"
                elif "1st Half" in label:
                    prefix = "ou_1h"
                else:
                    prefix = "ou"
                line_txt = ("%g" % m["line"]).replace(".", "_")
                m["market_key"] = f"{prefix}_{line_txt}"
    # Post-process corner keys: the line is in the selection text too.
    #
    # The exact titles the book uses on the Corners tab have not been
    # confirmed against a live board, so the mapping is kept in one obvious
    # place: adding a title is a one-line change here.
    for m in cleaned:
        key = m.get("market_key") or ""
        if key.startswith("corners_") and key.endswith("_pending"):
            line = m.get("line")
            if line is None:
                continue
            line_txt = ("%g" % line).replace(".", "_")
            m["market_key"] = key[: -len("_pending")] + f"_{line_txt}"
    return cleaned


def _maybe_set_ou_line(market: dict, selection: str, odds: Optional[float] = None) -> None:
    label = market["market_label"]
    key = market.get("market_key") or ""
    if key.startswith("corners_") and key.endswith("_pending"):
        # A corner market takes its line from the selection ('Over 9.5').
        m = _OU_SELECTION_RE.match(selection.strip())
        if m:
            market["line"] = float(m.group(2))
        return
    if (
        label in (
            "Over/Under",
            "2nd Half - Over/Under",
            "1st Half - Over/Under",
            "Home Team Over/Under",
            "Away Team Over/Under",
        )
        or label.endswith("Over/Under")
    ):
        m = _OU_SELECTION_RE.match(selection)
        if m and market.get("line") is None:
            market["line"] = float(m.group(2))


def _match_panel_start(texts: list[str]) -> int:
    """Index of the '<Home> vs <Away>' header of the viewed match.

    The live board renders every match card with its own clock and score, so
    extraction must anchor on the panel of the match actually being viewed.
    """
    for i, t in enumerate(texts):
        if " vs " in t and len(t) < 80:
            return i
    return 0


def _clock_span(texts: list[str], start: int = 0) -> Optional[tuple[int, int]]:
    """Index span of the match clock in the text list.

    SportyBet renders the clock either as one token ('75:11', '90:00 +03:42')
    or split across three tokens ('73', ':', '52'). Both must be recognised,
    because a split clock looks exactly like a score.
    """
    for i in range(start, len(texts)):
        if _CLOCK_RE.match(texts[i]) or _CLOCK_PLUS_RE.match(texts[i]):
            return (i, i + 1)
    for i in range(start, len(texts) - 2):
        a, b, c = texts[i], texts[i + 1], texts[i + 2]
        if b != ":" or not a.isdigit() or not c.isdigit():
            continue
        # A split clock has minutes 0..120 and seconds 0..59.
        if int(a) <= 120 and int(c) <= 59:
            return (i, i + 3)
    return None


def _score_candidates(texts: list[str]) -> list[tuple[int, int, int]]:
    """All (index, home, away) score readings, single-token and split."""
    cands: list[tuple[int, int, int]] = []
    for i, t in enumerate(texts):
        sm = _SCORE_RE.match(t)
        if sm:
            cands.append((i, int(sm.group(1)), int(sm.group(2))))
    for i in range(len(texts) - 2):
        a, b, c = texts[i], texts[i + 1], texts[i + 2]
        if b == ":" and a.isdigit() and c.isdigit():
            cands.append((i, int(a), int(c)))
    return cands


# A football score never reaches this; the clock (and stray numbers) do.
_MAX_PLAUSIBLE_GOALS = 25


def extract_score_from_dom(texts: list[str]) -> dict:
    """Extract the live score from the scoreboard region of the snapshot texts.

    SportyBet renders the score either as one token ('2:1', '2 : 1') or as three
    consecutive tokens ('2', ':', '1'). The HT/FT split scores use the same
    encoding, so the first match is the current score.

    The clock uses the same encoding, so the reading is anchored on the clock
    position: the score follows the clock in the match header. Without that
    anchor a split clock ('73', ':', '52') is read as a 73-2 score.
    """
    info: dict = {"home_goals": None, "away_goals": None}
    board = [t.strip() for t in texts if t.strip()]
    cands = _score_candidates(board)
    if not cands:
        return info

    panel = _match_panel_start(board)
    span = _clock_span(board, panel)
    if span is not None:
        clock_end = span[1]
        after = [c for c in cands if c[0] >= clock_end]
        # The score sits immediately after the clock in the header block.
        for c in after:
            if c[0] - clock_end <= 4:
                info["home_goals"], info["away_goals"] = c[1], c[2]
                return info
        # Otherwise take the first plausible reading just after the clock.
        for c in after:
            if c[0] - clock_end <= 12 and c[1] <= _MAX_PLAUSIBLE_GOALS and c[2] <= _MAX_PLAUSIBLE_GOALS:
                info["home_goals"], info["away_goals"] = c[1], c[2]
                return info

    # No clock anchor: take the first plausible reading, else the first reading.
    for c in cands:
        if c[1] <= _MAX_PLAUSIBLE_GOALS and c[2] <= _MAX_PLAUSIBLE_GOALS:
            info["home_goals"], info["away_goals"] = c[1], c[2]
            return info
    c = cands[0]
    info["home_goals"], info["away_goals"] = c[1], c[2]
    return info


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
    for token, period in _PERIOD_TOKENS:
        if token in texts:
            info["period"] = period
            break
    panel = _match_panel_start(texts)
    # A bare period label ('2nd', '1st') is only trusted next to the clock:
    # it sits immediately before it, separated by at most a '|' separator.
    if info["period"] is None:
        span = _clock_span(texts, panel)
        if span is not None:
            lo = max(0, span[0] - _PERIOD_WINDOW)
            for t in texts[lo:span[0]]:
                alias = _PERIOD_ALIASES.get(t.strip().lower())
                if alias is not None:
                    info["period"] = alias
                    break
    for i in range(panel, len(texts)):
        t = texts[i]
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
        pm = _CLOCK_PLUS_RE.match(t)
        if pm and info["clock_label"] is None:
            base = int(pm.group(1))
            stoppage = int(pm.group(2))
            info["clock_minute"] = base + stoppage
            info["clock_label"] = t
            break
    else:
        # A split clock ('73', ':', '52') is read as minutes:seconds.
        span = _clock_span(texts, panel)
        if span is not None and span[1] - span[0] == 3:
            i = span[0]
            mins, secs = int(texts[i]), int(texts[i + 2])
            info["clock_minute"] = mins + secs / 60.0
            info["clock_label"] = f"{mins}:{secs:02d}"
    score = extract_score_from_dom(texts)
    info["home_goals"] = score["home_goals"]
    info["away_goals"] = score["away_goals"]
    if info["home_goals"] is None or info["away_goals"] is None:
        # The book publishes the current score in its own market title
        # ('Rest of the Match (current score 1:0)'), which is a reliable
        # fallback when the scoreboard block itself is not rendered.
        for t in texts:
            m = re.search(r"current score (\d+)\s*:\s*(\d+)", t)
            if m:
                info["home_goals"] = int(m.group(1))
                info["away_goals"] = int(m.group(2))
                break
    return info


def extract_scoreboard_js() -> str:
    """Return JS source that reads the live scoreboard from the DOM.

    The scoreboard is NOT exposed in the accessibility tree, and it has two
    layouts:

      - the inline panel: the clock and period in one text node
        ('.m-result-title', e.g. "66' H2") and the score in sibling
        '.m-value-1' elements;
      - a dedicated match page: the clock ('88:50'), the period ('2nd') and
        the score ('1 : 1') as plain text after the '<Home> vs <Away>'
        header.

    Both are read here, the class-based one first.
    """
    return r"""
(() => {
  const res = {
    home_goals: null, away_goals: null,
    clock_label: null, clock_minute: null, period: null
  };

  // --- layout 1: the inline panel ---------------------------------------
  const detail = document.querySelector('.m-eventDetail')
    || document.querySelector('.m-tracker') || document.body;
  const titleEl = detail.querySelector('.m-result-title');
  const title = titleEl ? (titleEl.innerText || '').trim() : '';
  if (title) {
    const cm = title.match(/(\d{1,3})(?::(\d{2}))?\s*['\u2019\u2032]?/);
    if (cm) {
      const mins = parseInt(cm[1], 10);
      const secs = cm[2] ? parseInt(cm[2], 10) : 0;
      res.clock_minute = mins + secs / 60.0;
      res.clock_label = title;
    }
    const pm = title.match(/\b(H1|H2|HT|1ST|2ND|FULL\s*TIME|FT)\b/i);
    if (pm) {
      const t = pm[1].toUpperCase();
      if (t === 'H1' || t === '1ST') res.period = '1h';
      else if (t === 'H2' || t === '2ND') res.period = '2h';
      else if (t === 'HT') res.period = 'ht';
      else res.period = 'ft';
    }
    const vals = Array.from(detail.querySelectorAll('.m-value-1'))
      .map(e => (e.innerText || '').trim())
      .filter(v => /^\d{1,2}$/.test(v));
    if (vals.length >= 1) res.home_goals = parseInt(vals[0], 10);
    if (vals.length >= 2) res.away_goals = parseInt(vals[1], 10);
  }

  // --- layout 2: a dedicated match page ---------------------------------
  if (res.home_goals === null || res.clock_minute === null) {
    let header = null;
    document.querySelectorAll('*').forEach(e => {
      if (!header && e.children.length === 0 && / vs /.test(e.textContent || '')
          && (e.textContent || '').length < 80) header = e;
    });
    if (header) {
      let block = header.parentElement;
      for (let i = 0; i < 3 && block; i++) {
        const txt = (block.innerText || '').replace(/\u00a0/g, ' ');
        if (txt && /\d/.test(txt)) {
          const clock = txt.match(/(\d{1,3}):(\d{2})/);
          if (clock && res.clock_minute === null) {
            res.clock_minute = parseInt(clock[1], 10) + parseInt(clock[2], 10) / 60.0;
            res.clock_label = clock[0];
          }
          const pm = txt.match(/\b(1st|2nd|HT|FT|Full Time)\b/i);
          if (pm && !res.period) {
            const t = pm[1].toLowerCase();
            if (t === '1st') res.period = '1h';
            else if (t === '2nd') res.period = '2h';
            else if (t === 'ht') res.period = 'ht';
            else res.period = 'ft';
          }
          // The score is spaced ('1 : 1'); the clock is not ('88:50'). The
          // clock matches the loose pattern first, so every candidate is
          // checked and the first plausible SCORE is taken.
          if (res.home_goals === null) {
            const re = /(\d{1,2})\s*:\s*(\d{1,2})/g;
            let m;
            while ((m = re.exec(txt)) !== null) {
              const h = parseInt(m[1], 10);
              const a = parseInt(m[2], 10);
              if (h <= 25 && a <= 25) {
                res.home_goals = h;
                res.away_goals = a;
                break;
              }
            }
          }
          if (res.home_goals !== null && res.clock_minute !== null) break;
        }
        block = block.parentElement;
      }
    }
  }
  return JSON.stringify(res);
})()
"""


def extract_score_clock_js() -> str:
    """Return JS source for WebBridge evaluate to extract score/clock from DOM.

    Works across all match formats by searching broadly for score and clock
    patterns in the Match Tracker section, not relying on specific CSS classes.
    """
    return """(()=>{
        const out={home_goals:null,away_goals:null,clock_label:null,clock_minute:null,period:null};
        const all=document.querySelectorAll('*');
        const scoreRe=/^(\\d{1,2})\\s*:\\s*(\\d{1,2})$/;
        const clockRe=/^(\\d{1,3}):(\\d{2})(?:\\s*\\+(\\d+):(\\d{2}))?$/;
        const plusRe=/^(\\d{1,3})\\+(\\d{1,2})$/;
        for(const el of all){
            if(el.children.length>0)continue;
            const t=(el.innerText||'').trim();
            if(!t)continue;
            const sm=t.match(scoreRe);
            if(sm&&out.home_goals===null){out.home_goals=parseInt(sm[1]);out.away_goals=parseInt(sm[2]);}
            const cm=t.match(clockRe);
            if(cm&&out.clock_label===null){
                const base=parseInt(cm[1]),sec=parseInt(cm[2]);
                if(cm[3]){out.clock_minute=base+parseInt(cm[3])+parseInt(cm[4])/60;out.clock_label=cm[1]+':'+cm[2]+' +'+cm[3]+':'+cm[4];}
                else{out.clock_minute=base+sec/60;out.clock_label=t;}
            }
            const pm=t.match(plusRe);
            if(pm&&out.clock_label===null){out.clock_minute=parseInt(pm[1])+parseInt(pm[2]);out.clock_label=t;}
            if(t==='Ended'||t==='FT'||t==='Full Time')out.period='ft';
            else if(t==='HT'||t==='Half Time')out.period='ht';
            else if(t==='2nd Half')out.period='2h';
            else if(t==='1st Half')out.period='1h';
        }
        return JSON.stringify(out);
    })()"""


def markets_from_js_entries(
    entries: list[dict],
    match_id: str = "live",
    home_team: str = "",
    away_team: str = "",
) -> list[dict]:
    """Turn the DOM-read market entries into the pipeline's market dicts.

    The DOM path and the snapshot path produce the same shape here, so the
    key resolution (header patterns, team-specific markets, the line for
    O/U families) is shared.
    """
    markets: list[dict] = []
    for entry in entries or []:
        title = (entry.get("market") or "").strip()
        if not title:
            continue
        sels = []
        for s in entry.get("selections", []) or []:
            name = (s.get("name") or "").strip()
            odds = s.get("odds")
            if not name or odds is None:
                continue
            try:
                odds = float(odds)
            except (TypeError, ValueError):
                continue
            if odds <= 1.0:
                continue
            sels.append({"name": name, "odds": odds})
        hdr = _match_header(title)
        if hdr is not None:
            key, line, period = hdr
        else:
            key = _team_over_under_key(title, home_team, away_team)
            line, period = None, "full"
        if not key:
            key = _slug(title)
        market = {
            "match_id": match_id,
            "market_key": key,
            "market_label": title,
            "selections": sels,
            "status": "open" if len(sels) >= 2 else "suspended",
            "line": line,
            "period": period,
        }
        for s in sels:
            _maybe_set_ou_line(market, s["name"], s["odds"])
        markets.append(market)
    return _post_process_markets(markets)


def extract_markets_js() -> str:
    """Return JS source that reads the live markets (headers + odds) from the DOM.

    This is the PRIMARY extraction path. The accessibility snapshot
    intermittently fails on the heavy live board (the daemon returns 502),
    so the markets are read from the DOM and the snapshot is only a
    fallback.

    Structure: each market is a '.m-table__wrapper' holding a title
    ('.m-table-header-title') and one or more '.m-table-row.m-outcome'
    rows. Each selection cell holds two '.m-table-cell-item' elements: the
    selection name and its price.
    """
    return r"""
(() => {
  const out = [];
  const wrappers = document.querySelectorAll('.m-table__wrapper');
  const ODDS = /^[0-9]{1,3}\.[0-9]{2}$/;
  wrappers.forEach(w => {
    const titleEl = w.querySelector('.m-table-header-title');
    const title = titleEl ? (titleEl.innerText || '').trim() : '';
    const sels = [];
    w.querySelectorAll('.m-table-row.m-outcome').forEach(row => {
      row.querySelectorAll('.m-table-cell').forEach(cell => {
        const items = Array.from(cell.querySelectorAll('.m-table-cell-item'))
          .map(i => (i.innerText || '').trim());
        const prices = items.filter(t => ODDS.test(t));
        if (!prices.length) return;
        const name = items.find(t => t && !ODDS.test(t)) || '';
        sels.push({name: name, odds: parseFloat(prices[prices.length - 1])});
      });
    });
    if (title) out.push({market: title, selections: sels});
  });
  return JSON.stringify(out);
})()
"""


def extract_stats_js() -> str:
    """Return JS source that reads the live statistics panel.

    The panel is opened by clicking the 'Statistics' control; its markup
    pairs each stat with a caption ('CORNERS') and the two side values,
    and the values are NOT in the same subtree as the caption, so each
    caption is walked up until a block with a readable home/away value is
    found.

    Returns a JSON string:
    {possession_pct, shots_on_target, corners, yellow_cards, red_cards,
     fouls, offsides, saves, attempts, conversion_rate}, each {home, away};
    a missing stat is simply absent.
    """
    return r"""
(() => {
  const out = {};
  const num = (root, side) => {
    for (const e of root.querySelectorAll('[class*=' + side + '-1]')) {
      const t = (e.innerText || '').trim();
      if (/^[0-9]+(\.[0-9]+)?$/.test(t)) return parseFloat(t);
    }
    return null;
  };
  const CAPTION = '[class*=__caption]';
  document.querySelectorAll(CAPTION).forEach(cap => {
    const name = (cap.innerText || '').trim().toLowerCase();
    let block = cap.parentElement;
    for (let i = 0; i < 4 && block; i++) {
      const h = num(block, 'home');
      const a = num(block, 'away');
      if (h !== null || a !== null) {
        out[name] = {home: h, away: a};
        return;
      }
      block = block.parentElement;
    }
  });
  // Possession and the card counts use their own value classes.
  const poss = Array.from(document.querySelectorAll('.sr-mx-ballpossession__value'))
    .map(e => (e.innerText || '').trim());
  if (poss.length >= 2) {
    const h = parseFloat(poss[0]);
    const a = parseFloat(poss[1]);
    if (!isNaN(h) && !isNaN(a)) out.possession_pct = {home: h, away: a};
  }
  const cards = Array.from(document.querySelectorAll('.sr-mx-penaltycards__value'))
    .map(e => (e.innerText || '').trim());
  if (cards.length >= 4) {
    const n = cards.map(v => parseFloat(v));
    if (n.every(v => !isNaN(v))) {
      out.yellow_cards = {home: n[0], away: n[2]};
      out.red_cards = {home: n[1], away: n[3]};
    }
  }
  return JSON.stringify(out);
})()
"""



def panel_is_ready(
    state: LiveMatchState,
    markets: list[dict],
    min_markets: int = 1,
) -> tuple[bool, str]:
    """Is the match panel fully rendered enough to price?

    The live board renders progressively: a snapshot taken too early carries
    the match header but not yet the score, the clock or the market table.
    Acting on such a snapshot means pricing against a missing score, which
    the model handles by abstaining — but it also means the loop can make a
    decision from half a board.

    The gate requires, in order of importance:
      1. a known live period (the model abstains without one),
      2. a known score (both sides),
      3. a known clock,
      4. a minimum number of open markets (`min_markets`). A live board
         renders dozens; a caller that knows the expected board size can
         raise this to catch a panel that is still filling in.

    Returns (ready, reason). The reason says which condition failed.
    """
    if state.period not in ("1h", "2h", "ht"):
        return (False, f"period not live (got {state.period!r})")
    if state.home.goals is None or state.away.goals is None:
        return (False, "score unknown")
    if state.clock_minute is None:
        return (False, "clock unknown")
    open_markets = [m for m in markets if m.get("status") == "open"]
    if len(open_markets) < min_markets:
        return (
            False,
            f"only {len(open_markets)} open markets (need {min_markets})",
        )
    return (True, "ready")


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

    # Account balance (e.g. 'NGN 166.11'). Scan the whole pre-tab board, not
    # just the first 15 nodes: the header layout changes between builds and
    # the anchored regex cannot pick up promo copy, so a wider window only
    # adds chances to find the real balance.
    balance_ngn: Optional[float] = None
    for t in board:
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
