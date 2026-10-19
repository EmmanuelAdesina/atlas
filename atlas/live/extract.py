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
    (re.compile(r"^2nd Half - Odd/Even$"), ("odd_even", None, "2h")),
    (re.compile(r"^2nd Half - Home Team Clean Sheet$"), ("home_clean_sheet", None, "2h")),
    (re.compile(r"^2nd Half - Away Team Clean Sheet$"), ("away_clean_sheet", None, "2h")),
    (re.compile(r"^2nd Half Home Team to Win to Nil$"), ("home_win_to_nil", None, "2h")),
    (re.compile(r"^2nd Half Away Team to Win to Nil$"), ("away_win_to_nil", None, "2h")),
    (re.compile(r"^2nd Half - Correct Score$"), ("correct_score", None, "2h")),
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
            if label.startswith("Handicap"):
                return (f"handicap_{m.group(1).replace(':', '_')}", None, "full")
            if label.startswith("Asian Handicap"):
                return (f"ah_{m.group(1)}", float(m.group(1)), "full")
            if key and "{n}" in key:
                # 'Any Team to lead by 2 Goals at any time' -> lead_by_2_any
                key = key.format(n=m.group(1))
            return (key, line, period)
    return None


_TEAM_OU_RE = re.compile(r"^(?P<team>.+?) Over/Under$")


def _team_over_under_key(label: str, home_team: str, away_team: str) -> Optional[str]:
    """Map a team-specific Over/Under title to home_ou / away_ou.

    The live board titles these markets with the team name ('Beitar
    Jerusalem FC Over/Under'), so the side can only be identified against
    the match's own team names.
    """
    m = _TEAM_OU_RE.match(label.strip())
    if not m:
        return None
    team = m.group("team").strip()
    if home_team and team.lower() == home_team.strip().lower():
        return "home_ou_pending"
    if away_team and team.lower() == away_team.strip().lower():
        return "away_ou_pending"
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
    # such as 'Beitar Jerusalem FC Over/Under'.
    home_team = ""
    away_team = ""
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
    return info


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


def extract_stats_js() -> str:
    """Return JS source for WebBridge evaluate to extract stats from DOM.

    Searches the Statistics section for stat labels and their home/away values.
    Works across match formats by looking for label text and nearby numbers.
    """
    return """(()=>{
        const out={};
        const labels=['BALL POSSESSION','CARDS','CORNERS','ON TARGET','SAVES','ATTEMPTS','FOULS','OFFSIDES','CONVERSION RATE'];
        const all=document.querySelectorAll('*');
        for(const el of all){
            if(el.children.length>0)continue;
            const t=(el.innerText||'').trim().toUpperCase();
            if(!t||t.length>25)continue;
            if(labels.includes(t)){
                const parent=el.parentElement;
                if(!parent)continue;
                const nums=[];
                const siblings=parent.querySelectorAll('*');
                for(const s of siblings){
                    if(s.children.length>0)continue;
                    const v=(s.innerText||'').trim();
                    const m=v.match(/^(\\d+(?:\\.\\d+)?)$/);
                    if(m)nums.push(parseFloat(m[1]));
                    if(nums.length>=6)break;
                }
                if(nums.length>=2){
                    if(t==='BALL POSSESSION')out.possession_pct={home:nums[0],away:nums[1]};
                    else if(t==='CORNERS')out.corners={home:nums[0],away:nums[1]};
                    else if(t==='ON TARGET')out.shots_on_target={home:nums[0],away:nums[1]};
                    else if(t==='SAVES')out.saves={home:nums[0],away:nums[1]};
                    else if(t==='ATTEMPTS')out.attempts={home:nums[0],away:nums[1]};
                    else if(t==='FOULS')out.fouls={home:nums[0],away:nums[1]};
                    else if(t==='OFFSIDES')out.offsides={home:nums[0],away:nums[1]};
                    else if(t==='CONVERSION RATE')out.conversion_rate={home:nums[0],away:nums[1]};
                    else if(t==='CARDS'&&nums.length>=4)out.yellow_cards={home:nums[0],away:nums[2]},out.red_cards={home:nums[1],away:nums[3]};
                }
            }
        }
        return JSON.stringify(out);
    })()"""


_STAT_KEYS = (
    "possession_pct",
    "shots_on_target",
    "corners",
    "yellow_cards",
    "red_cards",
    "fouls",
    "offsides",
    "saves",
    "attempts",
    "conversion_rate",
)

# Stats-panel label -> result key. Values may sit after the label
# ('CORNERS', '5', '1'), before it ('12', '8', 'ATTEMPTS'), or split
# around it (cards). Possession/conversion skip a '%' token per value.
_STAT_LABEL_KEYS = {
    "BALL POSSESSION": "possession_pct",
    "CONVERSION RATE": "conversion_rate",
    "CORNERS": "corners",
    "ON TARGET": "shots_on_target",
    "SAVES": "saves",
    "ATTEMPTS": "attempts",
    "FOULS": "fouls",
    "OFFSIDES": "offsides",
}
_STAT_LABELS_ALL = frozenset(_STAT_LABEL_KEYS) | {"CARDS"}


def _numeric_token_values(tokens: list[str], contiguous: bool = False) -> list[float]:
    """Collect numeric tokens from `tokens`, skipping '%'/'yellow'/'red' noise.

    Stops at the first known stat label (so values never bleed across rows)
    or at an unrecognised word once a number has been collected. With
    `contiguous`, the run must be unbroken numbers (used scanning backwards,
    where the previous row's values sit further away than this row's).
    """
    vals: list[float] = []
    for t in tokens:
        up = t.upper()
        if up in _STAT_LABELS_ALL:
            break
        if up in {"YELLOW", "RED"} or t in {"%"}:
            if contiguous:
                break
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)$", t)
        if m:
            vals.append(float(m.group(1)))
            if len(vals) >= 4:
                break
        elif vals or contiguous:
            break
    return vals


def _pair_after_label(
    board: list[str], label: str, start: int
) -> tuple[Optional[float], Optional[float]]:
    """'LABEL', home, away — values follow the label."""
    for i in range(start, len(board)):
        if board[i] == label:
            vals = _numeric_token_values(board[i + 1 :])
            if len(vals) >= 2:
                return vals[0], vals[1]
    return None, None


def _pair_before_label(
    board: list[str], label: str, start: int
) -> tuple[Optional[float], Optional[float]]:
    """home, away, 'LABEL' — values precede the label.

    Prefers values immediately after the label (label-first rows where an
    older row's numbers sit further back), then falls back to a contiguous
    backward scan.
    """
    for i in range(start, len(board)):
        if board[i] == label:
            after = _numeric_token_values(board[i + 1 : i + 3])
            if len(after) >= 2:
                return after[0], after[1]
            vals = _numeric_token_values(
                board[max(start, i - 8) : i][::-1], contiguous=True
            )
            if len(vals) >= 2:
                return vals[1], vals[0]
    return None, None


def _cards_fallback(board: list[str], start: int) -> dict:
    """MLS cards layout: numbers ordered home_yellow home_red away_yellow away_red.

    Handles the label-first row ('CARDS', '0', '0', '3', '0'), the split row
    ('0', '0', 'CARDS', '3', '0') and the all-before row.
    """
    empty = {"yellow_cards": {"home": None, "away": None},
             "red_cards": {"home": None, "away": None}}
    for i in range(start, len(board)):
        if board[i] != "CARDS":
            continue
        after = _numeric_token_values(board[i + 1 :])
        before = _numeric_token_values(
            board[max(start, i - 8) : i][::-1], contiguous=True
        )
        if len(after) >= 4:
            return {
                "yellow_cards": {"home": after[0], "away": after[2]},
                "red_cards": {"home": after[1], "away": after[3]},
            }
        if len(before) >= 2 and len(after) >= 2:
            return {
                "yellow_cards": {"home": before[1], "away": after[0]},
                "red_cards": {"home": before[0], "away": after[1]},
            }
        if len(before) >= 4:
            return {
                "yellow_cards": {"home": before[3], "away": before[1]},
                "red_cards": {"home": before[2], "away": before[0]},
            }
    return empty


def _extract_stats_primary(board: list[str], start: int) -> dict:
    """Layout-specific pass over the statistics panel texts.

    SportyBet rows mix two layouts: some labels precede their values
    ('CORNERS', '3', '1') and some follow them ('18', '20', 'FOULS'). Cards
    split around the label ([home_red, home_yellow, 'CARDS', away_yellow,
    away_red]). Values alternate home/away.
    """
    stats: dict = {key: {"home": None, "away": None} for key in _STAT_KEYS}
    claimed: set[int] = set()

    def pair_after(label: str) -> tuple[Optional[int], Optional[int]]:
        for i in range(start, len(board) - 2):
            if board[i] == label and board[i + 1].isdigit() and board[i + 2].isdigit():
                claimed.update((i + 1, i + 2))
                return int(board[i + 1]), int(board[i + 2])
        return None, None

    def pair_before(label: str) -> tuple[Optional[int], Optional[int]]:
        for i in range(start + 2, len(board)):
            if board[i] == label and board[i - 2].isdigit() and board[i - 1].isdigit():
                if (i - 2) in claimed or (i - 1) in claimed:
                    continue
                return int(board[i - 2]), int(board[i - 1])
        return None, None

    for i in range(start, len(board) - 4):
        if board[i] == "BALL POSSESSION" and board[i + 1].isdigit() and board[i + 3].isdigit():
            stats["possession_pct"] = {
                "home": float(board[i + 1]),
                "away": float(board[i + 3]),
            }
            claimed.update((i + 1, i + 3))
            break

    for i in range(start + 2, len(board) - 2):
        if (
            board[i] == "CARDS"
            and board[i - 2].isdigit()
            and board[i - 1].isdigit()
            and board[i + 1].isdigit()
            and board[i + 2].isdigit()
            and (i - 2) not in claimed
            and (i - 1) not in claimed
        ):
            stats["yellow_cards"] = {
                "home": int(board[i - 1]),
                "away": int(board[i + 1]),
            }
            stats["red_cards"] = {
                "home": int(board[i - 2]),
                "away": int(board[i + 2]),
            }
            claimed.update((i - 2, i - 1, i + 1, i + 2))
            break

    h_corners, a_corners = pair_after("CORNERS")
    stats["corners"] = {"home": h_corners, "away": a_corners}
    h_sot, a_sot = pair_after("ON TARGET")
    stats["shots_on_target"] = {"home": h_sot, "away": a_sot}
    h_saves, a_saves = pair_after("SAVES")
    stats["saves"] = {"home": h_saves, "away": a_saves}
    h_attempts, a_attempts = pair_after("ATTEMPTS")
    stats["attempts"] = {"home": h_attempts, "away": a_attempts}
    h_conv, a_conv = pair_after("CONVERSION RATE")
    stats["conversion_rate"] = {"home": h_conv, "away": a_conv}
    h_fouls, a_fouls = pair_before("FOULS")
    stats["fouls"] = {"home": h_fouls, "away": a_fouls}
    h_offsides, a_offsides = pair_before("OFFSIDES")
    stats["offsides"] = {"home": h_offsides, "away": a_offsides}
    return stats


def extract_stats_from_texts(texts: list[str]) -> dict:
    """Parse the live statistics panel from snapshot texts, any match format.

    Runs the layout-specific pass first, then retries any field it missed
    with a label-agnostic scan that handles 'label home away',
    'home away label', and MLS rows whose values carry '%' or
    'yellow'/'red' tokens. MLS cards arrive as four numbers ordered
    home_yellow home_red away_yellow away_red.
    """
    board = [t.strip() for t in texts if t.strip()]
    try:
        start = board.index("Statistics")
    except ValueError:
        start = 0

    stats = _extract_stats_primary(board, start)

    def fill(key: str, finder) -> None:
        pair = stats.get(key) or {"home": None, "away": None}
        if pair["home"] is None or pair["away"] is None:
            h, a = finder()
            if h is not None and a is not None:
                stats[key] = {"home": h, "away": a}

    fill("possession_pct", lambda: _pair_after_label(board, "BALL POSSESSION", start))
    fill("shots_on_target", lambda: _pair_after_label(board, "ON TARGET", start))
    fill("corners", lambda: _pair_after_label(board, "CORNERS", start))
    fill("saves", lambda: _pair_after_label(board, "SAVES", start))
    fill("attempts", lambda: _pair_after_label(board, "ATTEMPTS", start))
    fill("conversion_rate", lambda: _pair_after_label(board, "CONVERSION RATE", start))

    def fouls() -> tuple[Optional[float], Optional[float]]:
        h, a = _pair_before_label(board, "FOULS", start)
        if h is None or a is None:
            return _pair_after_label(board, "FOULS", start)
        return h, a

    def offsides() -> tuple[Optional[float], Optional[float]]:
        h, a = _pair_before_label(board, "OFFSIDES", start)
        if h is None or a is None:
            return _pair_after_label(board, "OFFSIDES", start)
        return h, a

    fill("fouls", fouls)
    fill("offsides", offsides)

    cards = _cards_fallback(board, start)
    if (
        stats["yellow_cards"]["home"] is None
        and stats["yellow_cards"]["away"] is None
        and cards["yellow_cards"]["home"] is not None
    ):
        stats["yellow_cards"] = cards["yellow_cards"]
        stats["red_cards"] = cards["red_cards"]
    return stats


def extract_stats_js() -> str:
    """Return a JS source string that extracts live match stats from the DOM.

    Evaluate via the WebBridge `evaluate` action. Finds the Statistics
    section, searches for the stat labels (BALL POSSESSION, CARDS, CORNERS,
    ON TARGET, SAVES, ATTEMPTS, FOULS, OFFSIDES, CONVERSION RATE) and reads
    the numeric values rendered next to each, so it works for layouts whose
    snapshot text order differs from SportyBet's. Cards numbers are ordered
    home_yellow home_red away_yellow away_red. Returns a JSON string:
    {possession_pct, shots_on_target, corners, yellow_cards, red_cards,
    fouls, offsides} (plus saves/attempts/conversion_rate when present),
    each {home, away}; a missing stat is simply absent from the JSON.
    """
    return r"""
(() => {
  const LABELS = ["BALL POSSESSION", "CARDS", "CONVERSION RATE", "CORNERS",
                  "ON TARGET", "SAVES", "ATTEMPTS", "FOULS", "OFFSIDES"];
  const out = {};
  const pair = (home, away) => ({home: home, away: away});
  const ownText = (el) => Array.from(el.childNodes)
    .filter((c) => c.nodeType === 3)
    .map((c) => c.textContent.trim())
    .join(" ")
    .trim();
  const firstNum = (s) => {
    const m = String(s).match(/\d+(?:\.\d+)?/);
    return m ? parseFloat(m[0]) : null;
  };
  const numsAfter = (el) => {
    const res = [];
    let sib = el.nextElementSibling;
    let hops = 0;
    while (sib && hops < 8 && res.length < 4) {
      const v = firstNum(sib.textContent);
      if (v !== null) res.push(v);
      sib = sib.nextElementSibling;
      hops += 1;
    }
    return res;
  };
  const numsBefore = (el) => {
    const res = [];
    let sib = el.previousElementSibling;
    let hops = 0;
    while (sib && hops < 8 && res.length < 4) {
      const v = firstNum(sib.textContent);
      if (v !== null) res.unshift(v);
      sib = sib.previousElementSibling;
      hops += 1;
    }
    return res;
  };

  // Find the Statistics section: the smallest element whose own text starts
  // with the word, so label lookups stay inside the panel.
  let statsRoot = null;
  for (const el of document.querySelectorAll("body *")) {
    if (/^statistics(\s|$)/i.test(ownText(el))) { statsRoot = el; break; }
  }
  const scope = statsRoot || document.body;

  for (const el of scope.querySelectorAll("*")) {
    const label = ownText(el).toUpperCase();
    if (!LABELS.includes(label)) continue;
    if (label === "BALL POSSESSION" && out.possession_pct === undefined) {
      const v = numsAfter(el);
      if (v.length >= 2) out.possession_pct = pair(v[0], v[1]);
    } else if (label === "CARDS" && out.yellow_cards === undefined) {
      const after = numsAfter(el);
      const before = numsBefore(el);
      if (after.length >= 4) {
        out.yellow_cards = pair(after[0], after[2]);
        out.red_cards = pair(after[1], after[3]);
      } else if (before.length >= 2 && after.length >= 2) {
        out.yellow_cards = pair(before[0], after[0]);
        out.red_cards = pair(before[1], after[1]);
      } else if (before.length >= 4) {
        out.yellow_cards = pair(before[0], before[2]);
        out.red_cards = pair(before[1], before[3]);
      }
    } else if (label === "CORNERS" && out.corners === undefined) {
      const v = numsAfter(el);
      if (v.length >= 2) out.corners = pair(v[0], v[1]);
    } else if (label === "ON TARGET" && out.shots_on_target === undefined) {
      const v = numsAfter(el);
      if (v.length >= 2) out.shots_on_target = pair(v[0], v[1]);
    } else if (label === "SAVES" && out.saves === undefined) {
      const v = numsAfter(el);
      if (v.length >= 2) out.saves = pair(v[0], v[1]);
    } else if (label === "ATTEMPTS" && out.attempts === undefined) {
      const v = numsAfter(el);
      if (v.length >= 2) out.attempts = pair(v[0], v[1]);
    } else if (label === "FOULS" && out.fouls === undefined) {
      const b = numsBefore(el);
      if (b.length >= 2) out.fouls = pair(b[0], b[1]);
      else {
        const v = numsAfter(el);
        if (v.length >= 2) out.fouls = pair(v[0], v[1]);
      }
    } else if (label === "OFFSIDES" && out.offsides === undefined) {
      const b = numsBefore(el);
      if (b.length >= 2) out.offsides = pair(b[0], b[1]);
      else {
        const v = numsAfter(el);
        if (v.length >= 2) out.offsides = pair(v[0], v[1]);
      }
    } else if (label === "CONVERSION RATE" && out.conversion_rate === undefined) {
      const v = numsAfter(el);
      if (v.length >= 2) out.conversion_rate = pair(v[0], v[1]);
    }
  }
  return JSON.stringify(out);
})()
"""


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
