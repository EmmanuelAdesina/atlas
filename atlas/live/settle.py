"""Settlement of a placed bet from the final score.

Once a match ends, the book settles from the final score. ATLAS must do the
same, or the ledger's P&L — and therefore the hard daily stop — drifts from
reality.

The mapping is per market family, because 'the home team won' is not the
same test for 1X2 as it is for 'Home to lead by 2 at any time' (which is
settled early, when the lead is first reached) or for 'Over 2.5' (which is a
goal count). The outcomes are:

    'won'  - the selection paid out
    'lost' - the selection did not
    'void' - the stake was returned (a push)

Early-settlement markets are settled by the path, not the end state: a 1UP
bet wins the moment the side leads by one, and the final score alone cannot
say whether that happened. For those, the caller supplies the observed
maximum lead (or the lead reached at any point); without it, the
settlement is reported as UNKNOWN rather than guessed, because guessing
wrong corrupts the P&L that drives the daily stop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

WON = "won"
LOST = "lost"
VOID = "void"
UNKNOWN = "unknown"


@dataclass
class Settlement:
    status: str
    reason: str
    payout_ngn: float = 0.0
    settled_odds: Optional[float] = None


def _diff(final_h: int, final_a: int) -> int:
    return final_h - final_a


def _total(final_h: int, final_a: int) -> int:
    return final_h + final_a


def settle_from_final_score(
    market_key: str,
    selection: str,
    final_home: int,
    final_away: int,
    stake_ngn: float,
    odds: float,
    observed_max_lead: Optional[dict[str, int]] = None,
) -> Settlement:
    """Settle one selection from the final score.

    `observed_max_lead` maps 'home'/'away' to the largest lead that side
    reached at any point. It is only needed for the early-settlement
    families; when it is missing for those, the result is UNKNOWN.
    """
    key = (market_key or "").strip().lower()
    sel = (selection or "").strip()
    sel_l = sel.lower()
    diff = _diff(final_home, final_away)
    total = _total(final_home, final_away)

    # --- 1X2 family ----------------------------------------------------
    if key in ("1x2", "1x2_1h", "1x2_2h", "rest_of_match_1x2", "1x2_never_down"):
        if sel_l in ("home", "1"):
            return _binary(diff > 0, "home win", stake_ngn, odds)
        if sel_l in ("draw", "x"):
            return _binary(diff == 0, "draw", stake_ngn, odds)
        if sel_l in ("away", "2"):
            return _binary(diff < 0, "away win", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised 1X2 selection {selection!r}")

    # --- 1UP / 2UP: early settlement, needs the observed lead -----------
    if key in ("1x2_1up", "1x2_2up"):
        need = 1 if key == "1x2_1up" else 2
        if sel_l in ("home", "1"):
            side = "home"
        elif sel_l in ("away", "2"):
            side = "away"
        elif sel_l in ("draw", "x"):
            # The draw leg of a 1UP market settles on the full-time draw.
            return _binary(diff == 0, "draw", stake_ngn, odds)
        else:
            return Settlement(UNKNOWN, f"unrecognised 1UP selection {selection!r}")
        if observed_max_lead is None or side not in observed_max_lead:
            return Settlement(
                UNKNOWN,
                f"{key} settles on the path (did {side} ever lead by {need}?); "
                f"no observed lead supplied",
            )
        lead = observed_max_lead[side]
        return _binary(
            lead >= need,
            f"{side} reached a lead of {lead} (needed {need})",
            stake_ngn,
            odds,
        )

    # --- Double Chance -------------------------------------------------
    if key in ("double_chance", "1h_double_chance", "2h_double_chance"):
        if sel_l in ("home or draw", "1x", "home/draw"):
            return _binary(diff >= 0, "home or draw", stake_ngn, odds)
        if sel_l in ("draw or away", "x2", "draw/away"):
            return _binary(diff <= 0, "draw or away", stake_ngn, odds)
        if sel_l in ("home or away", "12", "home/away"):
            return _binary(diff != 0, "home or away", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised double chance {selection!r}")

    # --- Draw No Bet (a draw is a push) --------------------------------
    if key in ("dnb", "1h_dnb", "2h_dnb"):
        if sel_l in ("home", "1"):
            if diff > 0:
                return _binary(True, "home win", stake_ngn, odds)
            if diff == 0:
                return Settlement(
                    VOID, "draw pushes", payout_ngn=stake_ngn, settled_odds=odds
                )
            return _binary(False, "away win", stake_ngn, odds)
        if sel_l in ("away", "2"):
            if diff < 0:
                return _binary(True, "away win", stake_ngn, odds)
            if diff == 0:
                return Settlement(
                    VOID, "draw pushes", payout_ngn=stake_ngn, settled_odds=odds
                )
            return _binary(False, "home win", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised DNB selection {selection!r}")

    # --- Over/Under ----------------------------------------------------
    m = re.match(r"^(ou|ou_1h|ou_2h|home_ou|away_ou)_(\d+(?:_\d+)?)$", key)
    if m:
        prefix = m.group(1)
        line = float(m.group(2).replace("_", "."))
        is_over = sel_l.startswith("over")
        is_under = sel_l.startswith("under")
        if not (is_over or is_under):
            return Settlement(UNKNOWN, f"unrecognised O/U selection {selection!r}")
        if prefix == "home_ou":
            count = final_home
        elif prefix == "away_ou":
            count = final_away
        else:
            count = total
        frac = round(line - int(line), 2)
        if frac in (0.25, 0.75):
            # Two lines: a half-win / half-loss is a partial settlement the
            # ledger models as a push at the lower line, so the safest
            # classification is UNKNOWN rather than a wrong win/loss.
            return Settlement(
                UNKNOWN,
                f"quarter line {line} needs split settlement",
            )
        if count > line:
            return _binary(is_over, f"count {count} vs line {line}", stake_ngn, odds)
        if count < line:
            return _binary(is_under, f"count {count} vs line {line}", stake_ngn, odds)
        return Settlement(
                VOID,
                f"count {count} equals line {line}",
                payout_ngn=stake_ngn,
                settled_odds=odds,
            )

    # --- Both teams to score -------------------------------------------
    if key in ("btts", "1h_btts", "2h_btts", "gg_ng"):
        scored_both = final_home > 0 and final_away > 0
        if sel_l in ("yes", "gg"):
            return _binary(scored_both, "both scored", stake_ngn, odds)
        if sel_l in ("no", "ng"):
            return _binary(not scored_both, "not both scored", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised BTTS selection {selection!r}")

    # --- Odd/Even ------------------------------------------------------
    if key in ("odd_even", "home_odd_even", "away_odd_even"):
        if key == "odd_even":
            count = total
        elif key == "home_odd_even":
            count = final_home
        else:
            count = final_away
        is_odd = count % 2 == 1
        if sel_l.startswith("odd"):
            return _binary(is_odd, f"count {count}", stake_ngn, odds)
        if sel_l.startswith("even"):
            return _binary(not is_odd, f"count {count}", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised parity selection {selection!r}")

    # --- Clean sheet ----------------------------------------------------
    if key in ("home_clean_sheet", "away_clean_sheet"):
        clean = final_away == 0 if key.startswith("home") else final_home == 0
        if sel_l in ("yes", "1"):
            return _binary(clean, "clean sheet", stake_ngn, odds)
        if sel_l in ("no", "2"):
            return _binary(not clean, "not a clean sheet", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised clean sheet {selection!r}")

    # --- Win to nil ------------------------------------------------------
    if key in ("home_win_to_nil", "away_win_to_nil"):
        if key.startswith("home"):
            won = final_home > final_away and final_away == 0
        else:
            won = final_away > final_home and final_home == 0
        if sel_l in ("yes", "1"):
            return _binary(won, "win to nil", stake_ngn, odds)
        if sel_l in ("no", "2"):
            return _binary(not won, "not win to nil", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised win to nil {selection!r}")

    # --- Exact goals (0/1/2/3+) ----------------------------------------
    if key in ("home_exact_goals", "away_exact_goals"):
        count = final_home if key.startswith("home") else final_away
        m2 = re.match(r"^(\d+)\+$", sel)
        if m2:
            return _binary(count >= int(m2.group(1)), f"count {count}", stake_ngn, odds)
        if sel.isdigit():
            return _binary(count == int(sel), f"count {count}", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised exact goals {selection!r}")

    # --- Correct score ---------------------------------------------------
    if key == "correct_score":
        m2 = re.match(r"^(\d+)\s*[:\-]\s*(\d+)$", sel)
        if not m2:
            return Settlement(UNKNOWN, f"unrecognised correct score {selection!r}")
        hit = final_home == int(m2.group(1)) and final_away == int(m2.group(2))
        return _binary(hit, "exact score", stake_ngn, odds)

    # --- Asian handicap --------------------------------------------------
    m3 = re.match(r"^(?:ah|asian_handicap)_([+-]?\d+(?:[._]\d+)?)$", key)
    if m3:
        line = float(m3.group(1).replace("_", "."))
        is_home = sel_l.startswith("home")
        is_away = sel_l.startswith("away")
        if not (is_home or is_away):
            return Settlement(UNKNOWN, f"unrecognised AH selection {selection!r}")
        m2 = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", sel)
        team_line = float(m2.group(1)) if m2 else (line if is_home else -line)
        adj = (final_home - final_away if is_home else final_away - final_home) + team_line
        if abs(adj) < 1e-9:
            return Settlement(
                VOID, "on the line", payout_ngn=stake_ngn, settled_odds=odds
            )
        frac = round(abs(team_line) - int(abs(team_line)), 2)
        if frac in (0.25, 0.75):
            return Settlement(UNKNOWN, "quarter Asian line needs split settlement")
        return _binary(adj > 0, f"adj {adj:+.2f}", stake_ngn, odds)

    # --- 3-way European handicap ----------------------------------------
    m4 = re.match(r"^handicap_(\d+)_(\d+)$", key)
    if m4:
        h = int(m4.group(1))
        a = int(m4.group(2))
        adj = _diff(final_home + h, final_away + a)
        if sel_l.startswith("home"):
            return _binary(adj > 0, "home handicap", stake_ngn, odds)
        if sel_l.startswith("draw"):
            return _binary(adj == 0, "draw handicap", stake_ngn, odds)
        if sel_l.startswith("away"):
            return _binary(adj < 0, "away handicap", stake_ngn, odds)
        return Settlement(UNKNOWN, f"unrecognised handicap {selection!r}")

    # --- Path families need the observed path ---------------------------
    if key.startswith("lead_by_") or key.startswith("goals_in_row_"):
        return Settlement(
            UNKNOWN,
            f"{key} settles on the path; the observed path was not supplied",
        )

    return Settlement(UNKNOWN, f"no settlement rule for market {market_key!r}")


def _binary(
    won: bool, reason: str, stake_ngn: float, odds: float
) -> Settlement:
    if won:
        return Settlement(
            WON,
            reason,
            payout_ngn=round(stake_ngn * (odds - 1.0), 2),
            settled_odds=odds,
        )
    return Settlement(LOST, reason, payout_ngn=0.0, settled_odds=odds)
