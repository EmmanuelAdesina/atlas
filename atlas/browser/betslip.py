"""Betslip driver: the DOM contract for placing a SportyBet live bet.

Everything here was established by driving the real live board with the Kimi
WebBridge on 2026-10-19, not by guessing. The selectors are the contract;
if SportyBet restyles, these break and the failure is loud (no slip found),
never silent.

Observed contract
-----------------
Odds cells      .m-table-row.m-outcome .m-table-cell-item   (text = decimal odds)
Stake input     input.m-input, placeholder 'min. 10'  -> min stake NGN 10,
                default NGN 100
Place button    button.af-button--primary labelled 'Place Bet'
                (also a .m-btn-wrapper wrapper with the same label)
Book Bet        writes a booking code for the current slip
Betslip features
  'F Flexi'      accumulator with partial payout (Flex Your Bet)
  'One Cut'      partial cashout of a placed bet
  'SportyInsure' stake insurance on a losing bet

Operational constraints found on the live board
-----------------------------------------------
- The live page NEVER fires 'load' (it holds a websocket open), so
  `navigate()` to the live board always times out and then unregisters the
  tab, which makes every later daemon call 502. Navigate client-side
  (`location.href = ...`) and hard-reload to repopulate the board.
- The accessibility `snapshot` action 502s on the live board (45+ matches).
  Extraction must use targeted `evaluate` JS, never a snapshot.
- A click on an odds cell adds the selection to the betslip; the slip shows
  'Total Stake', 'Potential Win', 'Place Bet' and 'Book Bet'.

Safety
------
`place_bet()` is the only method that moves money. It refuses to run unless
`allow_submit=True` was passed to the constructor, so a stray call in a
research script cannot spend the account balance. Nothing in ATLAS enables
it today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

MIN_STAKE_NGN = 10.0
DEFAULT_STAKE_NGN = 100.0

ODDS_ROW_SELECTOR = ".m-table-row.m-outcome"
ODDS_CELL_SELECTOR = ".m-table-cell-item"
STAKE_INPUT_SELECTOR = "input.m-input"
PLACE_BET_SELECTOR = "button.af-button--primary"


class SubmissionNotAllowed(RuntimeError):
    """Raised when money-moving is attempted without an explicit opt-in."""


class BetslipError(RuntimeError):
    """Raised when the betslip could not be found or driven."""


@dataclass
class SlipSelection:
    market: str
    selection: str
    odds: float
    stake_ngn: float = 0.0
    potential_win: float = 0.0


@dataclass
class SlipState:
    selections: list[SlipSelection] = field(default_factory=list)
    total_stake_ngn: float = 0.0
    potential_win_ngn: float = 0.0
    can_place_bet: bool = False
    raw_text: str = ""


def validate_stake(stake_ngn: float, balance_ngn: Optional[float] = None) -> str:
    """Return '' if the stake is acceptable, else the reason it is not.

    Mirrors what SportyBet enforces: a NGN 10 floor, and a stake the account
    balance can actually cover.
    """
    if stake_ngn < MIN_STAKE_NGN:
        return f"stake {stake_ngn:.2f} below minimum {MIN_STAKE_NGN:.2f} NGN"
    if balance_ngn is not None and stake_ngn > balance_ngn:
        return (
            f"stake {stake_ngn:.2f} exceeds available balance {balance_ngn:.2f} NGN"
        )
    return ""


# --- JS payloads (kept as constants so they are reviewable in one place) ---

JS_ADD_ODDS = """
(() => {
  const rows = Array.from(document.querySelectorAll('%(row)s'));
  for (const row of rows) {
    const cells = Array.from(row.querySelectorAll('%(cell)s'));
    for (const c of cells) {
      if ((c.textContent || '').trim() === '%(odds)s') {
        c.click();
        return JSON.stringify({ok: true, odds: (c.textContent||'').trim()});
      }
    }
  }
  return JSON.stringify({ok: false, error: 'odds cell not found'});
})()
"""

JS_SET_STAKE = """
(() => {
  const el = document.querySelector('%(input)s');
  if (!el) return JSON.stringify({ok: false, error: 'stake input not found'});
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, '%(stake)s');
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  el.dispatchEvent(new Event('blur', { bubbles: true }));
  return JSON.stringify({ok: true, value: el.value});
})()
"""

JS_READ_SLIP = """
(() => {
  const t = document.body.innerText;
  const lines = t.split('\\n').map(s => s.trim()).filter(Boolean);
  const i = lines.findIndex(l => /Total Stake/i.test(l));
  const seg = i >= 0 ? lines.slice(Math.max(0, i - 14), i + 10) : [];
  const stakeEl = document.querySelector('%(input)s');
  const place = Array.from(document.querySelectorAll('button, [role=button]'))
    .filter(b => /place bet/i.test((b.innerText||'') + ' ' + (b.getAttribute('aria-label')||'')));
  return JSON.stringify({
    seg,
    stake: stakeEl ? stakeEl.value : null,
    canPlace: place.some(b => !b.disabled),
    hasPlaceButton: place.length > 0
  });
})()
"""

JS_CLEAR = """
(() => {
  const panels = Array.from(document.querySelectorAll('*')).filter(e =>
    /Total Stake/i.test(e.innerText || '') && (e.innerText || '').length < 900);
  let removed = 0;
  for (const p of panels.slice(0, 3)) {
    p.querySelectorAll('button, [class*=remove], [class*=delete], [class*=close], [class*=clear], svg, i').forEach(b => {
      const lbl = ((b.getAttribute('aria-label')||'') + ' ' + (b.getAttribute('title')||'') + ' ' +
                   (b.className||'').toString() + ' ' + (b.innerText||'')).toLowerCase();
      if (/remove|delete|close|clear|times|\\u2715|\\u00d7|m-icon-/.test(lbl)) {
        try { b.click(); removed++; } catch (e) {}
      }
    });
    if (removed) break;
  }
  return JSON.stringify({removed: removed});
})()
"""

JS_PLACE_BET = """
(() => {
  const btn = Array.from(document.querySelectorAll('%(place)s, button, [role=button]'))
    .find(b => /place bet/i.test((b.innerText||'') + ' ' + (b.getAttribute('aria-label')||'')));
  if (!btn) return JSON.stringify({ok: false, error: 'place bet button not found'});
  if (btn.disabled) return JSON.stringify({ok: false, error: 'place bet button is disabled'});
  btn.click();
  return JSON.stringify({ok: true});
})()
"""


class BetslipDriver:
    """Drives the SportyBet betslip through a WebBridge adapter.

    `adapter` must expose `evaluate(code) -> str` (the WebBridgeAdapter does).
    """

    def __init__(self, adapter: Any, allow_submit: bool = False) -> None:
        self.adapter = adapter
        self.allow_submit = bool(allow_submit)

    # --- low level -----------------------------------------------------

    def _js(self, template: str, **fmt: Any) -> dict:
        import json

        code = template % fmt
        raw = self.adapter.evaluate(code)
        # WebBridge returns {"type": "string", "value": "<json>"};
        # a fake or direct adapter may return the JSON string itself.
        if isinstance(raw, dict):
            if "value" in raw:
                raw = raw["value"]
            else:
                return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise BetslipError(f"betslip returned non-JSON: {raw[:200]}")
        return raw

    # --- operations ----------------------------------------------------

    def add_odds(self, odds: float) -> dict:
        """Click the odds cell with this decimal price to add it to the slip."""
        return self._js(
            JS_ADD_ODDS, row=ODDS_ROW_SELECTOR, cell=ODDS_CELL_SELECTOR,
            odds=f"{float(odds):.2f}",
        )

    def set_stake(self, stake_ngn: float) -> dict:
        reason = validate_stake(stake_ngn)
        if reason:
            raise BetslipError(reason)
        return self._js(
            JS_SET_STAKE, input=STAKE_INPUT_SELECTOR, stake=f"{float(stake_ngn):.2f}"
        )

    def read_slip(self) -> SlipState:
        data = self._js(JS_READ_SLIP, input=STAKE_INPUT_SELECTOR)
        stake_raw = data.get("stake")
        stake = 0.0
        if stake_raw:
            try:
                stake = float(str(stake_raw).replace(",", ""))
            except ValueError:
                stake = 0.0
        return SlipState(
            selections=[],
            total_stake_ngn=stake,
            potential_win_ngn=0.0,
            can_place_bet=bool(data.get("canPlace")),
            raw_text=" | ".join(data.get("seg", [])),
        )

    def clear(self) -> dict:
        """Remove every selection from the slip. Safe: never submits."""
        return self._js(JS_CLEAR)

    def place_bet(self) -> dict:
        """Submit the slip. Requires an explicit opt-in at construction.

        This is the only money-moving call in ATLAS. It stays disabled until
        somebody deliberately constructs the driver with allow_submit=True.
        """
        if not self.allow_submit:
            raise SubmissionNotAllowed(
                "BetslipDriver(allow_submit=True) is required to place a bet"
            )
        slip = self.read_slip()
        if not slip.can_place_bet:
            raise BetslipError("betslip is not placeable (no enabled Place Bet button)")
        return self._js(JS_PLACE_BET, place=PLACE_BET_SELECTOR)
