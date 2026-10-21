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

import json
import re
import time
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
    """One selection on the slip, with the odds the book ACCEPTED."""

    name: str
    market: str = ""
    odds: float = 0.0


@dataclass
class SlipState:
    selections: list[SlipSelection] = field(default_factory=list)
    total_stake_ngn: float = 0.0
    potential_win_ngn: float = 0.0
    can_place_bet: bool = False
    # The book shows 'Accept Changes' when the price moved and the slip
    # needs the new price acknowledged before it can be placed.
    needs_accept: bool = False
    raw_text: str = ""

    @property
    def accepted_odds(self) -> Optional[float]:
        """The odds the book accepted for the single selection on the slip.

        With more than one selection there is no single accepted price, so
        the caller must check the selection list instead.
        """
        if len(self.selections) == 1:
            return self.selections[0].odds
        return None


_ODDS_TOKEN_RE = re.compile(r"\b([0-9]{1,3}\.[0-9]{2})\b")


def parse_slip_text(text: str) -> tuple[list[SlipSelection], float, float]:
    """Parse the betslip's text into (selections, total_stake, potential_win).

    The slip renders as: selection name, event, market title, odds, then
    'Total Stake', the currency, then 'Potential Win', the payout. The
    currency sits between the label and the amount, so the label and its
    amount are NOT adjacent.

    The odds that matter are the last odds token BEFORE 'Total Stake': the
    ones after it belong to the payout, not to the price. (The board's own
    prices are also in the text, so this has to be the LAST one before the
    stake label.)
    """
    joined = " | ".join(
        ln.strip() for ln in (text or "").split("\n") if ln.strip()
    )
    stake = 0.0
    potential = 0.0
    head = joined

    m_stake = re.search(r"Total\s*Stake", joined, re.IGNORECASE)
    if m_stake:
        head = joined[: m_stake.start()]
    m_pot = re.search(
        r"Pot(?:ential)?\.?\s*Win\s*(?:\||\s)*([0-9][0-9,.]*)", joined, re.IGNORECASE
    )
    if m_pot:
        try:
            potential = float(m_pot.group(1).replace(",", ""))
        except ValueError:
            potential = 0.0

    odds_tokens = _ODDS_TOKEN_RE.findall(head)
    selections: list[SlipSelection] = []
    if odds_tokens:
        # The accepted price is the last odds token before the stake label:
        # the earlier ones belong to the event or the market title.
        selections.append(
            SlipSelection(name="", market="", odds=float(odds_tokens[-1]))
        )
    return selections, stake, potential


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
  const t = document.body.innerText || '';
  const lines = t.split('\\n').map(s => s.trim()).filter(Boolean);
  const i = lines.findIndex(l => /Total Stake/i.test(l));
  const seg = i >= 0 ? lines.slice(Math.max(0, i - 14), i + 10) : [];
  const stakeEl = document.querySelector('%(input)s');
  const visible = (b) => b && b.offsetParent !== null;
  const place = Array.from(document.querySelectorAll('button, [role=button]'))
    .filter(b => /place bet/i.test((b.innerText||'') + ' ' + (b.getAttribute('aria-label')||'')));
  // When the price has moved the book shows 'Accept Changes' instead of
  // 'Place Bet'; the slip is only placeable once that is acknowledged.
  const accept = Array.from(document.querySelectorAll('button, [role=button]'))
    .filter(b => /accept changes/i.test((b.innerText||'') + ' ' + (b.getAttribute('aria-label')||'')));
  return JSON.stringify({
    seg: seg,
    stake: stakeEl ? stakeEl.value : null,
    canPlace: place.some(b => !b.disabled && visible(b)),
    hasPlaceButton: place.length > 0,
    needsAccept: accept.length > 0
  });
})()
"""

JS_CLEAR = """
(() => {
  const visible = (e) => e && e.offsetParent !== null;
  // The slip carries a 'Remove All' control when it holds selections; the
  // per-selection remove buttons are the fallback.
  const all = Array.from(document.querySelectorAll('button, [role=button], a, span'))
    .find(b => visible(b) && /^remove all$/i.test((b.innerText||'').trim()));
  if (all) { all.click(); return JSON.stringify({ok: true, how: 'remove all'}); }
  let removed = 0;
  const panels = Array.from(document.querySelectorAll('*')).filter(e =>
    /Total Stake/i.test(e.innerText || '') && (e.innerText || '').length < 900);
  for (const p of panels.slice(0, 3)) {
    p.querySelectorAll('button, [class*=remove], [class*=delete], [class*=close], [class*=clear], svg, i').forEach(b => {
      const lbl = ((b.getAttribute('aria-label')||'') + ' ' + (b.getAttribute('title')||'') + ' ' +
                   (b.className||'').toString() + ' ' + (b.innerText||'')).toLowerCase();
      if (/remove|delete|close|clear|times|\u2715|\u00d7|m-icon-/.test(lbl)) {
        try { b.click(); removed++; } catch (e) {}
      }
    });
    if (removed) break;
  }
  return JSON.stringify({ok: true, how: 'per-selection', removed: removed});
})()
"""

JS_SLIP_EMPTY = """
(() => {
  const t = document.body.innerText || '';
  return JSON.stringify({empty: !/Total\\s*Stake/i.test(t)});
})()
"""

JS_PLACE_BET = """
(() => {
  const b = Array.from(document.querySelectorAll('button, [role=button]'))
    .find(x => /place bet/i.test((x.innerText||'') + ' ' + (x.getAttribute('aria-label')||''))
           && x.offsetParent !== null && !x.disabled);
  if (!b) return JSON.stringify({ok: false, error: 'no enabled Place Bet button'});
  b.setAttribute('data-atlas-place-bet', '1');
  return JSON.stringify({ok: true, selector: '[data-atlas-place-bet="1"]'});
})()
"""


JS_MARK_ACCEPT = """
(() => {
  const b = Array.from(document.querySelectorAll('button, [role=button]'))
    .find(x => /accept changes/i.test((x.innerText||'') + ' ' + (x.getAttribute('aria-label')||''))
           && x.offsetParent !== null && !x.disabled);
  if (!b) return JSON.stringify({ok: false, error: 'no Accept Changes button'});
  b.setAttribute('data-atlas-accept', '1');
  return JSON.stringify({ok: true, selector: '[data-atlas-accept="1"]'});
})()
"""


JS_MARK_CONFIRM = """
(() => {
  const b = Array.from(document.querySelectorAll('button, [role=button]'))
    .find(x => /^confirm$/i.test((x.innerText||'').trim())
           && x.offsetParent !== null && !x.disabled);
  if (!b) return JSON.stringify({ok: false, error: 'no Confirm button'});
  b.setAttribute('data-atlas-confirm', '1');
  return JSON.stringify({ok: true, selector: '[data-atlas-confirm="1"]'});
})()
"""


JS_REAL_MODE = """
(() => {
  const visible = (e) => e && e.offsetParent !== null;
  const box = document.querySelector('.switch-box');
  if (!box) return JSON.stringify({ok: false, error: 'no mode switch'});
  // The active mode is the one whose label sits behind the highlighted
  // shadow ('show-highlight'); the clickable control is an 'inside-btn'.
  const labels = Array.from(box.querySelectorAll('.shadow-text, .inside-btn'));
  const active = labels.find(l => /show-highlight/.test((l.className||'').toString()));
  const real = labels.find(l => /^REAL$/i.test((l.textContent||'').trim()));
  if (!real) return JSON.stringify({ok: false, error: 'no REAL control'});
  const activeText = active ? (active.textContent||'').trim() : '';
  if (activeText.toUpperCase() === 'REAL') return JSON.stringify({ok: true, already: true});
  const btn = real.className.includes('inside-btn') ? real :
    (real.parentElement && real.parentElement.querySelector('.inside-btn'));
  if (!btn || !visible(btn)) return JSON.stringify({ok: false, error: 'REAL control not clickable'});
  btn.click();
  return JSON.stringify({ok: true, switchedFrom: activeText});
})()
"""

JS_MODE_STATE = """
(() => {
  const box = document.querySelector('.switch-box');
  if (!box) return JSON.stringify({mode: null});
  const active = Array.from(box.querySelectorAll('.shadow-text'))
    .find(l => /show-highlight/.test((l.className||'').toString()));
  if (active) return JSON.stringify({mode: (active.textContent||'').trim()});
  // After a switch the highlight sits on the other side, so fall back to
  // the button that is not marked inactive.
  const btn = Array.from(box.querySelectorAll('.inside-btn'))
    .find(b => !/inactive/.test((b.className||'').toString()));
  return JSON.stringify({mode: btn ? (btn.textContent||'').trim() : null});
})()
"""


class _PlaceBetMarker:
    """Marks the Place Bet button so a REAL browser click can target it.

    A synthetic `element.click()` from injected script is not always accepted
    by the book: the click is untrusted, and some handlers only fire on a
    trusted event. The extension's own click action performs a real browser
    click, so the button is first tagged with a unique attribute and that
    attribute is used as the selector.
    """

    MARKER = "data-atlas-place-bet"


def _click_selector_via_adapter(adapter: Any, selector: str) -> Any:
    return adapter.click(selector)


class BetslipDriver:
    """Drives the SportyBet betslip through a WebBridge adapter.

    `adapter` must expose `evaluate(code) -> str` (the WebBridgeAdapter does).
    """

    def __init__(self, adapter: Any, allow_submit: bool = False) -> None:
        self.adapter = adapter
        self.allow_submit = bool(allow_submit)

    # --- low level -----------------------------------------------------

    def _js(self, template: str, **fmt: Any) -> dict:
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

    def read_slip(self, wait_s: float = 3.0) -> SlipState:
        """Read the slip, including the odds the book ACCEPTED.

        The accepted odds matter: the book can move the price between the
        click and the acceptance, and a bet placed at a worse price than the
        one that was priced is not the bet that was approved.

        The book enables the submit button a moment after the stake is set,
        so this waits for the button rather than reading it too early and
        concluding that the slip is not placeable.
        """
        deadline = time.monotonic() + max(0.0, wait_s)
        state = self._read_slip_once()
        while (
            not state.can_place_bet
            and time.monotonic() < deadline
        ):
            time.sleep(0.3)
            state = self._read_slip_once()
        return state

    def _read_slip_once(self) -> SlipState:
        data = self._js(JS_READ_SLIP, input=STAKE_INPUT_SELECTOR)
        text = " | ".join(data.get("seg", []))
        selections, parsed_stake, potential = parse_slip_text(text)
        stake_raw = data.get("stake")
        stake = 0.0
        if stake_raw:
            try:
                stake = float(str(stake_raw).replace(",", ""))
            except ValueError:
                stake = 0.0
        if not stake:
            stake = parsed_stake
        return SlipState(
            selections=selections,
            total_stake_ngn=stake,
            potential_win_ngn=potential,
            can_place_bet=bool(data.get("canPlace")),
            needs_accept=bool(data.get("needsAccept")),
            raw_text=text,
        )

    def accept_changes(self) -> dict:
        """Accept a moved price on the slip.

        The book shows 'Accept Changes' instead of 'Place Bet' when the
        selection's price has moved. The caller must have checked the new
        price against the priced odds BEFORE calling this.

        The click is a REAL browser click (the extension's click action), for
        the same reason as the placement: a synthetic click is untrusted and
        the book ignores it.
        """
        marked = self._real_click(JS_MARK_ACCEPT, "Accept Changes")
        if not marked.get("ok"):
            return {"ok": False, "error": marked.get("error", "accept failed")}
        try:
            self._evaluate(
                "(() => { const b = document.querySelector('[data-atlas-accept]');"
                " if (b) b.removeAttribute('data-atlas-accept'); return 1; })()"
            )
        except Exception:
            pass
        # The slip settles into a placeable state.
        for _ in range(20):
            time.sleep(0.3)
            if self.read_slip(wait_s=0.0).can_place_bet:
                break
        return {"ok": True}

    JS_ADD_SELECTION = """
    (() => {
      const market = '%(market)s';
      const selection = '%(selection)s';
      const ODDS = /^[0-9]{1,3}\\.[0-9]{2}$/;
      for (const w of document.querySelectorAll('.m-table__wrapper')) {
        const title = ((w.querySelector('.m-table-header-title')||{}).innerText || '').trim();
        if (title !== market) continue;
        for (const cell of w.querySelectorAll('.m-table-cell')) {
          const items = Array.from(cell.querySelectorAll('.m-table-cell-item'))
            .map(i => (i.innerText || '').trim());
          const name = items.find(t => t && !ODDS.test(t));
          const prices = items.filter(t => ODDS.test(t));
          if (name === selection && prices.length) {
            const item = Array.from(cell.querySelectorAll('.m-table-cell-item'))
              .filter(i => ODDS.test((i.innerText||'').trim())).pop();
            item.click();
            return JSON.stringify({ok: true, market: title, selection: name,
                                  price: prices[prices.length-1]});
          }
        }
      }
      return JSON.stringify({ok: false, error: 'selection not found on the board'});
    })()
    """

    def add_selection(self, market_label: str, selection: str) -> dict:
        """Add a selection to the slip BY NAME and report the accepted price.

        Finding it by price is not safe: a live price moves between the
        moment it is priced and the moment it is clicked, so the price
        lookup fails exactly when the price has changed — which is the case
        that must be caught, not the case that must be clicked.
        """
        return self._js(
            self.JS_ADD_SELECTION,
            market=str(market_label).replace("'", ""),
            selection=str(selection).replace("'", ""),
        )

    JS_ACCEPT_CHANGES = r"""
    (() => {
      const b = Array.from(document.querySelectorAll('button, [role=button]'))
        .find(x => /accept changes/i.test((x.innerText||'') + ' ' + (x.getAttribute('aria-label')||''))
               && x.offsetParent !== null);
      if (!b) return JSON.stringify({ok: false, error: 'no accept changes button'});
      b.click();
      return JSON.stringify({ok: true});
    })()
    """

    def _evaluate(self, code: str) -> Any:
        """Evaluate JS and return the unwrapped value."""
        raw = self.adapter.evaluate(code)
        data = raw
        if isinstance(raw, dict):
            if "tree" in raw or "url" in raw or "value" in raw:
                data = raw
            elif "data" in raw:
                data = raw["data"]
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def slip_is_empty(self) -> bool:
        """Is the slip free of selections?

        A leftover selection silently turns the next bet into a multiple, so
        this is checked before every placement.
        """
        try:
            raw = self._evaluate(JS_SLIP_EMPTY)
        except CaptureError:
            return False
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return False
        else:
            data = raw if isinstance(raw, dict) else {}
        return bool(data.get("empty"))

    def clear(self) -> dict:
        """Remove every selection from the slip. Safe: never submits."""
        result = self._js(JS_CLEAR)
        # The book needs a moment to drop the selections.
        for _ in range(8):
            if self.slip_is_empty():
                break
            time.sleep(0.3)
        return result

    def _real_click(self, marker_js: str, label: str) -> dict:
        """Mark an element and click it with a REAL browser click.

        A synthetic `element.click()` from injected script is untrusted and
        the book ignored it, so the extension's own click action is used on
        a uniquely-marked element.
        """
        marked = self._js(marker_js)
        if not isinstance(marked, dict) or not marked.get("ok"):
            return {"ok": False, "error": (marked or {}).get("error", f"no {label}")}
        try:
            self.adapter.click(marked.get("selector"))
        except Exception as e:
            return {"ok": False, "error": f"click failed: {e}"}
        return {"ok": True}

    def mode(self) -> Optional[str]:
        """The betslip's mode: 'REAL' or 'SIM'."""
        try:
            raw = self._evaluate(JS_MODE_STATE)
        except CaptureError:
            return None
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None
        else:
            data = raw if isinstance(raw, dict) else {}
        m = data.get("mode")
        return str(m).upper() if m else None

    def set_real_mode(self) -> dict:
        """Switch the betslip to REAL mode.

        The book can be left in SIM (simulation) mode, and a simulation slip
        is rejected by the order API with a bare 'Something went wrong...'.
        A real bet requires REAL mode, so this is checked before every
        placement.
        """
        try:
            raw = self._evaluate(JS_REAL_MODE)
        except CaptureError as e:
            return {"ok": False, "error": f"mode switch failed: {e}"}
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {"ok": False, "error": "mode switch returned nothing"}
        else:
            data = raw if isinstance(raw, dict) else {}
        if not data.get("ok"):
            return data
        time.sleep(0.8)
        mode = self.mode()
        if mode not in (None, "REAL"):
            return {"ok": False, "error": f"still in {mode} mode"}
        return {"ok": True, "mode": mode or "REAL"}

    def place_bet(self) -> dict:
        """Submit the slip. Requires an explicit opt-in at construction.

        This is the only money-moving call in ATLAS. It stays disabled until
        somebody deliberately constructs the driver with allow_submit=True.

        The book takes a bet in TWO steps: 'Place Bet' opens an 'About to pay'
        confirmation, and the bet is only placed when that is confirmed. Both
        steps are real browser clicks, and the result is verified from the
        slip itself — a placed bet empties the slip.
        """
        if not self.allow_submit:
            raise SubmissionNotAllowed(
                "BetslipDriver(allow_submit=True) is required to place a bet"
            )
        slip = self.read_slip()
        if not slip.can_place_bet:
            raise BetslipError("betslip is not placeable (no enabled Place Bet button)")
        # The book can be left in SIM mode, and a simulation slip is rejected
        # by the order API. A real bet needs REAL mode.
        mode = self.set_real_mode()
        if not mode.get("ok"):
            return {"ok": False, "error": mode.get("error", "could not set REAL mode")}
        marked = self._real_click(JS_PLACE_BET, "Place Bet")
        if not marked.get("ok"):
            return {"ok": False, "error": marked.get("error", "place bet failed")}
        # The confirmation dialog: 'About to pay | NGN 10.00 | Cancel | Confirm'.
        time.sleep(1.0)
        confirmed = self._real_click(JS_MARK_CONFIRM, "Confirm")
        if not confirmed.get("ok"):
            return {
                "ok": False,
                "error": confirmed.get("error", "no confirmation dialog appeared"),
            }
        # Verify from the book: a placed bet empties the slip.
        placed = False
        for _ in range(24):
            time.sleep(0.5)
            if self.slip_is_empty():
                placed = True
                break
        for attr in ("data-atlas-place-bet", "data-atlas-confirm"):
            try:
                self._evaluate(
                    "(() => { const b = document.querySelector('[%s]');"
                    " if (b) b.removeAttribute('%s'); return 1; })()"
                    % (attr, attr)
                )
            except Exception:
                pass
        return {
            "ok": placed,
            "error": None if placed else "the slip did not empty after Confirm",
        }
