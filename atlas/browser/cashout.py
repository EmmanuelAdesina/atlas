"""Cashout driver: read and execute SportyBet cashouts.

The DOM contract below was captured from the live book on 2026-10-19, when
the account held an open live position:

  - the betslip nav has a 'Cashout' tab, which is a leaf element whose text
    is exactly 'Cashout' (and shows a count badge, e.g. 'Cashout1');
  - the cashout panel holds one item per cashable position, in elements
    whose class contains 'm-cashout-item' (inside 'm-cashout-list');
  - the item text contains 'Bet Stake <amount>', 'Pot. Win <amount>' and
    'Full Cashout <amount>';
  - the action is a control reading 'Cashout now'.

Every selector is optional by design: if the book changes its markup, the
driver reports 'no offers' instead of raising, because a cashout driver
that throws inside the loop would take the loop down.

A partial cashout ('One Cut') is a PLACEMENT-time control on the betslip,
not a separate cashout flow, so it is detected and reported but not
automated: its exact settlement behaviour has not been confirmed against a
live position.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

# The item text is flattened, so the label and the amount can be separated
# by a newline, a pipe or a colon.
_SEP = r"\s*[:|]?\s*"
FULL_CASHOUT_RE = re.compile(
    r"Full Cashout" + _SEP + r"([0-9][0-9,.]*)", re.IGNORECASE
)
STAKE_RE = re.compile(r"Bet Stake" + _SEP + r"([0-9][0-9,.]*)", re.IGNORECASE)
POT_WIN_RE = re.compile(r"Pot\.?\s*Win" + _SEP + r"([0-9][0-9,.]*)", re.IGNORECASE)
MULTI_RE = re.compile(r"(\d+)\s*\+?\s*of\s*(\d+)", re.IGNORECASE)


@dataclass
class CashoutOffer:
    """One cashable position the book is offering to buy back."""

    index: int
    stake_ngn: float
    offered_ngn: float
    potential_win_ngn: float = 0.0
    legs_known: int = 0
    legs_total: int = 0
    label: str = ""

    def describe(self) -> str:
        legs = (
            f" ({self.legs_known}/{self.legs_total} legs up)"
            if self.legs_total
            else ""
        )
        return (
            f"offer {self.index}: stake NGN {self.stake_ngn:.2f}, "
            f"full cashout NGN {self.offered_ngn:.2f}{legs}"
        )


class CashoutDriver:
    """Reads and executes cashouts through a WebBridge adapter."""

    JS_OPEN_TAB = r"""
    (() => {
      const el = Array.from(document.querySelectorAll('*'))
        .find(e => e.children.length === 0 && /^Cashout/i.test((e.textContent||'').trim()));
      if (!el) return JSON.stringify({ok: false, error: 'no cashout tab'});
      el.click();
      return JSON.stringify({ok: true});
    })()
    """

    JS_READ = r"""
    (() => {
      const items = Array.from(document.querySelectorAll('[class*=m-cashout-item], [class*=m-cashout-list]'));
      const out = [];
      items.forEach((el, i) => {
        const txt = (el.innerText || '').trim();
        if (!txt) return;
        const btns = Array.from(el.querySelectorAll('button, [role=button]'))
          .map(b => ((b.innerText||'') + '|' + (b.getAttribute('aria-label')||'')).trim())
          .filter(Boolean);
        out.push({i, txt: txt.replace(/\n/g, ' | '), btns: btns.slice(0, 5)});
      });
      const oneCut = Array.from(document.querySelectorAll('*'))
        .some(e => /one cut/i.test((e.innerText||'')) && (e.innerText||'').length < 200);
      return JSON.stringify({items: out, oneCut: oneCut});
    })()
    """

    JS_EXECUTE = r"""
    (() => {
      const idx = %(index)d;
      const items = Array.from(document.querySelectorAll('[class*=m-cashout-item], [class*=m-cashout-list]'));
      const el = items[idx];
      if (!el) return JSON.stringify({ok: false, error: 'no cashout item'});
      const btn = Array.from(el.querySelectorAll('button, [role=button]'))
        .find(b => /cash ?out now/i.test((b.innerText||'') + ' ' + (b.getAttribute('aria-label')||'')));
      if (!btn) return JSON.stringify({ok: false, error: 'no cashout button'});
      btn.click();
      return JSON.stringify({ok: true});
    })()
    """

    def __init__(self, adapter: Any, allow_execute: bool = False) -> None:
        self.adapter = adapter
        self.allow_execute = allow_execute

    def _evaluate(self, code: str) -> Any:
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

    def open_cashout_tab(self) -> bool:
        try:
            raw = self._evaluate(self.JS_OPEN_TAB)
        except Exception:
            return False
        try:
            return bool(json.loads(raw).get("ok")) if isinstance(raw, str) else False
        except (TypeError, json.JSONDecodeError):
            return False

    def read_offers(self) -> list[CashoutOffer]:
        """Every cashable position the book is currently offering."""
        self.open_cashout_tab()
        try:
            raw = self._evaluate(self.JS_READ)
        except Exception:
            return []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return []
        offers: list[CashoutOffer] = []
        for item in (data or {}).get("items", []):
            txt = item.get("txt") or ""
            m_full = FULL_CASHOUT_RE.search(txt)
            m_stake = STAKE_RE.search(txt)
            if not m_full:
                continue
            m_legs = MULTI_RE.search(txt)
            offers.append(
                CashoutOffer(
                    index=int(item.get("i", len(offers))),
                    stake_ngn=float(m_stake.group(1).replace(",", "")) if m_stake else 0.0,
                    offered_ngn=float(m_full.group(1).replace(",", "")),
                    potential_win_ngn=(
                        float(POT_WIN_RE.search(txt).group(1).replace(",", ""))
                        if POT_WIN_RE.search(txt)
                        else 0.0
                    ),
                    legs_known=int(m_legs.group(1)) if m_legs else 0,
                    legs_total=int(m_legs.group(2)) if m_legs else 0,
                    label=txt[:120],
                )
            )
        return offers

    def partial_cashout_available(self) -> bool:
        """Whether the 'One Cut' control is present on the slip."""
        try:
            raw = self._evaluate(self.JS_READ)
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return False
        return bool(isinstance(data, dict) and data.get("oneCut"))

    def execute(self, index: int) -> dict:
        """Execute a cashout. Refuses unless explicitly enabled."""
        if not self.allow_execute:
            return {
                "ok": False,
                "error": "cashout execution is disabled "
                "(CashoutDriver(allow_execute=True) required)",
            }
        try:
            raw = self._evaluate(self.JS_EXECUTE % {"index": index})
            return json.loads(raw) if isinstance(raw, str) else {"ok": False}
        except Exception as e:
            return {"ok": False, "error": str(e)}
