"""
Stale-slip recovery — the 'Unavailable' / 'Accept Changes' wedge.

Ground truth from the live book (2026-10-28, FC Goa v Sporting Club Delhi):
after a click, the book can suspend the market, leaving the slip holding
one selection marked Unavailable with an 'Accept Changes' button instead
of 'Place Bet'. In that state:

  - synthetic element.click() from injected script is IGNORED by this UI
    build, so the old JS_CLEAR (which clicks per-selection remove buttons
    synthetically) silently did nothing — the slip stayed wedged across
    cycles;
  - the only removals that actually happen are REAL browser clicks,
    one control per call, via the marked-selector flow (_real_click).

So the sweep marks exactly ONE control per invocation — a marked
attribute is re-applied each call, never accumulated — in this order:

  1. the per-selection delete icon of a card (the stale one first). This is
     the control that actually removes a row on the live build;
  2. a 'Remove All' control, only when no per-selection icon exists. On the
     live slip it was clicked ten times and removed nothing (2026-10-29);
  3. an 'Accept Changes' control as the LAST resort. The wedge can hide
     every remove control behind that button; acknowledging the price move
     is not a bet — it only makes the stale card removable/placeable
     again, and the runner's odds gate still judges the accepted price
     before any money moves.

Two DOM traps made the earlier sweep a no-op, both found on the live slip
(2026-10-29, AD Sporting Hortaleza v Sevilla):

  - the book keeps a hidden PRINT COPY of the slip (.m-betslip-print,
    'Sharing Code' markup) that repeats every row and every delete icon at
    0x0. It has client rects, so a getClientRects()-based visibility test
    accepted it and every click landed on a hidden node;
  - the row is `.m-item`, not `.m-bet-item` (the latter only exists in the
    print copy), and the row holds a checkbox, a sport icon, a LIVE badge
    AND the delete icon — a generic 'm-icon-' match clicked the checkbox.

Every successful branch MUST return `selector`: _real_click clicks the
marked element through the extension. A marker that reports ok without a
selector is unclickable — the bug that made the first sweep a no-op.
"""

JS_MARK_SLIP_REMOVE = r"""
(() => {
  // ONE marker at a time: previous marks are cleared so adapter.click
  // never re-clicks a stale widget (the slip re-renders and old nodes die).
  document.querySelectorAll('[data-atlas-slip-remove]').forEach(
    (e) => e.removeAttribute('data-atlas-slip-remove')
  );
  // Visibility must be judged by the element's BOX, not by getClientRects()
  // existing: the book keeps a hidden print copy of the slip
  // (.m-betslip-print, 'Sharing Code' markup) that repeats every selection
  // and every delete icon at 0x0. It has client rects, so the old check
  // accepted it, the sweep clicked a hidden icon ten times, and the real
  // slip stayed wedged (2026-10-29, Hortaleza v Sevilla).
  const visible = (e) => {
    if (!e || !e.isConnected) return false;
    if (e.closest && e.closest('.m-betslip-print')) return false;
    const r = e.getBoundingClientRect ? e.getBoundingClientRect() : null;
    if (!r || r.width <= 0 || r.height <= 0) return false;
    const cs = window.getComputedStyle(e);
    return cs.visibility !== 'hidden' && cs.display !== 'none'
      && cs.opacity !== '0';
  };
  const mark = (el, how) => {
    el.setAttribute('data-atlas-slip-remove', '1');
    return JSON.stringify({
      ok: true,
      how: how,
      selector: '[data-atlas-slip-remove="1"]'
    });
  };
  const cand = Array.from(
    document.querySelectorAll('button,[role=button],a,span,i,svg,div')
  ).filter(visible);
  const attr = (b) => (
    (b.getAttribute('aria-label') || '') + ' ' +
    (b.getAttribute('title') || '') + ' ' +
    (b.className || '').toString()
  ).toLowerCase();
  // The slip row carries several icons (checkbox, sport, LIVE badge,
  // delete). Only the delete one removes anything: a generic 'm-icon-'
  // match used to click the CHECKBOX first and toggle the selection
  // instead of deleting it.
  const deleteish = (b) => /delete|remove|trash/.test(attr(b))
    || /^[\u2715\u00d7]\s*$/.test((b.innerText || '').trim());
  // The selection row is `.m-item` (NOT `.m-bet-item`, which only exists in
  // the hidden print copy). Delete icons exist nowhere else on the page, so
  // they identify the rows on their own; the row text tells us which one is
  // the stale card.
  const row = (b) => b.closest('.m-item,.m-bet-item,[class*=bet-item]');
  const stale = (b) => {
    const zone = row(b);
    return zone && /unavailable|accept changes|suspended/i.test(
      zone.innerText || ''
    );
  };
  // 1) Per-selection delete, stale card first. This is the control that
  // actually removes a row: 'Remove All' was clicked ten times on the live
  // slip (2026-10-29) and removed nothing.
  const removes = cand.filter(deleteish).filter(
    (b) => !/remove all/i.test((b.innerText || '').trim())
  );
  const target = removes.find(stale) || removes[0] || null;
  if (target) return mark(target, 'per-selection');
  // 2) 'Remove All' when no per-selection control exists.
  const all = cand.find(
    (b) => /^remove all$/i.test((b.innerText || '').trim())
  );
  if (all) return mark(all, 'all');
  // 3) LAST resort: no remove control at all. Acknowledge the moved price
  // so the stale card can re-render into a removable/placeable state.
  const accept = cand.find(
    (b) => b.matches && b.matches('button,[role=button],a')
      && /accept changes/i.test((b.innerText || '').trim())
  );
  if (accept) return mark(accept, 'accept');
  return JSON.stringify({ok: false, error: 'no remove control visible in slip'});
})()
"""

JS_MARK_SLIP_REMOVE_NAME = '[data-atlas-slip-remove="1"]'

# Bounded sweep: one delete icon per selection. The slip holds at most a
# handful of selections; SWEEP_MAX_CLICKS covers them all with headroom
# ('Remove All' and 'Accept Changes' are only reached when no delete icon
# is left, and each of those is a single extra click).
SWEEP_MAX_CLICKS = 10
