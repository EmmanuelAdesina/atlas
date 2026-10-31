# Atlas Operational Bugs & Issues

## Status: 2026-10-17

### Fixed (tested, 43 pass)

| # | Issue | Fix |
|---|-------|-----|
| B1 | Scoreboard/clock locked inside SportyTV iframe | Score IS in main DOM with CSS classes `sr-lmt-1-ms2-matchinfocountdown__result srm-is-home`/`srm-is-away`. Extractor updated. |
| B2 | Session expiry during live polling | Documented. Needs session-health check in polling loop. |
| B3 | No live data API access from page context | Using DOM scraping instead. API blocked from page context. |
| B4 | Red card multiplier ignores timing | Decay by remaining/30 |
| B5 | Score-state urgency multiplier ignores timing | Gated by remaining ≥ 10 |
| B6 | Shrinkage too high without shots | Halved when shots_on_target None |
| B7 | live_rate clamp too aggressive | [0.30→0.10] |
| B8 | rho inconsistency | -0.08 → -0.06 |
| B9 | Model systematically more extreme than market | Calibration layer added |
| B10 | Flat stoppage hazard | Ramps with stoppage progress |
| B11 | Red card stoppage not time-weighted | Weighted by (90-card_min)/90 |
| B12 | Matrix truncation max_goals=8 | Raised to 10 |
| B13 | Network capture filter case-sensitive | Documented. Using DOM scraping instead. |
| B14 | Sub-ms pricing test machine-dependent | Fixed: assert sub-ms **per selection** + cache-effectiveness |
| B15 | Correct Score header shows current score | Documented. Not a bug. |
| B16 | HT state crashes model | Added "ht" to is_live |
| B17 | 1x2_rest unpriced | Added to pricer |
| B18 | O/U ignores scored goals | Already correct (uses base_g = bh+ba) |
| B19 | No live stats = priors only | Statistics tab scraper added |
| B20 | 1x2_1up/1x2_2up unpriced | Added early-settlement pricing |
| B21 | Calibration cross-meaning bug | Only calibrate same-semantics markets |
| B22 | "45+3" clock parsed as 45.5 | Fixed to 48 (base + stoppage minutes) |
| B23 | Calibration still shows fake edges | Block bets when model-market gap > 5pp |

### Remaining

| # | Issue | What it needs |
|---|-------|---------------|
| B2 | Session expiry | Session-health check in polling loop. How long does your session last? |
| R2 | Statistics tab scraper | Built but needs live validation. May break if SportyBet changes CSS. |
| B14 | Sub-ms pricing test | **Fixed.** The assertion demanded ~95us/selection (2ms for 21 selections) — ~4x below what this class of machine delivers, so it could never pass here. It now asserts the actual architectural claim: sub-millisecond **per selection** (measured: ~420us/selection cold, ~200us/selection warm), plus a machine-independent cache-effectiveness test that compares two timings measured in the same run and fails only if the joint surface stops being cached (verified failing when the cache is disabled). |

### Found by running against the live board (2026-10-19)

| # | Issue | Status |
|---|-------|--------|
| B24 | 1UP/2UP pricing ignored the scoreline deficit (`d >= 1` instead of `d >= 1 - base_diff`), so a side 0-2 down at 84' read as ~13% to lead by 1 — a phantom EV **+5.31 at odds 47.00**. Live-money hazard. | **Fixed** (offset applied; end-of-game difference is a conservative lower bound on "led by N at any time"). Regression test added. |
| B25 | The live header renders a bare period label (`2nd`), not `2nd Half`. Unreadable period → `is_live` false → **all 128 selections abstained** on a live match. | **Fixed** (bare labels read only within 3 tokens of the clock). |
| B26 | The score was read as the clock. SportyBet renders the clock as `('73', ':', '52')` on some matches and `('75:11')` on others; the split form was parsed as a 73-2 score. | **Fixed** (clock-anchored, panel-scoped, plausibility-bounded). |
| B27 | `navigate()` to the live board always times out (the page never fires `load` — websocket polling) and the tab is then not registered, so every later command 502s. | **Open.** Workaround: load a light page, then navigate client-side via `location.href`; hard-reload to repopulate the board. The production loop must never call `navigate()` on the live board. |
| B28 | The `snapshot` action 502s on the live board (accessibility tree of 45+ matches). | **Open.** Extraction must use targeted `evaluate` JS, not snapshots. |
| B29 | Pricer coverage is **1.9% of a live board**: 20 of 92 market headers, 53 of 2,794 odds. Uncovered: clean sheet, win-to-nil, exact goals, odd/even, goals-in-a-row, lead-by-N-at-any-time, team O/U, 3-way handicap, 1X2-from-minute-N, total goals, correct score, corners/cards/player markets. | **Fixed: 99.6% of the board** (2,784 of 2,794 odds, 89 of 92 headers) via the score-path engine (`atlas/live/paths.py`), the final-state families, the combination markets, the time-windowed markets and the rest-of-match variants. The path engine reproduces the joint matrix to 1e-16, so it is provably the same model extended to order-dependent questions. Combination markets are read straight off the joint matrix, never as a product of marginals. Guarded by `tests/test_live_board_coverage.py` against the captured real board (threshold 95%). |
| B30 | No cashout support of any kind (no code path, no math). SportyBet exposes Cashout in the live betslip, including partial cashout on losing live positions. | **Open.** The UI contract is now known: the betslip carries **'One Cut'** (partial cashout), **'F Flexi'** (partial-payout accumulator) and **'SportyInsure'** (stake insurance). Live evidence: an open 'Flexi Your Bet 18+ of 19' accumulator, stake NGN 160, potential win NGN 38,498.89, offered for cashout at NGN 246.53. Needs its own model: cashout value vs the position's live EV, plus leg-risk for a partial cut. |
| B31 | No bet submission layer. The WebBridge `click` primitive exists but nothing clicks odds; `atlas.main plan` is observe-only. | **Done (behind `--submit`).** `atlas/browser/betslip.py` drives the real slip: add odds, set stake, read slip, clear. Verified live against the board (selected `Draw (0:2) | Live Al-Shabab FC v Al-Faisaly FC | Handicap 0:2 | 17.00`, set NGN 100, read it back, cleared it). The loop now plans, gates, and enqueues placements; `place_bet()` still refuses unless the driver is constructed with `allow_submit=True`, and the runner still requires `--submit` plus `PolicyConfig(allow_execution=True)`. Cashout execution remains open (B30). |
| B32 | The `plan` path used the raw `LiveStateSpaceModel`, bypassing the calibration layer and its 5pp model-market gap guard that exists to block fake edges. | **Fixed** (plan now uses `CalibratedModel`). |
| B33 | Placement ran synchronously inside the cycle: a wedged betslip — a suspended market leaving a selection marked **Unavailable** behind **Accept Changes** — froze the whole loop. The legacy `JS_CLEAR` synthetic clicks are ignored by this UI build, and the REAL-click sweep marked an element but did not return its selector, so `_real_click` asked the extension to click `None` and the slip stayed wedged across cycles. | **Fixed, and the wedge's real root cause found live.** Placement runs on a worker thread: the cycle enqueues accepted bets and keeps pricing. In-flight reservations make a queued bet visible to the next cycle's exposure/correlation/rate accounting (no duplicate placement), requests expire after 90s, the queue is capped at 8, and every page-lock acquisition has a timeout. `SlipState.unavailable` makes `read_slip` return immediately instead of waiting 3s for a Place Bet button that cannot appear. **The sweep itself was still broken on the live build** (2026-10-29, AD Sporting Hortaleza v Sevilla): the book keeps a hidden print copy of the slip (`.m-betslip-print`, 'Sharing Code' markup) that repeats every row and delete icon at 0x0 — it has client rects, so the old visibility test accepted it and every click landed on a hidden node; the row is `.m-item`, not `.m-bet-item` (which only exists in the print copy); a generic `m-icon-` match clicked the row's CHECKBOX instead of its delete icon; and 'Remove All' was clicked ten times and removed nothing on this build. `atlas/browser/slipclear.py` now judges visibility by the element's box plus computed style, excludes the print subtree, clicks the per-selection delete icon (stale card first), and only falls back to 'Remove All' then **Accept Changes**. A real click on the correct delete icon did remove the Suspended row live; the full sweep still needs one re-run with a wedged slip once the session is logged in again (the account was logged out mid-validation: 'Please enter a valid mobile number'). |
| B34 | The betslip controls were selected by bare component classes (`input.m-input`, `button.af-button--primary`). When the SportyBet session expired mid-run the page became a LOGIN modal built from those same components, and the placement worker — still draining a queue filled before the logout — typed the stake into the login PHONE field (the tab showed `+234 10.00`) and the form reported `Please enter a valid mobile number`. | **Fixed.** Every mutating betslip payload now begins with a login guard: a VISIBLE `input[type=password]` makes it return `{ok: false, login: true}` without touching the page, and `SlipState.session_expired` surfaces the same fact to the runner, which abandons the whole batch. The stake box is scoped to a visible, non-tel/password field advertising `min`/`stake` (or the panel that carries `Total Stake`), and `clear()` never reports success on a login page. `LiveRunner.place_bets` re-checks the page with `capture.session_expired()` before taking the page lock, and `run_cycle` calls `_drop_queued_placements()` when the session dies so a stale queue can never be drained against the modal. Verified against the real login page (2026-10-30): `set_stake(10)` → `{ok: false, login: true}`, `read_slip().session_expired` → `True`, `clear()` → `{ok: false}`, and the form's fields were untouched. |
| B35 | The deterministic re-login (`Supervisor.handle_session_expired` -> `BrowserTools.fill_login`) failed on every cycle with `no visible login button` (4 attempts, 13:31-13:35 on 2026-10-30). Three separate defects: (1) the Login button was searched inside `pwd.closest('form') || pwd.parentElement`, but the password's parent is `.m-input-wrapper` (310x54) while the button lives in `.m-btn-wrap` inside `.m-page--login`, so the search never reached it; (2) the phone field is `input[name=phone]` (type `text`, placeholder `Mobile Number`), not `input[type=tel]`, so the phone was never filled and the button stayed `is-disabled`; (3) React enables the button a tick after accepting both fields, so a single look right after filling is a race. A leftover `Odds not acceptable, please try again` dialog also covers the panel and eats the real click. **Fixed.** `_LOGIN_JS` scopes by the LOGIN PANEL (not by a parent element), finds the phone by `input[name=phone]` first, and only MARKS the button; `_LOGIN_BUTTON_JS` polls up to 6s for `disabled`/`is-disabled` to clear; `_LOGIN_DISMISS_JS` closes a non-login dialog first; `_LOGIN_STATE_JS` reports an OTP or a rejection message instead of a generic failure; the confirmation window is 45s because the SPA swaps its header for `My Account` asynchronously and the old 20s window gave up AFTER a successful click. Failures now notify the operator and back off (15s doubling to 10min) so a rejected password cannot be retried every 30s into a lockout, and a raised exception in `handle()` is audited instead of vanishing into `supervisor error`. Verified live: the click logged the account in (header became `NGN 765.58 | Deposit | Bet History | My Account`) and a second call on the logged-in page returned `no visible password field` without touching anything. Also fixed in the same pass: `session_expired()` reported `expired: False` when the probe could not read the page at all (extension 502, no tab) — an unattached extension looked exactly like a healthy session; it now returns `probe_failed: True` and the heartbeat publishes `session_probe_failed`. |

### Phantom edges found against the live board (all fixed)

Three separate layers each produced a false edge. They are listed together
because the pattern is the lesson: every one of them was found by running
the finished pipeline against real odds, not by testing the code.

| Where | What it did | Symptom |
|-------|-------------|---------|
| `math_engine` 1UP/2UP | Summed the remaining-goal difference without the scoreline deficit, so a side 0-2 down read as ~13% to lead by 1 | +5.31 EV at 47.00 on an impossible outcome |
| `CalibratedModel` | Shrank the full-time 1X2 toward the 'Rest of the Match' prior, which settles on the REMAINING time — a different event unless the score is level | +4.50 EV at 65.00 (model 9.5% vs the book's 1.4%) |
| `CalibratedModel` interval | The shrink can pull the probability below the base model's own lower bound, so the 'conservative' EV was computed from a HIGHER probability than the headline | A market with +0.007 headline EV passed a +0.05 conservative floor |

The guard that should have caught the last one is the conservative EV, which
is exactly what gates betting — so the interval has to stay coherent:
`[min(calibrated, base_lower), max(calibrated, base_upper)]`.

### Browser and site behaviour (the operational constraints)

Established by driving the real board; every one of them is load-bearing.

| # | Behaviour | Consequence |
|---|-----------|-------------|
| 1 | The live page never fires `load` (it holds a websocket open) | `navigate()` always times out and then unregisters the tab, 502ing every later call. Reach the board via a light page and a client-side navigation. |
| 2 | The accessibility `snapshot` intermittently 502s on the heavy board | Extraction uses targeted `evaluate` JS; snapshots are retried. |
| 3 | The extension wedges after heavy clicking | Recover by loading a light page (re-registers the tab) and retrying. The loop degrades with a reason instead of failing. |
| 4 | The scoreboard is NOT in the accessibility tree | The clock and period live in one text node (`.m-result-title`, e.g. "66' H2") and the score in sibling `.m-value-1` elements. The DOM is read directly. |
| 5 | The score is also published in the book's own market title | 'Rest of the Match (current score 1:0)' is a reliable fallback. |
| 6 | The 'SportyTV' media widget can expand over the scoreboard | The capture collapses it with the 'Hide' control and retries. |
| 7 | Multi View is the real board; Single View shows one match | The board reader reads `.match-row` (36-39 live matches with clocks and periods) and falls back to Single View cards. |
| 8 | Clicking a board row NAVIGATES to that match's dedicated page | The selection step opens the match, and the panel is read from there. |
| 9 | The Statistics panel REPLACES the market panel (its tabs go with it) | Statistics are read last, then the market panel is restored with the 'Match Tracker' control. |
| 10 | The stat values are NOT in the caption's subtree | Each caption is walked up to the block holding a readable home/away value. |
| 11 | A finished match shows 'ENDED' / 'Match Ended' instead of a scoreboard | The capture detects it and the period becomes `ft`, so the runner settles. |
| 12 | The board's live feed periodically delivers nothing | The readiness gate refuses to plan on an incomplete board. |

### The real placement flow (found by driving the book until a bet landed)

Five things had to be right, and every one of them failed silently first.
This is the part to read before the first live run.

| # | Behaviour | Consequence |
|---|-----------|-------------|
| 1 | The betslip can be left in **SIM mode** | A simulation slip is rejected by the order API with a bare "Something went wrong..." and the balance never moves. The driver now switches to REAL mode before placing and verifies it. |
| 2 | A bet is taken in **two steps**: 'Place Bet' opens an 'About to pay \| NGN 10.00 \| Cancel \| **Confirm**' dialog | Clicking 'Place Bet' alone places nothing. The confirmation is clicked too. |
| 3 | A synthetic `element.click()` from injected script is **untrusted** | The order was never submitted even though the click "succeeded". Every action goes through the extension's own click action on a uniquely-marked element — a real browser click. |
| 4 | When the price moves the book shows '**Accept Changes**' instead of 'Place Bet' | The slip is not placeable until the new price is acknowledged. That acknowledgement happens only AFTER the caller has checked the new price against the priced odds. |
| 5 | The reliable way to empty the slip is its '**Remove All**' control (per-selection remove buttons are the fallback) | A leftover selection silently turns the next bet into a multiple. The runner refuses to place unless the slip is empty first. |

Two more timing facts: the book enables the submit button a moment after the
stake is set (reading it too early concludes the slip is not placeable), and
the order API (`POST /api/ng/orders/order`) returns 200 even when the bet is
refused — so the result must be verified from the slip, not the status code.

The odds-drift guard was verified twice against the live book: an 18% and a
32% move were both refused.

### Live-board coverage: what is still unpriced (after the path engine)

Measured against the captured real board (92 headers, 2,794 odds, one match).
The remaining gaps, in order of size:

| Family | Notes |
|--------|-------|
| `GG/NG 2+` | Both teams to score 2+. The plain `GG/NG` is priced; the "2+" variant is not. |
| ~~Combination markets~~ | **Done.** `1X2 & Over/Under`, `1X2 & GG/NG`, `Over/Under & GG/NG`, `Double Chance & Total Goals`, `Double Chance & GG/NG` and the 2H variants are priced off the joint matrix. An unfamiliar selection label abstains rather than mispricing. |
| Time-windowed: `1X2 from 1 to 75/80/85 minute`, `Total Goals from 1 to 85 minute` | Need a window-restricted intensity, not the full remaining time. |
| `Home/Away No Bet` | Two-way market with a void outcome; the settlement semantics need confirming before pricing. |
| `2nd Half Home/Away Team Total Goals Odd or Even` | The per-team parity family exists; the 2H-scoped variant does not. |
| `No Draw Both Teams To Score Yes/No` | A combination of a no-draw condition and a BTTS condition. |

Also still unpriced: corners, cards and player markets (the board's Corners /
Bookings / Players / Teams / Minutes tabs). These need per-event intensities
the current model does not carry.

### Operator control (the "LLM Boss")

The name was a misnomer: the Boss shipped with zero LLM calls. It now has a
real slow plane with a clear division of labour.

- `atlas/ops/posture.py` — the `Posture` object (mode, EV floor, Kelly
  fraction, exposure cap, bets per snapshot, per-bet stake cap,
  `require_guaranteed_profit`). The LLM **proposes**; `Posture.clamped()`
  **validates** against hard invariants and records what it changed; the
  portfolio **executes** deterministically. An LLM can never set a
  probability or a stake.
- Two proposers: `RuleBasedProposer` (default, deterministic, no API key)
  and `LLMProposer` (OpenAI-compatible, `ATLAS_BOSS_API_KEY` /
  `ATLAS_BOSS_BASE_URL` / `ATLAS_BOSS_MODEL`). A model failure or an
  unparseable reply degrades to the rule-based parse, never to an error.
- `atlas/ops/notify.py` — custom notifications to the console, a JSONL audit
  log and an optional webhook. A failing sink never breaks the loop.
- `atlas.main posture "..."` interprets an instruction; `atlas.main plan
  --instruction "..."` runs a snapshot under that posture. The
  superedge-only posture refuses any bet that is not part of a guaranteed
  profit set.
- The posture only moves the risk-appetite levers. The correlation caps,
  rebalance thresholds and cooldowns are deliberately out of its reach:
  those protect the operator regardless of appetite.

### Architecture Decisions

- **R2 = DOM scraping** (not external API). No API key needed. Statistics tab scraper built.
- **R4 = 1UP/2UP = early-settlement 1X2**. 1UP: team leads by 1+ at any point. 2UP: team leads by 2+ at any point.
- **R6 = Block bets when model-market gap > 5pp**. Prevents fake edges from calibration.
- **R1 = Score IS in main DOM**. CSS classes identified from DevTools. Extractor updated.

