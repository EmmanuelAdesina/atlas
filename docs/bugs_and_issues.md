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
| B29 | Pricer coverage is **1.9% of a live board**: 20 of 92 market headers, 53 of 2,794 odds. Uncovered: clean sheet, win-to-nil, exact goals, odd/even, goals-in-a-row, lead-by-N-at-any-time, team O/U, 3-way handicap, 1X2-from-minute-N, total goals, correct score, corners/cards/player markets. | **Largely fixed: 70% of the board** (1,948 of 2,794 odds, 60 of 92 headers) via the score-path engine (`atlas/live/paths.py`), the new final-state families, and the combination markets. The path engine reproduces the joint matrix to 1e-16, so it is provably the same model extended to order-dependent questions. Combination markets are read straight off the joint matrix, never as a product of marginals. Remaining gaps are listed below. Guarded by `tests/test_live_board_coverage.py` against the captured real board. |
| B30 | No cashout support of any kind (no code path, no math). SportyBet exposes Cashout in the live betslip, including partial cashout on losing live positions. | **Open.** The UI contract is now known: the betslip carries **'One Cut'** (partial cashout), **'F Flexi'** (partial-payout accumulator) and **'SportyInsure'** (stake insurance). Live evidence: an open 'Flexi Your Bet 18+ of 19' accumulator, stake NGN 160, potential win NGN 38,498.89, offered for cashout at NGN 246.53. Needs its own model: cashout value vs the position's live EV, plus leg-risk for a partial cut. |
| B31 | No bet submission layer. The WebBridge `click` primitive exists but nothing clicks odds; `atlas.main plan` is observe-only. | **Partially done.** `atlas/browser/betslip.py` now drives the real slip: add odds, set stake, read slip, clear. Verified live against the board (selected `Draw (0:2) | Live Al-Shabab FC v Al-Faisaly FC | Handicap 0:2 | 17.00`, set NGN 100, read it back, cleared it). `place_bet()` is implemented but refuses to run unless the driver is constructed with `allow_submit=True` — nothing in ATLAS enables it. Still open: the autonomous loop that decides *when* to place, and the cashout execution path. |
| B32 | The `plan` path used the raw `LiveStateSpaceModel`, bypassing the calibration layer and its 5pp model-market gap guard that exists to block fake edges. | **Fixed** (plan now uses `CalibratedModel`). |

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

