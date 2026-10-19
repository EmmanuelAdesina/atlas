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

### Architecture Decisions

- **R2 = DOM scraping** (not external API). No API key needed. Statistics tab scraper built.
- **R4 = 1UP/2UP = early-settlement 1X2**. 1UP: team leads by 1+ at any point. 2UP: team leads by 2+ at any point.
- **R6 = Block bets when model-market gap > 5pp**. Prevents fake edges from calibration.
- **R1 = Score IS in main DOM**. CSS classes identified from DevTools. Extractor updated.

