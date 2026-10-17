# Atlas Operational Bugs & Issues

## Critical (blocks live operation)

### B1. Scoreboard/clock locked inside SportyTV iframe
- **What:** Live clock and score render inside a SportyTV video overlay iframe, not the main DOM.
- **Impact:** `LiveMatchState.clock_minute` and `score` are None → model abstains on everything. The entire loop is blind without these.
- **Evidence:** Boliviar vs Club Olimpia live test — odds extracted perfectly, clock/score returned None from both accessibility tree and DOM queries.
- **Fix needed:** Either (a) navigate WebBridge into the SportyTV iframe URL directly, (b) intercept the WebSocket/API feed that populates the scoreboard, or (c) use the SportyFM audio stream metadata if it carries score data.

### B2. Session expiry during live polling
- **What:** SportyBet session expired mid-test, page redirected to login.
- **Impact:** Any polling loop will go blind after session timeout. No detection or recovery.
- **Fix needed:** Session-health check (detect login redirect), alert operator, auto-refresh token if possible.

### B3. No live data API access from page context
- **What:** `fetch()` and XHR to `sportybet.com/api/...` from `evaluate` return `{"bizCode":19003,"message":"Forbidden"}`. The page's own requests succeed (they include auth headers/tokens), but injected JS cannot replicate them.
- **Impact:** Cannot pull structured match data programmatically from the page. Reduced to DOM scraping.
- **Fix needed:** Use WebBridge `network` capture to intercept the page's own API responses, or find the auth token in localStorage/cookies and inject it into fetch headers.

## High (math/model correctness)

### B4. Red card multiplier ignores timing
- **File:** `atlas/live/math_engine.py:425-429`
- **What:** A single red card gives `rc_h=0.68` (32% reduction) and `rc_a=1.32` (32% boost) regardless of when it occurred.
- **Impact:** At 90+3' with 2 minutes remaining, a red card has negligible impact on the match, but the model applies the full 32% swing. Overstates the effect of late red cards.
- **Fix:** Apply decay factor: `decay = min(1.0, remaining_minutes / 30.0)`. At 90+3', decay ≈ 0.07, so the multiplier is nearly 1.0.

### B5. Score-state urgency multiplier ignores timing
- **File:** `atlas/live/math_engine.py:438-445`
- **What:** Trailing team gets `ss_h=1.12` (12% urgency boost) regardless of clock minute.
- **Impact:** At 90+3', there's no time for urgency to manifest. The boost is meaningless but still applied.
- **Fix:** Gate by remaining time: `if remaining_minutes < 10: ss_multiplier = 1.0`.

### B6. Shrinkage weight too high for sparse stats
- **File:** `atlas/live/math_engine.py:409-411`
- **What:** `w_stat = minutes_played / (minutes_played + 65)` = 0.59 at 93.7'. But live stats are corners-only (no shots on target).
- **Impact:** Model gives 59% weight to a corner-only xG proxy, which is too aggressive. Corners are a weak proxy for expected goals.
- **Fix:** When `shots_on_target` is None, reduce shrinkage: `w_stat *= 0.5`.

### B7. live_rate clamp too aggressive
- **File:** `atlas/live/math_engine.py:413-414`
- **What:** Corner-only xG proxy gives `live_rate_h=0.10` and `live_rate_a=0.03`, clamped to 0.30 and 0.25.
- **Impact:** Model treats very low corner counts as "average" attacking output. The clamp [0.30, 3.80] is too aggressive for sparse stats.
- **Fix:** Relax to [0.10, 3.80] or make conditional on stat availability.

### B8. rho inconsistency between prior inference and model
- **File:** `atlas/live/math_engine.py:343`
- **What:** `infer_prematch_lambdas_from_1x2` uses `rho=-0.08` while `LiveStateSpaceModel` uses `rho=-0.06`.
- **Impact:** Small but systematic bias in the prior.
- **Fix:** Use `rho=-0.06` consistently.

## Medium (calibration/overconfidence)

### B9. Model is systematically more extreme than the market
- **What:** Backtest on Fortuna/Stuttgart showed model probabilities are more extreme than market-implied in both directions — overconfident in favorites (Under 1.5: model 0.934 vs market 0.773) and overconfident against longshots (Draw 2H: model 0.025 vs market 0.090).
- **Impact:** EV estimates are inflated. The +0.148 EV on Under 1.5 becomes -0.049 at the market-implied probability.
- **Fix:** Add a calibration layer that shrinks model probabilities toward market-implied when confidence is "low". Track Brier score and log loss.

### B10. Stoppage hazard is flat
- **File:** `atlas/live/math_engine.py:294-305`
- **What:** Stoppage hazard of 1.36 is applied uniformly regardless of how far into stoppage we are.
- **Impact:** In reality, hazard increases as stoppage progresses (teams take more risks). Flat hazard underestimates late-goal probability.
- **Fix:** Make hazard a function of stoppage minute: `hazard = 1.0 + 0.5 * (stoppage_elapsed / stoppage_total)`.

### B11. Stoppage estimate doesn't condition on red card timing
- **File:** `atlas/live/math_engine.py:236`
- **What:** Red card coefficient of 0.55 adds 0.55 minutes to expected stoppage regardless of when the card occurred.
- **Impact:** A red card at 10' adds as much stoppage as one at 85'. The estimate should be conditional on card timing.
- **Fix:** Weight the red card coefficient by `(90 - card_minute) / 90`.

### B12. Matrix truncation redistributes tail probability
- **File:** `atlas/live/math_engine.py:531-538`
- **What:** `max_goals=8` truncation means tail probability is redistributed across the matrix.
- **Impact:** For very small mu values (0.028, 0.042), negligible. For larger mu, could bias the distribution.
- **Fix:** Increase `max_goals` to 10+ or use a proper tail correction.

## Low (operational polish)

### B13. Network capture filter is case-sensitive
- **What:** `network cmd=list filter="liveChannel"` returned 0 requests. The filter may need exact case or the request was made before capture started.
- **Impact:** Makes API interception unreliable.
- **Fix:** Try filter without case sensitivity, or capture all requests and filter client-side.

### B14. Sub-ms pricing test is machine-dependent
- **What:** `test_fast_plane_surface_pricing_runs_sub_millisecond` asserts < 2ms wall-clock. Passes on fast machines, fails on this Windows box (9.25ms).
- **Impact:** Flaky CI. Not a logic bug.
- **Fix:** Warm the surface before timing, or use a relative threshold.

### B15. Correct Score market header shows current score
- **What:** DOM showed `m-table-header-title :: Correct Score [3:2]` — this appears to be the current score displayed as a market header, not a betting market.
- **Impact:** Could be used as a score source, but it's ambiguous and may not always be present.
- **Fix:** Investigate whether this is a reliable score source or just a UI artifact.
