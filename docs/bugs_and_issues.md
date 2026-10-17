# Atlas Operational Bugs & Issues

This list was originally assembled from a browser/live-feed investigation. Statuses below distinguish code fixes from operational issues that still need a real SportyBet/browser integration test. Quantitative changes are regression-tested, but that does **not** establish out-of-sample accuracy or profitability.

## Critical — operational prerequisites still open

### B1. Scoreboard/clock locked inside SportyTV iframe — unresolved
- **What:** Live clock and score rendered inside a SportyTV video overlay iframe rather than the main DOM in the recorded test.
- **Impact:** If the observer cannot extract score and clock, the model cannot safely price live markets and should abstain.
- **Next step:** Test direct iframe navigation, capture the page's scoreboard API/network feed, or validate another trusted score source. Do not infer live score from an ambiguous market header.

### B2. Session expiry during live polling — unresolved
- **What:** The recorded SportyBet session expired mid-test and redirected to login.
- **Impact:** Polling can lose market data without an operator noticing.
- **Next step:** Add a session-health signal, detect login redirects, alert the operator, and validate any refresh mechanism. No automatic live execution is enabled.

### B3. No live data API access from page context — unresolved
- **What:** Direct `fetch()`/XHR calls from the inspected page context were rejected while the page's own requests succeeded.
- **Impact:** Structured match data may remain unavailable if network response capture is not integrated.
- **Next step:** Validate supported browser-bridge network capture and authentication handling. Do not scrape or persist credentials as a workaround.

## High — math/model correctness

### B4. Red-card multiplier ignored timing — addressed in model
- Red-card intensity adjustments now decay with estimated remaining playing time (bounded by `remaining_minutes / 30`). A late dismissal therefore has less effect on remaining-goal intensity.
- Regression coverage checks early-versus-late card effects. The coefficient and decay horizon remain heuristic.

### B5. Score-state urgency ignored timing — addressed in model
- Leading/trailing multipliers are symmetric with respect to home/away and are applied only in late match states with at least ten estimated minutes remaining.
- Regression coverage checks both score directions and the late-game gate. The multipliers are heuristic, not fitted to a historical sample.

### B6. Shrinkage too high for sparse live stats — addressed in model
- Live event evidence is shrunk separately by team; when SOT is absent, that side receives only 20% of the ordinary time-based evidence weight. Possession alone does not create shot pressure.
- Regression coverage checks sparse inputs. The proxy weights and shrinkage horizon still need empirical calibration.

### B7. Sparse-stat live-rate clamp — partially addressed
- The lower clamp was relaxed from 0.30 to 0.10 goals per 90, avoiding the old artificial floor for very low corner/shot proxies. The upper bound remains 3.80.
- The proxy and remaining clamp bounds are still assumptions pending an out-of-sample fit.

### B8. Rho mismatch in prematch inversion — addressed
- Prematch 1X2 inversion now receives the same Dixon–Coles rho as the live model; the default is `-0.06` in both paths.
- Covered by prior-inversion/model regression tests.

## Medium — calibration and distribution tails

### B9. Model probabilities more extreme than market — mitigated, not empirically validated
- `CalibratedModel` is now used in the state-space `live-demo` and `boss` paths. It only blends against the fair probability for the exact same canonical market and selection; it does not substitute a related market with different settlement scope.
- Push/split-settlement estimates are left unchanged, and ordinary settlement scenario bounds are preserved through the blend. Regression coverage tests exact-market matching and split-settlement skipping.
- The confidence-based shrinkage percentages remain a heuristic. The cited single-fixture comparison is not enough to establish calibration; broad chronological out-of-sample Brier/log-loss testing is still required.

### B10. Stoppage hazard was flat — addressed in model
- Second-half stoppage hazard now ramps with elapsed stoppage, while the hazard integral remains normalized so a full-match kickoff maps to the full-match prior.
- Regression coverage checks the ramp and full-match normalization. The ramp shape is still heuristic.

### B11. Stoppage estimate ignored red-card timing — addressed in model
- The red-card contribution to the expected second-half stoppage baseline is weighted by the fraction of the half remaining when the card occurred. If the card minute is missing, the model falls back to 45'.
- Regression coverage covers card timing provenance/cache behavior; the stoppage coefficient still needs empirical validation.

### B12. Joint score-matrix truncation — reduced, residual tail approximation remains
- The live model uses a 0–10 goal support for ordinary intensities (0–6 for very small means); prematch inversion uses 0–12, rather than the previously noted 0–8 surface.
- The matrix is normalized over the retained support, so omitted tail mass is still redistributed. This is negligible for ordinary remaining-goal means but can matter for unusually large means; a proper tail treatment or explicit error bound remains future work.

## Low — operational polish

### B13. Network capture filter may be case-sensitive — unresolved
- A `liveChannel` filter returned no requests in the recorded browser test; this could reflect case sensitivity or capture starting too late.
- **Next step:** Validate case-insensitive/client-side filtering and capture timing in the supported browser bridge.

### B14. Fast-plane microbenchmark machine dependence — improved, still environment-sensitive
- The performance test now materializes the shared joint/scenario surface before timing cached selection pricing, matching the documented after-setup target.
- The test retains a wall-clock threshold, so CI/runtime contention can still affect it. Treat it as a performance guardrail, not a real-time service-level guarantee.

### B15. Correct Score header may display the current score — unresolved
- The recorded DOM showed a `Correct Score [3:2]` header, but this is ambiguous and may be a market/UI artifact.
- **Next step:** Validate against an independent scoreboard before treating it as a score source.
