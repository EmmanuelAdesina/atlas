# Atlas Engineering Context

Source of truth for how Atlas is built right now. Read this before changing code.

## Objective

Proprietary, AI-driven live sports betting operation. First venue: **SportyBet Nigeria (`sportybet.com/ng`, NGN)** via a desktop browser (Kimi WebBridge). Mode: **one selected live match**, live stats + live odds as primary inputs, value evaluated mathematically, execution only when evidence and risk constraints justify it.

## Two-Plane Architecture (Fast Plane vs. Slow Plane)

- **Fast Plane (Hot Path — `< 1ms` pure deterministic math, zero LLM calls)**:
  `WebBridge Snapshot -> extract.py -> observe.py -> math_engine.py (Bivariate Poisson / Dixon-Coles + Shin De-Vig) -> estimate.py -> decisions.py`.
- **Slow Plane (`atlas/live/boss.py` — Executive Portfolio Manager / CRO Control Plane)**:
  Reads live telemetry, PnL/calibration reports, and operational anomalies asynchronously; enforces the **Asymmetric Ratchet Law** (can tighten `min_ev`, reduce `kelly_fraction`, lower `max_exposure_per_match`, engage `kill_switch`, block matches/markets, or inject bounded prior shifts $\Delta\lambda \in [-0.25, +0.25]$; can NEVER loosen hard safety floors or enable live execution autonomously).

## Non-Negotiables

- Observations vs. estimates vs. assumptions are kept strictly apart. Unknowns stay `None` and produce abstention, never invented numbers.
- Raw snapshots are archived BEFORE mapping (append-only, `data/raw/`).
- Bookmaker prices are stored as observed decimal odds; Atlas probabilities live in `atlas/live/math_engine.py` and `atlas/live/estimate.py`.
- Suspended/locked selections are never priced by Atlas; they land in `skipped` with reasons so availability stays auditable.
- Correlation: canonical `(market_key, selection)` pairs are mapped to fundamental risk factors (`home_goals`, `away_goals`, `total_goals`, `low_tempo_under`, `home_supremacy`, `away_supremacy`, `home_corners`, `away_corners`) — max 1 bet per correlated factor group, max 3 per match.
- V1 is observe-only (`allow_execution=False`). No bet placement code path exists.

## Layout

- `atlas/browser/webbridge.py` — WebBridge adapter, documented actions only. Read + navigate. No execution API by design.
- `atlas/collectors/sportybet.py` — archives browser-supplied snapshots.
- `atlas/mappers/sportybet_mapper.py` — snapshot -> `SportMarket` (decimal odds, line, period, status, provenance).
- `atlas/models/sport.py` — `SportMarket`/`SportSelection` (+implied, overround, Proportional/Power/Shin de-vigged `fair_probs`, `shin_z` adverse-selection index).
- `atlas/live/extract.py` — snapshot tree -> markets + full `LiveMatchState` (`extract_live_match_state`: clock, score, HT score, red/yellow cards, corners, shots on target, attempts, saves, fouls, free kicks, offsides, possession, substitutions, NGN balance).
- `atlas/live/observe.py` — `LiveMatchState` & `TeamStats` (freshness, red-card state, live match stats).
- `atlas/live/math_engine.py` — Sub-millisecond pure-Python quantitative kernel:
  - De-vigging (`devig_proportional`, `devig_power`, `devig_shin`),
  - Non-linear piecewise clock hazard $\Lambda(t)$ & smooth survival stoppage-time estimator (`estimate_remaining_minutes`),
  - Empirical Bayes live-stat shrinkage + Markov red-card & score-state hazard multipliers (`compute_posterior_intensities`),
  - Coherent Bivariate Poisson / Dixon-Coles remaining-goal matrix (`build_remaining_score_matrix`) & $O(N)$ marginal surface pricing (`price_from_joint_matrix`) across 14+ SportyBet market families (`1x2`, `1x2_1h`, `1x2_2h`, `rest_of_match_1x2`, `double_chance`, `dnb`, `ou_*`, `ou_1h_*`, `ou_2h_*`, `home_ou_*`, `away_ou_*`, `btts`, `next_goal`, `handicap_*`, `ah_*`),
  - Exact push / quarter-line EV (`expected_value_detailed`) & conservative fractional Kelly (`fractional_kelly_stake`).
- `atlas/live/estimate.py` — `Estimate` (with `[p_lower, p_upper]` uncertainty bounds) + `MarketModel` protocol, `LiveStateSpaceModel`, `NoMoreGoalsBaseline`, and expanded `check_implications` (cross-market Fréchet/logical consistency).
- `atlas/live/loop.py` — breakeven/EV/de-vig/canonical correlation-group filter (`selection_correlation_groups`, `rank_and_filter`).
- `atlas/live/decisions.py` — SportyBet Nigeria (`NGN`) policy gate, fractional Kelly NGN stake sizing, factor correlation guard, and JSONL decision log.
- `atlas/live/calibration.py` — Post-settlement grading (`grade_decisions`): Brier score, log-loss, calibration gap, realized ROI by market family and confidence grade.
- `atlas/live/boss.py` — Slow-plane Executive Portfolio Manager / CRO (`BossController`, `diagnose_live_operation`) with asymmetric risk ratchets.

## Commands

- `python3 -m pytest -q` — full 44-test suite.
- `python3 -m atlas.main discover --snapshot <markets.json> --source-url <url>`
- `python3 -m atlas.main live-demo --fixture tests/fixtures/fortuna_stuttgart_90plus.json --log data/decisions/demo.jsonl`
- `python3 -m atlas.main boss --fixture tests/fixtures/fortuna_stuttgart_90plus.json --log data/decisions/boss_audit.jsonl`
