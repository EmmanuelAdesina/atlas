# Atlas Engineering Context

Source of truth for how Atlas is built right now. Read this before changing code.

## Objective

Proprietary, AI-driven live sports betting operation. First venue: **SportyBet**
via a desktop browser (Kimi WebBridge). Mode: **one selected live match**,
live stats + live odds as primary inputs, value evaluated mathematically,
execution only when evidence and risk constraints justify it.

## Non-negotiables

- Observations vs estimates vs assumptions are kept apart. Unknowns stay
  `None` and produce abstention, never invented numbers.
- Raw snapshots are archived BEFORE mapping (append-only, `data/raw/`).
- Bookmaker prices are stored as observed decimal odds; Atlas probabilities
  live in the estimate layer (`atlas/live/estimate.py`).
- Suspended/locked selections are never priced by Atlas; they land in
  `skipped` with reasons so availability stays auditable.
- Correlation: one bet per correlated group, max 3 per match.
- V1 is observe-only (`allow_execution=False`). No bet placement code path exists.

## Layout

- `atlas/browser/webbridge.py` — WebBridge adapter, documented actions only.
  Read + navigate. No execution API by design.
- `atlas/collectors/sportybet.py` — archives browser-supplied snapshots.
- `atlas/mappers/sportybet_mapper.py` — snapshot -> `SportMarket` (decimal odds,
  line, period, status, provenance).
- `atlas/models/sport.py` — `SportMarket`/`SportSelection` (+implied/overround/fair).
- `atlas/live/extract.py` — snapshot tree -> markets + clock/score/period.
  Lock == absence of price.
- `atlas/live/observe.py` — `LiveMatchState` (freshness, red-card state).
- `atlas/live/estimate.py` — `Estimate` + `MarketModel` protocol,
  `NoMoreGoalsBaseline` (narrow Under family, confidence capped low),
  `check_implications` (cross-market consistency).
- `atlas/live/loop.py` — breakeven/EV/de-vig/rank-filter math.
- `atlas/live/decisions.py` — policy gate + JSONL decision log.
- Legacy Polymarket files remain on disk but are NOT V1 (may become a venue later).

## What works (tested)

- Snapshot ingest -> canonical markets with de-vigged fair probs (`discover`).
- Real snapshot-tree extraction incl. locks, clock `90:00 +03:42`, score (tests
  run on the recorded Fortuna/Stuttgart read).
- Baseline estimates with abstention paths; implication checks; policy gate
  (duplicates, exposure caps, confidence bar); decision JSONL.
- `live-demo` replay of the recorded 90+3' observation: Under 1.5 EV +0.183
  but confidence low -> PASS; implications consistent; FT 0:1 recorded
  separately as settlement, never as input.

## Not implemented

- Live polling loop against the browser (adapter exists, loop wiring next).
- Minute/corner/booking/player models (baseline is goals-Under only).
- Calibration tracking over settled history (log schema ready, analysis pending).
- Any bet placement, cashout handling, accepted-odds verification.
- Claimed edge: none. The baseline informs and calibrates; it must earn stakes.

## Commands

- `python -m pytest -q` — full suite.
- `python -m atlas.main discover --snapshot <markets.json> --source-url <url>`
- `python -m atlas.main live-demo --fixture tests/fixtures/fortuna_stuttgart_90plus.json --log data/decisions/demo.jsonl`
