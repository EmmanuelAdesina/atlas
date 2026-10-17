# Atlas

**Live, Single-Match Quantitative Sports Betting Intelligence (SportyBet Nigeria V1)**

Atlas is a quantitative decision intelligence system built around a strict separation between a low-millisecond mathematical Fast Plane and an asymmetric-ratchet Executive Control Slow Plane. The regression target prices a 21-selection surface in under 2 ms after setup; optional pre-match odds inversion is precomputed outside that hot path.

## Quickstart

```bash
# Run full test suite
python3 -m pytest -v

# Replay the recorded Fortuna Cologne vs VfB Stuttgart II (90+3') live observation
python3 -m atlas.main live-demo \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/demo.jsonl

# Output structured Slow-Plane Executive Boss diagnostics & post-settlement calibration JSON
python3 -m atlas.main boss \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/boss_audit.jsonl
```

See [`docs/mission.md`](docs/mission.md) and [`docs/engineering.md`](docs/engineering.md) for architecture and mathematical invariants.
