"""Atlas agent plane — AI-first operation.

This plane REPLACES the deterministic decision core with an opencode agent
that owns pricing, reasoning, and execution. It uses the live venue page
through the WebBridge daemon (the agent explores markets and reads the live
match statistics from the tab itself), then places bets by driving the same
logged-in session.

What survives from the deterministic era, and why:
  - the append-only ledger (ops/ledger.py): the evidence product. The agent
    is obligated to record every placed bet here; nothing in this plane
    reprices or refuses its calls.
  - single-shot execution culture: the tab tool never retries a money
    action; a wedged page is reported, not clicked twice.
  - observe-only by construction: no code path arms money interaction
    without ATLAS_AGENT_MONEY=1 in the operator's environment.

Design boundary mirroring atlas/browser/webbridge.py: this plane knows
about browser mechanics and process supervision only. Domain reasoning
lives in the agent's model, not in Python.
"""
