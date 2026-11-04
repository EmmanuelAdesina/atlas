"""atlas.transport — browserless external transport for SportyBet.

Sensors only. Nothing in this package can place, confirm or cancel an
order except atlas.transport.executor.DirectOrderExecutor, which is disabled
by default and deliberately not wired into the live runner. Money still
moves through exactly one audited place per deployment mode, and the
ledger remains the system of record for stakes and their exact times.
See docs/browserless-transport-design.md.
"""
