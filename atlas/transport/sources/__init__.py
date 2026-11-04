"""Read-only sources built on the transport client.

Every source here is a SENSOR. None of them can place, confirm or cancel an
order, or move money. The executor lives behind its own boundary (§9 of
docs/browserless-transport-design.md) and is not imported by anything in
this package.
"""
