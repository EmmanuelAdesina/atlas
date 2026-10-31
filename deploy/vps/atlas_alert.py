#!/usr/bin/env python3
"""Watchdog & cron alert entrypoint.

Sends a WARN-level notification through atlas.ops.notify.Notifier, which
routes to the SMTP email sink (ATLAS_SMTP_* env) with its normal dedup and
severity gating. Payloads stay free of secrets.

Usage: atlas_alert.py "title" ["body"]
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.path.insert(0, "/opt/atlas")
    from atlas.ops.notify import Notifier

    title = sys.argv[1] if len(sys.argv) > 1 else "watchdog event"
    body = sys.argv[2] if len(sys.argv) > 2 else ""
    notifier = Notifier(
        log_path="/opt/atlas/data/decisions/notifications.jsonl", console=False
    )
    notifier.warn(title, body, source="watchdog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
