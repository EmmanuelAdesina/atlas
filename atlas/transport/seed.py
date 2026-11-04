"""Seed the session vault from an operator-supplied cookie export.

Usage:
  python -m atlas.transport.seed --cookies cookies.json [--path data/ops/session.json]

The cookies file is a JSON object: {"cookieName": "value", ...}. It is NOT
created by this program — export it from a logged-in browser session
(DevTools → Application → Cookies, or one CDP Network.getAllCookies pull).
After seeding, keep or delete the export file as you see fit; the vault is
the runtime artifact. The file is read, never printed, and its path is not
logged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlas.transport.session import seed_vault, VAULT_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cookies", required=True, help="JSON object of cookie name→value")
    parser.add_argument("--path", default=str(VAULT_PATH))
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.cookies).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("seed: cookies file must be a JSON object", file=sys.stderr)
        return 2
    cookies = {str(k): str(v) for k, v in raw.items() if str(v).strip()}
    if not cookies:
        print("seed: no non-empty cookies found — refusing to seed", file=sys.stderr)
        return 2
    out = seed_vault(cookies, Path(args.path))
    print(f"seed: vault written to {out} (0600). Cookies are never logged.")
    print("verify with: python -m atlas.transport.session_probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
