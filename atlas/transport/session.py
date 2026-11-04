"""Session vault: SportyBet cookies without a browser.

Design (see docs/browserless-transport-design.md §7):

  * The vault is SEEDED, never minted. Seeding is an operator action — one
    CDP ``Network.getAllCookies`` pull or a manual export. No programmatic
    login exists here or is planned.
  * ``data/ops/session.json``, mode 0600, directory created 0700.
  * Cookie values are never logged, never telemetered, never embedded in
    error messages. The loggable view is :func:`redacted_headers`.
  * Validation is a cheap authenticated probe — an HTTP error or an
    auth-shaped bizCode flips the vault state to EXPIRED in memory; callers
    fail the whole cycle closed instead of reading an account that is not
    there.
  * Nothing here retries mutations or "repairs" a bad session. The only
    recovery is a re-seed, performed by the operator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

VAULT_PATH = Path(
    os.environ.get("ATLAS_SESSION_VAULT", "data/ops/session.json")
)

# Header profile captured from the live site (see
# tests/fixtures/api/realbetlist_request_headers.json, cookie removed).
# These are the exact shapes the book's edge expects; do not "improve" them.
DEFAULT_HEADER_PROFILE: dict[str, str] = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36"
    ),
    "accept": "application/json, text/plain, */*",
    "accept-language": "en",
    "platform": "web",
    "clientid": "web",
    "operid": "2",
    "sec-ch-ua": '"Google Chrome";v="155", "Chromium";v="155", "Not(A:Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
}

SECRET_KEYS = ("cookie", "set-cookie")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Cookie/auth values -> name + [redacted]. Safe for logs and errors."""
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SECRET_KEYS or "authorization" in key.lower():
            out[key] = "[redacted]"
        else:
            out[key] = value
    return out


@dataclass
class Session:
    """Decrypted-at-rest cookie jar + header profile. In-memory only."""

    cookies: dict[str, str]
    seeded_at: str
    header_profile: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HEADER_PROFILE))

    def cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def request_headers(self, referer: Optional[str] = None) -> dict[str, str]:
        headers = dict(self.header_profile)
        headers["cookie"] = self.cookie_header()
        if referer:
            headers["referer"] = referer
        return headers

    def redacted_headers(self, referer: Optional[str] = None) -> dict[str, str]:
        return redact_headers(self.request_headers(referer))


def seed_vault(
    cookies: dict[str, str],
    path: Path = VAULT_PATH,
    header_profile: Optional[dict[str, str]] = None,
) -> Path:
    """Write the vault atomically with 0600 perms. Operator-invoked only."""
    if not cookies:
        raise ValueError("refusing to seed an empty cookie jar")
    payload = {
        "version": 1,
        "seeded_at": _utcnow_iso(),
        "header_profile": header_profile or dict(DEFAULT_HEADER_PROFILE),
        "cookies": cookies,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass  # best effort on non-posix filesystems
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    os.chmod(path, 0o600)
    return path


def load_vault(path: Path = VAULT_PATH) -> Session:
    """Load the vault. Raises FileNotFoundError/ValueError — never invents one."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cookies = raw.get("cookies") or {}
    if not isinstance(cookies, dict) or not cookies:
        raise ValueError(f"session vault {path} carries no cookies")
    profile = raw.get("header_profile") or dict(DEFAULT_HEADER_PROFILE)
    # Enforce the captured profile's shape; operator-supplied extras are
    # dropped rather than trusted blindly.
    merged = dict(DEFAULT_HEADER_PROFILE)
    for key in DEFAULT_HEADER_PROFILE:
        if isinstance(profile.get(key), str):
            merged[key] = profile[key]
    return Session(
        cookies={str(k): str(v) for k, v in cookies.items()},
        seeded_at=str(raw.get("seeded_at") or "unknown"),
        header_profile=merged,
    )


def vault_age_days(session: Session) -> Optional[float]:
    try:
        seeded = datetime.fromisoformat(session.seeded_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    if seeded.tzinfo is None:
        seeded = seeded.replace(tzinfo=timezone.utc)
    return (now - seeded).total_seconds() / 86400.0
