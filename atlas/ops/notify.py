"""Custom notification dispatch for the Boss.

The operator wants ATLAS to talk back: posture changes, sure-thing arbs, the
daily stop, drift, an open position needing attention. Notifications go to
four sinks, all optional and all failing soft — a notification channel going
down must never take the betting loop down with it:

  - the console (rich), for a live dashboard;
  - a JSONL audit file, so every notification is replayable;
  - an HTTP webhook, for a phone push / Telegram / Slack style endpoint;
  - SMTP email (ATLAS_SMTP_* env), for unattended-operation alerting —
    WARN and above by default, deduped per title so a stuck condition
    emails once per cooldown instead of every cycle.

Every notification carries a severity and a structured payload, so the
message can be rendered differently per channel while the content stays
identical.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx


class Severity(str, Enum):
    INFO = "info"
    GOOD = "good"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Notification:
    title: str
    body: str
    severity: Severity = Severity.INFO
    payload: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "severity": self.severity.value,
            "payload": self.payload,
            "created_at": self.created_at,
        }

    def to_text(self) -> str:
        return f"[{self.severity.value.upper()}] {self.title}: {self.body}"


class Notifier:
    """Fan a notification out to the configured sinks.

    A sink that raises is logged and skipped; the others still receive it.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        webhook_url: Optional[str] = None,
        webhook_token: Optional[str] = None,
        console: bool = True,
        timeout: float = 10.0,
        email_dedup_s: float = 1800.0,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.webhook_url = webhook_url or os.environ.get("ATLAS_WEBHOOK_URL", "") or None
        self.webhook_token = (
            webhook_token or os.environ.get("ATLAS_WEBHOOK_TOKEN", "") or None
        )
        self.console = console
        self.timeout = timeout
        self.sent: list[Notification] = []
        # --- email sink (SMTP) ------------------------------------------------
        # Unattended operation needs an alert channel that reaches a phone:
        # the heartbeat proves the loop ran, but only alerting proves someone
        # HEARD. Config via env: ATLAS_SMTP_HOST/PORT/USER/PASS/TO. Only WARN
        # and above email by default (the INFO per-cycle refusals would flood
        # any inbox); the same title re-fires are deduped for a cooldown so a
        # stuck condition emails once, not every 30s cycle.
        self.smtp_host = os.environ.get("ATLAS_SMTP_HOST", "") or None
        self.smtp_port = int(os.environ.get("ATLAS_SMTP_PORT", "587") or 587)
        self.smtp_user = os.environ.get("ATLAS_SMTP_USER", "") or None
        self.smtp_pass = os.environ.get("ATLAS_SMTP_PASS", "") or None
        self.smtp_to = [
            a.strip() for a in (os.environ.get("ATLAS_SMTP_TO", "") or "").split(",") if a.strip()
        ]
        min_sev = (os.environ.get("ATLAS_SMTP_MIN_SEVERITY", "warn") or "warn").lower()
        self.email_min_severity = (
            Severity(min_sev) if min_sev in Severity._value2member_map_ else Severity.WARN
        )
        self.email_dedup_s = email_dedup_s
        self._email_last: dict[str, float] = {}

    def _email_enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_pass and self.smtp_to)

    def _to_email(self, n: Notification) -> None:
        if not self._email_enabled():
            return
        order = {Severity.INFO: 0, Severity.GOOD: 0, Severity.WARN: 1, Severity.CRITICAL: 2}
        if order[n.severity] < order[self.email_min_severity]:
            return
        now = time.monotonic()
        last = self._email_last.get(n.title, 0.0)
        if now - last < self.email_dedup_s:
            return
        import smtplib
        from email.header import Header
        from email.mime.text import MIMEText

        msg = MIMEText(
            f"{n.body}\n\npayload: {json.dumps(n.payload, ensure_ascii=False, default=str)}\n"
            f"at: {n.created_at}\n",
            "plain",
            "utf-8",
        )
        prefix = {"info": "ATLAS", "good": "ATLAS OK", "warn": "ATLAS WARN", "critical": "ATLAS CRITICAL"}
        msg["Subject"] = f"{prefix.get(n.severity.value, 'ATLAS')}: {n.title}"
        msg["From"] = self.smtp_user
        msg["To"] = ", ".join(self.smtp_to)
        if self.smtp_port == 465:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=self.timeout) as s:
                s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as s:
                s.starttls()
                s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
        self._email_last[n.title] = now

    # --- individual sinks -------------------------------------------------

    def _to_console(self, n: Notification) -> None:
        if not self.console:
            return
        try:
            from rich.console import Console

            console = Console()
            style = {
                Severity.INFO: "cyan",
                Severity.GOOD: "bold green",
                Severity.WARN: "yellow",
                Severity.CRITICAL: "bold red",
            }[n.severity]
            console.print(f"[{style}]{n.title}[/{style}] {n.body}")
        except Exception:
            print(n.to_text())

    def _to_file(self, n: Notification) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")

    def _to_webhook(self, n: Notification) -> None:
        if not self.webhook_url:
            return
        headers = {"Content-Type": "application/json"}
        if self.webhook_token:
            headers["Authorization"] = f"Bearer {self.webhook_token}"
        httpx.post(
            self.webhook_url,
            json=n.to_dict(),
            headers=headers,
            timeout=self.timeout,
        )

    # --- public API -------------------------------------------------------

    def send(
        self,
        title: str,
        body: str,
        severity: Severity = Severity.INFO,
        **payload: Any,
    ) -> Notification:
        """Deliver a notification. Never raises."""
        n = Notification(
            title=title, body=body, severity=severity, payload=dict(payload)
        )
        self.sent.append(n)
        for sink in (self._to_console, self._to_file, self._to_webhook, self._to_email):
            try:
                sink(n)
            except Exception:
                # A broken sink must not break the loop.
                continue
        return n

    def info(self, title: str, body: str, **payload: Any) -> Notification:
        return self.send(title, body, Severity.INFO, **payload)

    def good(self, title: str, body: str, **payload: Any) -> Notification:
        return self.send(title, body, Severity.GOOD, **payload)

    def warn(self, title: str, body: str, **payload: Any) -> Notification:
        return self.send(title, body, Severity.WARN, **payload)

    def critical(self, title: str, body: str, **payload: Any) -> Notification:
        return self.send(title, body, Severity.CRITICAL, **payload)
