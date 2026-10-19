"""Custom notification dispatch for the Boss.

The operator wants ATLAS to talk back: posture changes, sure-thing arbs, the
daily stop, drift, an open position needing attention. Notifications go to
three sinks, all optional and all failing soft — a notification channel
going down must never take the betting loop down with it:

  - the console (rich), for a live dashboard;
  - a JSONL audit file, so every notification is replayable;
  - an HTTP webhook, for a phone push / Telegram / Slack style endpoint.

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
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.webhook_url = webhook_url or os.environ.get("ATLAS_WEBHOOK_URL", "") or None
        self.webhook_token = (
            webhook_token or os.environ.get("ATLAS_WEBHOOK_TOKEN", "") or None
        )
        self.console = console
        self.timeout = timeout
        self.sent: list[Notification] = []

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
        for sink in (self._to_console, self._to_file, self._to_webhook):
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
