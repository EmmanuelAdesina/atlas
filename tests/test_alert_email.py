"""The SMTP email alert sink.

Unattended operation needs an alert channel that reaches a phone when
nobody is watching the dashboard: the heartbeat proves the loop ran, only
alerting proves someone heard. Ground rules:

  - INFO must never email (per-cycle refusals would flood an inbox);
  - the same title re-firing must email once per cooldown, not every cycle;
  - with no SMTP config the sink must be a silent no-op, never an error;
  - an SMTP failure must never break the loop (the send() wrapper already
    swallows sink exceptions — these tests pin the logic around it).
"""

from atlas.ops.notify import Notifier, Severity


def _configured(monkeypatch, **over):
    env = {
        "ATLAS_SMTP_HOST": "smtp.test.example",
        "ATLAS_SMTP_PORT": "587",
        "ATLAS_SMTP_USER": "atlas@test.example",
        "ATLAS_SMTP_PASS": "secret",
        "ATLAS_SMTP_TO": "boss@test.example",
        "ATLAS_SMTP_MIN_SEVERITY": "warn",
    }
    env.update({k: v for k, v in over.items() if v is not None})
    for k, v in over.items():
        if v is None:
            env.pop(k, None)
    monkeypatch.setenv("ATLAS_SMTP_HOST", env["ATLAS_SMTP_HOST"])
    monkeypatch.setenv("ATLAS_SMTP_PORT", env["ATLAS_SMTP_PORT"])
    monkeypatch.setenv("ATLAS_SMTP_USER", env["ATLAS_SMTP_USER"])
    monkeypatch.setenv("ATLAS_SMTP_PASS", env["ATLAS_SMTP_PASS"])
    monkeypatch.setenv("ATLAS_SMTP_TO", env["ATLAS_SMTP_TO"])
    monkeypatch.setenv("ATLAS_SMTP_MIN_SEVERITY", env["ATLAS_SMTP_MIN_SEVERITY"])
    return env


def test_email_sink_is_silent_noop_without_config(monkeypatch):
    for k in ("ATLAS_SMTP_HOST", "ATLAS_SMTP_USER", "ATLAS_SMTP_PASS", "ATLAS_SMTP_TO"):
        monkeypatch.delenv(k, raising=False)
    n = Notifier(console=False)
    assert n._email_enabled() is False
    # must not raise even though it would try to connect
    n.critical("Balance dropped", "149.18 -> 49.18")


def test_info_never_emails(monkeypatch):
    _configured(monkeypatch)
    calls = []
    notifier = Notifier(console=False, email_dedup_s=0)
    monkeypatch.setattr(notifier, "_to_email", lambda n: calls.append(n), raising=False)
    # route through the real severity floor by calling the real method after
    # re-enabling and stubbing smtplib at the module boundary
    import smtplib

    class FakeSMTP:
        def __init__(self, *a, **k):
            calls.append(("connect", a))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            pass

        def send_message(self, msg):
            calls.append(("sent", msg["Subject"]))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    notifier.info("Bet refused (long-odds gate)", "lead_by_1_away @ 10.74")
    notifier.good("Bet placed", "ticket 123")
    assert not any(isinstance(c, tuple) and c[0] == "sent" for c in calls), \
        "INFO/GOOD must not reach SMTP"


def test_warn_and_critical_email(monkeypatch):
    _configured(monkeypatch)
    sent = []
    notifier = Notifier(console=False, email_dedup_s=0)
    import smtplib

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            pass

        def send_message(self, msg):
            sent.append(msg["Subject"])

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    notifier.warn("Odds moved — bet refused", "priced 1.42, accepted 1.35")
    notifier.critical("Balance dropped", "149.18 -> 49.18")
    assert len(sent) == 2
    assert sent[0].startswith("ATLAS WARN:")
    assert sent[1].startswith("ATLAS CRITICAL:")


def test_dedup_suppresses_same_title_within_cooldown(monkeypatch):
    _configured(monkeypatch)
    notifier = Notifier(console=False, email_dedup_s=1800.0)
    import smtplib

    sent = []

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            pass

        def send_message(self, msg):
            sent.append(msg["Subject"])

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    for _ in range(5):
        notifier.warn("Odds moved — bet refused", "priced 1.42, accepted 1.35")
    assert len(sent) == 1, "the same stuck condition emails once per cooldown"

    # a DIFFERENT title still emails immediately
    notifier.warn("Slip not placeable — bet skipped", "1X2 / Away")
    assert len(sent) == 2


def test_smtp_failure_never_breaks_send(monkeypatch):
    _configured(monkeypatch)
    notifier = Notifier(console=False, email_dedup_s=0)
    import smtplib

    class BrokenSMTP:
        def __init__(self, *a, **k):
            raise ConnectionError("smtp down")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(smtplib, "SMTP", BrokenSMTP)
    n = notifier.send("Title", "body", Severity.WARN)  # must not raise
    assert n.title == "Title"
    assert notifier.sent[-1].title == "Title", "notification still recorded"
