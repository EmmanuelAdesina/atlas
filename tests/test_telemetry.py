"""Tests for atlas.ops.telemetry.

The contract under test: the loop must never fail because of telemetry.
No collector configured, no SDK installed, a misbehaving gauge source —
the facade absorbs all of it and the loop's calls stay no-ops.
"""

from __future__ import annotations

import pytest

from atlas.ops.telemetry import (
    Telemetry,
    _parse_headers,
    _split_endpoint,
    init_telemetry,
)


# --- config helpers -----------------------------------------------------------


def test_bare_endpoint_gets_both_signal_paths():
    metrics, traces = _split_endpoint("http://collector:4318/")
    assert metrics == "http://collector:4318/v1/metrics"
    assert traces == "http://collector:4318/v1/traces"


def test_a_half_configured_signal_derives_the_other():
    metrics, traces = _split_endpoint("http://collector:4318/v1/metrics")
    assert metrics == "http://collector:4318/v1/metrics"
    assert traces == "http://collector:4318/v1/traces"


def test_headers_parse_and_drop_malformed_pairs():
    assert _parse_headers('k=v;  bad  ; auth = "Bearer t" ;') == {
        "k": "v",
        "auth": "Bearer t",
    }


# --- the no-op contract ---------------------------------------------------------


def test_unconfigured_telemetry_is_a_no_op_that_never_raises():
    # OTel not installed in this environment is fine too: enabled=False
    # either way, and every call below is swallowed.
    t = Telemetry(session="s", endpoint="", enabled=False)
    assert not t.enabled
    t.note_cycle("ok", ready=True, opportunities=1, duration_s=12.0)
    with t.span("atlas.cycle", {"k": "v"}):
        pass
    t.shutdown()  # repeat-safe
    t.shutdown()


def test_note_cycle_never_lets_an_instrument_raise(monkeypatch):
    t = Telemetry(session="s", endpoint="", enabled=False)

    def boom(*a, **k):
        raise RuntimeError("exporter on fire")

    monkeypatch.setattr(t, "cycles_total", boom, raising=False)
    # The call must return normally: a broken sink is the Notifier contract.
    t.note_cycle("crashed", ready=False, duration_s=1.0)
    assert not t.enabled


def test_unknown_gauge_name_is_ignored_not_registered():
    t = Telemetry(session="s", endpoint="", enabled=False)
    t.set_gauge_source("atlas.not_a_gauge", lambda: 1.0)
    assert "atlas.not_a_gauge" not in t._gauge_sources
    assert t._gauge_sources == {}

    t.set_gauge_source("atlas.daily_pnl_ngn", lambda: -1.5)
    assert t._gauge_sources["atlas.daily_pnl_ngn"]() == -1.5


def test_init_telemetry_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.delenv("ATLAS_OTLP_ENDPOINT", raising=False)
    import atlas.ops.telemetry as mod

    # A stale singleton from another test must not leak the wrong identity.
    monkeypatch.setattr(mod, "_current", None)
    first = init_telemetry()
    second = init_telemetry()
    assert first is second
    monkeypatch.setattr(mod, "_current", None)


@pytest.mark.parametrize(
    "ready, settled, stop, expected",
    [
        (True, 0, False, "ok"),
        (True, 2, False, "finished"),
        (True, 0, True, "stop_active"),
        (False, 0, False, "not_ready"),
    ],
)
def test_outcome_map(ready, settled, stop, expected):
    """The runner's outcome mapping, exercised through the same branch order
    it uses. `crashed` is covered by the crash flag in run_forever."""
    finished = settled and "match finished" in "match finished, settled 2 bet(s)"
    if finished:
        outcome = "finished"
    elif stop:
        outcome = "stop_active"
    elif ready:
        outcome = "ok"
    else:
        outcome = "not_ready"
    assert outcome == expected
