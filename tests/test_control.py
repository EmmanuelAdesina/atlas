"""Pins for the control dashboard's one bind-address resolution."""

import inspect

import atlas.live.control as control


def test_resolve_bind_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("ATLAS_HOST", raising=False)
    monkeypatch.delenv("ATLAS_PORT", raising=False)
    assert control.resolve_bind() == ("127.0.0.1", 8077)


def test_resolve_bind_honors_env_then_explicit_args(monkeypatch):
    monkeypatch.setenv("ATLAS_HOST", "10.9.9.9")
    monkeypatch.setenv("ATLAS_PORT", "9999")
    assert control.resolve_bind() == ("10.9.9.9", 9999)
    # an explicit argument beats the environment: the banner must print
    # the SAME address uvicorn binds (2026-11-02: the banner printed the
    # unresolved `http://None:None` and cost a probing cycle)
    assert control.resolve_bind("0.0.0.0", 1234) == ("0.0.0.0", 1234)


def test_serve_resolves_through_the_same_helper():
    """serve() and the banner must share one precedence — a single
    source of truth for the dashboard's bind address."""
    src = inspect.getsource(control.serve)
    assert "resolve_bind(host, port)" in src
