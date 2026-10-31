"""The daemon URL is configurable: the split-plane deployment points the
loop on an always-on box at a remote browser daemon (env var or --daemon-url),
while a plain local run keeps 127.0.0.1:10086 untouched."""

import importlib

import atlas.browser.webbridge as wb


def test_default_daemon_url_is_local():
    a = wb.WebBridgeAdapter(session="t")
    assert a.daemon_url == "http://127.0.0.1:10086/command"


def test_env_var_moves_the_daemon(monkeypatch):
    monkeypatch.setenv(
        "ATLAS_WEBBRIDGE_DAEMON_URL", "http://100.93.10.10:10086/command"
    )
    mod = importlib.reload(wb)
    try:
        assert (
            mod.DEFAULT_DAEMON_URL == "http://100.93.10.10:10086/command"
        )
        a = mod.WebBridgeAdapter(session="t")
        assert a.daemon_url == "http://100.93.10.10:10086/command"
    finally:
        monkeypatch.delenv("ATLAS_WEBBRIDGE_DAEMON_URL")
        importlib.reload(wb)


def test_explicit_flag_beats_everything():
    a = wb.WebBridgeAdapter(
        session="t", daemon_url="http://127.0.0.1:9999/command"
    )
    assert a.daemon_url == "http://127.0.0.1:9999/command"
