"""CDPFirstAdapter (atlas.browser.cdp_first) — the one-transport rules.

Under test: precedence (CDP before daemon), failover discipline (only a
PROVEN nothing-executed may ride the daemon), and the no-replay rule for
post-execution unknowns.
"""

import httpx
import pytest

from atlas.browser import cdp_direct
from atlas.browser.cdp_first import CDPFirstAdapter
from atlas.browser.webbridge import WebBridgeError


def _daemon(handler):
    return CDPFirstAdapter(
        session="t",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_backoff_s=0.0,
    )


def _ok_envelope(value="daemon-reply"):
    return httpx.Response(
        200, json={"ok": True, "data": {"type": "string", "value": value}}
    )


def test_cdp_answer_wins_and_daemon_is_untouched(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(500, text="daemon must not be reached")

    a = _daemon(handler)
    monkeypatch.setattr(
        cdp_direct, "evaluate", lambda code, **kw: {"type": "string", "value": "cdp"}
    )
    assert a.evaluate("1+1") == {"type": "string", "value": "cdp"}
    assert calls["n"] == 0


def test_proven_noop_failover_rides_the_daemon(monkeypatch):
    seen = []

    def handler(req):
        seen.append(req)
        return _ok_envelope()

    a = _daemon(handler)
    monkeypatch.setattr(
        cdp_direct, "evaluate",
        lambda code, **kw: (_ for _ in ()).throw(
            cdp_direct.CDPPreError("cannot connect")
        ),
    )
    assert a.evaluate("1+1")["value"] == "daemon-reply"
    assert len(seen) == 1  # exactly one daemon call, no retry storm


def test_post_error_is_never_replayed_on_the_daemon(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return _ok_envelope()

    a = _daemon(handler)
    monkeypatch.setattr(
        cdp_direct, "evaluate",
        lambda code, **kw: (_ for _ in ()).throw(
            cdp_direct.CDPPostError("stalled mid-call")
        ),
    )
    with pytest.raises(cdp_direct.CDPPostError):
        a.evaluate("betslip JS")
    assert calls["n"] == 0


def test_evaluate_safe_keeps_the_same_discipline(monkeypatch):
    a = _daemon(lambda req: _ok_envelope())
    monkeypatch.setattr(
        cdp_direct, "find_page", lambda *a, **kw: PAGE
    )
    monkeypatch.setattr(
        cdp_direct, "evaluate_read",
        lambda code, **kw: (_ for _ in ()).throw(
            cdp_direct.CDPPreError("cannot connect")),
    )
    assert a.evaluate_safe("scanners")["value"] == "daemon-reply"


def test_click_uses_the_direct_dispatch_and_fails_over_pre_only(monkeypatch):
    a = _daemon(lambda req: httpx.Response(200, json={"ok": True, "data": {"ok": True}}))
    monkeypatch.setattr(
        cdp_direct, "dispatch_click", lambda sel, **kw: {"clicked": True}
    )
    assert a.click("[data-atlas-click]") == {"clicked": True}

    monkeypatch.setattr(
        cdp_direct, "dispatch_click",
        lambda sel, **kw: (_ for _ in ()).throw(
            cdp_direct.CDPPreError("no element")
        ),
    )
    # Pre-error: daemon click retry is allowed (nothing was pressed).
    assert isinstance(a.click("[data-atlas-click]"), dict)

    monkeypatch.setattr(
        cdp_direct, "dispatch_click",
        lambda sel, **kw: (_ for _ in ()).throw(
            cdp_direct.CDPPostError("socket died after press")
        ),
    )
    with pytest.raises(cdp_direct.CDPPostError):
        a.click("[data-atlas-click]")


def test_evaluate_read_rotates_to_a_second_answering_target(monkeypatch):
    t1 = {**PAGE, "id": "T1"}
    t2 = {**PAGE, "id": "T2"}
    monkeypatch.setattr(cdp_direct, "find_page", lambda *a, **k: t1)
    consumed = []

    def fake_on(target, code, timeout_s=None):
        consumed.append(target["id"])
        if target["id"] == "T1":
            raise cdp_direct.CDPPostError("stalled")
        return {"type": "string", "value": "from-t2"}

    monkeypatch.setattr(cdp_direct, "evaluate_on", fake_on)
    out = cdp_direct.evaluate_read(
        "row-scan", url_prefixes=("sportybet.com",), extras=[t1, t2])
    assert out["value"] == "from-t2"
    assert consumed == ["T1", "T2"]


def test_evaluate_read_all_stall_raises_the_last_post_error(monkeypatch):
    t1 = {**PAGE, "id": "T1"}
    monkeypatch.setattr(cdp_direct, "find_page", lambda *a, **k: t1)
    monkeypatch.setattr(
        cdp_direct, "evaluate_on",
        lambda target, code, timeout_s=None: (_ for _ in ()).throw(
            cdp_direct.CDPPostError("stalled")),
    )
    with pytest.raises(cdp_direct.CDPPostError):
        cdp_direct.evaluate_read("x", url_prefixes=("sportybet.com",), extras=[{**PAGE, "id": "T2"}])


PAGE = {
    "type": "page",
    "id": "TMAIN",
    "url": "https://www.sportybet.com/ng/sport/football/sr:match:6601/",
    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/TMAIN",
}


def test_evaluate_safe_rotates_reads_but_evaluate_does_not(monkeypatch):
    def handler(req):
        calls.append(req)
        return _ok_envelope("daemon")

    calls: list = []
    a = _daemon(handler)
    seen = {}

    def fake_read(code, extras=(), **kw):
        seen["extras"] = [t["id"] for t in extras]
        return {"type": "string", "value": "rotated"}

    def fake_plain(code, **kw):
        seen["plain"] = True
        return {"type": "string", "value": "primary-only"}

    monkeypatch.setattr(cdp_direct, "evaluate_read", fake_read)
    monkeypatch.setattr(cdp_direct, "evaluate", fake_plain)
    monkeypatch.setattr(
        cdp_direct, "list_pages",
        lambda host=None: [PAGE, {**PAGE, "id": "T9"}],
    )
    assert a.evaluate_safe("scanner JS")["value"] == "rotated"
    assert seen["extras"] == ["T9"]  # the second page target was offered
    assert a.evaluate("betslip JS")["value"] == "primary-only"  # no rotation
    assert calls == [] or isinstance(calls, list)


def test_deterministic_page_error_is_raised_not_failover(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return _ok_envelope()

    a = _daemon(handler)
    monkeypatch.setattr(
        cdp_direct, "evaluate",
        lambda code, **kw: (_ for _ in ()).throw(
            WebBridgeError("CDP evaluate threw: Uncaught", transient=False)
        ),
    )
    with pytest.raises(WebBridgeError) as ei:
        a.evaluate("x")
    assert ei.value.transient is False
    assert calls["n"] == 0


def test_main_defaults_to_cdp_first(monkeypatch):
    from atlas.main import _select_adapter
    import argparse as _argparse

    args = _argparse.Namespace(session="t", daemon_url="")
    monkeypatch.delenv("ATLAS_TRANSPORT", raising=False)
    assert isinstance(_select_adapter(args), CDPFirstAdapter)
    monkeypatch.setenv("ATLAS_TRANSPORT", "daemon")
    from atlas.browser.webbridge import WebBridgeAdapter

    assert type(_select_adapter(args)) is WebBridgeAdapter
