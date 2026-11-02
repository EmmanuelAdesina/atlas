"""Direct DevTools transport (atlas.browser.cdp_direct).

Fakes ride the module's seams (_http_json / _ws_create); the semantics
under test are the ones that settled the 2026-11-02 outage: deadline
protection, event draining, send-free failover classification, and the
{'type','value'} envelope parity with the daemon contract.
"""

import json

import pytest

from atlas.browser import cdp_direct
from atlas.browser.webbridge import WebBridgeError


class FakeWS:
    """Cans: strings are frames, callables get the last frame, exceptions
    are raised from recv."""

    def __init__(self, cans):
        self.cans = list(cans)
        self.sent: list[dict] = []
        self.closed = False
        self.last_timeout = 0.0

    def send(self, frame):
        self.sent.append(json.loads(frame))

    def recv(self):
        if not self.cans:
            raise RuntimeError("fake starved")
        can = self.cans.pop(0)
        if isinstance(can, BaseException):
            raise can
        if callable(can):
            can = can(self.sent[-1])
        return json.dumps(can) if isinstance(can, dict) else can

    def settimeout(self, t):
        self.last_timeout = t

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_budget(monkeypatch):
    cdp_direct._created_at.clear()
    yield
    cdp_direct._created_at.clear()


PAGE = {
    "type": "page",
    "id": "T1",
    "url": "https://www.sportybet.com/ng/sport/football/live_list/",
    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/T1",
}
MATCH_PAGE = {
    "type": "page",
    "id": "T2",
    "url": "https://www.sportybet.com/ng/sport/football/sr:match:6601/",
    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/T2",
}


def eval_result(value):
    return json.dumps({"id": 1, "result": {"result": {"type": "string", "value": value}}})


def test_call_drains_events_until_the_matching_id():
    ws = FakeWS(
        [
            {"id": 9, "method": "Page.frameStartedLoading"},
            {"id": 10, "method": "Runtime.consoleAPICalled"},
            eval_result("ok"),
        ]
    )
    out = cdp_direct.call(ws, "Runtime.evaluate", {}, timeout=5)
    assert out["result"]["value"] == "ok"
    assert ws.closed is False


def test_call_stall_is_post_error_and_transient():
    ws = FakeWS([RuntimeError("no bytes")]*6)
    with pytest.raises(cdp_direct.CDPPostError) as ei:
        cdp_direct.call(ws, "Runtime.evaluate", {}, timeout=0.3)
    assert ei.value.transient is True


def test_connect_failure_is_pre_error():
    def refuse(url, timeout):
        raise OSError("connection refused")

    monkey_refuse = cdp_direct._ws_create
    try:
        cdp_direct._ws_create = refuse
        with pytest.raises(cdp_direct.CDPPreError):
            cdp_direct.connect(PAGE)
    finally:
        cdp_direct._ws_create = monkey_refuse


def test_evaluate_returns_the_type_value_envelope(monkeypatch):
    monkeypatch.setattr(cdp_direct, "list_pages", lambda host=None: [PAGE])
    fake = FakeWS([eval_result("hi")])
    monkeypatch.setattr(cdp_direct, "_ws_create", lambda url, timeout: fake)
    out = cdp_direct.evaluate(
        "1+1", url_prefixes=("sportybet.com",)
    )
    assert out == {"type": "string", "value": "hi"}
    assert fake.sent[0]["method"] == "Runtime.evaluate"
    assert fake.sent[0]["params"]["returnByValue"] is True


def test_page_exception_is_deterministic_not_transient(monkeypatch):
    monkeypatch.setattr(cdp_direct, "list_pages", lambda host=None: [PAGE])
    det = json.dumps(
        {"id": 1, "result": {"exceptionDetails": {"text": "Uncaught"}, "result": {}}}
    )
    monkeypatch.setattr(cdp_direct, "_ws_create", lambda url, timeout: FakeWS([det]))
    with pytest.raises(WebBridgeError) as ei:
        cdp_direct.evaluate("throw new Error('x')", url_prefixes=("sportybet.com",))
    assert ei.value.transient is False


def test_no_matching_page_creates_the_board_tab(monkeypatch):
    created = {}

    def fake_new(url, method="GET", timeout=0):
        created["new"] = url
        return {
            "id": "NEW",
            "url": "about:blank",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/NEW",
        }

    monkeypatch.setattr(cdp_direct, "list_pages", lambda host=None: [])
    monkeypatch.setattr(cdp_direct, "_http_json", fake_new)
    # about:blank landing -> Page.navigate over the socket, then evaluate.
    fake = FakeWS(
        [
            json.dumps({"id": 1, "result": {}}),  # Page.navigate ack
            eval_result("board1"),
        ]
    )
    monkeypatch.setattr(cdp_direct, "_ws_create", lambda url, timeout: fake)
    out = cdp_direct.evaluate(
        "document.title",
        url_prefixes=("sportybet.com",),
        create_url="https://www.sportybet.com/ng/sport/football/live_list/",
    )
    assert out["value"] == "board1"
    assert "json/new?" in created["new"]
    assert fake.sent[0]["method"] == "Page.navigate"


def test_create_budget_blocks_tab_spam(monkeypatch):
    import time as _time

    monkeypatch.setattr(cdp_direct, "list_pages", lambda host=None: [])
    for _ in range(cdp_direct.MAX_CREATES_PER_MINUTE):
        cdp_direct._created_at.append(_time.monotonic())
    with pytest.raises(cdp_direct.CDPPreError):
        cdp_direct.open_page("https://www.sportybet.com/x/")


def test_match_panel_target_wins_over_a_plain_board_tab(monkeypatch):
    monkeypatch.setattr(
        cdp_direct, "list_pages", lambda host=None: [PAGE, MATCH_PAGE]
    )
    seen = {}
    real_eval_on = cdp_direct.evaluate_on

    def spy(target, code, timeout_s=None):
        seen["target"] = target
        return {"type": "string", "value": "1"}

    monkeypatch.setattr(cdp_direct, "evaluate_on", spy)
    atlas_prefixes = ("/sr:match:", "/football/live/", PAGE["url"])
    cdp_direct.evaluate("1", url_prefixes=atlas_prefixes, host_hint="sportybet.com")
    assert seen["target"]["id"] == "T2"


def test_dispatch_click_presses_and_releases(monkeypatch):
    monkeypatch.setattr(cdp_direct, "list_pages", lambda host=None: [MATCH_PAGE])
    rect = json.dumps(
        {"id": 1, "result": {"result": {
            "type": "string",
            "value": json.dumps({"x": 320.5, "y": 210.0, "w": 60, "h": 22}),
        }}}
    )
    ws = FakeWS([rect, {"id": 1, "result": {}}, {"id": 1, "result": {}}])
    monkeypatch.setattr(cdp_direct, "_ws_create", lambda url, timeout: ws)
    out = cdp_direct.dispatch_click(
        "[data-atlas-click]", url_prefixes=("/sr:match:",), host_hint="sportybet.com"
    )
    assert out["clicked"] is True
    methods = [f["method"] for f in ws.sent]
    assert methods == ["Runtime.evaluate", "Input.dispatchMouseEvent", "Input.dispatchMouseEvent"]
    pressed = ws.sent[1]["params"]
    assert pressed["type"] == "mousePressed" and pressed["x"] == 320.5 and pressed["y"] == 210.0


def test_dispatch_click_missing_control_is_pre_error(monkeypatch):
    monkeypatch.setattr(cdp_direct, "list_pages", lambda host=None: [MATCH_PAGE])
    nx = json.dumps({"id": 1, "result": {"result": {"type": "string", "value": "ATLAS-NX"}}})
    monkeypatch.setattr(cdp_direct, "_ws_create", lambda url, timeout: FakeWS([nx]))
    with pytest.raises(cdp_direct.CDPPreError):
        cdp_direct.dispatch_click("[nothing]", url_prefixes=("/sr:match:",))
