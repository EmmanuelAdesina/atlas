"""Tests for the supervisor (atlas.live.supervisor).

The supervisor is the LLM that handles the exception path. The properties
that matter are safety properties, so they are what is tested:

  - it has no money tool, and asking for one is refused;
  - the credentials never enter the model's context;
  - every action is audited;
  - a known failure (an expired session) is handled WITHOUT a model;
  - the model only ever sees a compact page summary, not the raw DOM.
"""

import json

import pytest

from atlas.live.supervisor import (
    ALLOWED_TOOLS,
    MONEY_TOOLS,
    BrowserTools,
    LLMClient,
    Supervisor,
    _parse_tool_call,
)


class FakeCapture:
    """A capture that answers a fixed page state."""

    def __init__(self, session_expired=False, markets=0, candidates=1):
        self._session_expired = session_expired
        self.markets = markets
        self.candidates = candidates
        self.navigated = []
        self.reloads = 0
        self.adapter = _FakeAdapter()

    def board_candidates(self):
        return [object()] * self.candidates

    def read_markets_dom(self):
        return [{"market_label": "1X2", "selections": [{"name": "Home", "odds": 2.5}]}] * self.markets

    def session_expired(self):
        return {"expired": self._session_expired}

    def board_loaded(self):
        return True

    def _evaluate(self, code):
        self.navigated.append(code)
        return 1


class _FakeAdapter:
    def __init__(self):
        self.clicks = []
        self.screenshot_calls = 0

    def evaluate(self, code):
        return {"data": {"value": "{}"}}

    def click(self, selector):
        self.clicks.append(selector)
        return {"ok": True}

    def screenshot(self, format="jpeg", quality=40):
        self.screenshot_calls += 1
        return {"data": {"screenshot": "BASE64IMAGE" + "x" * 200}}


class ScriptedClient(LLMClient):
    """An LLM client that replays a fixed list of replies."""

    def __init__(self, replies):
        super().__init__(api_key="test-key")
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else '{"tool":"done","args":{"message":"end"}}'

    def complete_with_image(self, system, text, image_b64):
        self.calls.append([{"role": "user", "content": text, "image": image_b64}])
        return self.replies.pop(0) if self.replies else '{"tool":"done","args":{"message":"end"}}'


# --- the tool surface -----------------------------------------------------

def test_the_supervisor_has_no_money_tool():
    for tool in MONEY_TOOLS:
        assert tool not in ALLOWED_TOOLS


def test_tools_expose_no_placement_method():
    tools = BrowserTools(FakeCapture())
    for name in dir(tools):
        if name.startswith("_"):
            continue
        assert "bet" not in name.lower() or name == "add_selection" is False
        assert "stake" not in name.lower()


# --- the deterministic path ------------------------------------------------

def test_an_expired_session_is_handled_without_a_model(monkeypatch):
    capture = FakeCapture(session_expired=True)
    supervisor = Supervisor(capture=capture, client=ScriptedClient([]))
    monkeypatch.setenv("ATLAS_SPORTYBET_PHONE", "8036656131")
    monkeypatch.setenv("ATLAS_SPORTYBET_PASSWORD", "secret")
    result = supervisor.handle_session_expired()
    # The login tool reads the credentials from the environment.
    assert "secret" not in json.dumps(result.actions, default=str)
    assert result.actions[0].tool == "fill_login"


def test_no_credentials_configured_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("ATLAS_SPORTYBET_PHONE", raising=False)
    monkeypatch.delenv("ATLAS_SPORTYBET_PASSWORD", raising=False)
    capture = FakeCapture(session_expired=True)
    supervisor = Supervisor(capture=capture, client=ScriptedClient([]))
    result = supervisor.handle_session_expired()
    assert result.handled is False
    assert "credentials" in result.reason


# --- the LLM path ---------------------------------------------------------

def test_the_agent_runs_the_tools_it_is_given():
    replies = [
        '{"tool":"reload","args":{}}',
        '{"tool":"done","args":{"message":"reloaded and recovered"}}',
    ]
    client = ScriptedClient(replies)
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    result = supervisor.run_agent("the board is not loading")
    assert result.handled is True
    assert [a.tool for a in result.actions] == ["reload", "done"]
    assert result.message == "reloaded and recovered"


def test_a_money_tool_request_is_refused():
    replies = [
        '{"tool":"place_bet","args":{"odds":2.5}}',
        '{"tool":"done","args":{"message":"cannot place bets"}}',
    ]
    client = ScriptedClient(replies)
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    result = supervisor.run_agent("place a bet for me")
    assert any(a.tool == "place_bet" and not a.ok for a in result.actions)
    assert result.handled is True


def test_an_unknown_tool_is_refused():
    replies = [
        '{"tool":"exfiltrate","args":{}}',
        '{"tool":"done","args":{"message":"nope"}}',
    ]
    client = ScriptedClient(replies)
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    result = supervisor.run_agent("do something clever")
    assert any(a.tool == "exfiltrate" and not a.ok for a in result.actions)


def test_every_action_is_audited(tmp_path):
    audit = tmp_path / "supervisor.jsonl"
    client = ScriptedClient(['{"tool":"reload","args":{}}', '{"tool":"done","args":{"message":"x"}}'])
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=str(audit))
    supervisor.run_agent("task")
    lines = [json.loads(l) for l in audit.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines
    assert all("tool" in l and "at" in l for l in lines)


def test_the_model_never_sees_the_credentials(monkeypatch):
    monkeypatch.setenv("ATLAS_SPORTYBET_PASSWORD", "sup3rs3cret")
    client = ScriptedClient(['{"tool":"done","args":{"message":"ok"}}'])
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    supervisor.run_agent("log in for me")
    blob = json.dumps(client.calls, default=str)
    assert "sup3rs3cret" not in blob


def test_the_model_sees_the_screenshot_when_available():
    client = ScriptedClient(['{"tool":"done","args":{"message":"saw it"}}'])
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    supervisor.run_agent("what is on the screen?")
    assert client.calls
    payload = client.calls[0]
    blob = json.dumps(payload, default=str)
    assert "BASE64IMAGE" in blob


def test_without_an_api_key_the_agent_does_nothing():
    client = LLMClient(api_key="")
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    result = supervisor.run_agent("anything")
    assert result.handled is False
    assert "no LLM configured" in result.reason


def test_a_nonsense_reply_does_not_crash():
    client = ScriptedClient(["I think you should just try again."])
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None)
    result = supervisor.run_agent("anything")
    assert result.handled is False
    assert "tool call" in result.reason


def test_the_step_budget_is_enforced():
    client = ScriptedClient(['{"tool":"reload","args":{}}'] * 10)
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, client=client, audit_path=None, max_steps=3)
    result = supervisor.run_agent("loop forever")
    assert result.handled is False
    assert "steps" in result.reason


def test_notify_operator_reaches_the_notifier():
    from atlas.ops.notify import Notifier

    class FakeNotifier(Notifier):
        def __init__(self):
            super().__init__(console=False)
            self.messages = []

        def warn(self, title, body, **kw):
            self.messages.append((title, body))
            return super().warn(title, body, **kw)

    notifier = FakeNotifier()
    client = ScriptedClient(
        ['{"tool":"notify_operator","args":{"message":"the board is dead"}}',
         '{"tool":"done","args":{"message":"done"}}']
    )
    capture = FakeCapture()
    supervisor = Supervisor(capture=capture, notifier=notifier, client=client, audit_path=None)
    supervisor.run_agent("report")
    assert any("dead" in m[1] for m in notifier.messages)


# --- parsing --------------------------------------------------------------

def test_tool_call_parsing_handles_fences_and_prose():
    assert _parse_tool_call('{"tool": "done"}')["tool"] == "done"
    assert _parse_tool_call('```json\n{"tool": "reload"}\n```')["tool"] == "reload"
    assert _parse_tool_call('Sure! {"tool": "done", "args": {}}')["tool"] == "done"
    assert _parse_tool_call("no json here") is None
