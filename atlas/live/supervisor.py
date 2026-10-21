"""The supervisor: an LLM that handles the exception path.

The fast plane (capture -> price -> plan -> place) is deterministic and
never calls a model. What it cannot do is handle the things nobody
predicted: a session that expired, a board that says "Something went
wrong", a media widget covering the scoreboard, an extension that stopped
answering, a market that suspended between pricing and placing.

Those are judgement calls about a page that changes shape without notice.
That is exactly what a language model is for, and exactly what hard-coded
selectors are bad at. So the supervisor drives the browser for the
exception path, under three constraints:

  1. It has NO money tool. It cannot place a bet, and it cannot change a
     stake. Placement stays behind the deterministic gates, so the odds
     verification and the daily-stop re-check remain meaningful.
  2. Every action it takes is written to an audit log.
  3. The credentials never enter the model's context. The login tool reads
     them from the environment and fills the form itself, so the password is
     never in a prompt, a response, or a log.

There is also a rule-based supervisor for the cases that are already known
(session expired). It runs first and deterministically; the LLM is the
fallback for the cases nobody has seen yet. The system therefore works
with no API key and gets better with one.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from atlas.live.capture import CaptureResult, LiveCapture
from atlas.ops.notify import Notifier

# The site root. The live board is NEVER navigated to directly: it never
# finishes loading, so the navigation is queued to a page that will not
# answer it.
LIGHT_PAGE = "https://sportybet.com/ng/"

# The tools the supervisor may use. There is deliberately no placement tool.
ALLOWED_TOOLS = (
    "read_page",
    "fill_login",
    "navigate",
    "reload",
    "restart_session",
    "screenshot",
    "notify_operator",
    "done",
)

MONEY_TOOLS = ("place_bet", "set_stake", "click_odds")

SYSTEM_PROMPT = """\
You supervise an automated sports-betting browser session for SportyBet.
A deterministic engine does the trading; you handle EXCEPTIONS only.

Your tools:
  read_page      - a compact summary of the current page
  fill_login     - fill and submit the login form (credentials come from the
                   environment; you never see or handle them)
  navigate       - go to a URL (client-side; never use the live-board URL
                   directly, it never finishes loading)
  reload         - reload the current page
  restart_session - discard the current tab and open a fresh one. Use this
                   when the page is FROZEN: a frozen tab cannot be
                   navigated, because the navigation is queued to the page
                   and the page's JS thread is what is stuck.
  screenshot     - a screenshot of the page
  notify_operator - send a message to the operator
  done           - finish, with a short explanation

RULES
  - You CANNOT place bets or change stakes. There is no tool for it, and
    asking for it is an error.
  - Never navigate directly to a /sport/live URL: use navigate() with the
    site root, then let the client-side navigation happen.
  - Work in as few steps as possible. If read_page shows a login form, call
    fill_login, then read_page again to confirm.
  - If a page shows an error the deterministic engine already knows about,
    prefer reload().
  - Finish with done and one sentence saying what you did and what the
    operator should know.

Reply with ONE JSON object per step, no prose:
  {"tool": "<name>", "args": {...}}
or to finish:
  {"tool": "done", "args": {"message": "..."}}
"""


@dataclass
class SupervisorAction:
    """One thing the supervisor did."""

    at: str
    tool: str
    args: dict
    result: Any = None
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "at": self.at,
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "ok": self.ok,
        }


@dataclass
class SupervisorResult:
    """What the supervisor decided and did."""

    handled: bool
    reason: str
    actions: list[SupervisorAction] = field(default_factory=list)
    message: str = ""

    def describe(self) -> str:
        steps = ", ".join(a.tool for a in self.actions)
        return f"handled={self.handled} {self.reason} [{steps}]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrowserTools:
    """The tool surface the supervisor may use.

    Every method here is read-only or navigational, EXCEPT `fill_login`,
    which writes credentials into a form and submits it. There is no way to
    place a bet through this interface.
    """

    def __init__(self, capture: LiveCapture) -> None:
        self.capture = capture
        self.actions: list[SupervisorAction] = []

    # --- tools -----------------------------------------------------------

    def read_page(self) -> dict:
        """A compact summary of the page — never the raw DOM."""
        from atlas.browser.webbridge import WebBridgeAdapter  # noqa: F401

        board = self.capture.board_candidates()
        markets = self.capture.read_markets_dom()
        session = self.capture.session_expired()
        priced = [m for m in markets if m.get("selections")]
        return {
            "url": self.capture.board_loaded(),
            "board_has_matches": len(board) > 0,
            "matches": len(board),
            "market_headers": len(markets),
            "markets_with_prices": len(priced),
            "session_expired": bool(session.get("expired")),
            "sample_markets": [m["market_label"] for m in priced[:8]],
        }

    def fill_login(self) -> dict:
        """Fill and submit the login form.

        The credentials come from the environment, so they are never part of
        the model's context.
        """
        phone = os.environ.get("ATLAS_SPORTYBET_PHONE", "")
        password = os.environ.get("ATLAS_SPORTYBET_PASSWORD", "")
        if not phone or not password:
            return {
                "ok": False,
                "error": "no credentials configured "
                "(ATLAS_SPORTYBET_PHONE / ATLAS_SPORTYBET_PASSWORD)",
            }
        from atlas.browser.webbridge import WebBridgeAdapter

        adapter = self.capture.adapter
        code = _LOGIN_JS.replace("__PHONE__", json.dumps(phone)).replace(
            "__PASSWORD__", json.dumps(password)
        )
        try:
            raw = adapter.evaluate(code)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if isinstance(data, dict) and "value" in data:
            try:
                return json.loads(data["value"])
            except json.JSONDecodeError:
                return {"ok": False, "error": "login returned nothing"}
        return data if isinstance(data, dict) else {"ok": False}

    def navigate(self, url: str) -> dict:
        """Navigate client-side (never a direct live-board navigation)."""
        if "/sport/live" in str(url):
            return {
                "ok": False,
                "error": "never navigate to the live board directly; "
                "use the site root and let the client-side navigation happen",
            }
        try:
            self.capture._evaluate(f"location.href={json.dumps(str(url))}; 1")
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "url": url}

    def reload(self) -> dict:
        try:
            self.capture._evaluate("location.reload(); 1")
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    def restart_session(self) -> dict:
        """Discard the current tab and open a fresh one.

        A frozen page's JS thread never replies, so every command times out
        and even a navigation is queued to a thread that will never run it.
        Closing the session is the only thing that frees the extension.

        The navigate that follows is EXPECTED to report a timeout: the
        SportyBet page never fires 'load', so the daemon waits and gives up
        while the navigation itself succeeds. The result is therefore
        verified, not assumed.
        """
        try:
            self.capture.adapter.close_session()
        except Exception as e:
            return {"ok": False, "error": f"close failed: {e}"}
        time.sleep(1.5)
        navigate_error = None
        try:
            self.capture.adapter.navigate(LIGHT_PAGE)
        except Exception as e:
            # A timeout here does not mean the navigation failed.
            navigate_error = str(e)
        time.sleep(3.0)
        # Verify with a cheap evaluate: if the page answers, the session is
        # back whatever the navigate said.
        try:
            self.capture._evaluate("document.title")
            return {
                "ok": True,
                "note": "session restarted",
                "navigate_reported": navigate_error,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"the session did not come back: {e}",
                "navigate_reported": navigate_error,
            }

    def screenshot(self) -> dict:
        """A screenshot of the page, base64-encoded.

        This is the capability a programmatic probe does not have: it shows
        what the operator would actually see, including a page that is
        rendered but not extractable — a frozen page, an unexpected modal, a
        statistics overlay covering the scoreboard. A probe that returns
        'no markets' when the board is full of them is exactly the failure
        this avoids.
        """
        try:
            data = self.capture.adapter.screenshot(format="jpeg", quality=40)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if isinstance(data, dict):
            data = data.get("data", data)
        b64 = None
        if isinstance(data, dict):
            for key in ("screenshot", "image", "data", "value", "base64"):
                if isinstance(data.get(key), str) and len(data[key]) > 100:
                    b64 = data[key]
                    break
        if b64 is None:
            return {"ok": False, "error": "no screenshot payload"}
        return {"ok": True, "image": b64}


_LOGIN_JS = """
(() => {
  const visible = (e) => e && e.offsetParent !== null &&
    getComputedStyle(e).display !== 'none' &&
    getComputedStyle(e).visibility !== 'hidden';
  const setNative = (el, value) => {
    const proto = el instanceof HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const pw = Array.from(document.querySelectorAll('input[type=password]')).filter(visible);
  if (!pw.length) return JSON.stringify({ok: false, error: 'no visible password field'});
  const pwd = pw[0];
  const form = pwd.closest('form') || pwd.parentElement;
  let tel = null;
  for (const c of Array.from(document.querySelectorAll("input[type=tel], input[inputmode=numeric]")).filter(visible)) {
    if (!form || form.contains(c)) { tel = c; break; }
  }
  if (tel) setNative(tel, __PHONE__);
  setNative(pwd, __PASSWORD__);
  let btn = null;
  for (const b of (form || document).querySelectorAll('button, [role=button], input[type=submit]')) {
    const label = ((b.innerText || b.value || '') + ' ' + (b.getAttribute('aria-label') || '')).trim();
    if (/^login$/i.test(label) && visible(b)) { btn = b; break; }
  }
  if (!btn) return JSON.stringify({ok: false, error: 'no visible login button'});
  btn.click();
  return JSON.stringify({ok: true});
})()
"""


class LLMClient:
    """A minimal chat client. OpenAI-compatible by default."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ATLAS_BOSS_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("ATLAS_BOSS_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.environ.get("ATLAS_BOSS_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, messages: list[dict]) -> str:
        if not self.configured:
            raise RuntimeError("no API key (set ATLAS_BOSS_API_KEY)")
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "messages": [{"role": "system", "content": system}] + messages,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]

    def complete_with_image(
        self, system: str, text: str, image_b64: str
    ) -> str:
        """Complete with a screenshot attached.

        The model can SEE the page, which is the whole point of the
        supervisor: a page can be rendered but not extractable, and only
        the model can tell the difference between "the board is empty" and
        "my probe is broken".
        """
        if not self.configured:
            raise RuntimeError("no API key (set ATLAS_BOSS_API_KEY)")
        import httpx

        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]


class Supervisor:
    """Handles the exception path, with the LLM as the fallback for the
    cases nobody has seen yet."""

    def __init__(
        self,
        capture: LiveCapture,
        notifier: Optional[Notifier] = None,
        client: Optional[LLMClient] = None,
        audit_path: Optional[str] = None,
        max_steps: int = 6,
    ) -> None:
        self.capture = capture
        self.notifier = notifier
        self.client = client or LLMClient()
        self.tools = BrowserTools(capture)
        self.max_steps = max_steps
        self.audit_path = (
            Path(audit_path) if audit_path else Path("data/decisions/supervisor.jsonl")
        )

    # --- audit -----------------------------------------------------------

    def _audit(self, action: SupervisorAction) -> None:
        self.tools.actions.append(action)
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(action.to_dict(), ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # --- the deterministic path first -------------------------------------

    def handle_session_expired(self) -> SupervisorResult:
        """Re-login without a model.

        The session expiring is a known case with a known fix, so it does not
        need a language model. The credentials come from the environment and
        never enter a prompt.
        """
        result = self.tools.fill_login()
        action = SupervisorAction(
            at=_now(), tool="fill_login", args={}, result=result, ok=bool(result.get("ok"))
        )
        self._audit(action)
        if not result.get("ok"):
            return SupervisorResult(
                handled=False,
                reason=f"login failed: {result.get('error')}",
                actions=[action],
            )
        # Confirm it worked.
        for _ in range(10):
            time.sleep(1.0)
            session = self.capture.session_expired()
            if not session.get("expired"):
                return SupervisorResult(
                    handled=True,
                    reason="re-logged in",
                    actions=[action],
                    message="Session expired; re-logged in automatically.",
                )
        return SupervisorResult(
            handled=False,
            reason="login submitted but the session still shows as expired",
            actions=[action],
        )

    # --- the LLM path ---------------------------------------------------

    def run_agent(self, task: str, context: Optional[dict] = None) -> SupervisorResult:
        """Let the model work the exception.

        The model gets a compact page summary and a list of tools, and
        returns one tool call per step.
        """
        if not self.client.configured:
            return SupervisorResult(
                handled=False,
                reason="no LLM configured (set ATLAS_BOSS_API_KEY)",
            )
        page = self.tools.read_page()
        # Give the model the SCREEN as well as the probe. A page can be
        # rendered but not extractable — the difference between "the board is
        # empty" and "my probe is broken" is only visible on screen.
        image = None
        try:
            shot = self.tools.screenshot()
            if shot.get("ok"):
                image = shot.get("image")
        except Exception:
            image = None
        reply: Optional[str] = None
        if image and hasattr(self.client, "complete_with_image"):
            # The reply to the image IS the first tool call: it must not be
            # thrown away.
            reply = self.client.complete_with_image(
                SYSTEM_PROMPT,
                json.dumps(
                    {"task": task, "page": page, "context": context or {}},
                    default=str,
                )[:3000],
                image,
            )
            messages = [{"role": "user", "content": reply}]
        else:
            messages = [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"task": task, "page": page, "context": context or {}},
                        default=str,
                    )[:4000],
                }
            ]
        actions: list[SupervisorAction] = []
        for _ in range(self.max_steps):
            if reply is None:
                reply = self.client.complete(SYSTEM_PROMPT, messages)
            call = _parse_tool_call(reply)
            reply = None
            if call is None:
                return SupervisorResult(
                    handled=False,
                    reason=f"the model did not return a tool call: {str(reply)[:160]}",
                    actions=actions,
                )
            tool = str(call.get("tool", ""))
            args = call.get("args") or {}
            if tool in MONEY_TOOLS:
                action = SupervisorAction(
                    at=_now(), tool=tool, args={}, ok=False,
                    result="refused: the supervisor has no money tool",
                )
                self._audit(action)
                actions.append(action)
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": "That tool does not exist. Choose another from the list.",
                    }
                )
                continue
            if tool not in ALLOWED_TOOLS:
                action = SupervisorAction(
                    at=_now(), tool=tool, args={}, ok=False,
                    result=f"unknown tool; allowed: {', '.join(ALLOWED_TOOLS)}",
                )
                self._audit(action)
                actions.append(action)
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Unknown tool. Use one of: {', '.join(ALLOWED_TOOLS)}.",
                    }
                )
                continue
            if tool == "done":
                message = str(args.get("message", ""))[:600]
                action = SupervisorAction(
                    at=_now(), tool="done", args={}, result=message
                )
                self._audit(action)
                actions.append(action)
                return SupervisorResult(
                    handled=True, reason="agent finished", actions=actions, message=message
                )
            if tool == "notify_operator":
                message = str(args.get("message", ""))[:600]
                action = SupervisorAction(
                    at=_now(), tool="notify_operator", args={}, result=message
                )
                self._audit(action)
                actions.append(action)
                if self.notifier is not None:
                    self.notifier.warn("Supervisor", message)
                continue
            outcome = self._call_tool(tool, args)
            action = SupervisorAction(
                at=_now(), tool=tool, args=args, result=outcome,
                ok=bool(isinstance(outcome, dict) and outcome.get("ok", True)),
            )
            self._audit(action)
            actions.append(action)
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"tool": tool, "result": outcome}, default=str
                    )[:3000],
                }
            )
        return SupervisorResult(
            handled=False,
            reason=f"the agent did not finish within {self.max_steps} steps",
            actions=actions,
        )

    def _call_tool(self, tool: str, args: dict) -> Any:
        try:
            if tool == "read_page":
                return self.tools.read_page()
            if tool == "fill_login":
                return self.tools.fill_login()
            if tool == "navigate":
                return self.tools.navigate(str(args.get("url", "")))
            if tool == "reload":
                return self.tools.reload()
            if tool == "restart_session":
                return self.tools.restart_session()
            if tool == "screenshot":
                return self.tools.screenshot()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": False, "error": f"unhandled tool {tool}"}

    # --- the entry point -------------------------------------------------

    def handle(self, result: CaptureResult, task: Optional[str] = None) -> SupervisorResult:
        """Deal with whatever the fast plane could not.

        The known cases are handled deterministically; the LLM is the
        fallback for the ones nobody has seen.
        """
        reason = result.reason or ""
        if "session expired" in reason:
            outcome = self.handle_session_expired()
            if outcome.handled:
                if self.notifier is not None:
                    self.notifier.info("Session restored", outcome.message)
                return outcome
            # The deterministic re-login did not work: let the model try.
            if self.client.configured:
                agent = self.run_agent(
                    task or "The SportyBet session expired and the automatic "
                    "re-login did not work. Get the session back.",
                    context={"deterministic_attempt": outcome.reason},
                )
                if agent.handled and self.notifier is not None:
                    self.notifier.info("Session restored", agent.message)
                return agent
            return outcome
        # Anything else that looks like an exception: let the model look.
        if task and self.client.configured:
            return self.run_agent(task, context={"cycle_reason": reason})
        return SupervisorResult(
            handled=False,
            reason=f"no supervisor action for: {reason}",
        )


def _parse_tool_call(reply: str) -> Optional[dict]:
    """Parse the model's reply as a single JSON tool call."""
    text = (reply or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
