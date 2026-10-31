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

    # --- login -----------------------------------------------------------

    def _eval_json(self, code: str) -> dict:
        """Run a JSON-returning payload and normalise the transport envelope.

        The extension answers `{"value": "<json>"}` and the CDP fallback
        answers `{"type": "string", "value": "<json>"}`; a payload that threw
        answers nothing. Every login step wants a dict, so unwrap both here
        rather than repeating the dance at each call site.
        """
        raw = self.capture.adapter.evaluate(code)
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if isinstance(data, dict) and "value" in data:
            data = data["value"]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"ok": False, "error": "payload returned nothing"}
        if not isinstance(data, dict):
            return {"ok": False, "error": "payload returned nothing"}
        return data

    @staticmethod
    def _phone_candidates(phone: str) -> list:
        """The phone shapes the book accepts behind its `+234` prefix.

        The configured number is tried first. The live form shows `+234` and
        validates the rest of the field, so a leading 0 may or may not belong
        there — only the page can decide, and it says so by enabling (or
        refusing to enable) the Login button.
        """
        out = [phone]
        if phone.startswith("0"):
            out.append(phone[1:])
        elif len(phone) == 10:
            out.append("0" + phone)
        return [c for c in out if c]

    def _wait_login_button(self, adapter: Any, timeout: float = 6.0) -> dict:
        """Wait for the Login button to lose `disabled`/`is-disabled`.

        React enables it a tick after it accepts both fields, so a single
        look right after filling is a race — and losing that race is what
        made the old payload report 'no visible login button' every cycle.
        """
        deadline = time.time() + timeout
        last: dict = {"ok": False, "error": "login button not found"}
        while time.time() < deadline:
            try:
                last = self._eval_json(_LOGIN_BUTTON_JS)
            except Exception as e:  # transient page work
                last = {"ok": False, "error": f"button probe failed: {e}"}
            if last.get("ok"):
                return last
            time.sleep(0.4)
        return last

    def fill_login(self) -> dict:
        """Log in with the configured credentials. Deterministic, no model.

        The credentials come from the environment, so they never enter a
        model's context, a prompt, or a log. The flow is exactly what the
        operator does by hand:

          1. dismiss a leftover dialog (it covers the panel and eats the
             real click),
          2. fill phone + password,
          3. wait for Login to enable,
          4. real-click it (the site ignores synthetic clicks),
          5. wait for the session to actually come back.

        Every failure says what the page looked like, because 'no visible
        login button' four cycles in a row is not a diagnosis.
        """
        phone = os.environ.get("ATLAS_SPORTYBET_PHONE", "")
        password = os.environ.get("ATLAS_SPORTYBET_PASSWORD", "")
        if not phone or not password:
            return {
                "ok": False,
                "error": "no credentials configured "
                "(ATLAS_SPORTYBET_PHONE / ATLAS_SPORTYBET_PASSWORD)",
            }
        adapter = self.capture.adapter
        steps: list = []

        # 1. An 'Odds not acceptable, please try again' dialog sits on top of
        #    the login panel; a real click at the Login button's coordinates
        #    lands on the overlay instead.
        try:
            dismiss = self._eval_json(_LOGIN_DISMISS_JS)
        except Exception as e:
            return {"ok": False, "error": f"could not inspect the page: {e}"}
        if dismiss.get("selector"):
            try:
                adapter.click(dismiss["selector"])
                steps.append(f"dismissed dialog: {dismiss.get('text', '')[:60]}")
                time.sleep(0.4)
            except Exception as e:
                steps.append(f"dialog click failed: {e}")

        # 2./3. Fill, then wait for the button. The phone shape is retried
        # because the prefix rule is the book's, not ours.
        selector = None
        last: dict = {"ok": False, "error": "login not attempted"}
        for candidate in self._phone_candidates(phone):
            code = _LOGIN_JS.replace("__PHONE__", json.dumps(candidate)).replace(
                "__PASSWORD__", json.dumps(password)
            )
            try:
                filled = self._eval_json(code)
            except Exception as e:
                return {"ok": False, "error": f"login fill failed: {e}", "steps": steps}
            if not filled.get("ok"):
                last = filled
                continue
            button = self._wait_login_button(adapter, timeout=6.0)
            if button.get("ok"):
                selector = button.get("selector")
                steps.append(
                    f"filled {filled.get('phone_field')} ({len(candidate)} digits)"
                )
                break
            last = {
                "ok": False,
                "error": button.get("error") or "login button stayed disabled",
                "hint": button.get("hint"),
                "button_class": button.get("cls"),
                "phone_field": filled.get("phone_field"),
            }
            steps.append(
                f"phone of {len(candidate)} digits refused: "
                f"{last.get('hint') or last.get('error')}"
            )
        if not selector:
            return {**last, "ok": False, "steps": steps}

        # 4. The submit needs a REAL browser click.
        start = time.time()
        try:
            adapter.click(selector)
        except Exception as e:
            return {"ok": False, "error": f"login click failed: {e}", "steps": steps}
        steps.append("clicked Login")

        # 5. The form stays in the DOM after a successful login (it is only
        #    hidden), so wait for the session check to stop seeing it. The
        #    SPA swaps the header for 'My Account' asynchronously and this
        #    build took longer than the old 20s window — the click HAD
        #    worked, the confirmation gave up first (2026-10-30).
        deadline = time.time() + 45.0
        while time.time() < deadline:
            time.sleep(1.5)
            try:
                session = self.capture.session_expired()
            except Exception:
                continue
            if session.get("hasAccount") and not session.get("visiblePassword"):
                steps.append(f"session restored in {time.time() - start:.0f}s")
                return {"ok": True, "steps": steps}
            if not session.get("expired") and not session.get("probe_failed"):
                steps.append(f"session restored in {time.time() - start:.0f}s")
                return {"ok": True, "steps": steps}

        # Not logged in. Say WHY: an OTP needs a human, a rejection does not.
        try:
            state = self._eval_json(_LOGIN_STATE_JS)
        except Exception:
            state = {}
        if state.get("otp"):
            return {
                "ok": False,
                "error": "the book is asking for a verification code (OTP) — "
                "a human has to enter it",
                "steps": steps,
            }
        if state.get("message"):
            return {
                "ok": False,
                "error": f"login rejected: {state['message']}",
                "steps": steps,
            }
        return {
            "ok": False,
            "error": "login did not take (the login form is still showing)",
            "steps": steps,
        }

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


# The login form is a modal built from the SAME components as the betslip
# (.m-input, .af-button--primary), and it sits UNDER any leftover dialog
# ("Odds not acceptable, please try again"). Three things broke the old
# payload, all found against the live page on 2026-10-30:
#
#   1. The Login button was searched inside `pwd.closest('form') ||
#      pwd.parentElement`. The password's parent is `.m-input-wrapper`
#      (310x54) and the button lives in `.m-btn-wrap` three levels up, so
#      the search never reached it -> "no visible login button", every cycle.
#   2. The phone field is `input[name=phone]` (type text, "Mobile Number"),
#      NOT `input[type=tel]`, so the phone was never filled.
#   3. The button is `is-disabled` until React has accepted BOTH fields; the
#      enabled state arrives a tick after the events, so it has to be polled.
#
# The payload therefore scopes by the LOGIN PANEL (not by a parent element),
# fills by name/placeholder, and only MARKS the button: the submit still
# needs a real browser click (the site ignores untrusted events).
_LOGIN_JS = r"""
(() => {
  const vis = (e) => {
    if (!e) return false;
    const r = e.getBoundingClientRect();
    if (!r || r.width <= 0 || r.height <= 0) return false;
    const cs = window.getComputedStyle(e);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const setNative = (el, value) => {
    const proto = el instanceof HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new Event('blur', { bubbles: true }));
  };
  // Scope: the login panel. Its password field is the anchor, so walk up to
  // the panel that OWNS the button instead of stopping at the input wrapper.
  const pwAll = Array.from(document.querySelectorAll('input[type=password]')).filter(vis);
  if (!pwAll.length) return JSON.stringify({ok: false, error: 'no visible password field'});
  const pwd = pwAll[0];
  let scope = pwd.closest('.m-page--login, [role=dialog], .m-pop-dialog__wrapper, .es-dialog');
  if (!scope) {
    scope = pwd;
    for (let i = 0; i < 8 && scope.parentElement; i++) scope = scope.parentElement;
  }
  // Phone: by name/placeholder first (the live field is input[name=phone]
  // with placeholder 'Mobile Number'), then type=tel, then the visible text
  // input that precedes the password in DOM order.
  const inputs = Array.from(scope.querySelectorAll('input')).filter(vis);
  const textish = (i) => {
    const t = (i.type || 'text').toLowerCase();
    return t !== 'password' && t !== 'checkbox' && t !== 'radio' && t !== 'submit';
  };
  // The book renders the SAME login twice: an inline form in the header
  // (input[name=phone] + input[name=psd]) and a modal (.m-page--login).
  // Both name the phone field `phone`, and the header also carries an
  // unrelated 'Mobile Number' box from Instant Registration, so the name
  // is the only reliable discriminator.
  let tel = Array.from(document.querySelectorAll('input[name=phone]'))
    .filter(vis).filter(textish)[0] || null;
  if (!tel) tel = inputs.filter(textish).find(
    (i) => /phone|mobile/i.test((i.name || '') + ' ' + (i.placeholder || ''))
  );
  if (!tel) tel = inputs.filter(textish).find((i) => (i.type || '').toLowerCase() === 'tel');
  if (!tel) {
    const all = Array.from(document.querySelectorAll('input')).filter(vis);
    const idx = all.indexOf(pwd);
    if (idx > 0 && textish(all[idx - 1])) tel = all[idx - 1];
  }
  if (!tel) return JSON.stringify({ok: false, error: 'no phone field on the login form'});
  // Replace, never append: a stale value (the betslip once typed '10.00'
  // into this field) would otherwise survive and keep the button disabled.
  setNative(tel, __PHONE__);
  setNative(pwd, __PASSWORD__);
  return JSON.stringify({
    ok: true,
    phone_field: (tel.name || tel.placeholder || ''),
    phone_value: tel.value,
    scope: String(scope.className || '').slice(0, 60)
  });
})()
"""

# The Login button turns enabled a tick after the fields are accepted, so it
# is polled. `is-disabled` is the class the live build toggles alongside the
# `disabled` property; both are checked.
_LOGIN_BUTTON_JS = r"""
(() => {
  const vis = (e) => {
    if (!e) return false;
    const r = e.getBoundingClientRect();
    if (!r || r.width <= 0 || r.height <= 0) return false;
    const cs = window.getComputedStyle(e);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const label = (b) => ((b.innerText || b.value || '').trim());
  const isLogin = (b) => /^login$/i.test(label(b));
  const all = Array.from(
    document.querySelectorAll('button, [role=button], input[type=submit]')
  ).filter(vis).filter(isLogin);
  if (!all.length) {
    return JSON.stringify({ok: false, error: 'no login button on the page'});
  }
  const btn = all.find(
    (b) => !b.disabled && !/(^|\s)is-disabled(\s|$)/.test(String(b.className || ''))
  );
  if (!btn) {
    const msg = (document.body.innerText || '')
      .match(/please enter a valid[^\n]*|incorrect[^\n]*|invalid[^\n]*/i);
    return JSON.stringify({
      ok: false,
      error: 'login button stayed disabled',
      hint: msg ? msg[0].trim() : null,
      disabled: all[0].disabled,
      cls: String(all[0].className || '')
    });
  }
  btn.setAttribute('data-atlas-login-btn', '1');
  return JSON.stringify({ok: true, selector: '[data-atlas-login-btn="1"]'});
})()
"""

# A leftover dialog ("Odds not acceptable, please try again") sits ON TOP of
# the login panel; a real click on Login lands on the overlay instead. Only
# dialogs WITHOUT a password field are dismissed, so the login panel itself
# can never be closed by this payload.
_LOGIN_DISMISS_JS = r"""
(() => {
  const vis = (e) => {
    if (!e) return false;
    const r = e.getBoundingClientRect();
    if (!r || r.width <= 0 || r.height <= 0) return false;
    const cs = window.getComputedStyle(e);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const dlgs = Array.from(document.querySelectorAll(
    '.es-dialog, .m-dialog, [role=dialog], .m-pop-dialog__wrapper'
  )).filter(vis);
  for (const d of dlgs) {
    if (d.querySelector('input[type=password]')) continue;
    if (d.querySelector('input[name=phone]')) continue;
    const btn = Array.from(d.querySelectorAll('button, [role=button]'))
      .find((b) => vis(b) && /^(ok|close|got it|confirm|dismiss)$/i.test((b.innerText || '').trim())
                   && !b.disabled);
    if (!btn) continue;
    btn.setAttribute('data-atlas-dismiss', '1');
    return JSON.stringify({
      ok: true,
      selector: '[data-atlas-dismiss="1"]',
      text: (d.innerText || '').trim().slice(0, 80)
    });
  }
  return JSON.stringify({ok: true, selector: null});
})()
"""

# Why did the click not take? A wrong password leaves an inline message; a
# new-device check leaves an OTP box. Those need different answers: one is a
# credentials problem, the other is a human problem.
_LOGIN_STATE_JS = r"""
(() => {
  const t = document.body.innerText || '';
  const msg = (t.match(/incorrect[^\n]*|invalid[^\n]*|not match[^\n]*|verification code[^\n]*/i) || [])[0];
  const otpField = Array.from(document.querySelectorAll(
    'input[name*=code], input[name*=otp], input[autocomplete=one-time-code]'
  )).some((i) => i.getBoundingClientRect().width > 0);
  const otp = otpField || /verification code|one-time code|enter the code/i.test(t);
  return JSON.stringify({
    otp: otp,
    message: msg ? msg.trim().slice(0, 140) : null
  });
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
        # Auto-login backoff: consecutive failures and the time before the
        # next attempt. A rejected password must not be retried every cycle.
        self._login_failures = 0
        self._login_block_until = 0.0

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
        # Back off after a failed login: retrying a rejected password every
        # 30s is how an account gets locked, and a page that needs an OTP
        # will never be fixed by trying harder.
        if time.time() < self._login_block_until:
            return SupervisorResult(
                handled=False,
                reason="auto-login is backing off after a failure "
                f"({int(self._login_block_until - time.time())}s left)",
            )
        result = self.tools.fill_login()
        action = SupervisorAction(
            at=_now(), tool="fill_login", args={}, result=result, ok=bool(result.get("ok"))
        )
        self._audit(action)
        if not result.get("ok"):
            self._login_failures += 1
            delay = min(600.0, 15.0 * (2 ** min(self._login_failures - 1, 5)))
            self._login_block_until = time.time() + delay
            # The operator has to hear this: an auto-login that cannot work
            # leaves the loop alive but unable to bet, which looks exactly
            # like "no opportunities".
            if self.notifier is not None:
                self.notifier.critical(
                    "Automatic login failed",
                    f"{result.get('error')} — retrying in {int(delay)}s "
                    f"(attempt {self._login_failures})",
                )
            return SupervisorResult(
                handled=False,
                reason=f"login failed: {result.get('error')}",
                actions=[action],
            )
        self._login_failures = 0
        self._login_block_until = 0.0
        # Confirm it worked. fill_login already waited for the header to
        # swap; this is the second opinion on a fresh probe.
        for _ in range(6):
            session = self.capture.session_expired()
            if not session.get("expired"):
                return SupervisorResult(
                    handled=True,
                    reason="re-logged in",
                    actions=[action],
                    message="Session expired; re-logged in automatically.",
                )
            time.sleep(1.0)
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

        Nothing raised in here may escape: the supervisor is the LAST line
        of defence, so a dead model endpoint (a 404 from OpenRouter) or a
        wedged tab must degrade to "not handled" and let the next cycle
        retry. It used to propagate, and an unhandled HTTPStatusError took
        the whole trading loop down with it (2026-10-29).
        """
        try:
            return self._handle(result, task)
        except Exception as e:  # noqa: BLE001 — see the docstring
            # Degrading to "not handled" keeps the loop alive, but a silent
            # degradation is how a broken supervisor goes unnoticed for a
            # day. Write it down.
            self._audit(
                SupervisorAction(
                    at=_now(),
                    tool="error",
                    args={"reason": (result.reason or "")[:200]},
                    result={"error": f"{type(e).__name__}: {e}"},
                    ok=False,
                )
            )
            return SupervisorResult(
                handled=False,
                reason=f"supervisor error: {type(e).__name__}: {e}",
            )

    def _handle(
        self, result: CaptureResult, task: Optional[str] = None
    ) -> SupervisorResult:
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
