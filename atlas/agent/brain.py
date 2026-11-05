"""Opencode as the Atlas brain.

The agent plane drives `opencode run` headlessly:

  opencode run --session <name> --format json --auto "<cycle prompt>"

Key properties:

- `--session` gives the brain CONTINUITY: the same match conversation
  carries across cycles, so it remembers what it learned about the match
  and its own open positions without being re-told.
- `--format json` emits one JSON event per line. Extraction is TOLERANT by
  design: every string field named "text" is harvested, and if the event
  schema drifts in a future opencode release the raw stdout is returned
  whole instead of silently dropping the AI's output.
- `--auto` approves tool permissions that are not explicitly denied. This
  is the autonomy switch — without it an unattended session stalls on a
  permission prompt on the VPS.

Kinematics of `--session`: in opencode 1.18.x `--session` CONTINUES an
existing session only — passing a name that does not exist fails with
"Session not found" (2026-11-05: twelve armed cycles died on exactly that,
before a single tab call). So the first cycle runs WITHOUT `--session`,
the created session id is harvested from the JSON events (the key
"sessionID"), stored on disk, and every later cycle resumes with it. A
stale id (opencode's store cleared, or the store file points at a session
from another machine) falls back to creating a fresh session in the same
cycle — the orchestrator logs which happened.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

OPENCODE_BIN_ENV = "ATLAS_AGENT_OPENCODE_BIN"
DEFAULT_OPENCODE_BIN = "opencode"
OPENCODE_URL_ENV = "ATLAS_AGENT_OPENCODE_URL"
AGENT_MODEL_ENV = "ATLAS_AGENT_MODEL"
AGENT_NAME_ENV = "ATLAS_AGENT_NAME"
TIMEOUT_ENV = "ATLAS_AGENT_TIMEOUT_S"

DEFAULT_CYCLE_TIMEOUT_S = 1800.0
# A killed cycle's messages can overflow stderr; keep the traceback usable.
STDERR_TAIL = 4000
# Cycles re-emit tool output the agent already saw. The orchestrator wants
# the final assistant text, not the whole event stream.
MAX_OUTPUT_CHARS = 20_000
# Resumed-with-a-dead-session marker from the CLI. Triggers a fresh create.
SESSION_NOT_FOUND = "Session not found"
# Where created opencode session ids are kept, per atlas session name, so
# orchestrator restarts resume the SAME conversation.
SESSION_STORE_DEFAULT = "data/ops/agent_session_ids.json"


def _harvest_session_ids(stdout: str) -> list[str]:
    """Every `sessionID`-keyed string in the JSON event stream, in order.

    Tolerant like _harvest_text_lines: opencode's event schema has moved
    between releases, but the session id is what continuation needs, so
    collect every value under a session-id-ish key.
    """
    ids: list[str] = []
    keys = {"sessionID", "sessionId", "session_id"}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in keys and isinstance(v, str) and v.strip():
                        ids.append(v)
                    else:
                        walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(obj)
    # preserve order, drop dupes
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


class SessionIdStore:
    """Persistent name -> opencode-session-id map."""

    def __init__(self, path: str = SESSION_STORE_DEFAULT) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, session: str) -> Optional[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        val = data.get(session) if isinstance(data, dict) else None
        return val if isinstance(val, str) and val.strip() else None

    def put(self, session: str, opencode_id: str) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[session] = opencode_id
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class BrainError(RuntimeError):
    """The opencode subprocess failed, timed out, or exited nonzero."""


def _harvest_text_lines(stdout: str) -> str:
    """Pull assistant text out of --format json event lines, tolerantly.

    Known shapes across opencode releases are covered by one rule: find
    every JSON object in the stream, recursively collect values under keys
    named "text" whose value is a string. If nothing parses (schema drift,
    partial line, plain-text output), the raw stdout comes back whole.
    """
    texts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "text" and isinstance(v, str) and v.strip():
                        texts.append(v)
                    else:
                        walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(obj)
    return "\n".join(texts)


@dataclass
class OpencodeBrain:
    session: str
    model: Optional[str] = None
    agent_name: Optional[str] = None
    # An opencode serve URL (e.g. http://127.0.0.1:4096) keeps session state
    # in the server process across orchestrator restarts. None = spawn and
    # drive the CLI process directly (session state lives on disk).
    attach_url: Optional[str] = None
    binary: str = DEFAULT_OPENCODE_BIN
    workdir: str = "."
    timeout_s: float = DEFAULT_CYCLE_TIMEOUT_S
    env: dict = field(default_factory=dict)
    id_store: SessionIdStore = field(default_factory=SessionIdStore)

    @classmethod
    def from_env(cls, session: str, **overrides: object) -> "OpencodeBrain":
        base = dict(
            model=os.environ.get(AGENT_MODEL_ENV) or None,
            agent_name=os.environ.get(AGENT_NAME_ENV) or None,
            attach_url=os.environ.get(OPENCODE_URL_ENV) or None,
            binary=os.environ.get(OPENCODE_BIN_ENV, DEFAULT_OPENCODE_BIN),
            timeout_s=float(
                os.environ.get(TIMEOUT_ENV, DEFAULT_CYCLE_TIMEOUT_S)
            ),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(session=session, **base)  # type: ignore[arg-type]

    def argv(self, message: str, opencode_session_id: Optional[str] = None) -> list[str]:
        argv = [self.binary, "run"]
        if self.attach_url:
            argv += ["--attach", self.attach_url]
        if opencode_session_id:
            argv += ["--session", opencode_session_id]
        argv += ["--format", "json", "--auto", "--title", f"atlas/{self.session}"]
        if self.model:
            argv += ["--model", self.model]
        if self.agent_name:
            argv += ["--agent", self.agent_name]
        argv.append(message)
        return argv

    def run(self, message: str) -> str:
        # 1) resume the stored opencode session if we have one
        stored = self.id_store.get(self.session)
        if stored:
            try:
                return self._exec(message, stored)
            except BrainError as e:
                if SESSION_NOT_FOUND not in str(e):
                    raise
                # The stored id points at a session this opencode install
                # does not know (cleared store, another machine). Forget it
                # and create fresh, below.
        # 2) create a brand-new session (no --session) and remember its id
        return self._exec(message, None)

    def _exec(self, message: str, opencode_session_id: Optional[str]) -> str:
        try:
            proc = subprocess.run(
                self.argv(message, opencode_session_id),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.workdir,
                env={**os.environ, **self.env},
            )
        except subprocess.TimeoutExpired as e:
            raise BrainError(
                f"opencode cycle timed out after {self.timeout_s:.0f}s "
                f"(session {self.session})"
            ) from e
        except FileNotFoundError as e:
            raise BrainError(
                f"opencode binary {self.binary!r} not found on PATH "
                f"(set {OPENCODE_BIN_ENV})"
            ) from e
        if proc.returncode != 0:
            raise BrainError(
                f"opencode exited {proc.returncode} (session {self.session}): "
                f"{(proc.stderr or proc.stdout)[-STDERR_TAIL:]}"
            )
        # Remember the session id this (or a previous) cycle created. On a
        # resume the store is refreshed with the same id; on a create it
        # caches the new one. Take the LAST id: the final events belong to
        # the session the answer came from.
        ids = _harvest_session_ids(proc.stdout)
        if ids:
            self.id_store.put(self.session, ids[-1])
        text = _harvest_text_lines(proc.stdout) or proc.stdout.strip()
        return text[:MAX_OUTPUT_CHARS]
