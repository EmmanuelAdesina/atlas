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

The brain supervises a SUBPROCESS with a timeout. A cycle that outlives
its budget is killed, logged as an error, and the orchestrator backs off —
it never leaves money actions dangling because the tab tool logs every
call BEFORE the daemon answers, and money actions are never retried
anywhere in this plane.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
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

    def argv(self, message: str) -> list[str]:
        argv = [self.binary, "run"]
        if self.attach_url:
            argv += ["--attach", self.attach_url]
        argv += ["--session", self.session, "--format", "json", "--auto"]
        if self.model:
            argv += ["--model", self.model]
        if self.agent_name:
            argv += ["--agent", self.agent_name]
        argv.append(message)
        return argv

    def run(self, message: str) -> str:
        try:
            proc = subprocess.run(
                self.argv(message),
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
        text = _harvest_text_lines(proc.stdout) or proc.stdout.strip()
        return text[:MAX_OUTPUT_CHARS]
