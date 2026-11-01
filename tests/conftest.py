"""Shared fixtures for the ATLAS test suite.

Two campaign worlds:

- Mechanics tests (the default, autouse) run in an explicit 'proven'
  campaign world with the verdict scorer replaced by a no-write double.
  Under the 2026-11-01 fail-closed rule a missing / collect / watch state
  file freezes real-money placement, and the runner recomputes AND
  REWRITES the verdict from the ledger every placement round — a mock
  test ledger always recomputes to 'collect', so the unsigned default
  would either freeze every placement mechanics test or clobber the
  pinned verdict file behind its back.
- Tests of THE WALL ITSELF pin real semantics: they monkeypatch
  atlas.ops.campaign.STATE_FILE themselves, and tests that exercise the
  real scorer declare it with the real_campaign_scorer fixture.
"""

import json

import pytest

import atlas.ops.campaign as campaign

_REAL_REFRESH = campaign.refresh_state_file


@pytest.fixture(autouse=True)
def campaign_proven_world(tmp_path_factory, monkeypatch):
    state = tmp_path_factory.mktemp("campaign") / "state.json"
    state.write_text(json.dumps({"action": "proven"}), encoding="utf-8")
    monkeypatch.setattr(campaign, "STATE_FILE", state)
    monkeypatch.setattr(
        campaign,
        "refresh_state_file",
        lambda ledger: {
            "action": "proven",
            "why": "test double: the pre-registered scorer is not exercised here",
        },
    )


@pytest.fixture
def real_campaign_scorer(monkeypatch):
    """Restore the REAL verdict scorer for tests that pin what
    refresh_state_file computes and persists. The state file follows the
    test's own monkeypatching."""
    monkeypatch.setattr(campaign, "refresh_state_file", _REAL_REFRESH)
