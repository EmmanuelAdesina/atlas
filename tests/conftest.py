"""Shared fixtures for the ATLAS test suite.

Unit tests exercise bet MECHANICS: stake ranges, the daily stop,
placement, settlement, the dashboard. The campaign WALL itself is pinned
in test_campaign.py. Under the 2026-11-01 fail-closed campaign rule
(missing / collect / watch / observe state all freeze real-money
placement), the module-level campaign STATE_FILE would freeze every
mechanics test that touches a BetLedger — so every test runs in an
explicit 'proven' campaign world.

A test that wants to pin the WALL overrides atlas.ops.campaign.STATE_FILE
itself (its monkeypatch stacks on top of this one and wins).
"""

import json

import pytest

import atlas.ops.campaign as campaign


@pytest.fixture(autouse=True)
def campaign_proven_world(tmp_path_factory, monkeypatch):
    state = tmp_path_factory.mktemp("campaign") / "state.json"
    state.write_text(json.dumps({"action": "proven"}), encoding="utf-8")
    monkeypatch.setattr(campaign, "STATE_FILE", state)
