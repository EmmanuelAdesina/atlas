"""Tests for the corner model and the corner market families.

Corners are a separate counting process with their own rate. The rate comes
from what was actually observed, and with no observation the model abstains
rather than inventing one.
"""

from datetime import datetime, timezone

import pytest

from atlas.live.corners import CornerModel
from atlas.live.estimate import LiveStateSpaceModel
from atlas.live.observe import LiveMatchState, TeamStats


def state(corners_h=None, corners_a=None, minute=70.0):
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=1, corners=corners_h),
        away=TeamStats(name="Away", goals=0, corners=corners_a),
        period="2h",
        clock_minute=minute,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


def test_no_observation_means_no_model():
    assert CornerModel.from_state(state(), remaining_minutes=20.0) is None


def test_too_early_means_no_model():
    assert (
        CornerModel.from_state(state(corners_h=1, corners_a=0, minute=3.0),
                                remaining_minutes=87.0)
        is None
    )


def test_the_rate_comes_from_the_observed_corners():
    m = CornerModel.from_state(
        state(corners_h=5, corners_a=3, minute=70.0), remaining_minutes=20.0
    )
    assert m is not None
    # 5 corners in 70 minutes -> the same rate over the 20 minutes left.
    assert m.home_lambda == pytest.approx(5 / 70.0 * 20.0)
    assert m.away_lambda == pytest.approx(3 / 70.0 * 20.0)


def test_no_corners_observed_gives_a_small_prior_rate():
    m = CornerModel.from_state(
        state(corners_h=0, corners_a=0, minute=70.0), remaining_minutes=20.0
    )
    assert m is not None
    assert 0.0 < m.home_lambda < 0.1


def test_the_total_pmf_is_normalised():
    m = CornerModel.from_state(
        state(corners_h=5, corners_a=3, minute=70.0), remaining_minutes=20.0
    )
    assert m is not None
    assert sum(m.remaining_total_pmf()) == pytest.approx(1.0, abs=1e-9)
    assert sum(m.diff_pmf().values()) == pytest.approx(1.0, abs=1e-9)


def test_the_total_pmf_is_centred_on_the_expected_total():
    m = CornerModel.from_state(
        state(corners_h=5, corners_a=3, minute=70.0), remaining_minutes=20.0
    )
    assert m is not None
    pmf = m.remaining_total_pmf()
    mean = sum(k * p for k, p in enumerate(pmf))
    expected = m.home_lambda + m.away_lambda
    assert mean == pytest.approx(expected, abs=0.05)


def test_corner_markets_price():
    model = LiveStateSpaceModel(now=datetime.now(timezone.utc))
    st = state(corners_h=5, corners_a=3, minute=70.0)
    est = model.estimate(st, "corners_ou_9_5", "Over 9.5")
    assert est.p is not None
    assert 0.0 < est.p < 1.0
    # Eight corners have happened; more are expected, so Over 9.5 is likely.
    assert est.p > 0.5


def test_corner_markets_abstain_without_observations():
    model = LiveStateSpaceModel(now=datetime.now(timezone.utc))
    est = model.estimate(state(), "corners_ou_9_5", "Over 9.5")
    assert est.p is None
    assert est.abstained is True


def test_corner_handicap_prices():
    model = LiveStateSpaceModel(now=datetime.now(timezone.utc))
    st = state(corners_h=5, corners_a=3, minute=70.0)
    est = model.estimate(st, "corners_ah_-1_5", "Home (-1.5)")
    assert est.p is not None
    assert 0.0 < est.p < 1.0
