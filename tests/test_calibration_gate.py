"""Degenerate-calibration guard (2026-11-02 fix).

A family with THIN evidence must never receive the isotonic tail's
fantasy p=1.000: `ou_13_5` serialized a degenerate PAVA tail (its few
observations all won) and the extrapolation priced p=1.000, which
`fractional_kelly_stake` sized to the brim — the mechanism behind the
Nashville blast. Thin families return RAW model_p, from thin air to
thin bets: the model's own number, not an extrapolation of 4-of-4.
"""
import pytest

from atlas.live.calibration import (
    MIN_CALIBRATION_OBS,
    CalibrationMap,
    CalibrationRecord,
)


def test_thin_count_family_returns_raw_model_p_not_1():
    """Counts < min_obs (even when a fit got serialized): raw model_p."""
    cal = CalibrationMap()
    cal._counts["ou_13_5"] = 4
    cal._maps["ou_13_5"] = ([0.71], [1.0])  # the degenerate all-won tail
    assert cal.is_calibrated("ou_13_5") is False
    # the ask sits ABOVE the lone calibration point: old code returned 1.0
    assert cal.calibrate("ou_13_5", 0.85) == pytest.approx(0.85)
    assert cal.calibrate("ou_13_5", 0.71) == pytest.approx(0.71)


def test_missing_fit_zero_sample_family_returns_raw_model_p():
    """Zero observations: no map at all -> raw model_p at every ask."""
    cal = CalibrationMap()
    assert cal.calibrate("ou_13_5", 0.71) == pytest.approx(0.71)
    assert cal.calibrate("handicap_0_3", 0.9) == pytest.approx(0.9)


def test_loaded_min_obs_is_serialized_not_default():
    """A stale map serialized with a small min_obs keeps ITS bar (and the
    counts gate still binds it); load() must not silently adopt 50."""
    m = CalibrationMap()
    m.min_obs = 1
    m._counts["ou_13_5"] = 4
    m._maps["ou_13_5"] = ([0.71], [1.0])
    m.save()
    loaded = CalibrationMap.load()
    assert loaded.min_obs == 1
    # counts(4) >= loaded min_obs(1) -> is_calibrated True -> the map RUNS.
    assert loaded.is_calibrated("ou_13_5") is True
    # ...which is exactly why the DECIDED bar for the live campaign is
    # MIN_CALIBRATION_OBS=50 at fit time; the guard's second leg (counts)
    # is what stops a 4/4 tail when the bar is still 50.
    d = CalibrationMap()
    d._counts["ou_13_5"] = 4
    d._maps["ou_13_5"] = ([0.71], [1.0])
    assert d.is_calibrated("ou_13_5") is False  # 4 < 50
    assert d.calibrate("ou_13_5", 0.85) == pytest.approx(0.85)


def test_properly_calibrated_family_still_interpolates():
    """The guard must not lobotomize legitimate calibration: 60 mixed
    records over one family -> is_calibrated True -> PAVA map is used."""
    cal = CalibrationMap()
    records = []
    outcomes = [1, 0] * 30  # 60 records, mixed wins/losses
    for i, w in enumerate(outcomes):
        records.append(
            CalibrationRecord(
                model_p=0.60 + 0.2 * (i % 2),
                actual_win=w,
                market_key="ou_1_5",
                placed_at="",
            )
        )
    cal.fit(records)
    assert cal.is_calibrated("ou_1_5") is True
    p = cal.calibrate("ou_1_5", 0.65)
    assert 0.0 < p < 1.0  # calibrated, not the raw 0.65, not 1.0


def test_min_calibration_obs_is_the_pre_registered_bar():
    """Pin the bar constant: 50 is pre-registered — do not tune down."""
    assert MIN_CALIBRATION_OBS == 50
