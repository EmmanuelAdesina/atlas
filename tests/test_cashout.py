"""Tests for the cashout valuation (atlas.ops.cashout)."""

import pytest

from atlas.ops.cashout import (
    Position,
    portfolio_cashout_decision,
    value_cashout,
    value_partial_cashout,
)


def pos(stake=100.0, odds=2.0, p=0.5, p_lower=None, bet_id="b1"):
    return Position(
        bet_id=bet_id,
        match_id="m1",
        market_key="1x2",
        selection="Home",
        stake_ngn=stake,
        odds=odds,
        model_p=p,
        model_p_lower=p_lower,
    )


def test_model_value_is_the_expected_payout():
    p = pos(stake=100.0, odds=2.0, p=0.5)
    assert p.model_value_ngn == pytest.approx(100.0)
    assert p.gross_payout_ngn == pytest.approx(200.0)


def test_conservative_value_uses_the_lower_bound():
    p = pos(stake=100.0, odds=2.0, p=0.5, p_lower=0.3)
    assert p.conservative_value_ngn == pytest.approx(60.0)
    assert p.model_value_ngn == pytest.approx(60.0)


def test_take_the_cashout_when_it_beats_the_model_value():
    # Model thinks the bet is worth 100; the book offers 130 to end it.
    v = value_cashout(pos(), offered_ngn=130.0)
    assert v.recommended is True
    assert v.cashout_discount == pytest.approx(0.30)
    assert v.upside_kept_ngn == 0.0


def test_hold_when_the_cashout_is_cheap():
    # Model value 100; the book only offers 70 to end it.
    v = value_cashout(pos(), offered_ngn=70.0)
    assert v.recommended is False
    assert v.cashout_discount == pytest.approx(-0.30)
    assert v.upside_kept_ngn == pytest.approx(30.0)


def test_a_conservative_posture_demands_a_premium():
    """A risk-averse posture will not sell a fair-priced bet cheaply."""
    fair = pos()
    assert value_cashout(fair, offered_ngn=100.0, min_discount=0.10).recommended is False
    assert value_cashout(fair, offered_ngn=115.0, min_discount=0.10).recommended is True


def test_an_aggressive_posture_takes_a_discounted_floor():
    """A risk-seeking posture takes a guaranteed floor even at a discount."""
    v = value_cashout(pos(), offered_ngn=80.0, min_discount=-0.20)
    assert v.recommended is True


def test_zero_value_position_does_not_divide_by_zero():
    v = value_cashout(pos(p=0.0), offered_ngn=10.0)
    assert v.cashout_discount == 0.0
    assert v.recommended is True  # any money beats a worthless position


def test_partial_cashout_reports_the_trade_it_implies():
    p = pos(stake=100.0, odds=2.0, p=0.5)  # model value 100
    # Cut 50% of the position for NGN 60: 60 now + 50 riding = 110.
    v = value_partial_cashout(p, offered_ngn=60.0, fraction=0.5)
    assert v.upside_kept_ngn == pytest.approx(50.0)
    assert v.recommended is True
    assert "0%" in v.reason or "50%" in v.reason


def test_partial_cashout_that_costs_money_is_rejected():
    p = pos(stake=100.0, odds=2.0, p=0.5)
    # Cut 50% for NGN 20: 20 now + 50 riding = 70, worse than holding 100.
    v = value_partial_cashout(p, offered_ngn=20.0, fraction=0.5)
    assert v.recommended is False


def test_partial_cashout_rejects_a_nonsense_fraction():
    with pytest.raises(ValueError):
        value_partial_cashout(pos(), offered_ngn=10.0, fraction=0.0)
    with pytest.raises(ValueError):
        value_partial_cashout(pos(), offered_ngn=10.0, fraction=1.0)


def test_portfolio_splits_takes_and_holds():
    losing = value_cashout(pos(bet_id="b1"), offered_ngn=120.0)
    winning = value_cashout(pos(bet_id="b2"), offered_ngn=60.0)
    take, hold = portfolio_cashout_decision([losing, winning])
    assert [v.bet_id for v in take] == ["b1"]
    assert [v.bet_id for v in hold] == ["b2"]
