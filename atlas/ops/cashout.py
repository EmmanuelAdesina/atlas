"""Cashout valuation.

SportyBet offers a cashout on live positions, and a partial one ('One Cut'),
plus 'SportyInsure' (stake insurance). Cashout is not a bonus: it converts a
variable position into a guaranteed floor, usually below fair value. The
question is never "is there a cashout" but "is the cashout worth more than
the position is worth".

The decision
------------
For a position with remaining expected value V and cashout value C:

    take the cashout  <=>  C > V

But V is not the model probability times the odds; it is what the position
is worth GIVEN the risk the operator refuses to carry. That is a risk
preference, and it is a posture decision, not a math decision. So the module
computes the break-even cashout value and compares it to the offered one:

    breakeven_cashout = stake * p_remaining_settle_value

Three reference points are produced, and the operator's posture picks
between them:

  - `model_value`: the expected payout if the position runs to settlement,
    using the model's live probability. The pure-math reference.
  - `risk_free_value`: the guaranteed floor — the best payout available by
    cashing out NOW, or by structuring the position so that some outcome
    guarantees a return. This is the floor a risk-averse posture uses.
  - `cashout_discount`: (offered - model_value) / model_value. Negative means
    the book is offering less than the position is worth, i.e. the cashout is
    a bad deal and the position is worth holding.

Partial cashout ('One Cut') is modelled as a split: cashing out a fraction
f of a position converts f of the variance into a guaranteed floor while
leaving (1 - f) exposed. The value of the split is linear in f under the
model, so the only reason to take a partial cut rather than a full one is
that the operator wants SOME of the floor without giving up the upside —
which is exactly a posture choice, and the module reports the trade it
implies rather than deciding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """An open bet being evaluated for cashout."""

    bet_id: str
    match_id: str
    market_key: str
    selection: str
    stake_ngn: float
    odds: float
    model_p: float = 0.0          # P(selection wins) at the current price
    model_p_lower: Optional[float] = None  # conservative P, for the floor

    @property
    def gross_payout_ngn(self) -> float:
        """Payout if the bet wins."""
        return self.stake_ngn * self.odds

    @property
    def model_value_ngn(self) -> float:
        """Expected payout if the position runs to settlement."""
        p = self.model_p if self.model_p_lower is None else min(
            self.model_p, self.model_p_lower
        )
        return self.stake_ngn * p * self.odds

    @property
    def conservative_value_ngn(self) -> float:
        """Expected payout using the conservative (lower-bound) probability."""
        p = self.model_p_lower if self.model_p_lower is not None else self.model_p
        return self.stake_ngn * p * self.odds


@dataclass
class CashoutValuation:
    """What the cashout is worth, and what it costs."""

    bet_id: str
    stake_ngn: float
    offered_ngn: float
    model_value_ngn: float
    conservative_value_ngn: float
    recommended: bool
    reason: str
    cashout_discount: float = 0.0    # (offered - model) / model; negative = bad deal
    upside_kept_ngn: float = 0.0

    def describe(self) -> str:
        action = "TAKE THE CASHOUT" if self.recommended else "HOLD"
        return (
            f"{action}: offered NGN {self.offered_ngn:.2f} vs model value NGN "
            f"{self.model_value_ngn:.2f} (conservative NGN "
            f"{self.conservative_value_ngn:.2f}), stake NGN {self.stake_ngn:.2f}"
            + (
                f", discount {self.cashout_discount:+.1%}"
                if self.cashout_discount
                else ""
            )
            + f" — {self.reason}"
        )


def value_cashout(
    position: Position,
    offered_ngn: float,
    min_discount: float = 0.0,
) -> CashoutValuation:
    """Value a full cashout against the position.

    `min_discount` is how much BETTER than the model the offer must be
    before the cashout is taken. A conservative posture sets a positive
    value (it demands a premium to give up the upside); an aggressive one
    sets a negative value (it will take a guaranteed floor at a discount).
    """
    model_value = position.model_value_ngn
    conservative = position.conservative_value_ngn
    discount = (
        (offered_ngn - model_value) / model_value if model_value > 0 else 0.0
    )
    # Take the cashout when it beats the model value by the required margin.
    take = offered_ngn >= model_value * (1.0 + min_discount)
    if take:
        reason = (
            f"cashout beats the model value by {discount:+.1%}, "
            f"meeting the {min_discount:+.1%} threshold"
        )
    else:
        reason = (
            f"cashout is {discount:+.1%} against the model value, "
            f"short of the {min_discount:+.1%} threshold; holding keeps the upside"
        )
    return CashoutValuation(
        bet_id=position.bet_id,
        stake_ngn=position.stake_ngn,
        offered_ngn=offered_ngn,
        model_value_ngn=model_value,
        conservative_value_ngn=conservative,
        recommended=take,
        reason=reason,
        cashout_discount=discount,
        upside_kept_ngn=0.0 if take else model_value - offered_ngn,
    )


def value_partial_cashout(
    position: Position,
    offered_ngn: float,
    fraction: float,
) -> CashoutValuation:
    """Value a PARTIAL cashout ('One Cut') of a fraction of the position.

    A partial cut converts part of the variance into a guaranteed floor and
    leaves the rest exposed. Under the model the split is linear, so this
    reports what the operator gives up and keeps rather than pretending
    there is a free lunch in the middle.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    model_value = position.model_value_ngn
    conservative = position.conservative_value_ngn
    # The cut returns a guaranteed amount now; the remainder still rides.
    guaranteed = offered_ngn
    remaining_value = model_value * (1.0 - fraction)
    total_if_cut = guaranteed + remaining_value
    discount_vs_hold = (
        (total_if_cut - model_value) / model_value if model_value > 0 else 0.0
    )
    take = total_if_cut > model_value
    return CashoutValuation(
        bet_id=position.bet_id,
        stake_ngn=position.stake_ngn,
        offered_ngn=offered_ngn,
        model_value_ngn=model_value,
        conservative_value_ngn=conservative,
        recommended=take,
        reason=(
            f"cutting {fraction:.0%} for NGN {offered_ngn:.2f} leaves NGN "
            f"{remaining_value:.2f} riding, total NGN {total_if_cut:.2f} vs "
            f"NGN {model_value:.2f} held ({discount_vs_hold:+.1%})"
        ),
        cashout_discount=discount_vs_hold,
        upside_kept_ngn=remaining_value,
    )


def portfolio_cashout_decision(
    valuations: list[CashoutValuation],
) -> tuple[list[CashoutValuation], list[CashoutValuation]]:
    """Split valuations into (take, hold).

    The portfolio takes a cashout when the position is losing more than the
    model expects, i.e. when the offered floor is above the model value.
    Taking cashout on a position that is merely worth less than its model
    value would be selling a good bet cheaply.
    """
    take = [v for v in valuations if v.recommended]
    hold = [v for v in valuations if not v.recommended]
    return take, hold
