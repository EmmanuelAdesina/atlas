"""Sure-thing arbitrage scanner over the priced board.

A 'superedge' in the strict sense is a portfolio that profits no matter how
the match ends: a set of mutually exclusive, exhaustive selections whose
inverse prices sum to less than one.

    Home 12.50, Draw 1.14, Away 9.00
      1/12.50 + 1/1.14 + 1/9.00 = 1.068  ->  the book takes 6.8%: no arb

    Home 2.10, Draw 3.40, Away 3.40
      1/2.10 + 1/3.40 + 1/3.40 = 1.064  ->  still a margin

The bookmaker's margin is the obstacle, and a live market is where it goes
wrong: the same outcome is quoted through several families (1X2, Rest of
Match, Double Chance, Draw No Bet) at prices that need not be mutually
consistent. When they are not, the inverse prices of a covering set drop
below one and a guaranteed profit exists.

This module finds that. It works on the model's own probabilities, so it
also refuses to fire when the model does not believe the family is
exhaustive — an arbitrage found on a mis-priced family is not an arbitrage,
it is a modelling error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

# Families whose selections are mutually exclusive and exhaustive: exactly
# one selection wins, always. Only these can carry a sure-thing arb.
EXHAUSTIVE_FAMILIES = {
    "1x2",
    "1x2_1h",
    "1x2_2h",
    "rest_of_match_1x2",
    "double_chance",
    "1h_double_chance",
    "2h_double_chance",
    "btts",
    "1h_btts",
    "2h_btts",
    "odd_even",
    "home_odd_even",
    "away_odd_even",
}

# The model's probabilities for an exhaustive family must sum to ~1.
EXHAUSIVENESS_TOLERANCE = 0.02

# Minimum edge over the book's implied margin before it is worth acting on.
DEFAULT_MIN_EDGE = 0.005


@dataclass
class Selection:
    market_key: str
    selection: str
    odds: float
    model_p: float = 0.0


@dataclass
class SurethingArb:
    """A covering set whose inverse prices sum to less than one."""

    family: str
    legs: list[tuple[str, float]] = field(default_factory=list)
    implied_total: float = 0.0
    unit_stake: float = 0.0
    guaranteed_return: float = 0.0
    profit: float = 0.0
    profit_pct: float = 0.0
    model_sum: float = 0.0

    def stakes(self) -> list[tuple[str, float, float]]:
        """(selection, stake, guaranteed payout) for a unit investment."""
        out = []
        for name, odds in self.legs:
            stake = 1.0 / odds
            out.append((name, stake, stake * odds))
        return out

    def describe(self) -> str:
        legs = ", ".join(f"{n} @ {o:.2f}" for n, o in self.legs)
        return (
            f"{self.family}: {legs} | implied {self.implied_total:.4f} "
            f"-> profit {self.profit_pct:+.2%} per unit "
            f"(model sum {self.model_sum:.3f})"
        )


def find_surething_arbs(
    selections: Iterable[Selection],
    min_edge: float = DEFAULT_MIN_EDGE,
    tolerance: float = EXHAUSIVENESS_TOLERANCE,
) -> list[SurethingArb]:
    """Find every exhaustive family whose inverse prices sum below one.

    `selections` is the priced board: every selection with its odds and the
    model's probability for it. Selections whose model probability is
    missing are ignored, because the exhaustiveness check needs them.
    """
    by_family: dict[str, list[Selection]] = {}
    for s in selections:
        if s.market_key in EXHAUSTIVE_FAMILIES and s.odds > 1.0:
            by_family.setdefault(s.market_key, []).append(s)

    found: list[SurethingArb] = []
    for family, sels in by_family.items():
        if len(sels) < 2:
            continue
        model_sum = sum(s.model_p for s in sels)
        if abs(model_sum - 1.0) > tolerance:
            # The model does not believe this family is exhaustive, so any
            # apparent arbitrage is a modelling error, not an edge.
            continue
        implied = sum(1.0 / s.odds for s in sels)
        if implied >= 1.0 - min_edge:
            continue
        legs = [(s.selection, s.odds) for s in sels]
        profit = 1.0 - implied
        found.append(
            SurethingArb(
                family=family,
                legs=legs,
                implied_total=implied,
                unit_stake=implied,
                guaranteed_return=1.0,
                profit=profit,
                profit_pct=profit,
                model_sum=model_sum,
            )
        )
    found.sort(key=lambda a: -a.profit_pct)
    return found


def find_cross_family_arbs(
    selections: Iterable[Selection],
    min_edge: float = DEFAULT_MIN_EDGE,
    tolerance: float = EXHAUSIVENESS_TOLERANCE,
) -> list[SurethingArb]:
    """Find covering sets built from MORE THAN ONE family.

    Two exhaustive families can be mixed: the union of a 3-way set and a
    2-way set still covers every outcome, and the two sets' prices are set
    independently by the book, so their inconsistencies are additive. This
    is where live boards actually leak, because the 1X2, Rest-of-Match and
    Double Chance prices of the same event are not forced to agree.

    For each pair of exhaustive families, every pair of legs (one from each)
    is tested: the pair covers the outcome space only when the two events
    are complementary, which is verified against the model by requiring the
    pair's model probabilities to sum to ~1.
    """
    by_family: dict[str, list[Selection]] = {}
    for s in selections:
        if s.market_key in EXHAUSTIVE_FAMILIES and s.odds > 1.0:
            by_family.setdefault(s.market_key, []).append(s)

    found: list[SurethingArb] = []
    families = sorted(by_family)
    for i, fam_a in enumerate(families):
        for fam_b in families[i + 1:]:
            for sel_a in by_family[fam_a]:
                for sel_b in by_family[fam_b]:
                    pair_sum = sel_a.model_p + sel_b.model_p
                    if abs(pair_sum - 1.0) > tolerance:
                        # Not complementary: the two legs do not cover the
                        # outcome space, so this is not a sure-thing.
                        continue
                    implied = 1.0 / sel_a.odds + 1.0 / sel_b.odds
                    if implied >= 1.0 - min_edge:
                        continue
                    profit = 1.0 - implied
                    found.append(
                        SurethingArb(
                            family=f"{fam_a} + {fam_b}",
                            legs=[(sel_a.selection, sel_a.odds),
                                  (sel_b.selection, sel_b.odds)],
                            implied_total=implied,
                            unit_stake=implied,
                            guaranteed_return=1.0,
                            profit=profit,
                            profit_pct=profit,
                            model_sum=pair_sum,
                        )
                    )
    found.sort(key=lambda a: -a.profit_pct)
    return found


def best_arbs(
    selections: Iterable[Selection],
    min_edge: float = DEFAULT_MIN_EDGE,
) -> list[SurethingArb]:
    """All sure-thing arbs, single-family and cross-family, best first."""
    sels = list(selections)
    combined = find_surething_arbs(sels, min_edge) + find_cross_family_arbs(
        sels, min_edge
    )
    combined.sort(key=lambda a: -a.profit_pct)
    return combined
