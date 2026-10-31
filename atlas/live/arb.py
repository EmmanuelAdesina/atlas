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

Two guards keep this honest, because a false "sure thing" is worse than no
signal at all — it invites the operator to size it like a hedge:

  - the pair must be STRUCTURALLY complementary (the selections partition
    the outcome space), not merely have model probabilities that sum to 1;
  - the implied total must be PLAUSIBLE. A 13% underround is a
    specification error, not free money.

Both were added after 2026-10-28, when 'btts No' (0.9308) was paired with
'rest_of_match_1x2 Home' (0.0693) — sum 1.0001 — and reported as a 13.33%
sure thing. The events overlap (home scores, away does not: both win) and
leave a gap (away scores, home does not: both lose).
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
EXHAUSTIVENESS_TOLERANCE = 0.02

# Minimum edge over the book's implied margin before it is worth acting on.
DEFAULT_MIN_EDGE = 0.005

# An underround this deep is not a market leak, it is a specification error.
# Real books run 1.02-1.12 overround; a genuine cross-market inconsistency
# is worth a few percent at most. The 2026-10-28 'arb' implied 0.8667 (a
# 13.3% underround) because the two legs were never complementary.
MIN_PLAUSIBLE_IMPLIED = 0.95


@dataclass(frozen=True)
class ArbLeg:
    """One leg of a covering set.

    The market key is part of the identity: a leg is (market, selection),
    never just the selection text — 'No' exists in a dozen unrelated
    families, so a name-only key would let a filter match the wrong bet.
    """

    market_key: str
    selection: str
    odds: float


# --- structural complementarity -----------------------------------------
# Two selections cover the outcome space only when they PARTITION it.
# A sum of model probabilities equal to 1.0 is not evidence of that: two
# overlapping events can sum to 1, and two disjoint events can sum to less.
#
# 2026-10-28, cycle 5: 'btts No' (model_p 0.9308) was paired with
# 'rest_of_match_1x2 Home' (0.0693) because they summed to 1.0001. The
# events overlap (home scores, away does not: BOTH win) and leave a gap
# (away scores, home does not: BOTH lose — 9.5% of the time at 1-0/85').
# The "sure thing" was a correlated double bet wearing a 13.3% profit.
#
# The atoms below are the only cross-family complements that exist on this
# board: a 1X2 side against the double chance that covers the other two.

_ONE_X_TWO_ATOMS = {
    "home": frozenset({"H"}),
    "1": frozenset({"H"}),
    "draw": frozenset({"D"}),
    "x": frozenset({"D"}),
    "away": frozenset({"A"}),
    "2": frozenset({"A"}),
    "home or draw": frozenset({"H", "D"}),
    "1x": frozenset({"H", "D"}),
    "home or away": frozenset({"H", "A"}),
    "12": frozenset({"H", "A"}),
    "draw or away": frozenset({"D", "A"}),
    "x2": frozenset({"D", "A"}),
}

# Which families describe the same stretch of the match.
_FAMILY_SCOPE = {
    "1x2": "full",
    "double_chance": "full",
    "1x2_1h": "1h",
    "1h_double_chance": "1h",
    "1x2_2h": "2h",
    "2h_double_chance": "2h",
    "rest_of_match_1x2": "rest",
}

_ONE_X_TWO_FAMILIES = {"1x2", "1x2_1h", "1x2_2h", "rest_of_match_1x2"}
_DOUBLE_CHANCE_FAMILIES = {"double_chance", "1h_double_chance", "2h_double_chance"}


def _outcome_atoms(selection: str) -> Optional[frozenset]:
    return _ONE_X_TWO_ATOMS.get((selection or "").strip().lower())


def are_complementary(
    family_a: str, selection_a: str, family_b: str, selection_b: str
) -> bool:
    """True only when the two selections partition {Home, Draw, Away}.

    Structural, not probabilistic: no model number can make two overlapping
    events complementary, and no model number should be trusted to try.
    """
    scope_a = _FAMILY_SCOPE.get(family_a)
    if scope_a is None or scope_a != _FAMILY_SCOPE.get(family_b):
        return False
    mixed = (
        family_a in _ONE_X_TWO_FAMILIES and family_b in _DOUBLE_CHANCE_FAMILIES
    ) or (
        family_b in _ONE_X_TWO_FAMILIES and family_a in _DOUBLE_CHANCE_FAMILIES
    )
    if not mixed:
        return False
    atoms_a = _outcome_atoms(selection_a)
    atoms_b = _outcome_atoms(selection_b)
    if atoms_a is None or atoms_b is None:
        return False
    return atoms_a | atoms_b == frozenset({"H", "D", "A"}) and not (atoms_a & atoms_b)


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
    legs: list[ArbLeg] = field(default_factory=list)
    implied_total: float = 0.0
    unit_stake: float = 0.0
    guaranteed_return: float = 0.0
    profit: float = 0.0
    profit_pct: float = 0.0
    model_sum: float = 0.0

    def stakes(self) -> list[tuple[str, float, float]]:
        """(selection, stake, guaranteed payout) for a unit investment."""
        out = []
        for leg in self.legs:
            stake = 1.0 / leg.odds
            out.append((leg.selection, stake, stake * leg.odds))
        return out

    def identity_set(self) -> set[tuple[str, str]]:
        """The (market_key, selection) pairs a filter must match on."""
        return {(leg.market_key, leg.selection) for leg in self.legs}

    def describe(self) -> str:
        legs = ", ".join(f"{leg.selection} @ {leg.odds:.2f}" for leg in self.legs)
        return (
            f"{self.family}: {legs} | implied {self.implied_total:.4f} "
            f"-> profit {self.profit_pct:+.2%} per unit "
            f"(model sum {self.model_sum:.3f})"
        )


def _plausible(implied: float) -> bool:
    return implied >= MIN_PLAUSIBLE_IMPLIED


def normalized_identity_set(arbs: Iterable[SurethingArb]) -> set[tuple[str, str]]:
    """The (market_key, selection) pairs of every arb leg, match-ready.

    Normalized once here so the money path's filter and any test agree on
    exactly what "this decision is an arb leg" means.
    """
    return {
        (mk.strip().lower(), sel.strip().lower())
        for arb in arbs
        for mk, sel in arb.identity_set()
    }


def find_surething_arbs(
    selections: Iterable[Selection],
    min_edge: float = DEFAULT_MIN_EDGE,
    tolerance: float = EXHAUSTIVENESS_TOLERANCE,
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
        if not _plausible(implied):
            # Too far below one to be a book's inconsistency: the family is
            # mis-specified (a missing leg, a wrong line) — refuse.
            continue
        legs = [ArbLeg(s.market_key, s.selection, s.odds) for s in sels]
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
    tolerance: float = EXHAUSTIVENESS_TOLERANCE,
) -> list[SurethingArb]:
    """Find covering sets built from MORE THAN ONE family.

    The union of a 3-way set and a 2-way set still covers every outcome, and
    the two sets' prices are set independently by the book, so their
    inconsistencies are additive. This is where live boards actually leak,
    because the 1X2 and Double Chance prices of the same event are not
    forced to agree.

    The pair must be complementary STRUCTURALLY: one 1X2 side against the
    double chance covering the other two, over the same stretch of the
    match. A model probability sum is then only a sanity check, never the
    admission test — a scalar sum cannot tell a partition from a
    coincidence.
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
                    if not are_complementary(
                        fam_a, sel_a.selection, fam_b, sel_b.selection
                    ):
                        continue
                    pair_sum = sel_a.model_p + sel_b.model_p
                    if abs(pair_sum - 1.0) > tolerance:
                        # Structurally complementary, but the model disagrees:
                        # one of the two prices is stale or mis-parsed.
                        continue
                    implied = 1.0 / sel_a.odds + 1.0 / sel_b.odds
                    if implied >= 1.0 - min_edge:
                        continue
                    if not _plausible(implied):
                        continue
                    profit = 1.0 - implied
                    found.append(
                        SurethingArb(
                            family=f"{fam_a} + {fam_b}",
                            legs=[
                                ArbLeg(sel_a.market_key, sel_a.selection, sel_a.odds),
                                ArbLeg(sel_b.market_key, sel_b.selection, sel_b.odds),
                            ],
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
