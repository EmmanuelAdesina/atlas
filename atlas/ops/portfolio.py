"""Same-match portfolio mathematics (Phase B core).

The idea: place N bets on ONE live match so the outcome distribution is
controlled -- 'guaranteed profit anyhow the match might end'. The truth of
that idea, in one identity:

    Consider a set of bets whose winning conditions PARTITION the match
    outcomes (every possible result makes exactly one leg win). Stake
    proportionally:  s_i = budget * (1/odds_i) / (1/odds_1 + ... + 1/odds_n).
    Then EVERY outcome returns the same amount:  R = budget / SUM(1/odds_i).

    The return-on-cover is exactly  ROI = 1/SUM(1/odds_i) - 1.

    SUM(1/odds_i) is the book's overround on that cover. On a single
    bookmaker, one market's overround is positive (the margin), so covering
    ALL outcomes of ONE market with one book is a guaranteed LOSS of the
    margin, no matter how the stakes are sliced. A same-match cover only
    beats the book when the prices across DIFFERENT market families (or
    after a live price lag) imply SUM(1/odds) < 1 -- a real arbitrage. Those
    windows exist on live books and are the honest version of the idea.

    The second honest version is the MIDDLE: complementary lines
    (e.g. Over 1.5 vs Under 3.5) where most outcomes win one leg and one
    narrow outcome (a 2-2 style score for total-goals middle) wins BOTH.
    Zero-loss Middles on one book are also margin-negative overall; the
    structure trades a small tail win for a guaranteed cost elsewhere. The
    math below quantifies exactly what a given leg set costs and floors.

Everything here is pure functions on prices -- no browser, no money path.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def cover_boot(odds: list[float]) -> float:
    """SUM(1/odds) over a set of legs. < 1.0 means a true arbitrage."""
    if not odds:
        raise ValueError("no odds given")
    if any(o <= 1.0 for o in odds):
        raise ValueError("decimal odds below 1.0 are impossible")
    return sum(1.0 / o for o in odds)


@dataclass
class CoverPlan:
    """Stake plan for a leg set that partitions the outcomes."""

    legs: list[str]
    odds: list[float]
    stakes: list[float]
    cover_boot: float
    """SUM(1/odds) -- the overround actually paid on this cover."""
    guaranteed_return: float
    """What EVERY partitioned outcome pays back (so profit = R - budget)."""
    budget: float

    @property
    def guaranteed_pnl(self) -> float:
        return self.guaranteed_return - self.budget

    @property
    def roi(self) -> float:
        if self.budget <= 0:
            return 0.0
        return self.guaranteed_pnl / self.budget


def equal_return_cover(legs: list[str], odds: list[float], budget: float) -> CoverPlan:
    """Stake a partitioning leg set so every outcome returns the same amount.

    This is the closed-form core: s_i = budget * (1/odds_i) / SUM(1/odds).
    Every outcome then pays back budget / SUM(1/odds). Whether that is
    profit is decided entirely by cover_boot: it is (budget/sum) - budget.
    The function BUILDS the plan; it never pledges the profit. Check
    plan.guaranteed_pnl > 0 (equivalently cover_boot < 1) before calling
    anything that moves money.
    """
    boot = cover_boot(odds)
    if budget <= 0:
        raise ValueError("budget must be positive")
    weights = [1.0 / o for o in odds]
    total = sum(weights)
    stakes = [budget * w / total for w in weights]
    return CoverPlan(
        legs=list(legs),
        odds=list(odds),
        stakes=stakes,
        cover_boot=boot,
        guaranteed_return=budget / boot,
        budget=budget,
    )


@dataclass
class Leg:
    """One selection: which match outcomes pay it, and at what price.

    `covers` refers to outcome labels in `outcomes`; a leg can cover any
    subset (a 1X2 Home covers only home-win; a Draw-No-Bet fallback can be
    modelled as two legs). Odds are what it would PAY (decimal), 0 stakes
    are allowed (the solver may return them).
    """

    label: str
    odds: float
    covers: tuple[str, ...]


@dataclass
class PortfolioEval:
    outcomes: list[str]
    legs: list[str]
    stakes: list[float]
    total_stake: float
    pnl_by_outcome: dict[str, float] = field(default_factory=dict)
    worst_pnl: float = 0.0
    best_pnl: float = 0.0

    def describe(self) -> str:
        rows = [f"  {o}: {p:+.2f}" for o, p in self.pnl_by_outcome.items()]
        return (
            f"stake {self.total_stake:.2f}; worst {self.worst_pnl:+.2f}, "
            f"best {self.best_pnl:+.2f}\n" + "\n".join(rows)
        )


def evaluate_portfolio(legs: list[Leg], stakes: list[float], outcomes: list[str]) -> PortfolioEval:
    """Exact outcome-by-outcome profit for a given stake vector."""
    if len(stakes) != len(legs):
        raise ValueError("one stake per leg")
    covered = {o for leg in legs for o in leg.covers}
    missing = [o for o in outcomes if o not in covered]
    if missing:
        raise ValueError(f"legs do not cover outcome(s): {missing}")
    total = sum(stakes)
    pnl: dict[str, float] = {}
    for o in outcomes:
        ret = 0.0
        for leg, stake in zip(legs, stakes):
            if o in leg.covers:
                ret += stake * leg.odds
        pnl[o] = ret - total
    values = list(pnl.values())
    return PortfolioEval(
        outcomes=list(outcomes),
        legs=[leg.label for leg in legs],
        stakes=list(stakes),
        total_stake=total,
        pnl_by_outcome=pnl,
        worst_pnl=min(values),
        best_pnl=max(values),
    )


def best_equal_floor_split(legs: list[Leg], outcomes: list[str], budget: float) -> PortfolioEval | None:
    """Stakes that maximise the WORST outcome's P&L (the 'floor').

    For a strict partition (each outcome covered by exactly one leg) the
    equal-return cover above is optimal; here we verify the partition and
    delegate. Non-partition covers (middles, correlated legs) need a real
    LP -- deliberately NOT hand-implemented: flag it and refuse, so nobody
    mistakes a heuristic for the optimum.
    """
    cover_count = {o: sum(1 for leg in legs if o in leg.covers) for o in outcomes}
    partition = all(c == 1 for c in cover_count.values())
    if not partition:
        return None  # LP territory; not solved here
    odds = [leg.odds for leg in legs]
    boot = cover_boot(odds)  # sum(1/odds)
    if budget <= 0:
        raise ValueError("budget must be positive")
    total = sum(1.0 / o for o in odds)
    stakes = [budget * (1.0 / o) / total for o in odds]
    return evaluate_portfolio(legs, stakes, outcomes)
