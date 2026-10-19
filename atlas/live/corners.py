"""Corner model.

Corners are their own counting process: a corner is a distinct event with
its own rate, not a goal. The live board offers a Corners tab with its own
markets (total corners over/under, corner handicaps, and exact corner
counts), and those markets are priced by a separate distribution.

The rate is estimated from what has actually been observed: the corners
already conceded by each side over the minutes played. With no corner
observation there is no rate, and the model abstains rather than inventing
one — a prior-only corner rate would be a guess dressed as a number.

The approximation this makes, stated plainly: corners are assumed to arrive
at a constant rate over the remaining time. In reality the rate rises
through the match, so the remaining count is slightly understated late on.
The alternative — a time-varying hazard fitted to a single observation — has
no data to fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from atlas.live.observe import LiveMatchState


@dataclass(frozen=True)
class CornerModel:
    """The remaining-corner distribution for both sides."""

    home_lambda: float
    away_lambda: float
    home_so_far: int
    away_so_far: int
    elapsed_minutes: float
    remaining_minutes: float

    @classmethod
    def from_state(
        cls, state: LiveMatchState, remaining_minutes: float
    ) -> Optional["CornerModel"]:
        """Estimate the corner rates from the observed corners.

        Returns None when there is no observation to base a rate on.
        """
        elapsed = state.clock_minute
        if elapsed is None or elapsed <= 5.0:
            return None
        home_c = state.home.corners
        away_c = state.away.corners
        if home_c is None or away_c is None:
            return None
        home_c = int(home_c)
        away_c = int(away_c)
        if home_c + away_c <= 0:
            # Corners were watched and none happened: the rate is near zero
            # but not exactly zero (it is a Poisson rate, and a zero rate
            # would make every over-market certain).
            base = 0.08  # corners per team per match
        else:
            base = None
        if base is not None:
            rate_home = base / 90.0
            rate_away = base / 90.0
        else:
            rate_home = home_c / elapsed
            rate_away = away_c / elapsed
        return cls(
            home_lambda=max(0.0, rate_home * remaining_minutes),
            away_lambda=max(0.0, rate_away * remaining_minutes),
            home_so_far=home_c,
            away_so_far=away_c,
            elapsed_minutes=elapsed,
            remaining_minutes=remaining_minutes,
        )

    # --- distributions ----------------------------------------------------

    @staticmethod
    def _poisson_pmf(lam: float, max_k: int = 14) -> list[float]:
        out = [0.0] * (max_k + 1)
        out[0] = math.exp(-lam)
        for k in range(1, max_k + 1):
            out[k] = out[k - 1] * lam / k
        return out

    def home_pmf(self) -> list[float]:
        return self._poisson_pmf(self.home_lambda)

    def away_pmf(self) -> list[float]:
        return self._poisson_pmf(self.away_lambda)

    def remaining_total_pmf(self) -> list[float]:
        """Distribution of the corners still to come, both sides together.

        The corners already conceded are NOT included; a caller pricing a
        whole-match market adds `home_so_far + away_so_far` to the count.
        """
        h = self.home_pmf()
        a = self.away_pmf()
        out = [0.0] * (len(h) + len(a) - 1)
        for i, ph in enumerate(h):
            for j, pa in enumerate(a):
                out[i + j] += ph * pa
        return out

    def diff_pmf(self) -> dict[int, float]:
        """Distribution of the corner DIFFERENCE over the whole match."""
        h = self.home_pmf()
        a = self.away_pmf()
        out: dict[int, float] = {}
        for i, ph in enumerate(h):
            for j, pa in enumerate(a):
                out[i - j] = out.get(i - j, 0.0) + ph * pa
        return out

    def describe(self) -> str:
        return (
            f"corners: home {self.home_so_far}/{self.away_so_far} so far, "
            f"lambda {self.home_lambda:.2f}/{self.away_lambda:.2f} remaining, "
            f"over {self.elapsed_minutes:.0f}' with {self.remaining_minutes:.0f}' left"
        )
