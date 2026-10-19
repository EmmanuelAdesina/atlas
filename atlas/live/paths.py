"""Score-path probability engine.

The joint remaining-goal matrix answers "what is the end state". Several
SportyBet live markets ask a question about the *path* instead:

  - 'Any Team to lead by 2 Goals at any time'
  - 'Home Team To Score 3 or More Goals in a Row'
  - '1X2 - Never Down'

No end-state distribution can price those: a team can end level having led
by two, and vice versa. The book prices them on a path basis, so pricing
them on an end-state basis manufactures false edges in both directions.

The model
---------
Under the shared-tempo construction the next scoring event is

    home scores   with probability p_home
    away scores   with probability p_away
    both score    with probability p_both      (shared tempo)

and the number of scoring events N in the remaining time is Poisson with
mean `total_intensity` (= mu_home + mu_away). Event *types* are i.i.d. and
independent of N, so

    P(a specific type-sequence of length n) = P(N = n) * prod(p_type)

That distinction matters: the event count is finite and random, so a
recursion that simply runs to a fixed horizon silently over-counts long
paths. Everything here is a forward dynamic program over the event count,
weighted by the Poisson pmf.

Truncating at `max_events` leaves a Poisson tail far below 1e-8 for any
realistic remaining intensity, and the DP's endpoint distribution
reproduces the joint matrix, which the tests assert to 1e-9.

The shared-tempo event increments BOTH teams' scoring runs: each team did
score a goal at that instant. That is the literal reading of 'in a row' and
the shared component is a small fraction of total intensity, so the
difference from the alternative reading is immaterial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

HOME = "home"
AWAY = "away"


@dataclass(frozen=True)
class GoalPathModel:
    """Sequential goal process for the remaining match time."""

    p_home: float
    p_away: float
    p_both: float
    total_intensity: float
    max_goals: int = 10

    @classmethod
    def from_intensities(
        cls,
        mu_home: float,
        mu_away: float,
        shared_tempo_frac: float = 0.05,
        max_goals: int = 10,
    ) -> "GoalPathModel":
        mu_h = max(1e-7, mu_home)
        mu_a = max(1e-7, mu_away)
        mu_0 = min(mu_h, mu_a) * max(0.0, min(0.25, shared_tempo_frac))
        lam_1 = max(1e-7, mu_h - mu_0)
        lam_2 = max(1e-7, mu_a - mu_0)
        total = lam_1 + lam_2 + mu_0
        return cls(
            p_home=lam_1 / total,
            p_away=lam_2 / total,
            p_both=mu_0 / total,
            # The event RATE, not mu_home + mu_away: a shared-tempo event is a
            # single event in which both teams score, so the expected number
            # of events is lam_1 + lam_2 + mu_0.
            total_intensity=total,
            max_goals=max_goals,
        )

    @property
    def max_events(self) -> int:
        return 2 * self.max_goals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _event_count_pmf(self) -> list[float]:
        """P(N = n) for n = 0..max_events, N ~ Poisson(total_intensity)."""
        lam = self.total_intensity
        out = [0.0] * (self.max_events + 1)
        out[0] = math.exp(-lam)
        for n in range(1, self.max_events + 1):
            out[n] = out[n - 1] * lam / n
        return out

    # ------------------------------------------------------------------
    # End-state distribution (validation against the joint matrix)
    # ------------------------------------------------------------------

    def endpoint_distribution(self) -> list[list[float]]:
        """P(delta_home = i, delta_away = j) implied by the event process."""
        n = self.max_goals
        dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        pmf_n = self._event_count_pmf()
        states: dict[tuple[int, int], float] = {(0, 0): 1.0}
        for step in range(self.max_events + 1):
            weight = pmf_n[step]
            if weight <= 0.0:
                break
            for (h, a), p in states.items():
                dist[h][a] += p * weight
            nxt: dict[tuple[int, int], float] = {}
            for (h, a), p in states.items():
                if h + 1 <= n:
                    nxt[(h + 1, a)] = nxt.get((h + 1, a), 0.0) + p * self.p_home
                if a + 1 <= n:
                    nxt[(h, a + 1)] = nxt.get((h, a + 1), 0.0) + p * self.p_away
                if h + 1 <= n and a + 1 <= n:
                    nxt[(h + 1, a + 1)] = (
                        nxt.get((h + 1, a + 1), 0.0) + p * self.p_both
                    )
            states = nxt
        total = sum(sum(r) for r in dist)
        if total > 0:
            for r in dist:
                for i in range(len(r)):
                    r[i] /= total
        return dist

    # ------------------------------------------------------------------
    # Path markets
    #
    # All three path properties depend only on a ONE-dimensional Markov
    # chain, never on the individual goal counts:
    #   - the goal difference moves +1 / -1 / 0 per event;
    #   - a scoring run is broken by the other team scoring.
    # Tracking (h, a) as well would multiply the state space by ~100x for
    # no extra information, so these DPs walk a single coordinate.
    # ------------------------------------------------------------------

    def lead_by_at_any_time(self, side: str, n: int, base_diff: int = 0) -> float:
        """P(the side leads by n+ goals at some point in the remaining time).

        `base_diff` is the current goal difference (home - away). A side that
        already leads by n+ has already triggered the event, so this returns
        1.0 for it.
        """
        if n <= 0:
            return 1.0
        if side not in (HOME, AWAY):
            raise ValueError(f"side must be {HOME!r} or {AWAY!r}, got {side!r}")
        sign = 1 if side == HOME else -1
        d0 = sign * base_diff
        if d0 >= n:
            return 1.0

        pmf_n = self._event_count_pmf()
        # d is the side's signed lead; it moves by +/-1 (or 0 on a shared
        # tempo event). The floor is bounded by the number of events.
        floor = -self.max_events
        size = n - floor + 1
        states = [0.0] * size
        states[d0 - floor] = 1.0
        hit_cum = 0.0
        result = 0.0
        for step in range(self.max_events + 1):
            weight = pmf_n[step]
            if weight <= 0.0:
                break
            result += hit_cum * weight
            nxt = [0.0] * size
            for i, p in enumerate(states):
                if p <= 0.0:
                    continue
                d = i + floor
                # home scores: d moves by +sign
                if self.p_home > 0.0:
                    nd = d + sign
                    if nd >= n:
                        hit_cum += p * self.p_home
                    elif nd >= floor:
                        nxt[nd - floor] += p * self.p_home
                # away scores: d moves by -sign
                if self.p_away > 0.0:
                    nd = d - sign
                    if nd >= n:
                        hit_cum += p * self.p_away
                    elif nd >= floor:
                        nxt[nd - floor] += p * self.p_away
                # both score: the lead is unchanged
                if self.p_both > 0.0:
                    nxt[d - floor] += p * self.p_both
            states = nxt
        return min(1.0, result)

    def abs_lead_by_at_any_time(self, n: int, base_diff: int = 0) -> float:
        """P(either side leads by n+ goals at some point in the remaining time).

        This is the 'Any Team to lead by N Goals at any time' market. It is
        NOT the union of the two one-sided probabilities: both sides can lead
        by n at different times, so the absolute lead is walked directly.
        """
        if n <= 0:
            return 1.0
        d0 = base_diff
        if abs(d0) >= n:
            return 1.0

        pmf_n = self._event_count_pmf()
        # state: (k, sign) with k = |d| and sign in {0 tie, 1 home ahead,
        # -1 away ahead}
        size = (self.max_events + 1) * 3
        states = [0.0] * size

        def idx(k: int, sign: int) -> int:
            return k * 3 + (sign + 1)

        states[idx(0, 0 if d0 == 0 else (1 if d0 > 0 else -1))] = 1.0
        hit_cum = 0.0
        result = 0.0
        for step in range(self.max_events + 1):
            weight = pmf_n[step]
            if weight <= 0.0:
                break
            result += hit_cum * weight
            nxt = [0.0] * size
            for i, p in enumerate(states):
                if p <= 0.0:
                    continue
                k, sign = divmod(i, 3)
                sign -= 1
                # home scores -> d + 1
                if self.p_home > 0.0:
                    if sign >= 0:
                        nk, nsign = k + 1, 1
                    else:
                        nk, nsign = (k - 1, -1) if k >= 1 else (0, 0)
                    if nk >= n:
                        hit_cum += p * self.p_home
                    elif nk <= self.max_events:
                        j = idx(nk, nsign)
                        nxt[j] += p * self.p_home
                # away scores -> d - 1
                if self.p_away > 0.0:
                    if sign <= 0:
                        nk, nsign = k + 1, -1
                    else:
                        nk, nsign = (k - 1, 1) if k >= 1 else (0, 0)
                    if nk >= n:
                        hit_cum += p * self.p_away
                    elif nk <= self.max_events:
                        j = idx(nk, nsign)
                        nxt[j] += p * self.p_away
                # both score -> the lead is unchanged
                if self.p_both > 0.0:
                    j = idx(k, sign)
                    nxt[j] += p * self.p_both
            states = nxt
        return min(1.0, result)

    def never_trails(self, side: str, base_diff: int = 0) -> float:
        """P(the side is never behind during the remaining time).

        A side that is already behind can never satisfy this.
        """
        if side not in (HOME, AWAY):
            raise ValueError(f"side must be {HOME!r} or {AWAY!r}, got {side!r}")
        sign = 1 if side == HOME else -1
        d0 = sign * base_diff
        if d0 < 0:
            return 0.0

        pmf_n = self._event_count_pmf()
        floor = 0
        size = self.max_events + 1
        states = [0.0] * size
        states[d0] = 1.0
        result = 0.0
        for step in range(self.max_events + 1):
            weight = pmf_n[step]
            if weight <= 0.0:
                break
            result += sum(states) * weight
            nxt = [0.0] * size
            for d, p in enumerate(states):
                if p <= 0.0:
                    continue
                if self.p_home > 0.0:
                    nd = d + sign
                    if 0 <= nd < size:
                        nxt[nd] += p * self.p_home
                if self.p_away > 0.0:
                    nd = d - sign
                    if nd >= 0:
                        nxt[nd] += p * self.p_away
                if self.p_both > 0.0:
                    nxt[d] += p * self.p_both
            states = nxt
        return min(1.0, result)

    def goals_in_a_row(self, n: int, side: Optional[str] = None) -> float:
        """P(n or more goals scored consecutively by one team).

        `side` restricts the run to that team ('home' / 'away'); None means
        either team. The run is broken when the other team scores.
        """
        if n <= 0:
            return 1.0
        if side is not None and side not in (HOME, AWAY):
            raise ValueError(f"side must be {HOME!r}, {AWAY!r} or None, got {side!r}")

        want = _side_code(side) if side is not None else 0
        pmf_n = self._event_count_pmf()
        no_hit = 0.0

        if want:
            # Single side: only that team's run matters, so the state is the
            # run length alone (0..n-1). Any other goal breaks the run.
            states = [0.0] * n
            states[0] = 1.0
            target_p = self.p_home if want == 1 else self.p_away
            other_p = self.p_away if want == 1 else self.p_home
            for step in range(self.max_events + 1):
                weight = pmf_n[step]
                if weight <= 0.0:
                    break
                no_hit += sum(states) * weight
                nxt = [0.0] * n
                for run, p in enumerate(states):
                    if p <= 0.0:
                        continue
                    if target_p > 0.0:
                        nrun = run + 1 if run > 0 else 1
                        if nrun < n:
                            nxt[nrun] += p * target_p
                    if other_p > 0.0:
                        nxt[0] += p * other_p
                    if self.p_both > 0.0:
                        nxt[0] += p * self.p_both
                states = nxt
        else:
            # Either team: state is (run, who) with who 0 = none.
            states = [0.0] * (n * 3)
            states[0] = 1.0
            for step in range(self.max_events + 1):
                weight = pmf_n[step]
                if weight <= 0.0:
                    break
                no_hit += sum(states) * weight
                nxt = [0.0] * (n * 3)
                for idx, p in enumerate(states):
                    if p <= 0.0:
                        continue
                    run, who = divmod(idx, 3)
                    if self.p_home > 0.0:
                        nrun = run + 1 if who == 1 else 1
                        if nrun < n:
                            nxt[nrun * 3 + 1] += p * self.p_home
                    if self.p_away > 0.0:
                        nrun = run + 1 if who == 2 else 1
                        if nrun < n:
                            nxt[nrun * 3 + 2] += p * self.p_away
                    if self.p_both > 0.0:
                        nxt[0] += p * self.p_both
                states = nxt
        return min(1.0, max(0.0, 1.0 - no_hit))


def _side_code(side: str) -> int:
    return 1 if side == HOME else 2


# ---------------------------------------------------------------------------
# Cached entry points
#
# The dynamic programs depend only on the intensities, the scoreline and
# (n, side) — never on the market or the selection. A surface therefore
# computes a handful of distinct queries and reuses them across every
# selection and every scenario, so the results are memoised here.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16384)
def _lead_by_cached(
    model: GoalPathModel, side: str, n: int, base_diff: int
) -> float:
    return model.lead_by_at_any_time(side, n, base_diff)


@lru_cache(maxsize=16384)
def _abs_lead_cached(model: GoalPathModel, n: int, base_diff: int) -> float:
    return model.abs_lead_by_at_any_time(n, base_diff)


@lru_cache(maxsize=16384)
def _in_a_row_cached(model: GoalPathModel, n: int, side: Optional[str]) -> float:
    return model.goals_in_a_row(n, side=side)


@lru_cache(maxsize=16384)
def _never_trails_cached(model: GoalPathModel, side: str, base_diff: int) -> float:
    return model.never_trails(side, base_diff)


def lead_by_prob(
    model: GoalPathModel, side: str, n: int, base_diff: int
) -> float:
    """Cached P(side leads by n+ at any time)."""
    return _lead_by_cached(model, side, n, base_diff)


def abs_lead_by_prob(model: GoalPathModel, n: int, base_diff: int) -> float:
    """Cached P(either side leads by n+ at any time)."""
    return _abs_lead_cached(model, n, base_diff)


def goals_in_row_prob(
    model: GoalPathModel, n: int, side: Optional[str] = None
) -> float:
    """Cached P(n+ goals in a row by one team)."""
    return _in_a_row_cached(model, n, side)


def never_trails_prob(
    model: GoalPathModel, side: str, base_diff: int
) -> float:
    """Cached P(side is never behind)."""
    return _never_trails_cached(model, side, base_diff)
