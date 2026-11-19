"""Simulated annealing baseline (Section 4.3.1 and Appendix A.2 of the paper).

The search state is the paper's plan matrix ``X^mu`` for every team: ``n`` tours, each
with a release point, ``n`` visit slots and a collect point. A slot holds a monitoring
point or ``REPEAT`` (repeat the previous entry), so Eq. (3f) holds by construction.
The energy is the mission time plus penalties for tours that break the energy limits
(Eqs. 3c-3d) and for monitoring points that no team visits (Eq. 3b).

Starting from a round-robin plan, random moves are accepted with the Metropolis rule
under logarithmic cooling. Every move needs a plan evaluation, so SA (and the GA, which
reuses this encoding) is meant for small instances, about ``n <= 10``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .problem import (Instance, MissionParams, Solution, TeamPlan, Tour, check_altitude,
                      check_plan, team_mission_time, tour_times)

REPEAT = -1                  # the slot repeats the previous entry of its tour
ENERGY_PENALTY = 50_000.0    # per tour that breaks Eq. (3c), and again per tour breaking (3d)
COVERAGE_PENALTY = 20_000.0  # per monitoring point that no team visits


def solve(instance: Instance, params: Optional[MissionParams] = None, *, steps: int = 20_000,
          t_max: float = 1000.0, t_min: float = 0.1, seed: Optional[int] = None) -> Solution:
    """Plan a mission with simulated annealing.

    Args:
        instance: Monitoring points and team start/end positions.
        params: Vehicle parameters and robustness margins.
        steps: Number of annealing steps (the paper used 100k-200k on ``n = 10``).
        t_max, t_min: Temperature range of the logarithmic schedule
            ``T_k = max(t_min, t_max * ln 2 / ln(1 + k))``.
        seed: Random seed.

    Returns:
        A :class:`~rspect.problem.Solution`; ``details["feasible"]`` tells whether the best
        plan found satisfies every constraint (penalties make infeasible plans possible).
    """
    params = params or MissionParams()
    check_altitude(instance, params)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    state = round_robin(instance)
    scores = [score_team(state, mu, instance, params) for mu in range(instance.m)]
    current = energy(scores, instance.n)
    best_state, best = state.copy(), current
    for k in range(1, steps + 1):
        temperature = max(t_min, t_max * math.log(2) / math.log(1 + k))
        candidate, candidate_scores = state.copy(), list(scores)
        for mu in random_move(candidate, instance, rng):
            candidate_scores[mu] = score_team(candidate, mu, instance, params)
        e = energy(candidate_scores, instance.n)
        if e <= current or rng.random() < math.exp((current - e) / temperature):
            state, scores, current = candidate, candidate_scores, e
            if e < best:
                best_state, best = state.copy(), e

    plan = best_state.plan()
    return Solution.from_plan(plan, instance, params, time.perf_counter() - started,
                              energy=best, feasible=not check_plan(plan, instance, params))


# ------------------------------------------------------------------------------ encoding
@dataclass
class PlanMatrix:
    """Plan matrices of all teams: tour ``i`` of team ``mu`` goes from
    ``release[mu, i]`` through the points in ``slots[mu, i]`` to ``collect[mu, i]``.

    Attributes:
        slots: ``(m, n, n)`` monitoring point indices, or ``REPEAT``.
        release: ``(m, n, 2)`` release positions on the ground.
        collect: ``(m, n, 2)`` collect positions on the ground.
    """

    slots: np.ndarray
    release: np.ndarray
    collect: np.ndarray

    def copy(self) -> "PlanMatrix":
        return PlanMatrix(self.slots.copy(), self.release.copy(), self.collect.copy())

    def team_plan(self, mu: int) -> TeamPlan:
        """Tours of team ``mu``; repeated entries are dropped and empty rows become trivial tours."""
        tours = []
        for i in range(self.slots.shape[1]):
            visits = [int(p) for p in self.slots[mu, i] if p != REPEAT]
            visits = [p for k, p in enumerate(visits) if k == 0 or p != visits[k - 1]]
            tours.append(Tour(np.append(self.release[mu, i], 0.0), tuple(visits),
                              np.append(self.collect[mu, i], 0.0)))
        return tours

    def plan(self) -> List[TeamPlan]:
        return [self.team_plan(mu) for mu in range(self.slots.shape[0])]


def round_robin(instance: Instance) -> PlanMatrix:
    """Initial plan (Section 4.3.1): the points are dealt to the teams in turn and every
    team visits its points one per tour, released and collected right below the point.
    Unused rows are trivial tours."""
    m, n = instance.m, instance.n
    slots = np.full((m, n, n), REPEAT, dtype=np.int64)
    release = np.zeros((m, n, 2))
    collect = np.zeros((m, n, 2))
    for mu in range(m):
        release[mu] = collect[mu] = instance.starts[mu, :2]
        for i, p in enumerate(range(mu, n, m)):
            slots[mu, i, 0] = p
            release[mu, i] = collect[mu, i] = instance.points[p, :2]
    return PlanMatrix(slots, release, collect)


def random_move(state: PlanMatrix, instance: Instance, rng: np.random.Generator) -> Tuple[int, ...]:
    """Apply one random move to ``state`` in place and return the teams it changed.

    With two or more teams, a cross-team move is chosen with probability 0.4: a slot's
    entry moves to a random slot of another team, or two teams swap the entries of two
    slots. Otherwise one to three moves change a single team, each of which displaces the
    release and collect points of a tour, puts a random monitoring point in a slot, or
    makes a slot repeat the previous entry.
    """
    m, n = state.slots.shape[:2]
    if m >= 2 and rng.random() < 0.4:
        a, b = (int(v) for v in rng.choice(m, size=2, replace=False))
        ia, ja, ib, jb = (int(v) for v in rng.integers(n, size=4))
        if rng.random() < 0.5:   # reassign: the entry of team a moves to team b
            state.slots[b, ib, jb] = state.slots[a, ia, ja]
            state.slots[a, ia, ja] = REPEAT
        else:                    # swap entries between teams a and b
            state.slots[a, ia, ja], state.slots[b, ib, jb] = state.slots[b, ib, jb], state.slots[a, ia, ja]
        return a, b

    mu = int(rng.integers(m))
    area = np.asarray(instance.area, dtype=float)
    for _ in range(int(rng.integers(1, 4))):
        i = int(rng.integers(n))
        if rng.random() < 0.5:   # displace the release and collect points of tour i
            for ground in (state.release, state.collect):
                ground[mu, i] = np.clip(ground[mu, i] + rng.uniform(-area, area), 0.0, area)
        elif rng.random() < 0.5:  # resample: a random monitoring point in a random slot
            state.slots[mu, i, int(rng.integers(n))] = int(rng.integers(n))
        else:                     # repeat: a random slot repeats the previous entry
            state.slots[mu, i, int(rng.integers(n))] = REPEAT
    return (mu,)


# ------------------------------------------------------------------------------ energy
@dataclass
class TeamScore:
    """Contribution of one team to the energy."""

    time: float                # tau(X^mu), Eq. (2)
    violations: int            # broken energy constraints (3c) and (3d), counted per tour
    visited: FrozenSet[int]    # monitoring points the team visits


def score_team(state: PlanMatrix, mu: int, instance: Instance, params: MissionParams) -> TeamScore:
    team = state.team_plan(mu)
    violations, visited = 0, set()
    for tour in team:
        if tour.is_trivial:
            continue
        visited.update(tour.points)
        tau_a, tau_g = tour_times(tour, instance.points, params)
        violations += int(tau_a + params.delta_a > params.max_flight_time)
        violations += int(tau_g + params.delta_g > params.max_flight_time)
    t = team_mission_time(team, instance.starts[mu], instance.ends[mu], instance.points, params)
    return TeamScore(t, violations, frozenset(visited))


def energy(scores: Sequence[TeamScore], n: int) -> float:
    """Mission time plus penalties: the SA energy (and minus the GA fitness)."""
    visited = frozenset().union(*(s.visited for s in scores))
    return (max(s.time for s in scores)
            + ENERGY_PENALTY * sum(s.violations for s in scores)
            + COVERAGE_PENALTY * (n - len(visited)))
