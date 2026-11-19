"""RSPECT, Algorithm 1 of the paper.

Step 1    Assign the monitoring points to teams and order them with a nearest-neighbour
          TSP. With several teams, points are taken in random order and each goes to the
          team that keeps the longest TSP path shortest (Eq. 4).
Step 2.1  Cut each team's sequence into maximal tours that respect the energy limits
          (Eqs. 3c-3d) and record the feasible collect points ``C_i`` of every tour.
Step 2.2  Choose one collect point per tour by solving a generalized TSP with GLNS,
          minimising the team's mission time (Eq. 2).

Usage::

    from rspect import make_instance, rspect
    solution = rspect.solve(make_instance(n=50, m=3, seed=1))
    print(solution.summary())
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import glns
from .problem import (Instance, MissionParams, Solution, TeamPlan, Tour, check_altitude,
                      flight_time, ground_projection, ground_time, recharge_time)


class InfeasibleMissionError(ValueError):
    """A monitoring point cannot be visited within the energy limits, even on its own."""


@dataclass
class TourDraft:
    """A tour built in Step 2.1, before its collect point is chosen.

    Attributes:
        release: Release point: the ground projection of the tour's first point.
        points: Monitoring points of the tour, in visiting order.
        candidates: Tour points whose ground projection is a feasible collect point (``C_i``).
    """

    release: np.ndarray
    points: List[int]
    candidates: List[int]


def solve(instance: Instance, params: Optional[MissionParams] = None, *, seed: int = 0,
          sigma: float = 0.001) -> Solution:
    """Plan a mission with RSPECT.

    Args:
        instance: Monitoring points and team start/end positions.
        params: Vehicle parameters and robustness margins (default: Table 1, no margins).
        seed: Seed for the random point order of Step 1 (used when ``m > 1``) and for GLNS.
        sigma: GLNS time-limit constant: a team with ``n_mu`` points gets ``sigma * n_mu**3`` s.

    Returns:
        A :class:`~rspect.problem.Solution`. ``details`` holds the Step 1 ``sequences``,
        the Step 2.1 ``drafts`` (tours with their collect candidates) and per-step
        ``timings``. The runtime counts GLNS's own solver time, not the Julia start-up
        and compilation paid on the first call (that is included in ``wall_time``).
    """
    params = params or MissionParams()
    check_altitude(instance, params)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    sequences = assign_points(instance, rng)                                    # Step 1
    t1 = time.perf_counter()
    drafts = [build_tours(sequence, instance.points, params) for sequence in sequences]  # Step 2.1
    t2 = time.perf_counter()
    plan: List[TeamPlan] = []
    step22 = 0.0
    for mu in range(instance.m):                                                # Step 2.2
        team, python_time, glns_time = choose_collect_points(
            drafts[mu], instance.starts[mu], instance.ends[mu], instance.points, params, sigma, seed)
        plan.append(team)
        step22 += python_time + glns_time

    timings = {"step1": t1 - start, "step2.1": t2 - t1, "step2.2": step22}
    return Solution.from_plan(plan, instance, params, sum(timings.values()),
                              sequences=sequences, drafts=drafts, timings=timings,
                              wall_time=time.perf_counter() - start)


# ------------------------------------------------------------------------------ Step 1
def nearest_neighbor_path(start: Sequence[float], candidates: np.ndarray,
                          end: Sequence[float]) -> Tuple[List[int], float]:
    """Nearest-neighbour TSP path from ``start`` through every row of ``candidates`` to ``end``.

    Returns the visiting order (row indices of ``candidates``) and the path length.
    """
    remaining = list(range(len(candidates)))
    order: List[int] = []
    here = np.asarray(start, dtype=float)
    length = 0.0
    while remaining:
        dist = np.linalg.norm(candidates[remaining] - here, axis=1)
        k = int(np.argmin(dist))
        length += float(dist[k])
        here = candidates[remaining[k]]
        order.append(remaining.pop(k))
    return order, length + float(np.linalg.norm(np.asarray(end, dtype=float) - here))


def assign_points(instance: Instance, rng: np.random.Generator) -> List[List[int]]:
    """Step 1 (Algorithm 1, lines 1-15): the visit sequence ``S^mu`` of every team."""
    points, m = instance.points, instance.m
    if m == 1:
        order, _ = nearest_neighbor_path(instance.starts[0], points, instance.ends[0])
        return [order]

    members: List[List[int]] = [[] for _ in range(m)]
    sequences: List[List[int]] = [[] for _ in range(m)]
    path_costs = [0.0] * m
    unassigned = list(range(instance.n))
    while unassigned:
        p = unassigned.pop(int(rng.integers(len(unassigned))))  # uniformly random point
        trial_orders, trial_costs = [], []
        for mu in range(m):  # NN-TSP path of every team with p added
            team_points = members[mu] + [p]
            order, cost = nearest_neighbor_path(instance.starts[mu], points[team_points],
                                                instance.ends[mu])
            trial_orders.append([team_points[k] for k in order])
            trial_costs.append(cost)
        # Eq. (4): give p to the team whose new path keeps the longest path shortest
        best = min(range(m), key=lambda mu: max(
            [path_costs[v] for v in range(m) if v != mu] + [trial_costs[mu]]))
        members[best].append(p)
        sequences[best] = trial_orders[best]
        path_costs[best] = trial_costs[best]
    return sequences


# ------------------------------------------------------------------------------ Step 2.1
def feasible_collect_points(release: np.ndarray, tour: Sequence[int], points: np.ndarray,
                            params: MissionParams) -> List[int]:
    """Tour points whose ground projection can serve as the collect point (lines 31-33):
    the UAV must reach it within ``max_flight_time - delta_a`` (Eq. 3c) and the UGV within
    ``max_flight_time - delta_g`` (Eq. 3d)."""
    path = points[list(tour)]
    feasible = []
    for q in tour:
        collect = ground_projection(points[q])
        uav_ok = flight_time(release, path, collect, params) + params.delta_a <= params.max_flight_time
        ugv_ok = ground_time(release, collect, params) + params.delta_g <= params.max_flight_time
        if uav_ok and ugv_ok:
            feasible.append(q)
    return feasible


def build_tours(sequence: Sequence[int], points: np.ndarray,
                params: MissionParams) -> List[TourDraft]:
    """Step 2.1 (Algorithm 1, lines 20-39): cut a sequence into maximal feasible tours.

    A tour is released below its first point and keeps taking the next points of the
    sequence as long as at least one feasible collect point remains.
    """
    drafts: List[TourDraft] = []
    k = 0
    while k < len(sequence):
        release = ground_projection(points[sequence[k]])
        tour = [sequence[k]]
        candidates = feasible_collect_points(release, tour, points, params)
        if not candidates:
            raise InfeasibleMissionError(f"point {sequence[k]} cannot be visited within the energy limit")
        k += 1
        while k < len(sequence):
            longer = tour + [sequence[k]]
            longer_candidates = feasible_collect_points(release, longer, points, params)
            if not longer_candidates:
                break  # sequence[k] starts the next tour
            tour, candidates = longer, longer_candidates
            k += 1
        drafts.append(TourDraft(release, tour, candidates))
    return drafts


# ------------------------------------------------------------------------------ Step 2.2
def collect_point_graph(drafts: Sequence[TourDraft], start: np.ndarray, end: np.ndarray,
                        points: np.ndarray, params: MissionParams):
    """The GTSP of Algorithm 1 (lines 41-49) for one team.

    Nodes are the start, the release point of every tour, every collect candidate, the end,
    and a dummy node that closes the start-to-end path into the cycle GLNS optimises.
    Clusters are the candidate sets ``C_i``; all other nodes are singleton clusters.

    Edge weights (missing edges are ``inf``):

    * start - first release point: UGV travel time;
    * release_i - candidate c: ``max(tau_a, tau_g)`` of tour i collected at c;
    * candidate c - release_{i+1}: ``max(UGV travel time, recharge time of Eq. 19)``;
    * last candidates - end: UGV travel time (no recharge after the last tour, Eq. 2);
    * end - dummy - start: zero.

    Returns ``(weights, clusters, candidate_nodes)``, where ``candidate_nodes[i][r]`` is the
    node of ``drafts[i].candidates[r]``.
    """
    size = 3 + len(drafts) + sum(len(d.candidates) for d in drafts)
    start_node, end_node, dummy_node = 0, size - 2, size - 1
    weights = np.full((size, size), np.inf)
    np.fill_diagonal(weights, 0.0)

    def connect(a: int, b: int, w: float) -> None:
        weights[a, b] = weights[b, a] = w

    release_nodes, candidate_nodes, clusters = [], [], [[start_node]]
    node = 1
    for draft in drafts:
        release_nodes.append(node)
        candidate_nodes.append(list(range(node + 1, node + 1 + len(draft.candidates))))
        clusters += [[node], candidate_nodes[-1]]
        node += 1 + len(draft.candidates)
    clusters += [[end_node], [dummy_node]]

    connect(start_node, release_nodes[0], ground_time(start, drafts[0].release, params))
    for i, draft in enumerate(drafts):
        path = points[draft.points]
        for q, c_node in zip(draft.candidates, candidate_nodes[i]):
            collect = ground_projection(points[q])
            tau_a = flight_time(draft.release, path, collect, params)
            tau_g = ground_time(draft.release, collect, params)
            connect(release_nodes[i], c_node, max(tau_a, tau_g))
            if i + 1 < len(drafts):
                transfer = ground_time(collect, drafts[i + 1].release, params)
                connect(c_node, release_nodes[i + 1], max(transfer, recharge_time(tau_a, tau_g, params)))
            else:
                connect(c_node, end_node, ground_time(collect, end, params))
    connect(end_node, dummy_node, 0.0)
    connect(dummy_node, start_node, 0.0)
    return weights, clusters, candidate_nodes


def choose_collect_points(drafts: Sequence[TourDraft], start: np.ndarray, end: np.ndarray,
                          points: np.ndarray, params: MissionParams, sigma: float,
                          seed: int) -> Tuple[TeamPlan, float, float]:
    """Step 2.2 (Algorithm 1, lines 40-53) for one team.

    Returns the team's tours, the Python time spent building the graph and reading the
    answer, and GLNS's own solver time.
    """
    if not drafts:  # a team without points drives straight to its end position
        return [], 0.0, 0.0
    t = time.perf_counter()
    weights, clusters, candidate_nodes = collect_point_graph(drafts, start, end, points, params)
    python_time = time.perf_counter() - t
    n_points = sum(len(d.points) for d in drafts)
    solution = glns.solve(weights, clusters, max_time=sigma * n_points ** 3, seed=seed)

    t = time.perf_counter()
    chosen = set(solution.tour)
    team: TeamPlan = []
    for draft, nodes in zip(drafts, candidate_nodes):
        q = next(c for c, node in zip(draft.candidates, nodes) if node in chosen)
        team.append(Tour(draft.release, tuple(draft.points), ground_projection(points[q])))
    return team, python_time + time.perf_counter() - t, solution.solver_time
