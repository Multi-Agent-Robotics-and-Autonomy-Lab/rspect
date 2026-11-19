"""The planning problem of Section 2 of the paper, shared by all four solvers.

* :class:`MissionParams` - vehicle parameters (defaults from Table 1).
* :class:`Instance` / :func:`make_instance` - monitoring points and team start/end positions.
* :class:`Tour` - one UAV flight: release point -> monitoring points -> collect point.
* :func:`team_mission_time` - the mission time ``tau(X^mu)`` of Eq. (2).
* :func:`check_plan` - the constraints of Eq. (3).
* :class:`Solution` - what every solver returns.

Conventions: coordinates are metres; monitoring points fly at ``z = flight_altitude``;
team positions and release/collect points are on the ground (``z = 0``). The
environment is obstacle-free, so ground projections are ``(x, y, 0)`` and distances
are Euclidean. A UAV takes off vertically, flies at the flight altitude and lands
vertically (VTOL, Assumption 1). A UGV drives in straight lines at constant speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MissionParams:
    """Vehicle and mission parameters. Defaults follow Table 1 of the paper.

    Attributes:
        uav_speed_h: Horizontal UAV speed [m/s].
        uav_speed_v: Vertical UAV speed, for take-off and landing [m/s].
        ugv_speed: UGV speed [m/s].
        flight_altitude: Minimum flight altitude [m]; monitoring points are placed here.
        max_flight_time: UAV endurance ``tau_a_bar`` [s].
        recharge_ratio: ``gamma`` of Eq. (19); ``0`` approximates battery swapping.
        delta_a: Robustness margin of the UAV flight time, Eq. (3c) [s].
        delta_g: Robustness margin of the UGV travel time, Eq. (3d) [s].
    """

    uav_speed_h: float = 10.0
    uav_speed_v: float = 2.0
    ugv_speed: float = 2.5
    flight_altitude: float = 100.0
    max_flight_time: float = 600.0
    recharge_ratio: float = 1.0
    delta_a: float = 0.0
    delta_g: float = 0.0

    @property
    def takeoff_landing_time(self) -> float:
        """Duration of one vertical take-off or landing, ``tau_TL``."""
        return self.flight_altitude / self.uav_speed_v


# ---------------------------------------------------------------------------- instances
@dataclass(frozen=True, eq=False)
class Instance:
    """``n`` monitoring points and the start/end positions of ``m`` teams.

    Attributes:
        points: ``(n, 3)`` monitoring points ``P_UAV``.
        starts: ``(m, 3)`` team start positions ``p_o`` (on the ground).
        ends: ``(m, 3)`` team end positions ``p_f`` (on the ground).
        area: Size ``(x_max, y_max)`` of the environment.
    """

    points: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    area: Tuple[float, float] = (4000.0, 4000.0)

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def m(self) -> int:
        return int(self.starts.shape[0])


def make_instance(n: int, m: int, seed: Optional[int] = None, *,
                  area: Tuple[float, float] = (4000.0, 4000.0), flight_altitude: float = 100.0,
                  starts: Optional[Sequence] = None, ends: Optional[Sequence] = None,
                  random_endpoints: bool = False) -> Instance:
    """Random instance: ``n`` points drawn uniformly over the area at the flight altitude.

    By default every team starts at ``(0, 0)`` and ends at the opposite corner. Pass
    ``starts`` and ``ends`` (one ``(x, y)`` per team, or a single ``(x, y)`` shared by all
    teams), or ``random_endpoints=True`` to draw them uniformly on the ground.
    """
    if not 1 <= m <= n:
        raise ValueError("need 1 <= m <= n")
    point_seed, endpoint_seed = np.random.SeedSequence(seed).spawn(2)
    corner = (float(area[0]), float(area[1]))
    xy = np.random.default_rng(point_seed).uniform((0.0, 0.0), corner, size=(n, 2))
    points = np.column_stack([xy, np.full(n, float(flight_altitude))])

    if starts is not None or ends is not None:
        if starts is None or ends is None:
            raise ValueError("pass both starts and ends")
        starts_xy, ends_xy = starts, ends
    elif random_endpoints:
        rng = np.random.default_rng(endpoint_seed)
        starts_xy = rng.uniform((0.0, 0.0), corner, size=(m, 2))
        ends_xy = rng.uniform((0.0, 0.0), corner, size=(m, 2))
    else:
        starts_xy, ends_xy = (0.0, 0.0), corner
    return Instance(points, _on_ground(starts_xy, m), _on_ground(ends_xy, m), corner)


def _on_ground(xy: Sequence, m: int) -> np.ndarray:
    arr = np.asarray(xy, dtype=float)
    if arr.ndim == 1:
        arr = np.tile(arr, (m, 1))
    if arr.shape[0] != m:
        raise ValueError(f"expected {m} positions")
    return np.column_stack([arr[:, 0], arr[:, 1], np.zeros(m)])


def ground_projection(point: Sequence[float]) -> np.ndarray:
    """``pi(p)``, the closest ground point to ``p``: ``(x, y, 0)`` without obstacles."""
    return np.array([point[0], point[1], 0.0])


def check_altitude(instance: Instance, params: MissionParams) -> None:
    """Raise ``ValueError`` unless every monitoring point is at ``params.flight_altitude``.

    Take-off and landing times come from ``params``, so the points must fly at that
    altitude. Every solver calls this first.
    """
    heights = instance.points[:, 2]
    if not np.allclose(heights, params.flight_altitude):
        raise ValueError(
            f"monitoring points must be at params.flight_altitude = {params.flight_altitude:g} m, "
            f"found {heights.min():g} to {heights.max():g} m; pass the same altitude to "
            "make_instance(flight_altitude=...) and MissionParams(flight_altitude=...)")


# ---------------------------------------------------------------------------- plans
@dataclass(frozen=True, eq=False)
class Tour:
    """One UAV flight, i.e. one row of the plan matrix ``X^mu``.

    The UGV releases the UAV at ``release``; the UAV visits ``points`` (indices into
    ``Instance.points``) in order and lands at ``collect``, where the UGV picks it up.
    A tour without points is *trivial*: an unused row that costs nothing.
    """

    release: np.ndarray
    points: Tuple[int, ...]
    collect: np.ndarray

    @property
    def is_trivial(self) -> bool:
        return len(self.points) == 0


TeamPlan = List[Tour]   # the tours of one team, in order
Plan = List[TeamPlan]   # one team plan per team


# ---------------------------------------------------------------------------- times
def ground_time(a: Sequence[float], b: Sequence[float], params: MissionParams) -> float:
    """UGV travel time ``tau_g`` between two ground points."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) / params.ugv_speed


def flight_time(release: Sequence[float], path: np.ndarray, collect: Sequence[float],
                params: MissionParams) -> float:
    """UAV flight time ``tau_a`` of a tour that visits the ``(k, 3)`` points ``path``.

    Take-off and landing (``2 tau_TL``) plus the horizontal distance
    release -> path -> collect flown at ``uav_speed_h``. Zero for an empty path.
    """
    if len(path) == 0:
        return 0.0
    xy = np.vstack([np.asarray(release)[:2], np.asarray(path)[:, :2], np.asarray(collect)[:2]])
    horizontal = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    return 2.0 * params.takeoff_landing_time + horizontal / params.uav_speed_h


def recharge_time(tau_a: float, tau_g: float, params: MissionParams) -> float:
    """Recharge time ``tau_c`` of Eq. (19): ``gamma * max(tau_a, tau_g)``."""
    return params.recharge_ratio * max(tau_a, tau_g)


def tour_times(tour: Tour, points: np.ndarray, params: MissionParams) -> Tuple[float, float]:
    """``(tau_a, tau_g)`` of a tour: UAV flight time and UGV release-to-collect time."""
    tau_a = flight_time(tour.release, points[list(tour.points)], tour.collect, params)
    return tau_a, ground_time(tour.release, tour.collect, params)


def team_mission_time(team: TeamPlan, start: Sequence[float], end: Sequence[float],
                      points: np.ndarray, params: MissionParams) -> float:
    """Mission time ``tau(X^mu)`` of one team, Eq. (2).

    The UGV drives from ``start`` to the first release point. Each tour lasts as long as
    the slower of the UAV flight and the UGV drive to the collect point. Between tours
    the UGV carries the UAV to the next release point while the UAV recharges. After the
    last tour the UGV drives to ``end`` (no recharge is needed). Trivial tours are skipped.
    """
    tours = [tour for tour in team if not tour.is_trivial]
    if not tours:
        return ground_time(start, end, params)
    total = ground_time(start, tours[0].release, params)
    for i, tour in enumerate(tours):
        tau_a, tau_g = tour_times(tour, points, params)
        total += max(tau_a, tau_g)
        if i + 1 < len(tours):
            transfer = ground_time(tour.collect, tours[i + 1].release, params)
            total += max(transfer, recharge_time(tau_a, tau_g, params))
    return total + ground_time(tours[-1].collect, end, params)


def team_times(plan: Plan, instance: Instance, params: MissionParams) -> List[float]:
    """``tau(X^mu)`` of every team."""
    return [team_mission_time(team, instance.starts[mu], instance.ends[mu], instance.points, params)
            for mu, team in enumerate(plan)]


def mission_time(plan: Plan, instance: Instance, params: MissionParams) -> float:
    """Overall mission time ``max_mu tau(X^mu)``: the objective of Eq. (3a)."""
    return max(team_times(plan, instance, params))


def tour_robustness(tour: Tour, points: np.ndarray, params: MissionParams) -> Tuple[float, float]:
    """Tour robustness of Definition 1: how much the UAV and UGV times of the tour can
    grow before the energy limit is violated."""
    tau_a, tau_g = tour_times(tour, points, params)
    return params.max_flight_time - tau_a, params.max_flight_time - tau_g


def check_plan(plan: Plan, instance: Instance, params: MissionParams, tol: float = 1e-6) -> List[str]:
    """Violations of the constraints of Eq. (3); an empty list means the plan is feasible."""
    problems: List[str] = []
    visited = set()
    limit = params.max_flight_time + tol
    for mu, team in enumerate(plan):
        for i, tour in enumerate(team):
            if tour.is_trivial:
                continue
            visited.update(tour.points)
            tau_a, tau_g = tour_times(tour, instance.points, params)
            if tau_a + params.delta_a > limit:
                problems.append(f"team {mu}, tour {i}: UAV time {tau_a:.1f} s + delta_a is over the limit (3c)")
            if tau_g + params.delta_g > limit:
                problems.append(f"team {mu}, tour {i}: UGV time {tau_g:.1f} s + delta_g is over the limit (3d)")
            if abs(tour.release[2]) > tol or abs(tour.collect[2]) > tol:
                problems.append(f"team {mu}, tour {i}: release/collect point is not on the ground (3e)")
    missing = sorted(set(range(instance.n)) - visited)
    if missing:
        problems.append(f"monitoring points never visited (3b): {missing}")
    return problems


# ---------------------------------------------------------------------------- results
@dataclass
class Solution:
    """Result returned by every solver (``rspect``, ``sa``, ``ga`` and ``bnc``).

    Attributes:
        plan: One list of tours per team.
        team_times: Mission time of every team, Eq. (2) [s].
        makespan: Overall mission time ``max(team_times)`` [s].
        runtime: Computation time of the solver [s].
        details: Solver-specific information.
    """

    plan: Plan
    team_times: List[float]
    makespan: float
    runtime: float
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: Plan, instance: Instance, params: MissionParams, runtime: float,
                  **details: Any) -> "Solution":
        times = team_times(plan, instance, params)
        return cls(plan, times, max(times), runtime, details)

    def summary(self) -> str:
        """Short human-readable description of the plan."""
        lines = [f"mission time {self.makespan:.0f} s, computed in {self.runtime:.2f} s"]
        for mu, (team, t) in enumerate(zip(self.plan, self.team_times)):
            tours = sum(not tour.is_trivial for tour in team)
            visits = sum(len(tour.points) for tour in team)
            lines.append(f"  team {mu + 1}: {tours} tour{'s' * (tours != 1)}, "
                         f"{visits} visit{'s' * (visits != 1)}, {t:.0f} s")
        return "\n".join(lines)
