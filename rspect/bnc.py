"""Branch and cut baseline (Section 4.2 and Appendix A.1 of the paper).

Eq. (3) is written as a mixed-integer quadratically constrained program (MIQCP) and
solved with Gurobi. This needs ``pip install gurobipy`` and a Gurobi license (the
license bundled with pip only allows tiny models). The model grows quickly with the
number of points, so it is only practical for a handful of them (about ``n <= 5``).

Variables, for team ``mu``, tour ``i`` and column ``j`` of the plan matrix ``X^mu``
(column 0 is the release point, 1..n the visit slots, n+1 the collect point):

* ``x[mu, i, j, d]``: position of column j, ``d`` in (x, y, z);
* ``z[mu, i, s, k]``: 1 if visit slot ``s`` of tour i is monitoring point k;
* ``T_max``: the overall mission time, which is minimised.

Unused tours are trivial rows. Placing them before the team's first real tour costs
nothing, so the optimum matches the mission time of Eq. (2).
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from .problem import Instance, MissionParams, Solution, Tour, check_altitude


def solve(instance: Instance, params: Optional[MissionParams] = None, *, time_limit: float = 200.0,
          mip_gap: float = 0.05, verbose: bool = False) -> Solution:
    """Plan a mission with branch and cut (Gurobi).

    Args:
        instance: Monitoring points and team start/end positions (keep ``n`` tiny).
        params: Vehicle parameters and robustness margins.
        time_limit: Gurobi time limit [s]; the best plan found so far is returned when hit.
        mip_gap: Relative optimality gap at which Gurobi stops.
        verbose: Print the Gurobi log.

    Returns:
        A :class:`~rspect.problem.Solution`; ``details`` holds the Gurobi ``status``,
        the model ``objective`` and the final ``mip_gap``.
    """
    params = params or MissionParams()
    check_altitude(instance, params)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as err:
        raise ImportError("rspect.bnc needs Gurobi: pip install gurobipy (and a license)") from err

    points, starts, ends = instance.points, instance.starts, instance.ends
    n, m = instance.n, instance.m
    last = n + 1                                      # index of the collect column
    sizes = (instance.area[0], instance.area[1], params.flight_altitude)
    big_m = [s + 100.0 for s in sizes]

    model = gp.Model("rspect_branch_and_cut")
    model.Params.OutputFlag = int(verbose)
    model.Params.NonConvex = 2
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit

    x = model.addVars(m, n, n + 2, 3, lb=0.0, name="x")
    for (_, _, _, d), var in x.items():
        var.UB = sizes[d]
    z = model.addVars(m, n, n, n, vtype=GRB.BINARY, name="z")
    # every UGV must at least drive from its start to its end: a valid bound that
    # lets Gurobi stop as soon as it finds a plan reaching it
    ugv_bound = max(np.hypot(*(ends[mu, :2] - starts[mu, :2])) for mu in range(m)) / params.ugv_speed
    t_max = model.addVar(lb=ugv_bound, name="T_max")

    def distance(dx, dy):
        """New variable equal to the planar distance sqrt(dx^2 + dy^2) (non-convex)."""
        d = model.addVar(lb=0.0)
        model.addQConstr(d * d == dx * dx + dy * dy)
        return d

    def new_var(expr):
        v = model.addVar(lb=0.0)
        model.addConstr(v == expr)
        return v

    # (3b) every monitoring point is visited by some team
    for k in range(n):
        model.addConstr(z.sum("*", "*", "*", k) >= 1)

    for mu, i in itertools.product(range(m), range(n)):
        for s in range(n):  # visit slot s of tour i sits in column s + 1
            used = z.sum(mu, i, s, "*")
            model.addConstr(used <= 1)
            for k, d in itertools.product(range(n), range(3)):  # slot at point k if z = 1 (big-M)
                model.addConstr(x[mu, i, s + 1, d] - points[k, d] <= big_m[d] * (1 - z[mu, i, s, k]))
                model.addConstr(x[mu, i, s + 1, d] - points[k, d] >= -big_m[d] * (1 - z[mu, i, s, k]))
            for d in range(3):  # (3f) a slot without a point repeats the previous column
                model.addQConstr(x[mu, i, s + 1, d] == gp.quicksum(points[k, d] * z[mu, i, s, k] for k in range(n))
                                 + (1 - used) * x[mu, i, s, d])
        model.addConstr(x[mu, i, 0, 2] == 0)     # (3e) release on the ground
        model.addConstr(x[mu, i, last, 2] == 0)  # (3e) collect on the ground
        active = z.sum(mu, i, "*", "*")          # a tour without visits collects where it releases
        for d in range(2):
            model.addConstr(x[mu, i, last, d] - x[mu, i, 0, d] <= sizes[d] * active)
            model.addConstr(x[mu, i, 0, d] - x[mu, i, last, d] <= sizes[d] * active)

    # mission time of every team, Eq. (2)
    for mu in range(m):
        tour_length = []
        for i in range(n):
            flight = gp.LinExpr()
            for j in range(last):  # leg from column j to column j + 1
                horizontal = distance(x[mu, i, j, 0] - x[mu, i, j + 1, 0], x[mu, i, j, 1] - x[mu, i, j + 1, 1])
                climb = model.addVar(lb=-sizes[2], ub=sizes[2])
                model.addConstr(climb == x[mu, i, j, 2] - x[mu, i, j + 1, 2])
                vertical = model.addVar(lb=0.0)
                model.addGenConstrAbs(vertical, climb)
                flight += horizontal / params.uav_speed_h + vertical / params.uav_speed_v
            tau_a = new_var(flight)
            tau_g = new_var(distance(x[mu, i, last, 0] - x[mu, i, 0, 0],
                                     x[mu, i, last, 1] - x[mu, i, 0, 1]) / params.ugv_speed)
            model.addConstr(tau_a + params.delta_a <= params.max_flight_time)  # (3c)
            model.addConstr(tau_g + params.delta_g <= params.max_flight_time)  # (3d)
            longest = model.addVar(lb=0.0)
            model.addGenConstrMax(longest, [tau_a, tau_g])
            tour_length.append(longest)
        transfers = []
        for i in range(n - 1):  # UGV carries the UAV to the next release point while it recharges
            drive = new_var(distance(x[mu, i + 1, 0, 0] - x[mu, i, last, 0],
                                     x[mu, i + 1, 0, 1] - x[mu, i, last, 1]) / params.ugv_speed)
            recharge = new_var(params.recharge_ratio * tour_length[i])  # Eq. (19)
            transfer = model.addVar(lb=0.0)
            model.addGenConstrMax(transfer, [drive, recharge])
            transfers.append(transfer)
        first = distance(x[mu, 0, 0, 0] - starts[mu, 0], x[mu, 0, 0, 1] - starts[mu, 1])
        final = distance(x[mu, n - 1, last, 0] - ends[mu, 0], x[mu, n - 1, last, 1] - ends[mu, 1])
        model.addConstr(t_max >= (first + final) / params.ugv_speed
                        + gp.quicksum(tour_length) + gp.quicksum(transfers))
    model.setObjective(t_max, GRB.MINIMIZE)

    _warm_start(x, z, instance)
    model.optimize()
    if model.SolCount == 0:
        if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            raise RuntimeError("the instance is infeasible: some monitoring point cannot be visited "
                               "within the energy limits (check max_flight_time, delta_a and delta_g)")
        raise RuntimeError(f"Gurobi found no plan before stopping (status {model.Status}); "
                           "try a longer time_limit")

    plan = []
    for mu in range(m):
        team = []
        for i in range(n):
            visits = [k for s in range(n) for k in range(n) if z[mu, i, s, k].X > 0.5]
            visits = [p for idx, p in enumerate(visits) if idx == 0 or p != visits[idx - 1]]
            release = np.array([x[mu, i, 0, 0].X, x[mu, i, 0, 1].X, 0.0])
            collect = np.array([x[mu, i, last, 0].X, x[mu, i, last, 1].X, 0.0])
            team.append(Tour(release, tuple(visits), collect))
        plan.append(team)
    status = {GRB.OPTIMAL: "optimal within the MIP gap", GRB.TIME_LIMIT: "time limit reached"}
    return Solution.from_plan(plan, instance, params, model.Runtime,
                              status=status.get(model.Status, f"Gurobi status {model.Status}"),
                              objective=model.ObjVal, mip_gap=model.MIPGap)


def _warm_start(x, z, instance: Instance) -> None:
    """Round-robin start: points are dealt to the teams in turn and visited one per tour,
    released and collected right below the point. Unused tours come first and sit at the
    team's start position, where they cost nothing."""
    n, m = instance.n, instance.m
    for var in z.values():
        var.Start = 0.0
    for mu in range(m):
        assigned = list(range(mu, n, m))
        unused = n - len(assigned)
        for i in range(n):
            if i >= unused:
                k = assigned[i - unused]
                p = instance.points[k]
                z[mu, i, 0, k].Start = 1.0
                ground, air = (p[0], p[1], 0.0), tuple(p)
            else:
                ground = air = tuple(instance.starts[mu])
            for j in range(n + 2):
                position = ground if j in (0, n + 1) else air
                for d in range(3):
                    x[mu, i, j, d].Start = position[d]
