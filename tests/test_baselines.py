"""The baselines are slow by nature, so these tests use tiny instances and short runs."""
import pytest

from rspect import MissionParams, bnc, check_plan, ga, make_instance, mission_time, rspect, sa

P = MissionParams()


def test_round_robin_start_is_feasible():
    instance = make_instance(6, 2, seed=1)
    assert check_plan(sa.round_robin(instance).plan(), instance, P) == []


def test_sa_improves_on_its_start():
    instance = make_instance(5, 1, seed=2)
    start = mission_time(sa.round_robin(instance).plan(), instance, P)
    solution = sa.solve(instance, steps=3_000, seed=0)
    assert solution.details["feasible"] and check_plan(solution.plan, instance, P) == []
    assert solution.makespan < start


@pytest.mark.parametrize("solver", [rspect, sa, ga, bnc])
def test_solvers_reject_points_at_another_altitude(solver):
    instance = make_instance(4, 1, seed=1, flight_altitude=150.0)  # params say 100 m
    with pytest.raises(ValueError, match="flight_altitude"):
        solver.solve(instance, P)


def test_ga_returns_a_plan_for_every_team():
    instance = make_instance(5, 2, seed=3)
    solution = ga.solve(instance, population=30, generations=10, seed=0)
    assert len(solution.plan) == 2
    assert solution.makespan == pytest.approx(max(solution.team_times))
    assert solution.details["feasible"] == (check_plan(solution.plan, instance, P) == [])


def test_branch_and_cut_on_a_tiny_instance(gurobi):
    instance = make_instance(2, 1, seed=1)
    solution = bnc.solve(instance, time_limit=30)
    assert check_plan(solution.plan, instance, P, tol=1e-3) == []
    assert solution.details["mip_gap"] <= 0.05 + 1e-9


def test_branch_and_cut_reports_infeasible_instances(gurobi):
    instance = make_instance(2, 1, seed=1)
    too_strict = MissionParams(delta_a=550.0)  # every flight needs 100 s, only 50 s left
    with pytest.raises(RuntimeError, match="infeasible"):
        bnc.solve(instance, too_strict, time_limit=30)


def test_branch_and_cut_is_not_worse_than_rspect(gurobi, glns):
    """An optimal plan (within the 5% gap) can't be worse than any feasible plan."""
    instance = make_instance(3, 1, seed=12)
    optimum = bnc.solve(instance, time_limit=30)
    assert optimum.details["mip_gap"] <= 0.05 + 1e-9
    assert optimum.makespan <= rspect.solve(instance).makespan / 0.95 + 1e-6
