import numpy as np
import pytest

from rspect import MissionParams, check_plan, make_instance, rspect
from rspect.rspect import (assign_points, build_tours, collect_point_graph, feasible_collect_points,
                           nearest_neighbor_path)

P = MissionParams()


def test_nearest_neighbor_path():
    points = np.array([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    order, length = nearest_neighbor_path([0.0, 0.0, 0.0], points, [4.0, 0.0, 0.0])
    assert order == [1, 2, 0] and length == pytest.approx(4.0)


@pytest.mark.parametrize("m", [1, 3])
def test_step1_assigns_every_point_once(m):
    instance = make_instance(30, m, seed=2)
    sequences = assign_points(instance, np.random.default_rng(0))
    assert len(sequences) == m
    assert sorted(p for sequence in sequences for p in sequence) == list(range(30))


def test_step21_tours_are_feasible_and_maximal():
    instance = make_instance(40, 1, seed=3)
    sequence = assign_points(instance, np.random.default_rng(0))[0]
    drafts = build_tours(sequence, instance.points, P)
    assert [p for d in drafts for p in d.points] == sequence  # tours cut the sequence in order
    assert all(d.candidates for d in drafts)                   # every tour can be collected
    for draft, following in zip(drafts, drafts[1:]):            # adding one more point breaks it
        assert not feasible_collect_points(draft.release, draft.points + [following.points[0]],
                                           instance.points, P)


def test_margins_give_more_tours():
    instance = make_instance(40, 1, seed=3)
    sequence = assign_points(instance, np.random.default_rng(0))[0]
    safe = MissionParams(delta_a=150.0, delta_g=150.0)
    assert len(build_tours(sequence, instance.points, safe)) > len(build_tours(sequence, instance.points, P))


@pytest.mark.parametrize("n, m", [(30, 1), (60, 3), (5, 5)])
def test_plans_are_feasible(glns, n, m):
    instance = make_instance(n, m, seed=n + m)
    solution = rspect.solve(instance)
    assert check_plan(solution.plan, instance, P) == []
    assert solution.makespan == pytest.approx(max(solution.team_times))


def test_plans_respect_margins(glns):
    instance = make_instance(40, 2, seed=5)
    params = MissionParams(delta_a=120.0, delta_g=90.0)
    assert check_plan(rspect.solve(instance, params).plan, instance, params) == []


def test_glns_finds_the_best_collect_points(glns):
    """The GTSP of Step 2.2 fixes the order of the tours, so its optimum can also be found
    by picking the best collect point of each tour on its own."""
    instance = make_instance(50, 2, seed=9)
    solution = rspect.solve(instance)
    for mu, drafts in enumerate(solution.details["drafts"]):
        weights, _, nodes = collect_point_graph(drafts, instance.starts[mu], instance.ends[mu],
                                                instance.points, P)
        releases = [candidates[0] - 1 for candidates in nodes]  # each release node precedes its candidates
        best = weights[0, releases[0]]
        for i, candidates in enumerate(nodes):
            following = releases[i + 1] if i + 1 < len(nodes) else len(weights) - 2  # next release or end
            best += min(weights[releases[i], c] + weights[c, following] for c in candidates)
        assert solution.team_times[mu] == pytest.approx(best, abs=0.1)
