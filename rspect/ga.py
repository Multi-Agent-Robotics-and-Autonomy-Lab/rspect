"""Genetic algorithm baseline (Section 4.3.2 and Appendix A.3 of the paper).

A chromosome holds the plan matrices of all teams, in the same encoding as the SA
baseline, and its fitness is minus the SA energy. Each generation

1. keeps the best plan (elitism),
2. takes the ``num_parents`` best plans as parents (steady-state selection),
3. builds every child from two consecutive parents, taking each team's plan matrix
   from either parent with equal probability (crossover at the team level), and
4. mutates a child with probability ``mutation_rate`` using the SA move.

Like SA, this is meant for small instances (about ``n <= 10``).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .problem import Instance, MissionParams, Solution, check_altitude, check_plan
from .sa import energy, random_move, round_robin, score_team


def solve(instance: Instance, params: Optional[MissionParams] = None, *, population: int = 200,
          generations: int = 100, num_parents: int = 75, mutation_rate: float = 0.5,
          seed: Optional[int] = None) -> Solution:
    """Plan a mission with the genetic algorithm.

    Args:
        instance: Monitoring points and team start/end positions.
        params: Vehicle parameters and robustness margins.
        population: Number of plans per generation (the paper used 300-500).
        generations: Number of generations (the paper used 100-200).
        num_parents: Parents kept by steady-state selection.
        mutation_rate: Probability that a child is mutated.
        seed: Random seed.

    Returns:
        A :class:`~rspect.problem.Solution`; ``details["feasible"]`` tells whether the best
        plan found satisfies every constraint.
    """
    params = params or MissionParams()
    check_altitude(instance, params)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    first = round_robin(instance)
    first_scores = [score_team(first, mu, instance, params) for mu in range(instance.m)]
    members = [(first.copy(), list(first_scores)) for _ in range(population)]
    energies = [energy(first_scores, instance.n)] * population

    for _ in range(generations):
        ranking = np.argsort(energies, kind="stable")
        parents = [members[i] for i in ranking[:min(num_parents, population)]]
        children, child_energies = [members[ranking[0]]], [energies[ranking[0]]]  # elitism
        for k in range(population - 1):
            (plan_1, scores_1), (plan_2, scores_2) = parents[k % len(parents)], parents[(k + 1) % len(parents)]
            child, scores = plan_1.copy(), list(scores_1)
            for mu in range(instance.m):  # team-level crossover
                if rng.random() < 0.5:
                    child.slots[mu] = plan_2.slots[mu]
                    child.release[mu] = plan_2.release[mu]
                    child.collect[mu] = plan_2.collect[mu]
                    scores[mu] = scores_2[mu]
            if rng.random() < mutation_rate:
                for mu in random_move(child, instance, rng):
                    scores[mu] = score_team(child, mu, instance, params)
            children.append((child, scores))
            child_energies.append(energy(scores, instance.n))
        members, energies = children, child_energies

    best = int(np.argmin(energies))
    plan = members[best][0].plan()
    return Solution.from_plan(plan, instance, params, time.perf_counter() - started,
                              energy=energies[best], feasible=not check_plan(plan, instance, params))
