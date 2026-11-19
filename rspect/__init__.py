"""RSPECT: Robust and Scalable Planner for Energy-Aware Coordination of UAV-UGV Teams.

Implementation accompanying Er, Kashiri and Yazicioglu, Journal of Intelligent & Robotic
Systems 112:91 (2026), https://doi.org/10.1007/s10846-026-02426-5.

Solvers (each one in its own module, each with a ``solve(instance, params, ...)`` function):

* :mod:`rspect.rspect` - RSPECT, the proposed algorithm (needs Julia + GLNS).
* :mod:`rspect.sa`     - simulated annealing baseline.
* :mod:`rspect.ga`     - genetic algorithm baseline.
* :mod:`rspect.bnc`    - branch and cut baseline (needs Gurobi).
"""
from . import bnc, ga, rspect, sa
from .problem import (Instance, MissionParams, Solution, Tour, check_plan, make_instance,
                      mission_time, team_times, tour_robustness)

__version__ = "1.0.0"

__all__ = ["Instance", "MissionParams", "Solution", "Tour", "bnc", "check_plan", "ga",
           "make_instance", "mission_time", "rspect", "sa", "team_times", "tour_robustness"]
