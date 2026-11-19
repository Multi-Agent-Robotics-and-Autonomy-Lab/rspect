"""Solve a tiny mission to (near) optimality with branch and cut.

Needs Gurobi (pip install gurobipy, plus a license). The model grows quickly with the
number of points, so keep it to a handful: 4 points with one team takes a few seconds.

    python examples/bnc_example.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone without installing

from rspect import bnc, make_instance  # noqa: E402
from rspect.plotting import plot_plan, save_figure  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Solve a tiny mission with branch and cut (Gurobi).")
    parser.add_argument("--n", type=int, default=4, help="number of monitoring points (keep it tiny)")
    parser.add_argument("--m", type=int, default=1, help="number of UAV-UGV teams")
    parser.add_argument("--seed", type=int, default=3, help="instance seed")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Gurobi time limit in seconds")
    parser.add_argument("--out", default="outputs/bnc_plan.png")
    args = parser.parse_args(argv)

    instance = make_instance(args.n, args.m, seed=args.seed)
    solution = bnc.solve(instance, time_limit=args.time_limit)
    print(solution.summary())
    print(f"Gurobi: {solution.details['status']}, MIP gap {100 * solution.details['mip_gap']:.1f}%")

    teams = "1 team" if args.m == 1 else f"{args.m} teams"
    ax = plot_plan(instance, solution, title=f"Branch and cut: {args.n} points, {teams}")
    save_figure(ax.figure, args.out)


if __name__ == "__main__":
    main()
