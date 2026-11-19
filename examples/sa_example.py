"""Plan a small mission with the simulated annealing baseline.

SA evaluates a full plan at every step, so keep instances small (about 10 points).

    python examples/sa_example.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone without installing

from rspect import make_instance, sa  # noqa: E402
from rspect.plotting import plot_plan, save_figure  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan a small mission with simulated annealing.")
    parser.add_argument("--n", type=int, default=6, help="number of monitoring points (keep it small)")
    parser.add_argument("--m", type=int, default=2, help="number of UAV-UGV teams")
    parser.add_argument("--seed", type=int, default=11, help="instance seed")
    parser.add_argument("--steps", type=int, default=20_000, help="annealing steps")
    parser.add_argument("--out", default="outputs/sa_plan.png")
    args = parser.parse_args(argv)

    instance = make_instance(args.n, args.m, seed=args.seed)
    solution = sa.solve(instance, steps=args.steps, seed=0)
    print(solution.summary())
    print("all constraints satisfied:", solution.details["feasible"])

    ax = plot_plan(instance, solution, title=f"Simulated annealing: {args.steps:,} steps")
    save_figure(ax.figure, args.out)


if __name__ == "__main__":
    main()
