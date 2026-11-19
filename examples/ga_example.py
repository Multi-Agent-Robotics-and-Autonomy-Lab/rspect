"""Plan a small mission with the genetic algorithm baseline.

The GA evaluates many plans per generation, so keep instances small (about 10 points).

    python examples/ga_example.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone without installing

from rspect import ga, make_instance  # noqa: E402
from rspect.plotting import plot_plan, save_figure  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan a small mission with a genetic algorithm.")
    parser.add_argument("--n", type=int, default=6, help="number of monitoring points (keep it small)")
    parser.add_argument("--m", type=int, default=2, help="number of UAV-UGV teams")
    parser.add_argument("--seed", type=int, default=11, help="instance seed")
    parser.add_argument("--population", type=int, default=200, help="plans per generation")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--out", default="outputs/ga_plan.png")
    args = parser.parse_args(argv)

    instance = make_instance(args.n, args.m, seed=args.seed)
    solution = ga.solve(instance, population=args.population, generations=args.generations, seed=0)
    print(solution.summary())
    print("all constraints satisfied:", solution.details["feasible"])

    ax = plot_plan(instance, solution,
                   title=f"Genetic algorithm: {args.population} plans x {args.generations} generations")
    save_figure(ax.figure, args.out)


if __name__ == "__main__":
    main()
