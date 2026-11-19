"""Plan a multi-team monitoring mission with RSPECT and plot it.

    python examples/rspect_quickstart.py
    python examples/rspect_quickstart.py --n 100 --m 4 --random-endpoints
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone without installing

from rspect import MissionParams, check_plan, make_instance, rspect  # noqa: E402
from rspect.plotting import plot_plan, save_figure  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan a mission with RSPECT and plot it.")
    parser.add_argument("--n", type=int, default=60, help="number of monitoring points")
    parser.add_argument("--m", type=int, default=3, help="number of UAV-UGV teams")
    parser.add_argument("--seed", type=int, default=4, help="instance seed")
    parser.add_argument("--random-endpoints", action="store_true",
                        help="give every team a random start and end position")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--out", default="outputs/rspect_plan.png")
    args = parser.parse_args(argv)

    instance = make_instance(args.n, args.m, seed=args.seed, random_endpoints=args.random_endpoints)
    params = MissionParams()  # Table 1: UAV 10 m/s (2 m/s vertical), UGV 2.5 m/s, 600 s flights

    print("Planning... (the first GLNS call compiles its Julia code, about 10 s)")
    solution = rspect.solve(instance, params)
    print(solution.summary())
    print("all constraints satisfied:", not check_plan(solution.plan, instance, params))

    ax = plot_plan(instance, solution, theme=args.theme,
                   title=f"RSPECT plan: {args.n} monitoring points, {args.m} teams")
    save_figure(ax.figure, args.out)


if __name__ == "__main__":
    main()
