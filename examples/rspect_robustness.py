"""Plan the same mission with and without robustness margins.

The margins delta_a and delta_g (Eqs. 3c-3d) reserve slack in every tour, so the plan
still works if flights or drives take up to that much longer than planned (Theorem 2).

    python examples/rspect_robustness.py --margin 150
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone without installing

from rspect import MissionParams, make_instance, rspect, tour_robustness  # noqa: E402
from rspect.plotting import plot_plans, save_figure  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare RSPECT plans with and without margins.")
    parser.add_argument("--n", type=int, default=40, help="number of monitoring points")
    parser.add_argument("--m", type=int, default=2, help="number of UAV-UGV teams")
    parser.add_argument("--seed", type=int, default=3, help="instance seed")
    parser.add_argument("--margin", type=float, default=150.0, help="delta_a = delta_g, in seconds")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--out", default="outputs/rspect_robustness.png")
    args = parser.parse_args(argv)

    instance = make_instance(args.n, args.m, seed=args.seed)
    solutions, titles = [], []
    for margin in (0.0, args.margin):
        params = MissionParams(delta_a=margin, delta_g=margin)
        solution = rspect.solve(instance, params)
        tours = [tour for team in solution.plan for tour in team if not tour.is_trivial]
        # Definition 1: how much longer each tour's flight and drive could take
        slack = [tour_robustness(tour, instance.points, params) for tour in tours]
        print(f"margin {margin:5.0f} s: {len(tours)} tours, mission time {solution.makespan:.0f} s, "
              f"smallest flight slack {min(s[0] for s in slack):.0f} s, "
              f"smallest drive slack {min(s[1] for s in slack):.0f} s")
        solutions.append(solution)
        titles.append(f"Margin {margin:.0f} s: {len(tours)} tours")

    save_figure(plot_plans(instance, solutions, titles, theme=args.theme), args.out)


if __name__ == "__main__":
    main()
