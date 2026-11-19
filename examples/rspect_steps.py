"""Show how RSPECT builds a plan, one panel per step of Algorithm 1.

    python examples/rspect_steps.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone without installing

from rspect import make_instance, rspect  # noqa: E402
from rspect.plotting import plot_steps, save_figure  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot the steps of RSPECT.")
    parser.add_argument("--n", type=int, default=40, help="number of monitoring points")
    parser.add_argument("--m", type=int, default=3, help="number of UAV-UGV teams")
    parser.add_argument("--seed", type=int, default=7, help="instance seed")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--out", default="outputs/rspect_steps.png")
    args = parser.parse_args(argv)

    instance = make_instance(args.n, args.m, seed=args.seed)
    solution = rspect.solve(instance)

    # what each step produced
    for mu, (sequence, drafts) in enumerate(zip(solution.details["sequences"], solution.details["drafts"])):
        sizes = [len(d.points) for d in drafts]
        candidates = [len(d.candidates) for d in drafts]
        print(f"team {mu + 1}: Step 1 -> {len(sequence)} points | Step 2.1 -> tours of {sizes} points "
              f"with {candidates} collect candidates")
    print(solution.summary())

    save_figure(plot_steps(instance, solution, theme=args.theme), args.out)


if __name__ == "__main__":
    main()
