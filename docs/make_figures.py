"""Regenerate the README figures (light and dark versions) from the RSPECT examples.

    python docs/make_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import rspect_quickstart  # noqa: E402
import rspect_robustness  # noqa: E402
import rspect_steps  # noqa: E402

for theme in ("light", "dark"):
    rspect_steps.main(["--theme", theme, "--out", str(ROOT / "docs" / f"steps_{theme}.png")])
    rspect_quickstart.main(["--n", "80", "--m", "4", "--seed", "2", "--random-endpoints",
                            "--theme", theme, "--out", str(ROOT / "docs" / f"plan_{theme}.png")])
    rspect_robustness.main(["--theme", theme, "--out", str(ROOT / "docs" / f"robustness_{theme}.png")])
