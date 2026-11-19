"""Top-view plots of mission plans.

* :func:`plot_plan` draws one plan: UAV flights (solid), UGV routes (dashed), release
  points (squares) and collect points (pentagons).
* :func:`plot_steps` shows how RSPECT builds a plan, one panel per step.

Each team has a fixed color (up to eight teams), a direct label (T1, T2, ...) and a
legend entry, so team identity never relies on color alone. ``theme`` is ``"light"``
or ``"dark"``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .problem import Instance, Plan, Solution, Tour

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "teams": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#6250d6", "#e34948"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink_2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "teams": ["#3987e5", "#d95926", "#199e70", "#c98500",
                       "#d55181", "#008300", "#9085e9", "#e66767"]},
}
KM = 1000.0  # plots are drawn in kilometres


def _system_sans() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial"]
    return next((name for name in candidates if name in installed), "DejaVu Sans")


FONT = _system_sans()


def plot_plan(instance: Instance, plan: Union[Plan, Solution], ax: Optional[Axes] = None, *,
              theme: str = "light", title: Optional[str] = None, legend: bool = True) -> Axes:
    """Draw a mission plan (a :class:`Solution` or a list of team plans) and return the axes."""
    plan = plan.plan if isinstance(plan, Solution) else plan
    colors = _team_colors(len(plan), theme)
    t = THEMES[theme]
    ax = _new_axes(ax, t)
    _draw_points(ax, instance.points, t["muted"], t)
    for mu, team in enumerate(plan):
        tours = [tour for tour in team if not tour.is_trivial]
        _draw_ugv_route(ax, instance.starts[mu], tours, instance.ends[mu], colors[mu])
        for tour in tours:
            _draw_flight(ax, tour, instance.points, colors[mu], t)
            _marker(ax, tour.release, "s", colors[mu], t, size=7)
            _marker(ax, tour.collect, "p", colors[mu], t, size=10)
        if tours and len(plan) > 1:
            _team_label(ax, tours[0].release, mu, t)
    _draw_endpoints(ax, instance, colors, t)
    _finish_axes(ax, instance, t, title)
    if legend:
        _legend(ax.figure, colors, ["flight", "route", "release", "collect", "start", "end"], t,
                anchor=ax)
    return ax


def plot_steps(instance: Instance, solution: Solution, *, theme: str = "light") -> Figure:
    """Four panels showing how RSPECT builds ``solution``: the monitoring points, Step 1
    (assignment and visit order), Step 2.1 (energy-feasible tours and their collect
    candidates) and Step 2.2 (selected collect points and UGV routes)."""
    sequences, drafts = solution.details["sequences"], solution.details["drafts"]
    colors = _team_colors(instance.m, theme)
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9))
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        _new_axes(ax, t)

    ax = axes[0]
    _draw_points(ax, instance.points, t["muted"], t, size=5)
    _draw_endpoints(ax, instance, colors, t)
    _finish_axes(ax, instance, t, f"Monitoring points (n = {instance.n})")

    ax = axes[1]
    for mu, sequence in enumerate(sequences):
        xy = np.vstack([instance.starts[mu, :2], instance.points[sequence, :2], instance.ends[mu, :2]]) / KM
        ax.plot(*xy.T, color=colors[mu], lw=1.5, solid_joinstyle="round", zorder=3)
        _draw_points(ax, instance.points[sequence], colors[mu], t, size=5)
    _draw_endpoints(ax, instance, colors, t)
    _finish_axes(ax, instance, t, "Step 1: assignment and visit order")

    ax = axes[2]
    for mu, team_drafts in enumerate(drafts):
        for draft in team_drafts:
            xy = np.vstack([draft.release[:2], instance.points[draft.points, :2]]) / KM
            ax.plot(*xy.T, color=colors[mu], lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            _draw_points(ax, instance.points[draft.points], colors[mu], t, size=5)
            _marker(ax, draft.release, "s", colors[mu], t, size=7)
            for q in draft.candidates:
                ax.plot(*(instance.points[q, :2] / KM), "p", ms=11, mfc="none", mec=colors[mu],
                        mew=1.3, zorder=4)
    _draw_endpoints(ax, instance, colors, t)
    _finish_axes(ax, instance, t, "Step 2.1: feasible tours")

    plot_plan(instance, solution, axes[3], theme=theme, title="Step 2.2: final plan", legend=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _legend(fig, colors, ["order", "flight", "route", "release", "candidate", "collect", "start", "end"],
            t, anchor=None)
    return fig


def save_figure(fig: Figure, path: Union[str, Path], dpi: int = 160) -> None:
    """Save a figure (creating the folder if needed) on its own background color."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {path}")


def plot_plans(instance: Instance, plans: Sequence[Union[Plan, Solution]], titles: Sequence[str], *,
               theme: str = "light") -> Figure:
    """Several plans of the same instance side by side, sharing one legend."""
    t = THEMES[theme]
    fig, axes = plt.subplots(1, len(plans), figsize=(4.6 * len(plans), 5.1), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    for ax, plan, title in zip(axes[0], plans, titles):
        plot_plan(instance, plan, ax, theme=theme, title=title, legend=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend(fig, _team_colors(instance.m, theme), ["flight", "route", "release", "collect", "start", "end"],
            t, anchor=None)
    return fig


# ------------------------------------------------------------------------------ drawing helpers
def _team_colors(m: int, theme: str) -> List[str]:
    palette = THEMES[theme]["teams"]
    if m > len(palette):
        raise ValueError(f"plots show at most {len(palette)} teams; plot a smaller instance")
    return palette[:m]


def _new_axes(ax: Optional[Axes], t: dict) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    return ax


def _marker(ax: Axes, point, shape: str, color: str, t: dict, size: float) -> None:
    ax.plot(point[0] / KM, point[1] / KM, shape, ms=size, color=color, mec=t["surface"], mew=1.5, zorder=5)


def _draw_points(ax: Axes, points: np.ndarray, color: str, t: dict, size: float = 4.5) -> None:
    if len(points):
        ax.plot(points[:, 0] / KM, points[:, 1] / KM, "o", ms=size, color=color, mec=t["surface"],
                mew=1.0, zorder=4)


def _draw_flight(ax: Axes, tour: Tour, points: np.ndarray, color: str, t: dict) -> None:
    xy = np.vstack([tour.release[:2], points[list(tour.points), :2], tour.collect[:2]]) / KM
    ax.plot(*xy.T, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    _draw_points(ax, points[list(tour.points)], color, t)


def _draw_ugv_route(ax: Axes, start, tours: Sequence[Tour], end, color: str) -> None:
    stops = [start[:2]] + [p for tour in tours for p in (tour.release[:2], tour.collect[:2])] + [end[:2]]
    xy = np.array(stops) / KM
    ax.plot(*xy.T, color=color, lw=1.5, ls=(0, (4, 3)), dash_capstyle="round", zorder=2)


def _draw_endpoints(ax: Axes, instance: Instance, colors: Sequence[str], t: dict) -> None:
    """Start (triangle up) and end (triangle down); in ink when all teams share them."""
    shared = len({tuple(p) for p in instance.starts}) == 1 and len({tuple(p) for p in instance.ends}) == 1
    for mu in range(instance.m):
        color = t["ink"] if shared else colors[mu]
        _marker(ax, instance.starts[mu], "^", color, t, size=11)
        _marker(ax, instance.ends[mu], "v", color, t, size=11)
        if shared:
            break


def _team_label(ax: Axes, point, mu: int, t: dict) -> None:
    ax.annotate(f"T{mu + 1}", (point[0] / KM, point[1] / KM), xytext=(6, 6), textcoords="offset points",
                fontsize=8.5, color=t["ink_2"], fontfamily=FONT, zorder=6,
                bbox={"boxstyle": "round,pad=0.15", "fc": t["surface"], "ec": "none", "alpha": 0.85})


def _finish_axes(ax: Axes, instance: Instance, t: dict, title: Optional[str]) -> None:
    w, h = instance.area[0] / KM, instance.area[1] / KM
    pad = 0.04 * max(w, h)
    ax.set_xlim(-pad, w + pad)
    ax.set_ylim(-pad, h + pad)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, w + 1e-9, 1.0))
    ax.set_yticks(np.arange(0, h + 1e-9, 1.0))
    ax.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(t["axis"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT)
    ax.set_xlabel("x (km)", color=t["ink_2"], fontsize=9, fontfamily=FONT)
    ax.set_ylabel("y (km)", color=t["ink_2"], fontsize=9, fontfamily=FONT)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=10.5, fontfamily=FONT, loc="left", pad=8)


def _legend(fig: Figure, colors: Sequence[str], roles: Sequence[str], t: dict, anchor: Optional[Axes]) -> None:
    ink = t["ink_2"]
    keys = {
        "order": Line2D([], [], color=ink, lw=1.5, label="Visit order (Step 1)"),
        "flight": Line2D([], [], color=ink, lw=2, label="UAV flight"),
        "route": Line2D([], [], color=ink, lw=1.5, ls=(0, (4, 3)), label="UGV route"),
        "release": Line2D([], [], marker="s", ls="none", ms=7, color=ink, mec=t["surface"], label="Release point"),
        "candidate": Line2D([], [], marker="p", ls="none", ms=10, mfc="none", mec=ink, label="Collect candidate"),
        "collect": Line2D([], [], marker="p", ls="none", ms=10, color=ink, mec=t["surface"], label="Collect point"),
        "start": Line2D([], [], marker="^", ls="none", ms=10, color=t["ink"], mec=t["surface"], label="Start"),
        "end": Line2D([], [], marker="v", ls="none", ms=10, color=t["ink"], mec=t["surface"], label="End"),
    }
    handles = [Line2D([], [], color=c, lw=3, label=f"Team {mu + 1}") for mu, c in enumerate(colors)]
    handles += [keys[r] for r in roles]
    style = {"frameon": False, "labelcolor": ink, "prop": {"family": FONT, "size": 9}, "handlelength": 2.2}
    if anchor is not None:  # single plot: legend to the right of the axes
        anchor.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, **style)
    else:                   # figure: legend below the panels, in one row or two even rows
        columns = len(handles) if len(handles) <= 6 else -(-len(handles) // 2)
        fig.legend(handles=handles, loc="lower center", ncol=columns, bbox_to_anchor=(0.5, 0.0), **style)
