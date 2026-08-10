"""Generate a compact geometric explanation of PLECTA matching."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(__file__))
from _style import BLUE, DARK, GRAY, GREEN, LIGHT_GRAY, ORANGE, apply_style, save_fig


def curve(ax, points, color=DARK, width=3.0, linestyle="solid", zorder=2):
    """Draw one cubic Bezier segment from four control points."""
    path = Path(points, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, linewidth=width,
                           linestyle=linestyle, capstyle="round", zorder=zorder))


def arrow(ax, start, end, color=BLUE, width=1.5, dashed=False, zorder=4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, color=color,
        linewidth=width, linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=0, shrinkB=0, zorder=zorder))


def tip(ax, xy, color=DARK):
    ax.add_patch(Circle(xy, 0.075, facecolor="white", edgecolor=color,
                        linewidth=1.35, zorder=5))


def panel_label(ax, letter, heading):
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, ha="left",
            va="top", fontsize=9, fontweight="bold")
    ax.text(0.13, 0.97, heading, transform=ax.transAxes, ha="left",
            va="top", fontsize=8.1, fontweight="bold")


def support_frame(ax):
    panel_label(ax, "a", "Local frame at a stub")
    curve(ax, [(0.20, 0.62), (0.90, 0.55), (1.62, 0.82), (2.18, 1.34)],
          color=LIGHT_GRAY, width=3.7)
    curve(ax, [(1.18, 0.68), (1.52, 0.81), (1.90, 1.10), (2.18, 1.34)],
          color=BLUE, width=4.6)
    tip(ax, (2.18, 1.34), BLUE)
    arrow(ax, (2.18, 1.34), (2.86, 2.03), BLUE, width=1.7)
    curve(ax, [(2.18, 1.34), (2.45, 1.60), (2.69, 1.82), (2.93, 1.92)],
          color=ORANGE, width=1.45, linestyle=(0, (3, 2)), zorder=3)

    ax.annotate("support window", xy=(1.48, 0.88), xytext=(0.36, 1.52),
                fontsize=7, color=BLUE, ha="left",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9))
    ax.text(2.42, 2.12, r"outward tangent $\mathbf{t}$", fontsize=7,
            color=BLUE, ha="center")
    ax.text(2.72, 1.58, r"curvature $\kappa$", fontsize=7,
            color=ORANGE, ha="left", rotation=18)
    ax.text(1.55, 0.24, "fit in arclength; read outward from the tip",
            fontsize=6.55, color=GRAY, ha="center")


def pair_cost(ax):
    panel_label(ax, "b", "Pairwise geometric cost")
    a = (0.54, 0.66)
    b = (3.00, 1.53)
    curve(ax, [(0.05, 0.30), (0.22, 0.40), (0.40, 0.54), tuple(a)],
          color=BLUE, width=3.8)
    curve(ax, [(3.48, 1.92), (3.32, 1.79), (3.13, 1.63), tuple(b)],
          color=ORANGE, width=3.8)
    tip(ax, a, BLUE)
    tip(ax, b, ORANGE)

    arrow(ax, a, (1.29, 1.14), BLUE, width=1.45)
    arrow(ax, b, (2.27, 1.37), ORANGE, width=1.45)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, linewidth=1.2,
            linestyle=(0, (4, 2)), zorder=1)
    ax.text(1.80, 1.22, r"chord $\mathbf{c}_{ab}$", fontsize=6.8,
            color=GRAY, rotation=19, ha="center")

    ax.add_patch(Arc((1.34, 1.08), 0.48, 0.38, angle=0, theta1=190,
                     theta2=224, color=GREEN, linewidth=1.25))
    ax.text(1.21, 0.92, r"$\phi_a$", fontsize=7.2, color=GREEN)
    ax.add_patch(Arc((2.52, 1.42), 0.52, 0.40, angle=0, theta1=166,
                     theta2=199, color=GREEN, linewidth=1.25))
    ax.text(2.43, 1.17, r"$\phi_b$", fontsize=7.2, color=GREEN)

    # Direct tangent mismatch is shown away from the chord to keep the roles distinct.
    arrow(ax, (1.46, 1.96), (2.06, 2.24), BLUE, width=1.15)
    arrow(ax, (2.66, 2.19), (2.08, 2.06), ORANGE, width=1.15)
    ax.add_patch(Arc((2.08, 2.10), 0.42, 0.32, theta1=13, theta2=25,
                     color=DARK, linewidth=1.15))
    ax.text(2.08, 2.34, r"direct mismatch $\theta$", fontsize=6.8,
            ha="center", color=DARK)

    ax.text(1.76, 0.22,
            r"cost combines $\theta$, chord turns $\phi_a,\phi_b$, distance and curvature",
            fontsize=6.35, ha="center", color=DARK)


def junction_matching(ax):
    panel_label(ax, "c", "Exact partial matching at a junction")
    center = (1.75, 1.26)
    tips = {
        "a": (0.63, 2.05),
        "b": (0.42, 0.68),
        "c": (2.73, 2.09),
        "d": (3.03, 0.62),
        "e": (1.69, 0.12),
    }
    colors = {"a": BLUE, "b": ORANGE, "c": BLUE, "d": ORANGE, "e": GRAY}
    controls = {
        "a": [(0.13, 2.40), (0.32, 2.31), (0.49, 2.17), tuple(tips["a"])],
        "b": [(0.03, 0.49), (0.17, 0.55), (0.31, 0.62), tuple(tips["b"])],
        "c": [(3.22, 2.40), (3.05, 2.31), (2.88, 2.20), tuple(tips["c"])],
        "d": [(3.42, 0.32), (3.27, 0.43), (3.13, 0.54), tuple(tips["d"])],
        "e": [(1.68, -0.18), (1.69, -0.06), (1.69, 0.04), tuple(tips["e"])],
    }
    for name, pts in controls.items():
        curve(ax, pts, color=colors[name], width=3.6 if name != "e" else 2.8)
        tip(ax, tips[name], colors[name])
        ax.text(tips[name][0] + (-0.13 if name in "ab" else 0.13),
                tips[name][1] + (0.10 if name in "ac" else -0.10), name,
                fontsize=7.1, fontweight="bold", color=colors[name], ha="center")

    ax.add_patch(Circle(center, 0.25, facecolor="#edf4fb", edgecolor=DARK,
                        linewidth=1.0, zorder=1))
    # Candidate pairings are light; the exact solution is emphasized.
    curve(ax, [tuple(tips["a"]), (1.15, 1.78), (2.15, 1.75), tuple(tips["d"])],
          color=LIGHT_GRAY, width=1.05, linestyle=(0, (2, 2)), zorder=0)
    curve(ax, [tuple(tips["b"]), (1.08, 1.05), (2.18, 1.55), tuple(tips["c"])],
          color=LIGHT_GRAY, width=1.05, linestyle=(0, (2, 2)), zorder=0)
    curve(ax, [tuple(tips["a"]), (1.13, 1.72), (2.14, 1.83), tuple(tips["c"])],
          color=GREEN, width=2.15, zorder=3)
    curve(ax, [tuple(tips["b"]), (1.10, 0.88), (2.30, 0.89), tuple(tips["d"])],
          color=GREEN, width=2.15, zorder=3)
    ax.text(1.77, 1.47, "minimum\ntotal cost", fontsize=6.1, color=GREEN,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.annotate("unmatched (price)", xy=tuple(tips["e"]), xytext=(2.23, 0.27),
                fontsize=6.6, color=GRAY, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.85, color=GRAY))
    ax.text(1.75, 2.48, "all feasible pairings scored jointly",
            fontsize=6.45, color=DARK, ha="center")


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05),
                             gridspec_kw={"wspace": 0.12})
    for ax in axes:
        ax.set_xlim(0, 3.5)
        ax.set_ylim(-0.22, 2.80)
        ax.set_aspect("equal")
        ax.axis("off")

    support_frame(axes[0])
    pair_cost(axes[1])
    junction_matching(axes[2])
    for ax in axes[:-1]:
        ax.plot([1.02, 1.02], [0.04, 0.94], transform=ax.transAxes,
                color="#e2e8f0", linewidth=0.85, clip_on=False)

    fig.subplots_adjust(left=0.018, right=0.992, top=0.985, bottom=0.04)
    save_fig(fig, "fig_plecta_geometry", bbox_inches="tight", pad_inches=0.03)
    svg = os.path.join(os.path.dirname(__file__), "..", "..", "figures",
                       "fig_plecta_geometry.svg")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.03)
    print(f"[saved] {os.path.normpath(svg)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
