"""Generate the PLECTA stage-and-gate schematic."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from _style import BLUE, DARK, GRAY, GREEN, ORANGE, apply_style, save_fig


def box(ax, x, y, w, h, title, detail, edge, face="white"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        edgecolor=edge, facecolor=face, linewidth=1.45))
    ax.text(x, y + 0.22, title, ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=edge)
    ax.text(x, y - 0.18, detail, ha="center", va="center", fontsize=6.45,
            color=DARK, linespacing=1.22)


def arrow(ax, start, end, color=GRAY, curved=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10.5, linewidth=1.2,
        color=color, linestyle=(0, (4, 2)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curved}"))


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.35))
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    box(ax, 1.95, 6.45, 3.45, 1.35, "Topology preconditioning",
        "skeletonize; prune spurs\nmerge nodes; form arms/stubs", BLUE,
        face="#edf4fb")
    box(ax, 5.85, 6.45, 3.00, 1.35, "Local frames",
        "24 px support\ntangent + curvature", BLUE, face="#edf4fb")
    arrow(ax, (3.70, 6.45), (4.32, 6.45), color=BLUE)

    ax.add_patch(FancyBboxPatch(
        (7.55, 1.05), 8.05, 6.55,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        edgecolor=GREEN, facecolor="#f5fbf7", linewidth=1.4,
        linestyle=(0, (5, 2.5))))
    ax.text(11.58, 7.28, "Eight coupled rounds", ha="center", color=GREEN,
            fontsize=8.4, fontweight="bold")

    box(ax, 9.45, 5.90, 3.05, 1.55, "Junction matching",
        "same node; different arms\nfinite cost; cost < 1.24\nexact partial matching",
        GREEN)
    arrow(ax, (7.37, 6.45), (7.90, 6.05), color=GREEN)
    box(ax, 13.75, 5.90, 2.45, 1.35, "Cycle guard",
        "remove links that\nclose a chain", GRAY)
    arrow(ax, (11.00, 5.90), (12.50, 5.90), color=GREEN)

    box(ax, 13.35, 3.65, 3.65, 1.70, "Gap matching",
        "unmatched + reliable\nd <= 85 px; theta <= 0.40\nphi <= 0.28; cost < 0.60",
        ORANGE, face="#fff8f1")
    arrow(ax, (13.75, 5.20), (13.55, 4.52), color=ORANGE)
    box(ax, 9.40, 3.65, 2.45, 1.35, "Cycle guard",
        "remove links that\nclose a chain", GRAY)
    arrow(ax, (11.50, 3.65), (10.65, 3.65), color=ORANGE)

    box(ax, 11.35, 1.70, 4.05, 1.25, "Chain-supported frames",
        "55 px support; update geometry\nanneal from 85% to full strictness",
        GREEN)
    arrow(ax, (9.40, 2.95), (10.35, 2.33), color=GREEN)
    arrow(ax, (9.35, 1.58), (8.45, 5.18), color=GREEN, curved=0.42,
          dashed=True)

    box(ax, 17.00, 4.65, 2.25, 1.70, "Instance output",
        "arm-link components\nshared node pixels\ndiscard short debris",
        DARK)
    arrow(ax, (15.22, 3.85), (15.88, 4.25), color=DARK)
    ax.text(17.00, 3.52, "after round 8", fontsize=6.5, color=GRAY,
            ha="center", style="italic")

    ax.text(0.30, 0.28,
            "Gap links affect identity but are not painted into the core centreline output.",
            fontsize=7.1, color=DARK, ha="left")
    fig.tight_layout(pad=0.25)
    save_fig(fig, "fig_plecta_gates", bbox_inches="tight")
    fig.savefig(os.path.join(os.path.dirname(__file__), "..", "..", "figures",
                             "fig_plecta_gates.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
