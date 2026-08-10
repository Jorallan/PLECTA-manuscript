"""Generate the PLECTA workflow and evidence-boundary schematic."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from _style import BLUE, DARK, GRAY, GREEN, ORANGE, apply_style, save_fig


def box(ax, x, y, w, h, text, edge, face="white", dashed=False):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        edgecolor=edge, facecolor=face, linewidth=1.5,
        linestyle=(0, (4, 2)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=8,
            fontweight="bold", linespacing=1.25)


def arrow(ax, start, end, color=GRAY, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
        color=color, linestyle=(0, (4, 2)) if dashed else "solid"))


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.35))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(0.25, 5.75, "Input-mask formation", color=BLUE,
            fontsize=9, fontweight="bold", ha="left")
    box(ax, 1.5, 4.55, 2.2, 0.95, "SEM\nmicrograph", DARK)
    box(ax, 4.45, 4.55, 2.55, 0.95, "Upstream mask\nextraction", BLUE,
        face="#edf4fb")
    box(ax, 7.55, 4.55, 2.35, 0.95, "Binary filament-\naxis mask", DARK)
    arrow(ax, (2.63, 4.55), (3.00, 4.55))
    arrow(ax, (5.75, 4.55), (6.35, 4.55))

    box(ax, 5.25, 2.75, 2.45, 0.95, "Procedural\naxis mask", GRAY,
        dashed=True)
    arrow(ax, (6.50, 2.75), (7.15, 4.02), color=GRAY, dashed=True)
    ax.text(5.25, 2.08, "development and evaluation", fontsize=6.8,
            color=GRAY, ha="center", style="italic")

    ax.add_patch(FancyBboxPatch(
        (9.00, 3.43), 2.85, 2.24,
        boxstyle="round,pad=0.08,rounding_size=0.10",
        edgecolor=GREEN, facecolor="#edf7f1", linewidth=1.7,
        linestyle=(0, (5, 2.5))))
    ax.text(10.43, 5.43, "Evaluated grouping", color=GREEN, fontsize=7.2,
            fontweight="bold", ha="center")
    box(ax, 10.43, 4.55, 2.35, 0.95, "PLECTA core\n(mask geometry only)",
        GREEN, face="white")
    arrow(ax, (8.75, 4.55), (9.22, 4.55), color=GREEN)

    box(ax, 14.10, 4.55, 2.75, 0.95, "Overlap-aware\n2-D instances", DARK)
    arrow(ax, (11.65, 4.55), (12.70, 4.55), color=GREEN)

    box(ax, 10.75, 1.25, 3.65, 1.05,
        "Image characterization\nand optional ribbon rendering", ORANGE,
        face="#fff3e8")
    arrow(ax, (14.10, 4.05), (11.65, 1.82), color=ORANGE)
    ax.plot([1.5, 1.5], [4.05, 1.25], color=ORANGE, linewidth=1.3,
            linestyle=(0, (4, 2)))
    arrow(ax, (1.5, 1.25), (8.88, 1.25), color=ORANGE, dashed=True)
    ax.text(4.25, 1.55, "optional grayscale image", fontsize=6.8,
            color=ORANGE, ha="center", style="italic")
    box(ax, 14.35, 1.25, 2.65, 0.95,
        "Widths, brightness,\nrendered masks", DARK)
    arrow(ax, (12.60, 1.25), (13.00, 1.25), color=ORANGE)

    ax.text(0.25, 0.35,
            "Only the green core contributed to the 128-scene held-out grouping result.",
            fontsize=7.2, color=DARK, ha="left")
    fig.tight_layout(pad=0.25)
    save_fig(fig, "fig_plecta_scope", bbox_inches="tight")
    fig.savefig(os.path.join(os.path.dirname(__file__), "..", "..", "figures",
                             "fig_plecta_scope.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
