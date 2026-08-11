"""Figure: how PLECTA responds to the input mask it is handed.

Same two-panel layout and shared coverage legend as the earlier development
sensitivity figure in this repository.  (a) varies the nominal binary
axis-mask width; (b) contrasts the degraded 1 px axis that the paper evaluates
against the generator's undegraded axis.

The sample is four development scenes per coverage level, so the figure shows
every scene as well as the mean and makes no interval claim at all:
exploratory, n = 4 scenes per point, no inference.

Data: results/plecta_mask_variants.json (committed).

    python scripts/figures/fig_plecta_sensitivity.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import (DENSITY_CYCLE, FIG_W, GRAY, PT_AXIS, PT_LEGEND,
                    PT_TICK, PT_TITLE, plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]
FIG_H = 3.15


def load():
    data = json.loads((REPO / "results" / "plecta_mask_variants.json")
                      .read_text(encoding="utf-8"))
    table = defaultdict(lambda: defaultdict(list))
    for row in data["per_scene"]:
        table[int(row["density"])][row["variant"]].append(float(row["f1"]))
    return data, table


def panel(ax, table, variants, xticklabels, title, xlabel):
    """Mean line per coverage level, with a descriptive min-max band.

    The band is the range across the four development scenes, not an
    interval estimate: four scenes cannot support one, and none is claimed.
    """
    x = np.arange(len(variants), dtype=float)
    for colour, density in zip(DENSITY_CYCLE, sorted(table)):
        means = [float(np.mean(table[density][v])) for v in variants]
        lows = [float(np.min(table[density][v])) for v in variants]
        highs = [float(np.max(table[density][v])) for v in variants]
        ax.fill_between(x, lows, highs, color=colour, alpha=0.16, lw=0,
                        zorder=2)
        ax.plot(x, means, color=colour, lw=2.0, marker="o", ms=5.0,
                mfc=colour, mec="white", mew=1.1, zorder=4,
                label=f"{density}%")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=PT_TICK)
    ax.set_xlim(-0.35, len(variants) - 0.65)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel(xlabel, fontsize=PT_AXIS)
    ax.set_title(title, loc="left", fontsize=PT_TITLE, fontweight="bold")
    ax.tick_params(labelsize=PT_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)


def main() -> int:
    plecta_style()
    data, table = load()
    n_per = data["n_per_density"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(left=0.100, right=0.985, top=0.918, bottom=0.300,
                        wspace=0.09)

    panel(ax_a, table, ["w1", "w2", "w3"], ["1 px", "2 px", "3 px"],
          "(a) Nominal axis-mask width", "Binary axis-mask width")
    panel(ax_b, table, ["w1", "clean"], ["Degraded\n(1 px, evaluated)",
                                         "Clean\naxis"],
          "(b) Degraded vs. clean axis", "Input axis-mask condition")
    ax_a.set_ylabel("Development common-fragment F₁", fontsize=PT_AXIS)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.014), fontsize=PT_LEGEND,
               title="Target areal coverage", title_fontsize=PT_LEGEND,
               handlelength=2.2, columnspacing=1.9, handletextpad=0.55)

    save_fig(fig, "fig_plecta_sensitivity", bbox_inches=None)
    plt.close(fig)
    for density in sorted(table):
        print(density, {v: round(float(np.mean(table[density][v])), 4)
                        for v in ("w1", "w2", "w3", "clean")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
