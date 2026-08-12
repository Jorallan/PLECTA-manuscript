"""Figure 7: PLECTA against DNAi, by coverage, on the same 50 scenes.

Rebuilt after review, and given a generator: the earlier version of this
figure was committed as a PDF with no script behind it and in a different
style from the rest of the set.

Two series were dropped on the author's instruction and one on style grounds:

  * the greedy minimum-turning-angle baseline -- our own reimplementation of a
    principle rather than a published method, and currently being re-swept, so
    its numbers may move;
  * the connected-component floor, which is a diagnostic rather than a
    comparator and belongs with the other diagnostics (Fig. 10);
  * the panel's three-line header, which is caption material.

What remains is the comparison the paper actually makes: PLECTA against a
published method run from its released source, at its shipped default and at
the best configuration found for it on *development* scenes, under both mask
conditions.

Plain means, n = 10 scenes per point, no interval drawn: at this stratum size
none is supportable, and the aggregate 50-scene paired difference -- which
does carry one -- is reported in the text, not here.

Data: results/plecta_dnai_comparison.json (committed).

    python scripts/figures/fig_comparators_density.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import (DARK, FIG_W, GRAY, GREEN, PT_AXIS, PT_LEGEND, PT_TICK,
                    PT_TITLE, plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]
FIG_H = 2.55

#: method key -> (legend label, colour, linestyle, marker)
SERIES = (
    ("plecta", "PLECTA", GREEN, "-", "o"),
    ("dnai_tuned", "DNAi, best development configuration", DARK, "-", "s"),
    ("dnai_default", "DNAi, shipped default", GRAY, (0, (4, 2)), "^"),
)

CONDITIONS = (("degraded", "a", "Degraded axis (evaluated)"),
              ("clean", "b", "Clean axis"))


def main() -> int:
    plecta_style()
    data = json.loads((REPO / "results" / "plecta_dnai_comparison.json")
                      .read_text(encoding="utf-8"))
    per_density = data["per_density"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
    fig.subplots_adjust(left=0.092, right=0.988, top=0.880, bottom=0.255,
                        wspace=0.075)

    n_per_point = set()
    reported = {}
    for ax, (condition, letter, name) in zip(axes, CONDITIONS):
        table = per_density[condition]
        for key, label, colour, dashes, marker in SERIES:
            densities = sorted(table[key], key=int)
            means = [table[key][d]["f1"]["mean"] for d in densities]
            n_per_point.update(table[key][d]["f1"]["n"] for d in densities)
            reported[(condition, key)] = [round(v, 3) for v in means]
            ax.plot([int(d) for d in densities], means, color=colour,
                    linestyle=dashes, lw=1.7, marker=marker, ms=4.8,
                    mfc="white", mew=1.4, zorder=4, label=label)
        ax.set_xticks([20, 30, 40, 50, 60])
        ax.set_xticklabels([f"{d}%" for d in (20, 30, 40, 50, 60)])
        ax.set_xlim(16, 64)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Target areal coverage", fontsize=PT_AXIS, labelpad=1.5)
        ax.set_title(f"({letter})  {name}", loc="left", fontsize=PT_TITLE,
                     fontweight="bold", pad=5.0)
        ax.tick_params(labelsize=PT_TICK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRAY)
    axes[0].set_ylabel("Mean common-fragment F₁", fontsize=PT_AXIS)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), fontsize=PT_LEGEND,
               handlelength=2.4, columnspacing=1.6, handletextpad=0.55)

    save_fig(fig, "fig_comparators_density", subdir="archive", bbox_inches=None)
    plt.close(fig)
    print("n per point:", sorted(n_per_point), "-- plain means, no interval")
    for key, values in sorted(reported.items()):
        print(" ", key, values)
    agg = data["per_method"]
    for condition in ("degraded", "clean"):
        print(" aggregate", condition,
              {k: round(agg[condition][k]["f1"]["mean"], 3)
               for k in ("plecta", "dnai_tuned", "dnai_default")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
