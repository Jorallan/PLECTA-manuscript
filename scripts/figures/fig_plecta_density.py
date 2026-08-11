"""Figure: PLECTA against target areal coverage, degraded and clean masks.

One axes, two series from the same 50 paired scenes, so the clean-versus-
degraded difference reads as a vertical gap rather than a comparison the
reader has to carry between panels.  Error-bar style, palette and legend
discipline follow the existing figures in this repository.

Everything explanatory lives in the LaTeX caption; the image carries only
axis labels, tick labels and a legend.

Data: results/plecta_robustness.csv (committed).

    python scripts/figures/fig_plecta_density.py
"""
from __future__ import annotations

import csv
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

from _style import (DARK, FIG_W, GRAY, GREEN, PT_AXIS, PT_LEGEND, PT_TICK,
                    plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]
RNG_SEED = 20260811
N_BOOT = 20000
FIG_H = 3.05

#: Condition is carried by line style as well as by hue, so the two series
#: stay distinguishable in greyscale: the degraded axis -- the input the paper
#: actually evaluates -- is dashed, the clean axis is a solid smooth line.
SERIES = (
    ("clean", "Clean axis", DARK, "-", "s"),
    ("degraded", "Degraded axis (evaluated)", GREEN, (0, (5, 2)), "o"),
)


def bootstrap_ci(values, seed=RNG_SEED, n=N_BOOT, alpha=0.05):
    """Percentile bootstrap of the mean, resampled within one stratum."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def load_paired():
    with open(REPO / "results" / "plecta_robustness.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["status"] == "ok":
            out[row["mask_variant"]][int(row["density"])].append(float(row["f1"]))
    return out


def main() -> int:
    plecta_style()
    paired = load_paired()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.098, right=0.985, top=0.965, bottom=0.155)

    n_per_point = None
    for key, label, colour, dashes, marker in SERIES:
        by_density = paired[key]
        densities = sorted(by_density)
        n_per_point = len(by_density[densities[0]])
        means = np.array([np.mean(by_density[d]) for d in densities])
        lo, hi = zip(*(bootstrap_ci(by_density[d]) for d in densities))
        ax.fill_between(densities, lo, hi, color=colour, alpha=0.18, lw=0,
                        zorder=2)
        ax.plot(densities, means, color=colour, linestyle=dashes, lw=1.8,
                marker=marker, ms=5.5, mfc="white", mew=1.6, zorder=4,
                label=label)

    ax.set_xticks([20, 30, 40, 50, 60])
    ax.set_xticklabels([f"{d}%" for d in (20, 30, 40, 50, 60)])
    ax.set_xlim(16, 64)
    ax.set_ylim(0.62, 1.0)
    ax.set_xlabel("Target areal coverage", fontsize=PT_AXIS)
    ax.set_ylabel("Common-fragment F₁", fontsize=PT_AXIS)
    ax.tick_params(labelsize=PT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.legend(frameon=False, loc="lower left", fontsize=PT_LEGEND,
              handlelength=2.6, labelspacing=0.5, borderpad=0.2)

    save_fig(fig, "fig_plecta_density", bbox_inches=None)
    plt.close(fig)

    factorial = json.loads(
        (REPO / "results" / "plecta_density_factorial.json")
        .read_text(encoding="utf-8"))["width_negative_control"]
    print("n per point:", n_per_point, " bootstrap:", N_BOOT, "seed", RNG_SEED)
    print("coverage ratio %.2f, F1 range %.3f"
          % (factorial["max_areal_coverage_ratio_within_geometry"],
             factorial["max_within_geometry_f1_range"]))
    for key, *_ in SERIES:
        print(key, {d: round(float(np.mean(v)), 4)
                    for d, v in sorted(paired[key].items())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
