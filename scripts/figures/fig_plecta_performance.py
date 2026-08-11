"""Figure: three things about PLECTA's accuracy that the tables cannot show.

The tables already give every mean.  Each panel here is chosen because it
carries information a mean cannot:

  (a) the spread of per-scene F1 across the 128 held-out scenes, and the
      overlap between coverage strata that the stratum means hide;
  (b) a negative control -- inside a fixed geometry, areal coverage moves by
      a factor of 2.49 and F1 does not move at all, so the abscissa of
      Fig. "density" is a proxy for difficulty and not a cause of it;
  (c) the joint distribution of precision and recall per scene, against the
      connected-component floor, which is not balanced at all.

Data, all committed: results/plecta_heldout.json,
results/plecta_density_factorial.json, results/plecta_cc_baseline.json.

    python scripts/figures/fig_plecta_performance.py
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

from _style import (DARK, DENSITY_CYCLE, FIG_W, GRAY, GREEN, PT_BODY,
                    PT_LABEL, PT_TINY, plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]
FIG_H = 2.55
JITTER_SEED = 20260811


def read(name):
    return json.loads((REPO / "results" / name).read_text(encoding="utf-8"))


def panel_spread(ax, heldout):
    """Mean, with two labelled descriptive bands over the held-out scenes.

    The outer band is the 10th-90th percentile of per-scene F1 and the inner
    one the interquartile range; both describe the spread across scenes and
    neither is an interval estimate for the mean.  With 32 scenes per stratum
    the percentiles are worth reading, and the overlap between strata is the
    point of the panel.
    """
    by_density = defaultdict(list)
    for row in heldout["per_scene"]:
        by_density[int(row["density"])].append(float(row["f1"]))
    order = sorted(by_density)
    x = np.arange(len(order), dtype=float)
    stats = {q: [float(np.percentile(by_density[d], q)) for d in order]
             for q in (10, 25, 75, 90)}
    means = [float(np.mean(by_density[d])) for d in order]
    ax.fill_between(x, stats[10], stats[90], color=GREEN, alpha=0.16, lw=0,
                    zorder=2, label="10th–90th percentile")
    ax.fill_between(x, stats[25], stats[75], color=GREEN, alpha=0.30, lw=0,
                    zorder=3, label="interquartile range")
    ax.plot(x, means, color=DARK, lw=1.8, marker="o", ms=4.6, mfc="white",
            mew=1.5, zorder=5, label="mean")
    for i, value in enumerate(means):
        ax.text(i, 1.012, f"{value:.3f}", ha="center", va="bottom",
                fontsize=PT_TINY, color=DARK)
    ax.legend(frameon=False, loc="lower left", fontsize=PT_TINY,
              handlelength=1.4, labelspacing=0.32, borderpad=0.15,
              handletextpad=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{d}%" for d in order], fontsize=PT_LABEL)
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(0.40, 1.07)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Target areal coverage", fontsize=PT_LABEL, labelpad=1.5)
    ax.set_ylabel("Common-fragment $F_1$", fontsize=PT_LABEL, labelpad=2.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return sum(len(v) for v in by_density.values())


def panel_control(ax, factorial):
    by_geometry = defaultdict(list)
    for row in factorial["per_scene"]:
        by_geometry[row["geometry"]].append(
            (float(row["areal_coverage"]), float(row["f1"]),
             row["length_density"]))
    ld_colour = {"ld020": DENSITY_CYCLE[0], "ld040": DENSITY_CYCLE[2],
                 "ld060": DENSITY_CYCLE[4]}
    seen = set()
    for points in by_geometry.values():
        points.sort()
        xs = [100.0 * p[0] for p in points]
        ys = [p[1] for p in points]
        colour = ld_colour[points[0][2]]
        label = None
        if points[0][2] not in seen:
            seen.add(points[0][2])
            label = {"ld020": "low", "ld040": "medium",
                     "ld060": "high"}[points[0][2]]
        ax.plot(xs, ys, color=colour, lw=1.2, marker="o", ms=3.4, mfc=colour,
                mec="white", mew=0.8, zorder=4, label=label, alpha=0.9)
    ax.set_xlim(8, 68)
    ax.set_ylim(0.40, 1.07)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([10, 30, 50])
    ax.set_xticklabels(["10%", "30%", "50%"], fontsize=PT_LABEL)
    ax.set_xlabel("Realised areal coverage", fontsize=PT_LABEL, labelpad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("low", "medium", "high") if k in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              title="centreline density", frameon=False, loc="lower left",
              fontsize=PT_TINY, title_fontsize=PT_TINY, handlelength=1.6,
              labelspacing=0.32, borderpad=0.15)


def panel_pr(ax, heldout, cc):
    pr = np.array([[float(r["precision"]), float(r["recall"])]
                   for r in heldout["per_scene"]])
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=(0, (3, 2.4)), zorder=2)
    ax.plot(pr[:, 0], pr[:, 1], linestyle="none", marker="o", ms=2.8,
            color=GREEN, alpha=0.5, zorder=4, label="per scene")
    ax.plot([pr[:, 0].mean()], [pr[:, 1].mean()], marker="o", ms=6.0,
            mfc="white", mec=DARK, mew=1.6, zorder=6, linestyle="none",
            label="mean")
    ax.plot([cc["summary"]["precision"]], [cc["summary"]["recall"]],
            marker="X", ms=7.0, color=DARK, zorder=6, linestyle="none",
            label="component floor")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.40, 1.07)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Precision", fontsize=PT_LABEL, labelpad=1.5)
    ax.set_ylabel("Recall", fontsize=PT_LABEL, labelpad=2.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=PT_TINY,
              handlelength=1.2, labelspacing=0.32, borderpad=0.15,
              handletextpad=0.5)


def main() -> int:
    plecta_style()
    heldout = read("plecta_heldout.json")
    factorial = read("plecta_density_factorial.json")
    cc = read("plecta_cc_baseline.json")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 3, left=0.070, right=0.988, top=0.865,
                          bottom=0.185, wspace=0.32,
                          width_ratios=[1.05, 1.0, 1.0])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    n_scenes = panel_spread(ax_a, heldout)
    panel_control(ax_b, factorial)
    panel_pr(ax_c, heldout, cc)

    for ax, letter, title in ((ax_a, "a", "Per-scene spread"),
                              (ax_b, "b", "Coverage is a proxy"),
                              (ax_c, "c", "Precision vs. recall")):
        ax.set_title(f"({letter}) {title}", loc="left",
                     fontsize=PT_BODY, fontweight="bold", pad=6.0)

    save_fig(fig, "fig_plecta_performance", bbox_inches=None)
    plt.close(fig)
    print("scenes:", n_scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
