"""The width negative control, redrawn so the null is legible, plus P vs R.

This is a proposed replacement for ``fig_plecta_performance``.  Panel (a) shows
the same 12-geometry factorial as before and panel (b) is unchanged; what
changed is how (a) argues.

The old panel drew one polyline per geometry against realised areal coverage.
The lines were flat, which *is* the result, but a flat line is the least
noticeable mark on a chart: a reader sees twelve horizontal segments and has no
way to tell whether flat means "small effect" or "no effect", nor how far the x
values actually travelled.  Three things fix that and all three are drawn here:

  * each geometry becomes a left-to-right ARROW, so the distance travelled
    along the coverage axis is a mark rather than something to be inferred from
    where the dots landed;
  * the null is given a SCALE.  A vertical reference on the right shows the F₁
    spread actually observed between geometries (0.209).  A reader can now see
    that the quantity that did not move is measured on an axis where a
    comparable quantity moves by a fifth of the range;
  * the claim is stated once, as a number, with its denominator: 12 of 12.

What the panel supports, exactly: bundle width -- and hence realised areal
coverage -- is not what makes a scene hard.  The stronger reading, that F₁ is
"sensitive to centreline density", is only half-supported by these 36 scenes and
is deliberately NOT claimed here: the density means do fall monotonically
(0.920, 0.878, 0.836) but the geometry-to-geometry spread inside one density
level reaches 0.144, which is larger than the 0.084 total spread between the
density means, on n = 4 geometries per level.  The density effect is carried by
the 50- and 128-scene evaluations, where n supports it; this figure's job is the
negative control alone.

Data, all committed: results/plecta_density_factorial.json,
results/plecta_heldout.json, results/plecta_cc_baseline.json.

    python scripts/figures/fig_plecta_control.py
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
from matplotlib.patches import FancyArrowPatch

from _style import (DARK, DENSITY_CYCLE, FIG_W, GRAY, GREEN, PT_ANNOT, PT_AXIS,
                    PT_LEGEND, PT_MIN, PT_TICK, PT_TITLE, plecta_style,
                    save_fig)

REPO = Path(__file__).resolve().parents[2]
FIG_H = 2.86

LEVEL = {"ld020": ("low", DENSITY_CYCLE[0]),
         "ld040": ("medium", DENSITY_CYCLE[2]),
         "ld060": ("high", DENSITY_CYCLE[4])}


def read(name):
    return json.loads((REPO / "results" / name).read_text(encoding="utf-8"))


def panel_control(ax, factorial):
    by_geometry = defaultdict(list)
    for row in factorial["per_scene"]:
        by_geometry[row["geometry"]].append(
            (100.0 * float(row["areal_coverage"]), float(row["f1"]),
             row["length_density"]))

    seen, f1s = set(), []
    for points in by_geometry.values():
        points.sort()
        level = points[0][2]
        name, colour = LEVEL[level]
        f1 = points[0][1]
        f1s.append(f1)
        # The arrow IS the message: how far coverage travelled, at a fixed F1.
        ax.add_patch(FancyArrowPatch(
            (points[0][0], f1), (points[-1][0], f1),
            arrowstyle="-|>,head_length=0.42,head_width=0.22", mutation_scale=10,
            color=colour, lw=1.5, shrinkA=0, shrinkB=0, zorder=4))
        ax.plot([p[0] for p in points], [f1] * len(points), linestyle="none",
                marker="o", ms=2.8, mfc=colour, mec="white", mew=0.7, zorder=5,
                label=name if level not in seen else None)
        seen.add(level)

    lo, hi = min(f1s), max(f1s)
    control = factorial["width_negative_control"]

    # The null needs a scale or "flat" means nothing: this is the F1 spread the
    # same axis shows between geometries, drawn on the same axis.  The three
    # free bands of the panel are used deliberately -- headline above the top
    # arrow, scale bar at the right, its label below the lowest arrow, legend
    # outside -- so nothing lands on the data.
    xr = 67.0
    ax.plot([xr, xr], [lo, hi], color=GRAY, lw=1.0, zorder=3)
    for yv in (lo, hi):
        ax.plot([xr - 1.2, xr + 1.2], [yv, yv], color=GRAY, lw=1.0, zorder=3)
    ax.annotate("", xy=(xr, hi), xytext=(xr, lo), zorder=3,
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
    ax.text(xr + 1.6, 0.745, "spread between geometries %.3f" % (hi - lo),
            fontsize=PT_MIN, color=GRAY, ha="right", va="bottom", zorder=6)

    ax.text(0.012, 0.995,
            "coverage ×%.2f within one geometry\n"
            "F₁ range %.3f — %d of %d geometries"
            % (control["max_areal_coverage_ratio_within_geometry"],
               control["max_within_geometry_f1_range"],
               control["n_geometries_with_zero_range"],
               control["n_geometries"]),
            transform=ax.transAxes, ha="left", va="top", fontsize=PT_ANNOT,
            color=DARK, fontweight="bold", linespacing=1.40)

    ax.set_xlim(8, 70)
    ax.set_ylim(0.700, 1.135)
    ax.set_yticks([0.8, 0.9, 1.0])
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.set_xticklabels(["%d%%" % d for d in (10, 20, 30, 40, 50, 60)])
    ax.set_xlabel("Realised areal coverage", fontsize=PT_AXIS, labelpad=1.5)
    ax.set_ylabel("Common-fragment F₁", fontsize=PT_AXIS, labelpad=2.0)
    ax.tick_params(labelsize=PT_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.spines["left"].set_bounds(0.700, 1.02)
    ax.spines["bottom"].set_bounds(10, 62)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("low", "medium", "high") if k in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              title="centreline density", frameon=False, loc="upper center",
              bbox_to_anchor=(0.46, -0.175), ncol=3, fontsize=PT_LEGEND,
              title_fontsize=PT_LEGEND, handlelength=1.0, labelspacing=0.24,
              columnspacing=1.2, borderpad=0.10, handletextpad=0.40)


def panel_pr(ax, heldout, cc):
    pr = np.array([[float(r["precision"]), float(r["recall"])]
                   for r in heldout["per_scene"]])
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=(0, (3, 2.4)), zorder=2)
    ax.plot(pr[:, 0], pr[:, 1], linestyle="none", marker="o", ms=2.8,
            color=GREEN, alpha=0.5, zorder=4, label="per scene")
    ax.plot([pr[:, 0].mean()], [pr[:, 1].mean()], marker="o", ms=5.6,
            mfc="white", mec=DARK, mew=1.5, zorder=6, linestyle="none",
            label="mean")
    ax.plot([cc["summary"]["precision"]], [cc["summary"]["recall"]],
            marker="X", ms=6.4, color=DARK, zorder=6, linestyle="none",
            label="component floor")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.40, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Precision", fontsize=PT_AXIS, labelpad=1.5)
    ax.set_ylabel("Recall", fontsize=PT_AXIS, labelpad=2.0)
    ax.tick_params(labelsize=PT_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.legend(frameon=False, loc="lower left", fontsize=PT_LEGEND,
              handlelength=1.2, labelspacing=0.30, borderpad=0.15,
              handletextpad=0.5)


def main() -> int:
    plecta_style()
    factorial = read("plecta_density_factorial.json")
    heldout = read("plecta_heldout.json")
    cc = read("plecta_cc_baseline.json")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = fig.add_axes([0.077, 0.255, 0.395, 0.640])
    ax_b = fig.add_axes([0.607, 0.255, 0.378, 0.640])
    panel_control(ax_a, factorial)
    panel_pr(ax_b, heldout, cc)
    ax_a.set_title("(a) Bundle width does not move the score", loc="left",
                   fontsize=PT_TITLE, fontweight="bold", pad=5.0)
    ax_b.set_title("(b) Precision vs. recall", loc="left", fontsize=PT_TITLE,
                   fontweight="bold", pad=5.0)

    save_fig(fig, "fig_plecta_control", bbox_inches=None)
    plt.close(fig)

    control = factorial["width_negative_control"]
    by_level = defaultdict(list)
    for row in factorial["per_scene"]:
        by_level[row["length_density"]].append(float(row["f1"]))
    print("negative control: coverage x%.2f, within-geometry F1 range %.3f, "
          "%d/%d geometries exactly zero"
          % (control["max_areal_coverage_ratio_within_geometry"],
             control["max_within_geometry_f1_range"],
             control["n_geometries_with_zero_range"], control["n_geometries"]))
    for level in sorted(by_level):
        v = np.array(by_level[level])
        print("  %s  mean %.3f  within-level spread %.3f  (n=%d scenes, "
              "%d geometries)" % (level, v.mean(), v.max() - v.min(), v.size,
                                  v.size // 3))
    allf1 = np.array([float(r["f1"]) for r in factorial["per_scene"]])
    print("  between-geometry spread overall %.3f" % (allf1.max() - allf1.min()))
    means = [np.mean(by_level[k]) for k in sorted(by_level)]
    print("  spread between density means %.3f" % (max(means) - min(means)))
    print("held-out scenes:", len(heldout["per_scene"]),
          " component floor P %.3f R %.3f"
          % (cc["summary"]["precision"], cc["summary"]["recall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
