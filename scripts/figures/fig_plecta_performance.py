"""Figure 10: the two controls that a table of means cannot carry.

Reduced after review, from three panels to two.  The recommendation and its
reasons are in the round-2 memo; in short:

  * KEPT -- the negative control.  Inside a fixed geometry, realised areal
    coverage moves by a factor of 2.49 and F1 does not move at all.  That is
    the only evidence that the abscissa of Figs. 6 and 7 is a *proxy* for
    difficulty rather than the cause of it, and nothing else in the paper
    says it.
  * KEPT -- precision against recall per scene.  A fragment-level F1 can be
    reached by a method that merges everything or splits everything, and this
    panel is the only place that shows PLECTA's errors are two-sided.  The
    connected-component floor is plotted here, where it belongs: it is a
    diagnostic (recall-only, precision near zero), not a comparator, which is
    why it was removed from Fig. 7.
  * DROPPED -- the per-scene F1 spread by coverage stratum.  It re-plotted
    Fig. 6's axes on a different population, and its one extra fact (the
    strata overlap) is a sentence in the text.  ``--panels abc`` restores it.

Data, all committed: results/plecta_heldout.json,
results/plecta_density_factorial.json, results/plecta_cc_baseline.json.

    python scripts/figures/fig_plecta_performance.py
    python scripts/figures/fig_plecta_performance.py --panels abc
"""
from __future__ import annotations

import argparse
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

from _style import (DARK, DENSITY_CYCLE, FIG_W, GRAY, GREEN, PT_ANNOT,
                    PT_AXIS, PT_LEGEND, PT_TICK, PT_TITLE, plecta_style,
                    save_fig)

REPO = Path(__file__).resolve().parents[2]


def read(name):
    return json.loads((REPO / "results" / name).read_text(encoding="utf-8"))


def panel_spread(ax, heldout, **_):
    """Mean per stratum with two descriptive bands over the held-out scenes.

    Neither band is an interval estimate for the mean: the outer one is the
    10th-90th percentile of per-scene F1 and the inner one the interquartile
    range, both describing spread across scenes.
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
    ax.plot(x, means, color=DARK, lw=1.7, marker="o", ms=4.4, mfc="white",
            mew=1.4, zorder=5, label="mean")
    ax.legend(frameon=False, loc="lower left", fontsize=PT_LEGEND,
              handlelength=1.4, labelspacing=0.32, borderpad=0.15,
              handletextpad=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{d}%" for d in order], fontsize=PT_TICK)
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(0.40, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Target areal coverage", fontsize=PT_AXIS, labelpad=1.5)
    ax.set_ylabel("Common-fragment F₁", fontsize=PT_AXIS, labelpad=2.0)
    ax.tick_params(labelsize=PT_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_control(ax, factorial, **_):
    """One line per geometry: coverage swept, the geometry held fixed."""
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
        colour = ld_colour[points[0][2]]
        label = None
        if points[0][2] not in seen:
            seen.add(points[0][2])
            label = {"ld020": "low", "ld040": "medium",
                     "ld060": "high"}[points[0][2]]
        ax.plot([100.0 * p[0] for p in points], [p[1] for p in points],
                color=colour, lw=1.2, marker="o", ms=3.2, mfc=colour,
                mec="white", mew=0.8, zorder=4, label=label, alpha=0.9)
    control = factorial["width_negative_control"]
    ax.text(0.985, 0.975,
            "coverage ×%.2f" "\n" "F₁ range %.3f"
            % (control["max_areal_coverage_ratio_within_geometry"],
               control["max_within_geometry_f1_range"]),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=PT_ANNOT, color=DARK, linespacing=1.45)
    ax.set_xlim(8, 68)
    ax.set_ylim(0.40, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([10, 30, 50])
    ax.set_xticklabels(["10%", "30%", "50%"], fontsize=PT_TICK)
    ax.set_xlabel("Realised areal coverage", fontsize=PT_AXIS, labelpad=1.5)
    ax.set_ylabel("Common-fragment F₁", fontsize=PT_AXIS, labelpad=2.0)
    ax.tick_params(labelsize=PT_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("low", "medium", "high") if k in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              title="centreline density", frameon=False, loc="lower left",
              fontsize=PT_LEGEND, title_fontsize=PT_LEGEND, handlelength=1.6,
              labelspacing=0.32, borderpad=0.15)


def panel_pr(ax, heldout, cc, **_):
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
    ax.legend(frameon=False, loc="lower left", fontsize=PT_LEGEND,
              handlelength=1.2, labelspacing=0.32, borderpad=0.15,
              handletextpad=0.5)


PANELS = {
    "a": (panel_spread, "Per-scene spread"),
    "b": (panel_control, "Coverage varied, geometry fixed"),
    "c": (panel_pr, "Precision vs. recall"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", default="bc",
                    help="which panels to draw, e.g. 'bc' (default) or 'abc'")
    args = ap.parse_args(argv)

    plecta_style()
    payload = {"heldout": read("plecta_heldout.json"),
               "factorial": read("plecta_density_factorial.json"),
               "cc": read("plecta_cc_baseline.json")}

    keys = [k for k in args.panels if k in PANELS]
    fig_h = 2.55 if len(keys) > 2 else 2.60
    fig = plt.figure(figsize=(FIG_W, fig_h))
    gs = fig.add_gridspec(1, len(keys), left=0.075, right=0.988, top=0.870,
                          bottom=0.175, wspace=0.255)
    for i, key in enumerate(keys):
        draw, name = PANELS[key]
        ax = fig.add_subplot(gs[0, i])
        draw(ax, **payload)
        ax.set_title(f"({chr(97 + i)}) {name}", loc="left",
                     fontsize=PT_TITLE, fontweight="bold", pad=5.0)

    save_fig(fig, "fig_plecta_performance", bbox_inches=None)
    plt.close(fig)
    control = payload["factorial"]["width_negative_control"]
    print("panels drawn:", keys)
    print("held-out scenes:", len(payload["heldout"]["per_scene"]))
    print("negative control: coverage ratio %.2f, F1 range %.3f"
          % (control["max_areal_coverage_ratio_within_geometry"],
             control["max_within_geometry_f1_range"]))
    print("component floor: precision %.3f recall %.3f"
          % (payload["cc"]["summary"]["precision"],
             payload["cc"]["summary"]["recall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
