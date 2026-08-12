"""The same three methods, the same scenes, two metrics that disagree.

Every comparator number elsewhere in this paper comes from one generator --
ours. This figure is the answer to the obvious objection: the same three
methods on scenes made by GraFT's own vendored generator, at its own published
settings, over a density ladder that reproduces its published range and then
continues into ours.

  (a) the manuscript's endpoint. PLECTA leads at every density and the margin
      widens; nothing crosses over.
  (b) GraFT's own Fig. 2 metric on the identical predictions. The ranking
      inverts at the dense end: GraFT is first and PLECTA last.
A third panel drew the ratio of objects emitted to objects present, which is
the mechanism: matched coverage pairs each reference filament with at most one
detection, so emitting fewer objects than a scene contains caps the score while
emitting more does not. It is gone. That ratio is one number per method and the
prose gives it, and (b) carries the point by itself once its axis is resolved to
the range the values actually occupy.

The two panels are not in tension. Matched coverage asks whether each true
filament was found; the pairwise score asks whether the pieces were put
together correctly. A method that shatters every filament at its crossings
still offers a distinct detection for each one and still scores near 1 in (b).

Data: results/plecta_graft_regime.json, written by
scripts/results/make_graft_regime_record.py.

    python scripts/figures/fig_graft_regime.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import (BLUE, FIG_W, FONE, GRAY, GREEN, ORANGE, PT_AXIS,
                    PT_LEGEND, PT_MIN, PT_TICK, PT_TITLE, plecta_style,
                    save_fig)

REPO = Path(__file__).resolve().parents[2]
FIG_H = 2.26

SERIES = (("plecta", "PLECTA", GREEN, "o"),
          ("graft", "GraFT", BLUE, "^"),
          ("dnai", "DNAi", ORANGE, "D"))

#: GraFT's published Fig. 2 stops here, in centreline px per image px. Marking
#: it is the whole reason the ladder was built to cross it.
PUBLISHED_LIMIT = 0.020


def panel(ax, xs, series, ylabel, ylim, yticks):
    for key, label, colour, marker in SERIES:
        ax.plot(xs, series[key], color=colour, lw=1.6, marker=marker, ms=3.8,
                mfc="white", mew=1.2, zorder=4, label=label, clip_on=False)
    ax.axvline(PUBLISHED_LIMIT, color=GRAY, lw=0.8, ls=(0, (2.6, 2.0)),
               zorder=2)
    ax.set_xlim(0, 0.060)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks([0, 0.02, 0.04, 0.06])
    ax.set_xlabel("centreline px per image px", fontsize=PT_AXIS, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=PT_AXIS, labelpad=2.0)
    ax.tick_params(labelsize=PT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)


def main() -> int:
    plecta_style()
    data = json.loads((REPO / "results" / "plecta_graft_regime.json")
                      .read_text(encoding="utf-8"))
    strata = data["by_stratum"]
    xs = [s["density_mean"] for s in strata]

    f1 = {k: [s["methods"][k]["f1"] for s in strata] for k, *_ in SERIES}
    fmc = {k: [s["methods"][k]["filament_matched_coverage"] for s in strata]
           for k, *_ in SERIES}
    #  No longer drawn -- the third panel is gone -- but still printed, because
    #  it is the mechanism the prose quotes.
    ratio = {k: [s["methods"][k]["n_instances_emitted"]
                 / s["methods"][k]["n_instances_true"] for s in strata]
             for k, *_ in SERIES}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y0, dy = 0.275, 0.575
    rects = ([0.088, y0, 0.375, dy], [0.598, y0, 0.375, dy])

    ax = fig.add_axes(rects[0])
    panel(ax, xs, f1, "Common-fragment " + FONE, (0.0, 1.0),
          [0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(a) this paper's endpoint", loc="left", fontsize=PT_TITLE,
                 fontweight="bold", pad=4.0)
    ax.annotate("GraFT's published" + chr(10) + "range ends here",
                (PUBLISHED_LIMIT, 0.70), textcoords="offset points",
                xytext=(5.0, 0.0), ha="left", va="center", fontsize=PT_MIN,
                color=GRAY, linespacing=1.4)

    #  A truncated axis, deliberately: every value lies above 0.75 and the
    #  differences that decide the ranking are hundredths. Stated in the
    #  caption. (a) is not truncated, because there the range is real.
    ax = fig.add_axes(rects[1])
    panel(ax, xs, fmc, "Filament matched coverage", (0.75, 1.005),
          [0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_title("(b) GraFT's own measure", loc="left", fontsize=PT_TITLE,
                 fontweight="bold", pad=4.0)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=PT_LEGEND,
               handlelength=2.0, columnspacing=1.6, handletextpad=0.45)

    save_fig(fig, "fig_graft_regime", bbox_inches=None)
    plt.close(fig)
    for s in strata:
        print("density %.4f  " % s["density_mean"] + "  ".join(
            "%s F1 %.3f fmc %.3f n/n %.2f"
            % (k, s["methods"][k]["f1"],
               s["methods"][k]["filament_matched_coverage"],
               s["methods"][k]["n_instances_emitted"]
               / s["methods"][k]["n_instances_true"]) for k, *_ in SERIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
