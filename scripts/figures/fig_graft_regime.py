"""The same three methods, the same scenes, two metrics that disagree.

Every comparator number elsewhere in this paper comes from one generator --
ours. This figure is the answer to the obvious objection: the same three
methods on scenes made by GraFT's own vendored generator, at its own published
settings, over a density ladder that reproduces its published range and then
continues into ours.

  (a) the manuscript's endpoint. PLECTA leads at every density and the margin
      widens; nothing crosses over.
  (b) GraFT's own Fig. 2 metric on the identical predictions, drawn twice. The
      solid lines are the measure as published, where the ranking inverts at
      the dense end: GraFT is first and PLECTA last. The dashed lines are the
      identical bipartite matching with one edge condition added -- a detection
      may claim a filament only when it covers half that filament's axis -- and
      the inversion disappears. It also carries an inset over the three
      sparsest strata: drawing the half-coverage lines forced the panel onto a
      full 0-1 axis, and on that axis the sparse end -- where all three
      published curves lie between 0.84 and 1.00 and cross each other -- is
      unreadable. The inset restores exactly that window, published measure
      only, because six curves in a box that size would not be legible.

Three things this figure deliberately does not draw.

*The emission ratio.* A panel once showed objects emitted over objects present,
which is the mechanism behind (b): matched coverage pairs each reference
filament with at most one detection, so emitting fewer objects than a scene
contains caps the score while emitting more does not. The dashed lines now make
that point as a measurement rather than an arithmetic aside.

*ARI.* Over these 40 scenes it tracks pairwise F1 to within 0.004 for PLECTA
and 0.03 for the others, so a panel would repeat (a).

*The variation-of-information decomposition.* Drawn as a third panel for one
revision, then removed. The comparators need an axis running to 1.8 bits while
PLECTA's whole trajectory lives below 0.42, so the panel showed the gap by
crushing the one method a reader most wants to look at. Six numbers in the
prose say the same thing without that distortion.

The two panels are not in tension. Matched coverage asks whether each true
filament was found; the pairwise score asks whether the pieces were put
together correctly. A method that shatters every filament at its crossings
still offers a distinct detection for each one and still scores near 1 in (b).
What the dashed lines add is that those detections are then mostly fragments:
the solid-to-dashed drop is the share of matches that do not survive being
asked to resemble the filament they were matched to.

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
from matplotlib.patches import Rectangle

from _style import (BLUE, FIG_W, FONE, GRAY, GREEN, LIGHT_GRAY, ORANGE, PT_AXIS,
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

#: The window panel (b)'s inset magnifies. Chosen to hold the three
#: sparsest strata, which is where the published measure's curves cross
#: each other and where a full 0-1 axis leaves them indistinguishable.
INSET_X = 0.0095
INSET_Y = (0.80, 1.045)


def panel(ax, xs, series, ylabel, ylim, yticks, dashed=None):
    for key, label, colour, marker in SERIES:
        ax.plot(xs, series[key], color=colour, lw=1.6, marker=marker, ms=3.8,
                mfc="white", mew=1.2, zorder=4, label=label, clip_on=False)
        if dashed is not None:
            #  Same colour, no marker: this is the same method under a stricter
            #  reading of the same measure, not a fourth series.
            ax.plot(xs, dashed[key], color=colour, lw=1.2,
                    ls=(0, (2.4, 1.8)), zorder=3, clip_on=False)
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
    fmc_half = {k: [s["methods"][k]["filament_matched_coverage_half"]
                    for s in strata] for k, *_ in SERIES}
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

    #  Full range, matching (a): the half-coverage lines reach 0.257, so the
    #  truncation this panel used to need is gone and both panels are now read
    #  on the same scale. What that costs is the sparse end, where all three
    #  published curves sit between 0.84 and 1.00 and cross each other; the
    #  inset restores exactly that window.
    ax = fig.add_axes(rects[1])
    panel(ax, xs, fmc, "Filament matched coverage", (0.0, 1.0),
          [0, 0.25, 0.5, 0.75, 1.0], dashed=fmc_half)
    ax.set_title("(b) GraFT's own measure", loc="left", fontsize=PT_TITLE,
                 fontweight="bold", pad=4.0)

    #  The inset carries the published measure only. Drawing the half-coverage
    #  lines in it as well would put six curves in a box this size, and the
    #  question the inset answers -- who is above whom before the ladder
    #  separates them -- is a question about the solid ones.
    inset = ax.inset_axes([0.125, 0.085, 0.475, 0.355], zorder=6)
    #  Opaque, or the half-coverage curves read straight through it.
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    for key, _label, colour, marker in SERIES:
        #  Clipped, unlike the parent panels: an inset that lets its lines run
        #  outside the box draws them across the panel it sits inside. The top
        #  limit is above 1.0 so the markers at exactly 1.000 are not sliced.
        inset.plot(xs, fmc[key], color=colour, lw=1.2, marker=marker, ms=3.0,
                   mfc="white", mew=1.0, zorder=4, clip_on=True)
    inset.set_xlim(-0.0004, INSET_X)
    inset.set_ylim(INSET_Y)
    inset.set_xticks([0.0, 0.004, 0.008])
    inset.set_xticklabels(["0", ".004", ".008"])
    inset.set_yticks([0.85, 0.90, 0.95, 1.00])
    inset.set_yticklabels([".85", "", ".95", ""])
    inset.tick_params(labelsize=PT_MIN, length=2.0, pad=1.2)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        inset.spines[side].set_color(GRAY)
    #  A light frame on the parent showing the window the inset magnifies.
    ax.add_patch(Rectangle((0.0, INSET_Y[0]), INSET_X, INSET_Y[1] - INSET_Y[0],
                           fill=False, ec=LIGHT_GRAY, lw=0.7, zorder=5))

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
    #  Printed because the prose now carries these instead of a panel, and a
    #  number quoted in the text should be reproducible from the generator
    #  that owns the figure it used to be drawn in.
    print(chr(10) + "pooled over the ladder, not drawn:")
    for key, label, *_ in SERIES:
        b = data["overall"][key]
        print("  %-7s ARI %.3f  VI split %.3f  VI merge %.3f  VI total %.3f"
              % (label, b["adjusted_rand_index"], b["vi_split_bits"],
                 b["vi_merge_bits"],
                 b["vi_split_bits"] + b["vi_merge_bits"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
