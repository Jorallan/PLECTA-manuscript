"""The three reported measures against coverage, both mask conditions, one figure.

Replaces ``fig_comparators`` (pairwise F1 alone, both conditions in one figure)
and ``fig_graft_regime`` (a second generator under two measures).  Those drew
one endpoint and left the reader to take on trust that the others agreed.  The
three columns here are the three measures the paper reports, and they are drawn
side by side precisely so that the agreement is visible rather than asserted:

  (1) **pairwise F1** counts pairs of fragments.  It sees an identity swap at a
      crossing -- the error this method exists to prevent -- and it is quadratic
      in fragmentation, so a broken mask is charged many times over for one
      missed join.
  (2) **detection F1** at DNAi's own IoU threshold of 0.1, with a 2 px
      placement tolerance, counts whole objects.  It is the most forgiving
      reading of a fragmented mask, and it is blind to the swap: it scores an
      inverted reconstruction 1.000 where pairwise scores 0.000.  It is the
      flattest of the three for exactly that reason.
  (3) **Crossing Fidelity** counts crossings: the share of reference crossings
      whose incident arms the prediction partitions exactly as the reference
      does.  It states the paper's claim directly, it is the only one of the
      three whose sample size is crossings rather than scenes, and it is
      mechanistically distinct from the other two because a crossing carries no
      information about a *gap*.

The two mask conditions were two figures and are now two rows of one: **a**,
the clean undegraded axis, above **b**, the degraded axis.  Stacking them puts
the comparison a reader actually makes -- what the same method does when the
mask degrades -- inside one frame and one column, rather than across a page
turn.  The columns keep their meaning down the rows, so a1 and b1 are the same
measure on the same scenes from two mask conditions.

**Five methods, and one of them is qualified.**  Basu is our reimplementation
of Stage B, no implementation of that paper having been found, which
is the star.  SIFNE carried a dagger here until 2026-08-26, for being run at
parameters we selected on development scenes; it was dropped because every
comparator was translated on those same scenes, so the mark implied a
distinction that does not exist.  The legend is ordered by overall F1,
descending, the same order the comparator table prints.

Variation of information is the fourth measure the paper reports and it does
not get a panel: PLECTA's whole trajectory lives under 0.9 bits against DNAi's
2.9, so a shared axis crushes it into a corner.  It used to be printed as two
columns of numbers inside panel (a); with five series that block is a table
rather than an annotation, and Table~\\ref{tab:plecta-comparators} is where a
table belongs.  The adjusted Rand index gets neither a panel nor a number: over
these cells it tracks pairwise F1 to within 0.016 everywhere, so a second line
would be drawn on top of the first.

Data: results/plecta_metric_panels.json, written by
scripts/results/make_metric_panels_record.py, which is gated on reproducing
every per-density mean the manuscript already prints and the overall mean each
comparator study already recorded.

    python scripts/figures/fig_metric_panels.py
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

from _style import (BLUE, FIG_W, FONE, GRAY, GREEN, INK, ORANGE, PT_AXIS,
                    PT_LEGEND, PT_TICK, PT_TITLE, ROSE, VIOLET, panel_tag,
                    plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]

#: The figure contains no filaments and carries its own legend, so BLUE and
#: ORANGE are plain categorical hues here.  GREEN keeps its one fixed meaning.
#: Ordered by overall F1, descending, as the comparator table is; the star is
#: that table's own mark and means the same thing here.
SERIES = (
    ("plecta", "PLECTA", GREEN, "o"),
    ("sifne", "SIFNE", VIOLET, "s"),
    ("basu", r"Basu$^{*}$", ROSE, "v"),
    ("graft", "GraFT", BLUE, "^"),
    ("dnai", "DNAi", ORANGE, "D"),
)

#: Two columns, not three. Crossing Fidelity was dropped from this figure on
#: 2026-08-25: it is already tabulated for these same 50 scenes and all five
#: methods in `tab:plecta-comparators`, so removing it here costs only the
#: per-coverage shape, and the manuscript itself calls it an in-house
#: diagnostic "used to say what was solved and not to rank". Detection F1 has
#: no other home in the synthetic comparison, so it stays. The record still
#: carries `crossing_fidelity` per stratum; nothing was recomputed.
COLUMNS = (
    ("f1", "Pairwise " + FONE),
    ("detection_f1", "Detection " + FONE),
)

#: Top row first: the clean axis above the degraded one.  Panels are tagged
#: (a)-(d) reading order; the column headings are shared, written once above
#: the top row, and the row identity is the y-axis label.
ROWS = (("clean", "Clean masks"), ("degraded", "Degraded masks"))

#: Inches, laid out left to right and bottom up.  The panels share one 0--1
#: scale, so the tick labels are written once per row, in the first column;
#: the left margin is wide enough for those and for the row's own label.
LEFT, PANEL_W, GAP = 0.62, 2.675, 0.215
BOTTOM, PANEL_H, ROW_GAP = 0.58, 1.66, 0.46
FIG_H = BOTTOM + 2 * PANEL_H + ROW_GAP + 0.52


def read():
    path = REPO / "results" / "plecta_metric_panels.json"
    return json.loads(path.read_text(encoding="utf-8"))


def panel(ax, cells, key, densities):
    for name, label, colour, marker in SERIES:
        xs = list(densities)
        ys = [cells[name][str(d)][key] for d in xs]
        ax.plot(xs, ys, color=colour, lw=1.5, marker=marker, ms=4.0,
                mfc="white", mew=1.2, zorder=4, label=label, clip_on=False)
    for y in (0.25, 0.50, 0.75):
        ax.axhline(y, color=GRAY, lw=0.4, alpha=0.30, zorder=1)
    #  Vertical counterparts at the coverage strata, same weight, so a value
    #  can be read down to its stratum as well as across to its level.
    for x in densities:
        ax.axvline(x, color=GRAY, lw=0.4, alpha=0.30, zorder=1)
    ax.set_xticks(list(densities))
    ax.set_xticklabels(["%d" % d for d in densities])
    ax.set_xlim(min(densities) - 3, max(densities) + 3)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(labelsize=PT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)


def draw(payload, name):
    densities = payload["densities"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    #  Shared column headings, once, above the top row.
    for column, (_key, title) in enumerate(COLUMNS):
        x = (LEFT + column * (PANEL_W + GAP) + 0.5 * PANEL_W) / FIG_W
        fig.text(x, (FIG_H - 0.17) / FIG_H, title, ha="center", va="bottom",
                 fontsize=PT_TITLE, color=INK)

    first = None
    for row, (condition, row_label) in enumerate(ROWS):
        cells = payload["panels"][condition]
        bottom = BOTTOM + (len(ROWS) - 1 - row) * (PANEL_H + ROW_GAP)
        for column, (key, title) in enumerate(COLUMNS):
            rect = [(LEFT + column * (PANEL_W + GAP)) / FIG_W, bottom / FIG_H,
                    PANEL_W / FIG_W, PANEL_H / FIG_H]
            ax = fig.add_axes(rect)
            panel(ax, cells, key, densities)
            panel_tag(ax, "abcd"[row * len(COLUMNS) + column], dy=1.02)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=PT_AXIS, labelpad=2.5)
                if first is None:
                    first = ax
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.text((LEFT + len(COLUMNS) * 0.5 * PANEL_W + 0.5 * GAP) / FIG_W, 0.240 / FIG_H,
             "Target areal coverage (%)", ha="center", va="bottom",
             fontsize=PT_AXIS, color=INK)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(SERIES), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.007),
               fontsize=PT_LEGEND, handlelength=2.0, columnspacing=1.4,
               handletextpad=0.45)

    save_fig(fig, name, bbox_inches=None)
    plt.close(fig)


def main() -> int:
    plecta_style()
    payload = read()
    draw(payload, "fig_metric_panels")
    for condition, _label in ROWS:
        cells = payload["panels"][condition]
        print("--", condition)
        for key, title in COLUMNS:
            print("   %-18s" % title.replace(FONE, "F1"),
                  "   ".join("%s %s" % (name, " ".join(
                      "%.3f" % cells[name][str(d)][key]
                      for d in payload["densities"]))
                      for name, _label, _c, _m in SERIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
