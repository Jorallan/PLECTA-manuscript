"""The three reported measures against coverage, one figure per mask condition.

Replaces ``fig_comparators`` (pairwise F1 alone, both conditions in one figure)
and ``fig_graft_regime`` (a second generator under two measures).  Those drew
one endpoint and left the reader to take on trust that the others agreed.  The
three panels here are the three measures the paper reports, and they are drawn
side by side precisely so that the agreement is visible rather than asserted:

  (a) **pairwise F1** counts pairs of fragments.  It sees an identity swap at a
      crossing -- the error this method exists to prevent -- and it is quadratic
      in fragmentation, so a broken mask is charged many times over for one
      missed join.
  (b) **detection F1** at DNAi's own IoU threshold of 0.1, with a 2 px
      placement tolerance, counts whole objects.  It is the most forgiving
      reading of a fragmented mask, and it is blind to the swap: it scores an
      inverted reconstruction 1.000 where pairwise scores 0.000.  It is the
      flattest of the three for exactly that reason.
  (c) **Crossing Fidelity** counts crossings: the share of reference crossings
      whose incident arms the prediction partitions exactly as the reference
      does.  It states the paper's claim directly, it is the only one of the
      three whose sample size is crossings rather than scenes, and it is
      mechanistically distinct from the other two because a crossing carries no
      information about a *gap*.  Its chance level -- the do-nothing prediction
      that leaves every arm unpaired -- is drawn as a rule, because a fidelity
      quoted without one is not readable.  That level is 0.088 and 0.080, not
      the 0.709 the manuscript used to print: 0.709 is the chance level of pair
      accuracy, a different statistic and one this figure no longer carries.

Two figures rather than six panels in one, because the mask condition is the
thing a reader compares across and stacking six panels would put the comparison
inside a single crowded frame.  Both are drawn by this one generator so the two
cannot drift apart.

Variation of information is the fourth measure the paper reports and it does
not get a panel: PLECTA's whole trajectory lives under 0.9 bits against DNAi's
2.9, so a shared axis crushes it into a corner.  Its totals at the two ends of
the range are printed inside panel (a) instead.  The adjusted Rand index gets
neither a panel nor a number here: over these 30 cells it tracks pairwise F1 to
within 0.016 everywhere, so a second line would be drawn on top of the first.
Section 3.3 states that agreement in one sentence, which is what it is worth.

Data: results/plecta_metric_panels.json, written by
scripts/results/make_metric_panels_record.py, which is gated on reproducing
every per-density mean the manuscript already prints.

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
                    PT_LEGEND, PT_MIN, PT_TICK, plecta_style, save_fig,
                    tagged_title)

REPO = Path(__file__).resolve().parents[2]

#: The figure contains no filaments and carries its own legend, so BLUE and
#: ORANGE are plain categorical hues here.  GREEN keeps its one fixed meaning.
SERIES = (
    ("plecta", "PLECTA", GREEN, "o"),
    ("graft", "GraFT", BLUE, "^"),
    ("dnai", "DNAi", ORANGE, "D"),
)

PANELS = (
    ("f1", "a", "pairwise " + FONE),
    ("detection_f1", "b", "detection " + FONE),
    ("crossing_fidelity", "c", "crossing fidelity"),
)

#: Inches, laid out left to right and bottom up.  The three panels share one
#: 0--1 scale, so the tick labels are written once, on (a).
LEFT, PANEL_W, GAP = 0.44, 1.72, 0.215
BOTTOM, PANEL_H = 0.58, 1.30
FIG_H = 2.18


def read():
    path = REPO / "results" / "plecta_metric_panels.json"
    return json.loads(path.read_text(encoding="utf-8"))


def panel(ax, cells, key, densities, full_n, chance=None, label_n=False):
    if chance is not None:
        ax.axhline(chance, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.text(densities[0] - 2.2, chance + 0.025, "chance", ha="left",
                va="bottom", fontsize=PT_MIN, color=INK, zorder=6)
    for name, label, colour, marker in SERIES:
        xs = list(densities)
        ys = [cells[name][str(d)][key] for d in xs]
        ax.plot(xs, ys, color=colour, lw=1.5, marker=marker, ms=4.0,
                mfc="white", mew=1.2, zorder=4, label=label, clip_on=False)
        #  The censored count is written once, on (a), where the curves are
        #  furthest apart.  It is a property of the scene set, not of the
        #  measure, so it holds for all three panels and the caption says so;
        #  repeating it three times put a label on a neighbouring curve.
        if not label_n:
            continue
        for d in xs:
            n = cells[name][str(d)]["n"]
            if n != full_n:
                ax.annotate("n=%d" % n, (d, cells[name][str(d)][key]),
                            textcoords="offset points", xytext=(-1.5, -5.5),
                            ha="right", va="top", fontsize=PT_MIN,
                            color=colour, zorder=6)
    for y in (0.25, 0.50, 0.75):
        ax.axhline(y, color=GRAY, lw=0.4, alpha=0.30, zorder=1)
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


def vi_block(ax, cells, densities):
    """The two ends of the VI trajectory, printed where a panel would not fit.

    Bottom left of panel (a): the region no curve enters in either condition,
    the lowest series being DNAi at 0.50 and 0.66 at the sparsest stratum.
    """
    columns = (0.30, 0.465, 0.63)
    ax.text(0.03, 0.245, "VI total (bits)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PT_MIN, color=GRAY, style="italic")
    for density, y in ((densities[0], 0.135), (densities[-1], 0.030)):
        ax.text(0.03, y, "%d %%" % density, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=PT_MIN, color=GRAY)
        for (name, _label, colour, _m), x in zip(SERIES, columns):
            ax.text(x, y, "%.2f" % cells[name][str(density)]["vi_total_bits"],
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=PT_MIN, color=colour)


def draw(payload, condition, name):
    densities = payload["densities"]
    cells = payload["panels"][condition]
    chance = payload["chance"][condition]["pooled"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    first = None
    for index, (key, tag, title) in enumerate(PANELS):
        rect = [(LEFT + index * (PANEL_W + GAP)) / FIG_W, BOTTOM / FIG_H,
                PANEL_W / FIG_W, PANEL_H / FIG_H]
        ax = fig.add_axes(rect)
        panel(ax, cells, key, densities, full_n=10, label_n=index == 0,
              chance=chance if key == "crossing_fidelity" else None)
        tagged_title(ax, tag, title, dy=1.02, gap=0.135)
        if first is None:
            vi_block(ax, cells, densities)
            first = ax
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.text((LEFT + 1.5 * PANEL_W + GAP) / FIG_W, 0.240 / FIG_H,
             "Target areal coverage (%)", ha="center", va="bottom",
             fontsize=PT_AXIS, color=INK)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.012), fontsize=PT_LEGEND,
               handlelength=2.0, columnspacing=1.6, handletextpad=0.45)

    save_fig(fig, name, bbox_inches=None)
    plt.close(fig)


def main() -> int:
    plecta_style()
    payload = read()
    for condition, name in (("degraded", "fig_metric_panels_degraded"),
                            ("clean", "fig_metric_panels_clean")):
        draw(payload, condition, name)
        cells = payload["panels"][condition]
        print("--", condition)
        for key, _tag, title in PANELS:
            print("   %-18s" % title.replace(FONE, "F1"),
                  "   ".join("%s %s" % (label, " ".join(
                      "%.3f" % cells[name][str(d)][key]
                      for d in payload["densities"]))
                      for name, label, _c, _m in SERIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
