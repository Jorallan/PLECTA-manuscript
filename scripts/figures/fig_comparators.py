"""One comparator figure: PLECTA against GraFT, DNAi and a greedy baseline.

Replaces ``fig_comparators_density`` (PLECTA vs DNAi only) and pulls in the
GraFT study, whose figures lived only in ``C:/Repos/comparisons/graft_comparison``
and never reached the manuscript.  Three separate comparator stories become one.

Everything is the same 50 scenes, the same input masks and one scorer
(``eval/core/common_metric.py``); the three source files agree on PLECTA's mean
to the last digit, which is the check that they describe one population.

  (a), (b)  plain means per coverage stratum, n = 10 per point unless the point
            is annotated otherwise.  No interval is drawn and none is claimed.
  (c)       the aggregate paired difference, which is the only quantity here
            that carries a bootstrap interval.

Two honesty constraints are built into the drawing rather than left to the
caption.  GraFT did not finish 3 of the 50 degraded scenes and 1 of the clean
ones inside its per-scene wall-clock budget; those scenes are excluded from its
means, they are the hardest of the densest stratum, so its means are optimistic.
Every point and every interval computed on fewer than the full complement is
labelled with its own n.

The greedy baseline is read from ``plecta_greedy_baseline.json``, not from the
``greedy_baseline`` arm of ``plecta_dnai_comparison.json``: the latter is the
older hand-set 28/20 configuration (0.670 / 0.686) and the former is the one
swept on development scenes, 24/15 (0.676 / 0.716), which is what the paper
reports.

Data, all committed: results/plecta_dnai_comparison.json,
results/plecta_graft_comparison.json, results/plecta_greedy_baseline.json.

    python scripts/figures/fig_comparators.py
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

from _style import (BLUE, DARK, FIG_W, GRAY, GREEN, ORANGE, PT_AXIS, PT_LEGEND,
                    PT_MIN, PT_TICK, PT_TITLE, plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]
FIG_H = 2.50
DENSITIES = (20, 30, 40, 50, 60)

#: The figure contains no filaments and carries its own legend, so BLUE and
#: ORANGE revert to plain categorical hues here, as they do in the pipeline
#: figure.  GREEN keeps its one fixed meaning: PLECTA.
PLECTA_C = GREEN
GREEDY_C = DARK
GRAFT_C = BLUE
DNAI_C = ORANGE
FLOOR_C = GRAY


def read(name):
    return json.loads((REPO / "results" / name).read_text(encoding="utf-8"))


def series_by_density(dnai, graft, greedy, condition):
    """(key, label, colour, marker) -> {density: (mean_f1, n)} for one mask."""
    pd = dnai["per_density"][condition]
    out = {
        "plecta": {d: (pd["plecta"][str(d)]["f1"]["mean"],
                       pd["plecta"][str(d)]["f1"]["n"]) for d in DENSITIES},
        "dnai": {d: (pd["dnai_tuned"][str(d)]["f1"]["mean"],
                     pd["dnai_tuned"][str(d)]["f1"]["n"]) for d in DENSITIES},
    }
    bd = greedy["conditions"][condition]["summary"]["by_density"]
    out["greedy"] = {d: (bd[str(d)]["greedy_f1"], bd[str(d)]["n"])
                     for d in DENSITIES}
    gb = graft["arms"][f"graft|{condition}"]["by_density"]
    out["graft"] = {d: (gb[str(d)]["f1"], gb[str(d)]["n"]) for d in DENSITIES}
    return out


#: The connected-component floor is deliberately absent.  It exists on exactly
#: these 50 scenes -- ``baseline_cc|native|*`` in the GraFT study's summary.json
#: -- but that file is not committed under results/, and the floor that *is*
#: committed (plecta_cc_baseline.json) was measured on the 128 held-out scenes
#: and has no 50 % stratum.  Drawing it here would put two populations on one
#: axis to save a reader one sentence of caption.  If it is wanted, the fix is
#: to add the arm to results/plecta_graft_comparison.json, not to reach across.
SERIES = (
    ("plecta", "PLECTA", PLECTA_C, "o", "-"),
    ("greedy", "Greedy continuation", GREEDY_C, "s", "-"),
    ("graft", "GraFT", GRAFT_C, "^", "-"),
    ("dnai", "DNAi", DNAI_C, "D", "-"),
)


def panel_density(ax, table, full_n):
    """F1 against coverage stratum; any point with n < full_n says so."""
    for key, label, colour, marker, dashes in SERIES:
        xs = list(DENSITIES)
        ys = [table[key][d][0] for d in xs]
        ax.plot(xs, ys, color=colour, linestyle=dashes, lw=1.6,
                marker=marker, ms=4.2, mfc="white", mew=1.3, zorder=4,
                label=label, clip_on=False)
        for d in xs:
            n = table[key][d][1]
            if n != full_n:
                ax.annotate("n=%d" % n, (d, table[key][d][0]),
                            textcoords="offset points", xytext=(-1.5, -8.5),
                            ha="right", va="top", fontsize=PT_MIN,
                            color=colour, zorder=6)
    ax.set_xticks(list(DENSITIES))
    ax.set_xticklabels(["%d%%" % d for d in DENSITIES])
    ax.set_xlim(17, 63)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Target areal coverage", fontsize=PT_AXIS, labelpad=1.5)
    ax.tick_params(labelsize=PT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)


#: (comparator key, label, colour) in the order they are stacked, best last.
GAPS = (("greedy", "Greedy", GREEDY_C),
        ("graft", "GraFT", GRAFT_C),
        ("dnai", "DNAi", DNAI_C))


def paired_gaps(dnai, graft, greedy):
    """{condition: {key: (diff, lo, hi, n, wins)}} for the F1 difference."""
    out = {}
    for condition in ("degraded", "clean"):
        row = {}
        p = dnai["paired"][condition]["plecta_minus_dnai_tuned__f1"]
        row["dnai"] = (p["mean_diff"], p["ci95_lo_stratified"],
                       p["ci95_hi_stratified"], p["n_paired_scenes"],
                       p["wins_a"])
        p = graft["paired"][condition]["metrics"]["f1"]
        row["graft"] = (p["mean_diff"], p["ci95_lo"], p["ci95_hi"], p["n"],
                        p["plecta_wins"])
        p = greedy["conditions"][condition]["summary"]["metrics"]["f1"]["paired"]
        row["greedy"] = (p["mean_difference"], p["ci_low"], p["ci_high"],
                         greedy["conditions"][condition]["summary"]["n_scenes"],
                         p["plecta_better"])
        out[condition] = row
    return out


def panel_gaps(ax, gaps):
    """Paired PLECTA - comparator difference, with its bootstrap interval."""
    ticks, labels = [], []
    y = 0.0
    for ci, (condition, marker, fill) in enumerate(
            (("degraded", "o", True), ("clean", "o", False))):
        for key, label, colour in GAPS:
            diff, lo, hi, n, wins = gaps[condition][key]
            ax.plot([lo, hi], [y, y], color=colour, lw=1.5, solid_capstyle="butt",
                    zorder=4)
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.16, y + 0.16], color=colour, lw=1.1,
                        zorder=4)
            ax.plot([diff], [y], marker=marker, ms=4.6, zorder=6,
                    mfc=colour if fill else "white", mec=colour, mew=1.3,
                    linestyle="none")
            note = "%d/%d" % (wins, n)
            ax.annotate(note, (hi, y), textcoords="offset points",
                        xytext=(4.0, 0), ha="left", va="center",
                        fontsize=PT_MIN, color=colour, zorder=6)
            ticks.append(y)
            labels.append(label)
            y -= 1.0
        y -= 0.55
    ax.axvline(0.0, color=GRAY, lw=0.8, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=PT_TICK)
    ax.set_ylim(ticks[-1] - 0.72, 0.80)
    ax.set_xlim(0.0, 0.74)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel("PLECTA − comparator, F₁", fontsize=PT_AXIS, labelpad=1.5)
    ax.tick_params(labelsize=PT_TICK)
    ax.tick_params(axis="y", length=0, pad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    # Group headers sit in the empty left margin of each group, where nothing
    # else is drawn, rather than on the right where they meet the win counts.
    for row, name in ((0, "degraded mask"), (3, "clean mask")):
        ax.text(0.005, ticks[row] + 0.60, name, ha="left", va="center",
                fontsize=PT_MIN, color=GRAY, style="italic")
    return ticks


def main() -> int:
    plecta_style()
    dnai = read("plecta_dnai_comparison.json")
    graft = read("plecta_graft_comparison.json")
    greedy = read("plecta_greedy_baseline.json")

    # One population or the figure is a lie: all three files must agree on
    # PLECTA, on the same 50 scenes, through the same scorer.
    anchors = [dnai["per_method"]["degraded"]["plecta"]["f1"]["mean"],
               graft["arms"]["plecta|degraded"]["overall"]["f1"],
               greedy["conditions"]["degraded"]["summary"]["metrics"]["f1"]
               ["plecta_mean"]]
    if max(anchors) - min(anchors) > 1e-9:
        raise SystemExit("sources disagree on PLECTA: %r" % (anchors,))

    # Explicit rects, not a uniform gridspec: (a) and (b) share one scale and
    # sit close together with the ticks written once, while (c) needs a wider
    # gap to its left for its own row labels.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y0, dy = 0.256, 0.634
    rects = {"a": [0.074, y0, 0.246, dy],
             "b": [0.344, y0, 0.246, dy],
             "c": [0.706, y0, 0.282, dy]}

    ax_a = None
    for key, condition, name in (("a", "degraded", "Degraded mask"),
                                 ("b", "clean", "Clean mask")):
        ax = fig.add_axes(rects[key], sharey=ax_a) if ax_a is not None \
            else fig.add_axes(rects[key])
        table = series_by_density(dnai, graft, greedy, condition)
        panel_density(ax, table, full_n=10)
        ax.set_title("(%s) %s" % (key, name), loc="left",
                     fontsize=PT_TITLE, fontweight="bold", pad=5.0)
        if ax_a is None:
            ax.set_ylabel("Mean common-fragment F₁", fontsize=PT_AXIS,
                          labelpad=2.0)
            ax_a = ax
        else:
            ax.tick_params(axis="y", labelleft=False)

    ax_c = fig.add_axes(rects["c"])
    gaps = paired_gaps(dnai, graft, greedy)
    panel_gaps(ax_c, gaps)
    ax_c.set_title("(c) Paired difference", loc="left", fontsize=PT_TITLE,
                   fontweight="bold", pad=5.0)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=PT_LEGEND,
               handlelength=2.0, columnspacing=1.15, handletextpad=0.45)

    save_fig(fig, "fig_comparators", bbox_inches=None)
    plt.close(fig)

    for condition in ("degraded", "clean"):
        table = series_by_density(dnai, graft, greedy, condition)
        print("--", condition)
        for key, label, _c, _m, _d in SERIES:
            print("   %-20s" % label,
                  " ".join("%d%%:%.3f(n=%d)" % (d, table[key][d][0],
                                                table[key][d][1])
                           for d in DENSITIES))
        for key, label, _c in GAPS:
            diff, lo, hi, n, wins = gaps[condition][key]
            print("   gap vs %-8s %+.3f [%.3f, %.3f] n=%d wins=%d"
                  % (label, diff, lo, hi, n, wins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
