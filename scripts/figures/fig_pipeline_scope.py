"""Figure 1: what this paper contributes, and where the evidence stops.

Restructured after review.  Three things the earlier version did not say
clearly enough:

  * the two inputs are declared **once**, at the top, and every consumer is
    fed from there.  The grayscale SEM image used to appear beside the
    downstream box, halfway down the diagram, as though it arrived from
    nowhere;
  * the hierarchy is explicit.  Pipeline B (PLECTA) is the contribution and
    gets the large band; pipeline A (CycleGAN + U-Net) is an *enabling
    example* -- one proof-of-concept route to a binary mask for real SEM --
    and gets a slim subordinate band;
  * the evidence boundary is a single dashed enclosure that begins at the
    binary mask and ends at the instance layers.  Everything the paper scores
    is inside it.  Width measurement and ribbon rendering sit outside and
    below, tagged with the reason they are off: they change the
    representation, not the identities, and identity is what the metric
    scores.

Text is deliberately minimal -- box names, step names, one tag per boundary.
The mechanism is in the Methods.

Fully self-contained; no data file is read.

    python scripts/figures/fig_pipeline_scope.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, BLUE_TINT, DARK, GRAY, GREEN, GREEN_TINT,
                    ORANGE, ORANGE_TINT, FIG_W, PT_ANNOT, PT_LEGEND, PT_TITLE,
                    elbow, harrow, plecta_style, rbox, save_fig, stage_box,
                    swatch_legend)

#: Drawing units per inch.  Fixed so that horizontal and vertical distances
#: mean the same thing and a box is not silently stretched.
UNITS_PER_IN = 13.60 / FIG_W
XMAX, YMAX = 13.60, 11.00
FIG_H = YMAX / UNITS_PER_IN


def band(ax, x0, x1, y0, y1, colour):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0, facecolor=colour, zorder=0))


def main() -> int:
    plecta_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

    # ── inputs, declared once ──────────────────────────────────────────
    band(ax, 0.18, 13.42, 9.58, 10.92, "#f1f3f6")
    ax.text(0.44, 10.72, "INPUTS", ha="left", va="center", fontsize=PT_ANNOT,
            fontweight="bold", color=GRAY, zorder=1)

    y_in, h_in = 10.05, 0.68
    x_proc, w_proc = 3.30, 3.90
    x_sem, w_sem = 9.90, 3.50
    rbox(ax, x_proc, y_in, w_proc, h_in, "Procedural binary axis mask",
         GRAY, face="white", fontsize=PT_ANNOT, ls=(0, (4, 2)), lw=1.4,
         bold=False)
    rbox(ax, x_sem, y_in, w_sem, h_in, "Grayscale SEM image", ORANGE,
         face=ORANGE_TINT, fontsize=PT_ANNOT, ls=(0, (4, 2)), lw=1.4,
         bold=False)

    # ── A: the enabling example, drawn subordinate ─────────────────────
    band(ax, 2.35, 13.20, 7.92, 9.38, BLUE_TINT)
    ax.text(2.60, 9.14, "A   Enabling example — a binary mask from real SEM",
            ha="left", va="center", fontsize=PT_ANNOT, fontweight="bold",
            color=BLUE, zorder=1)

    yA, hA = 8.40, 0.72
    boxesA = [
        (4.30, 2.30, "CycleGAN", BLUE, BLUE_TINT, True, "solid"),
        (7.25, 2.90, "SEM-like synthetic\nimages", GRAY, "white", False,
         (0, (4, 2))),
        (10.05, 2.05, "U-Net", BLUE, BLUE_TINT, True, "solid"),
        (12.35, 2.05, "SEM-derived\naxis mask", DARK, "white", True, "solid"),
    ]
    for xc, w, text, edge, face, bold, ls in boxesA:
        rbox(ax, xc, yA, w, hA, text, edge, face=face, fontsize=PT_ANNOT,
             bold=bold, ls=ls, lw=1.6 if edge == DARK else 1.4)
    for (x0, w0), (x1, w1) in zip([(b[0], b[1]) for b in boxesA[:-1]],
                                  [(b[0], b[1]) for b in boxesA[1:]]):
        harrow(ax, x0 + w0 / 2 + 0.04, yA, x1 - w1 / 2 - 0.04, yA, lw=1.2,
               mscale=9)

    top_A = yA + hA / 2
    elbow(ax, [(x_proc, y_in - h_in / 2 - 0.03), (x_proc, 9.44),
               (3.95, 9.44), (3.95, top_A + 0.03)],
          colour=GRAY, lw=1.2, ls="solid", mscale=9)
    elbow(ax, [(x_sem, y_in - h_in / 2 - 0.03), (x_sem, 9.52),
               (4.62, 9.52), (4.62, top_A + 0.03)],
          colour=ORANGE, lw=1.2, ls="solid", mscale=9)

    # ── B: the contribution ────────────────────────────────────────────
    band(ax, 0.18, 13.42, 0.62, 7.90, GREEN_TINT)
    ax.text(0.44, 7.64,
            "B   PLECTA — overlap-aware instance segmentation  (this paper)",
            ha="left", va="center", fontsize=PT_TITLE, fontweight="bold",
            color=GREEN, zorder=1)

    # the two mask sources meet, outside the evidence boundary
    junc = (6.80, 7.18)
    elbow(ax, [(1.72, y_in - h_in / 2 - 0.03), (1.72, junc[1]),
               (junc[0] - 0.23, junc[1])], colour=GRAY, lw=1.2, ls="solid",
          mscale=9)
    elbow(ax, [(boxesA[-1][0], yA - hA / 2 - 0.03),
               (boxesA[-1][0], junc[1]), (junc[0] + 0.23, junc[1])],
          colour=DARK, lw=1.2, ls="solid", mscale=9)
    ax.add_patch(Circle(junc, 0.21, facecolor="white", edgecolor=DARK,
                        linewidth=1.3, zorder=5))
    ax.text(junc[0], junc[1], "OR", ha="center", va="center",
            fontsize=PT_LEGEND, fontweight="bold", color=DARK, zorder=6)

    # ── the evidence boundary ──────────────────────────────────────────
    ev_l, ev_r, ev_b, ev_t = 0.50, 13.10, 2.40, 6.92
    ax.add_patch(FancyBboxPatch((ev_l, ev_b), ev_r - ev_l, ev_t - ev_b,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=1.4, linestyle=(0, (5, 3)),
                                edgecolor=GREEN, facecolor="none", zorder=1))
    ax.text(ev_l + 0.22, ev_t - 0.12,
            "EVALUATED:  mask in  →  instances out", ha="left", va="top",
            fontsize=PT_ANNOT, fontweight="bold", color=GREEN, zorder=4)

    x_mask, y_mask, w_mask, h_mask = 6.80, 6.22, 5.30, 0.74
    rbox(ax, x_mask, y_mask, w_mask, h_mask,
         "Binary filament-axis mask", DARK, face="white", fontsize=PT_ANNOT,
         lw=1.7, bold=True)
    harrow(ax, junc[0], junc[1] - 0.22, x_mask, y_mask + h_mask / 2 + 0.03,
           colour=DARK, lw=1.4, mscale=10)

    # ── the five geometric steps, and the round loop ───────────────────
    stage_y, stage_h = 4.86, 1.54
    pad, gap = 0.20, 0.16
    stage_w = (ev_r - ev_l - 2 * pad - 4 * gap) / 5.0
    centres = [ev_l + pad + stage_w / 2 + i * (stage_w + gap) for i in range(5)]
    steps = [
        ("STEP 1", "Skeleton graph:\narms, stubs,\njunction nodes"),
        ("STEP 2", "Frame at\neach stub"),
        ("STEP 3", "Exact matching\nat each junction"),
        ("STEP 4", "Gated bridging\nof mask gaps"),
        ("STEP 5", "Paint instance\nlayers"),
    ]
    for xc, (tag, name) in zip(centres, steps):
        stage_box(ax, xc, stage_y, stage_w, stage_h, tag, name, GREEN,
                  fontsize=PT_ANNOT, tag_fontsize=PT_ANNOT)
    for a, b in zip(centres[:-1], centres[1:]):
        harrow(ax, a + stage_w / 2 + 0.04, stage_y, b - stage_w / 2 - 0.04,
               stage_y, colour=GREEN, lw=1.2, mscale=9)

    elbow(ax, [(x_mask, y_mask - h_mask / 2 - 0.03), (x_mask, 5.78),
               (centres[0], 5.78), (centres[0], stage_y + stage_h / 2 + 0.03)],
          colour=DARK, lw=1.4, ls="solid", mscale=10)

    loop_y = stage_y - stage_h / 2 - 0.44
    ax.plot([centres[3], centres[3]], [stage_y - stage_h / 2 - 0.03, loop_y],
            color=GREEN, lw=1.2, ls=(0, (4, 2.4)), zorder=2)
    ax.plot([centres[1], centres[3]], [loop_y, loop_y], color=GREEN, lw=1.2,
            ls=(0, (4, 2.4)), zorder=2)
    harrow(ax, centres[1], loop_y, centres[1], stage_y - stage_h / 2 - 0.03,
           colour=GREEN, lw=1.2, ls=(0, (4, 2.4)), mscale=9)
    ax.text(centres[2], loop_y - 0.14, "8 rounds", ha="center", va="top",
            fontsize=PT_ANNOT, color=GREEN, style="italic", zorder=4)

    x_out, y_out, w_out, h_out = 2.95, 2.94, 4.70, 0.84
    rbox(ax, x_out, y_out, w_out, h_out,
         "Overlap-aware instance layers\n(a crossing pixel has two owners)",
         DARK, face="white", fontsize=PT_ANNOT, lw=1.7)
    elbow(ax, [(centres[4], stage_y - stage_h / 2 - 0.03),
               (centres[4], y_out), (x_out + w_out / 2 + 0.04, y_out)],
          colour=DARK, lw=1.4, ls="solid", mscale=10)

    # ── downstream, outside the boundary ───────────────────────────────
    y_dn, h_dn = 1.38, 0.80
    x_dn, w_dn = 4.60, 5.10
    rbox(ax, x_dn, y_dn, w_dn, h_dn,
         "Optional: width measurement\nand ribbon rendering", ORANGE,
         face=ORANGE_TINT, fontsize=PT_ANNOT, lw=1.6)
    x_rb, w_rb = 9.95, 3.30
    rbox(ax, x_rb, y_dn, w_rb, h_dn, "Rendered ribbon\ninstances", DARK,
         face="white", fontsize=PT_ANNOT, lw=1.5)
    harrow(ax, x_dn + w_dn / 2 + 0.04, y_dn, x_rb - w_rb / 2 - 0.04, y_dn,
           colour=ORANGE, lw=1.3, mscale=10)
    harrow(ax, x_out, y_out - h_out / 2 - 0.03, x_out, y_dn + h_dn / 2 + 0.03,
           colour=DARK, lw=1.4, mscale=10)
    elbow(ax, [(x_sem, y_in - h_in / 2 - 0.03), (x_sem, 9.66), (13.32, 9.66),
               (13.32, 2.10), (6.15, 2.10), (6.15, y_dn + h_dn / 2 + 0.03)],
          colour=ORANGE, lw=1.2, ls="solid", mscale=9)
    ax.text(x_dn, y_dn - h_dn / 2 - 0.14,
            "default off — changes the representation, not the identities",
            ha="center", va="top", fontsize=PT_ANNOT, color=ORANGE,
            style="italic", zorder=4)

    # ── legend ─────────────────────────────────────────────────────────
    swatch_legend(ax,
                  [(BLUE, "Learned component"),
                   (GREEN, "PLECTA geometric step"),
                   (ORANGE, "Needs the SEM image"),
                   (DARK, "Data artefact")],
                  [0.52, 3.75, 7.30, 10.55], 0.28, fontsize=PT_LEGEND)

    save_fig(fig, "fig_pipeline_scope", bbox_inches=None)
    plt.close(fig)
    print("fig_pipeline_scope  %.3f x %.3f in" % (FIG_W, FIG_H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
