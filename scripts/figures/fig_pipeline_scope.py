"""Figure: scope -- what PLECTA is, what feeds it, and what the result covers.

Same two-band schematic as the earlier scope figure in this repository, and the
same colour semantics and legend idiom, but Pipeline B now carries what PLECTA
actually does rather than the superseded four-stage pipeline.  The separate
"input-mask formation" scope figure is folded in here: the input-output
contract, the OR junction between a procedural mask and a segmenter mask, and
the boundary of the evaluated result all live in this one figure.

Colour semantics (legend, bottom): blue = learned component, green = PLECTA
geometric step, orange = dependency on the grayscale image, dark = data
artefact.

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
                    LIGHT_GRAY, ORANGE, ORANGE_TINT, FIG_W,
                    elbow, harrow, plecta_style, rbox, save_fig, stage_box,
                    swatch_legend)

FIG_H = 5.55
XMAX, YMAX = 13.60, 11.60


def main() -> int:
    plecta_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    fig.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.006)

    # ── bands ──────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.18, 10.02), 13.24, 1.42,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0, facecolor=BLUE_TINT, zorder=0))
    ax.text(0.42, 11.22, "PIPELINE A  —  binary segmentation of SEM images",
            ha="left", va="center", fontsize=9.6, fontweight="bold",
            color=BLUE, zorder=1)

    ax.add_patch(FancyBboxPatch((0.18, 0.16), 13.24, 9.62,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0, facecolor=GREEN_TINT, zorder=0))
    ax.text(0.42, 9.56,
            "PIPELINE B  —  PLECTA: overlap-aware instance reconstruction",
            ha="left", va="center", fontsize=9.6, fontweight="bold",
            color=GREEN, zorder=1)

    # ── Pipeline A, unchanged ──────────────────────────────────────────
    yA, hA = 10.62, 0.76
    boxesA = [
        (1.60, 2.80, "Procedural binary masks\n+ real SEM images", GRAY,
         "white", False, (0, (4, 2)), 7.0),
        (4.505, 2.45, "CycleGAN\nimage translation", BLUE, BLUE_TINT, True,
         "solid", 7.4),
        (7.185, 2.35, "SEM-like synthetic\ntraining images", GRAY, "white",
         False, (0, (4, 2)), 7.4),
        (9.665, 2.05, "U-Net\nsegmentation", BLUE, BLUE_TINT, True, "solid",
         7.4),
        (12.145, 2.35, "Binary CNT\nbundle-axis mask", DARK, "white", True,
         "solid", 7.4),
    ]
    for xc, w, text, edge, face, bold, ls, fs in boxesA:
        rbox(ax, xc, yA, w, hA, text, edge, face=face, fontsize=fs, bold=bold,
             ls=ls, lw=2.0 if edge == DARK else 1.5)
    for (x0, w0), (x1, w1) in zip([(b[0], b[1]) for b in boxesA[:-1]],
                                  [(b[0], b[1]) for b in boxesA[1:]]):
        harrow(ax, x0 + w0 / 2 + 0.04, yA, x1 - w1 / 2 - 0.04, yA, lw=1.4)

    # ── Pipeline B input tier: two mask sources, one OR junction ───────
    y_in = 8.45
    junc_x = 6.05
    rbox(ax, 2.45, y_in, 3.15, 0.86, "Procedural binary\naxis mask", GRAY,
         face="white", fontsize=7.4, ls=(0, (4, 2)), lw=1.5, bold=False)
    ax.text(2.45, y_in + 0.43 + 0.10, "(development and evaluation path)",
            ha="center", va="bottom", fontsize=6.9, color=GRAY,
            style="italic", zorder=4)
    harrow(ax, 2.45 + 3.15 / 2 + 0.05, y_in, junc_x - 0.22, y_in, lw=1.4)

    elbow(ax, [(boxesA[-1][0], yA - hA / 2 - 0.03), (boxesA[-1][0], 9.10),
               (junc_x, 9.10), (junc_x, y_in + 0.23)],
          colour=DARK, lw=1.6, ls="solid")
    ax.text(junc_x + 0.32, 9.18, "SEM-derived mask", ha="left", va="bottom",
            fontsize=7.1, color=DARK, style="italic", zorder=4)

    ax.add_patch(Circle((junc_x, y_in), 0.20, facecolor="white",
                        edgecolor=DARK, linewidth=1.5, zorder=5))
    ax.text(junc_x, y_in, "OR", ha="center", va="center", fontsize=6.9,
            fontweight="bold", color=DARK, zorder=6)

    x_mask, w_mask = 9.95, 4.20
    rbox(ax, x_mask, y_in, w_mask, 0.86,
         "Binary filament-axis mask\n(the only input PLECTA reads)", DARK,
         face="white", fontsize=7.4, lw=1.9, bold=True)
    harrow(ax, junc_x + 0.22, y_in, x_mask - w_mask / 2 - 0.05, y_in, lw=1.4)

    # ── the evaluated core ─────────────────────────────────────────────
    panel_l, panel_r = 0.45, 13.15
    stage_y, stage_h = 6.45, 1.72
    panel_top = stage_y + stage_h / 2 + 0.42
    panel_bot = stage_y - stage_h / 2 - 1.18
    ax.add_patch(FancyBboxPatch((panel_l, panel_bot), panel_r - panel_l,
                                panel_top - panel_bot,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=1.4, linestyle=(0, (5, 3)),
                                edgecolor=GREEN, facecolor="none", zorder=1))
    ax.text(panel_l + 0.20, panel_top - 0.10,
            "Evaluated grouping  —  mask geometry only",
            ha="left", va="top", fontsize=7.5, fontweight="bold",
            style="italic", color=GREEN, zorder=4)

    pad, gap = 0.18, 0.15
    stage_w = (panel_r - panel_l - 2 * pad - 4 * gap) / 5.0
    centres = [panel_l + pad + stage_w / 2 + i * (stage_w + gap)
               for i in range(5)]
    steps = [
        ("STEP 1", "Skeleton graph:\narms, stubs and\njunction nodes"),
        ("STEP 2", "Frame at each\nstub: tangent $\\mathbf{t}$,\n"
                   "curvature $\\kappa$"),
        ("STEP 3", "Exact matching\nat each junction,\nwith a priced\n"
                   "unmatched option"),
        ("STEP 4", "Gated global\nmatching across\nmask gaps"),
        ("STEP 5", "Paint overlap-\naware instance\nlayers"),
    ]
    for xc, (tag, name) in zip(centres, steps):
        stage_box(ax, xc, stage_y, stage_w, stage_h, tag, name, GREEN,
                  fontsize=7.2, tag_fontsize=6.8)
    for a, b in zip(centres[:-1], centres[1:]):
        harrow(ax, a + stage_w / 2 + 0.04, stage_y, b - stage_w / 2 - 0.04,
               stage_y, colour=GREEN, lw=1.3)

    harrow(ax, x_mask, y_in - 0.43 - 0.03, x_mask, panel_top + 0.04,
           colour=DARK, lw=1.6)

    # the refinement loop: steps 3 and 4 are re-made against re-fitted frames
    loop_y = stage_y - stage_h / 2 - 0.52
    ax.plot([centres[3], centres[3]], [stage_y - stage_h / 2 - 0.03, loop_y],
            color=GREEN, lw=1.3, ls=(0, (4, 2.4)), zorder=2)
    ax.plot([centres[1], centres[3]], [loop_y, loop_y], color=GREEN, lw=1.3,
            ls=(0, (4, 2.4)), zorder=2)
    harrow(ax, centres[1], loop_y, centres[1], stage_y - stage_h / 2 - 0.03,
           colour=GREEN, lw=1.3, ls=(0, (4, 2.4)))
    ax.text(centres[2], loop_y - 0.14,
            "8 rounds — frames re-fitted along the chain assembled so far "
            "(24 px locally, 55 px along a chain).\nThe fixed schedule reaches "
            "full strictness only in round 8, and that round is returned.",
            ha="center", va="top", fontsize=6.9, color=GREEN, style="italic",
            zorder=4, linespacing=1.35)

    # ── output, and the optional downstream branch ─────────────────────
    y_out = 2.42
    x_out, w_out = 3.35, 4.55
    rbox(ax, x_out, y_out, w_out, 1.02,
         "Overlap-aware 2-D instance layers\n(a crossing pixel has two owners)",
         DARK, face="white", fontsize=7.4, lw=1.9)
    elbow(ax, [(centres[4], panel_bot - 0.03), (centres[4], y_out + 0.90),
               (x_out, y_out + 0.90), (x_out, y_out + 0.54)],
          colour=DARK, lw=1.5, ls="solid")

    x_gray, y_gray, w_gray = 8.85, 3.94, 2.85
    rbox(ax, x_gray, y_gray, w_gray, 0.78, "Paired grayscale\nSEM image",
         ORANGE, face=ORANGE_TINT, fontsize=7.4, lw=1.6, ls=(0, (4, 2)))
    x_char, w_char = 8.85, 3.45
    rbox(ax, x_char, y_out, w_char, 1.02,
         "Optional downstream:\nwidth measurement and\nribbon rendering",
         ORANGE, face=ORANGE_TINT, fontsize=7.4, lw=1.8)
    harrow(ax, x_gray, y_gray - 0.39 - 0.03, x_char, y_out + 0.54,
           colour=ORANGE, lw=1.6)
    harrow(ax, x_out + w_out / 2 + 0.04, y_out, x_char - w_char / 2 - 0.04,
           y_out, colour=DARK, lw=1.4)

    x_widths, w_widths = 12.20, 2.35
    rbox(ax, x_widths, y_out, w_widths, 1.02,
         "Widths and\nrendered ribbon\nmasks", DARK, face="white",
         fontsize=7.4, lw=1.7)
    harrow(ax, x_char + w_char / 2 + 0.04, y_out, x_widths - w_widths / 2 - 0.04,
           y_out, colour=ORANGE, lw=1.4)
    ax.text(x_char, y_out - 0.51 - 0.13,
            "default off; not part of the held-out grouping result",
            ha="center", va="top", fontsize=6.9, color=ORANGE, style="italic",
            zorder=4)

    # ── legend ─────────────────────────────────────────────────────────
    swatch_legend(ax,
                  [(BLUE, "Learned component"),
                   (GREEN, "PLECTA geometric step"),
                   (ORANGE, "Grayscale-image input"),
                   (DARK, "Data artefact")],
                  [0.55, 3.55, 6.95, 10.05], 0.60, fontsize=7.2)

    save_fig(fig, "fig_pipeline_scope", bbox_inches=None)
    plt.close(fig)
    print("saved fig_pipeline_scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
