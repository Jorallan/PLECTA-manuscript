"""Figure: the task PLECTA solves, and the contract of its output.

Panel (a) states the ambiguity on real pixels: a crossing is one blob of mask,
and nothing local says which arm continues into which.  Panels (b) and (c) give
the input-output contract on a whole real scene.  Panel (d) shows what
"overlap-aware" means at pixel level: the crossing pixels are written into both
filaments, so the output is a set of layers, not a partition.

All pixels come from ``results/plecta_figure_data.json``.

    python scripts/figures/fig_plecta_task.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from _style import (FIG_W, INSTANCE_CYCLE, PT_BODY, PT_LABEL, PT_TINY,
                    FIL_A, FIL_B, JUNCTION, CHORD, INK,
                    fig_prose, fig_title, load_figure_data, pixel_axes,
                    plecta_style, save_fig, show_layers, stack, unpack)


def draw_pixels(ax, mask, colour, z=2):
    for r, c in zip(*np.nonzero(mask)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colour,
                               edgecolor="none", zorder=z))


def draw_split_pixels(ax, mask, first, second, z=4):
    for r, c in zip(*np.nonzero(mask)):
        x0, y0 = c - 0.5, r - 0.5
        ax.add_patch(Polygon([(x0, y0), (x0 + 1, y0), (x0, y0 + 1)],
                             facecolor=first, edgecolor="none", zorder=z))
        ax.add_patch(Polygon([(x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)],
                             facecolor=second, edgecolor="none", zorder=z))


def arrow_between(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="-|>,head_length=0.5,head_width=0.28", mutation_scale=11,
        color=CHORD, lw=1.1, shrinkA=0, shrinkB=0, zorder=10))


def main() -> int:
    plecta_style()
    data = load_figure_data()
    cr = data["crossing"]
    proc = data["procedural"]

    panel_in = 1.30
    # bottom=0.34 in: panel (g) carries an axis label and tick labels below
    # its box, and ``stack`` measures panels, not what hangs under them.
    rows, fig_h = stack([("r1", panel_in, 0), ("r2", panel_in, 0)],
                        bottom=0.34)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    w, h = panel_in / FIG_W, panel_in / fig_h

    # Four square panels across the page.  The step is derived from the panel
    # width rather than typed in, so the fourth panel lands inside the canvas
    # instead of hanging over the right edge and being cut off by the save.
    x_left, x_right = 0.045, 0.995
    step = (x_right - x_left - w) / 3.0
    col = [x_left + i * step for i in range(4)]

    # ── (a) the ambiguity, on real pixels ──────────────────────────────
    r1 = rows["r1"]
    ax_a = pixel_axes(fig, [col[0], r1["panel"], w, h], cr["size"])
    draw_pixels(ax_a, unpack(cr["mask"]), INK)
    draw_pixels(ax_a, unpack(cr["node_px"]), JUNCTION, z=3)
    for dx, dy in ((0.80, 0.14), (0.16, 0.86), (0.86, 0.84), (0.10, 0.16)):
        ax_a.text(dx * cr["size"], dy * cr["size"], "?", fontsize=PT_BODY,
                  fontweight="bold", color=CHORD, ha="center", va="center",
                  zorder=6)

    # ── (b) input: one whole binary axis mask ──────────────────────────
    mask = unpack(proc["mask_w1"])
    ax_b = pixel_axes(fig, [col[1], r1["panel"], w, h], mask.shape[0])
    show_layers(ax_b, [mask], [INK], background=np.zeros_like(mask))

    # ── (c) output: overlapping instances ──────────────────────────────
    layers = [unpack(p) for p in cr["layers"]]
    ladder = proc["ladder"]
    scene = next(row for row in ladder if row["density"] == proc["meta"].get(
        "coverage_percent", 30)) if False else None
    inst = [unpack(p) for p in proc["gt_layers"]]
    ax_c = pixel_axes(fig, [col[2], r1["panel"], w, h], mask.shape[0])
    show_layers(ax_c, inst,
                [INSTANCE_CYCLE[i % len(INSTANCE_CYCLE)] for i in range(len(inst))],
                background=np.zeros_like(mask))

    # ── (d) the same crossing, as layers ───────────────────────────────
    ax_d = pixel_axes(fig, [col[3], r1["panel"], w, h], cr["size"])
    shared = layers[0] & layers[1] if len(layers) >= 2 else np.zeros_like(layers[0])
    draw_pixels(ax_d, layers[0] & ~shared, FIL_A)
    if len(layers) >= 2:
        draw_pixels(ax_d, layers[1] & ~shared, FIL_B)
    draw_split_pixels(ax_d, shared, FIL_A, FIL_B)

    y_arrow = r1["panel"] + h / 2.0
    for i in range(3):
        arrow_between(fig, col[i] + w + 0.006, col[i + 1] - 0.006, y_arrow)

    for x, letter, name in zip(col, "abcd",
                               ("The ambiguity", "The input", "The reference",
                                "Overlap")):
        fig_title(fig, x - 0.015, r1["title"], letter, name)


    # ── row 2: what "layers, not a partition" costs and buys ───────────
    r2 = rows["r2"]
    stage4 = None
    ax_e = pixel_axes(fig, [col[0], r2["panel"], w, h], cr["size"])
    only_a = layers[0] & ~shared
    draw_pixels(ax_e, only_a, FIL_A)
    draw_pixels(ax_e, shared, FIL_A, z=4)

    ax_f = pixel_axes(fig, [col[1], r2["panel"], w, h], cr["size"])
    if len(layers) >= 2:
        draw_pixels(ax_f, layers[1] & ~shared, FIL_B)
        draw_pixels(ax_f, shared, FIL_B, z=4)

    ax_g = fig.add_axes([col[2] + 0.055, r2["panel"] + 0.030,
                         x_right - col[2] - 0.090, h - 0.060])
    counts = [int(layers[0].sum()), int(layers[1].sum()) if len(layers) > 1 else 0,
              int(shared.sum())]
    labels = ["instance A", "instance B", "claimed by both"]
    colours = [FIL_A, FIL_B, CHORD]
    ypos = np.arange(3)[::-1]
    ax_g.barh(ypos, counts, height=0.55, color=colours, zorder=3)
    for yi, v in zip(ypos, counts):
        ax_g.text(v + 1.2, yi, str(v), va="center", ha="left",
                  fontsize=PT_TINY, color=INK)
    ax_g.set_yticks(ypos)
    ax_g.set_yticklabels(labels, fontsize=PT_LABEL)
    ax_g.set_xlim(0, max(counts) * 1.25)
    ax_g.set_xlabel("pixels in the %d x %d crop" % (cr["size"], cr["size"]),
                    fontsize=PT_LABEL, labelpad=1.0)
    for s in ("top", "right", "left"):
        ax_g.spines[s].set_visible(False)
    ax_g.tick_params(axis="y", length=0, pad=1.5)

    fig_title(fig, col[0] - 0.015, r2["title"], "e", "Layer A")
    fig_title(fig, col[1] - 0.015, r2["title"], "f", "Layer B")
    fig_title(fig, col[2] + 0.040, r2["title"], "g", "Layer sizes")


    save_fig(fig, "fig_plecta_task", subdir="archive", bbox_inches=None)
    plt.close(fig)
    print("fig height:", round(fig_h, 3), "reference instances:", len(inst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
