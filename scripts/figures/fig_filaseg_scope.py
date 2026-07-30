"""FIG: FilaSeg scope schematic.

Two-lane horizontal pipeline schematic showing exactly what FilaSeg covers
and how it relates to the upstream CycleGAN/U-Net segmentation pipeline.

Pipeline A (top lane): procedural masks + real SEM images -> CycleGAN ->
SEM-like synthetic training images -> U-Net -> binary CNT bundle-axis mask.

Pipeline B (bottom lane): a binary axis mask -- either the Pipeline-A
output, or a procedurally generated mask entering directly (the
development/evaluation path) -- feeds an OR junction, then Stages 1-3
(vectorization, fragment cleaning, geometric reconnection), which are
visually boxed in a dashed panel to show they consume only the binary axis
mask. Stage 3's instances then drop into Stage 4, which is the only stage
that additionally needs the paired grayscale SEM image (drawn entering
from the left, in the grayscale-input colour, nowhere else in the
diagram).

Colour semantics: blue = learned components (CycleGAN, U-Net), green =
geometric FilaSeg stages (1-3), orange = grayscale-image dependency
(Stage 4 and its grayscale input), dark/gray = data artifacts
(masks/images/outputs). Pure vector schematic (no external raster),
theme-neutral, publication style, consistent with the shared _style.py
palette (fsblue/fsorange/fsgreen).
"""
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, save_fig, BLUE, ORANGE, GREEN, GRAY, LIGHT_GRAY, DARK

BLUE_TINT = "#eaf1f8"
GREEN_TINT = "#eaf4ee"
ORANGE_TINT = "#fdf1e7"


def rbox(ax, xc, yc, w, h, title, edge, face="white", lw=1.6,
          fontsize=7.8, fontcolor=DARK, ls="solid", bold=True,
          rounding=0.10, zorder=3, linespacing=1.3):
    box = FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, linestyle=ls, edgecolor=edge, facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(xc, yc, title, ha="center", va="center", fontsize=fontsize,
             fontweight="bold" if bold else "normal", color=fontcolor,
             zorder=zorder + 1, linespacing=linespacing)
    return box


def stage_box(ax, xc, yc, w, h, tag, name, color, fontsize=7.6):
    box = FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.8, edgecolor=color, facecolor="white", zorder=3,
    )
    ax.add_patch(box)
    ax.text(xc, yc + h / 2 - 0.20, tag, ha="center", va="center",
             fontsize=6.8, fontweight="bold", color=color, zorder=4)
    ax.text(xc, yc - 0.12, name, ha="center", va="center",
             fontsize=fontsize, fontweight="bold", color=DARK, zorder=4, linespacing=1.35)


def harrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.5, ls="solid", mscale=12, zorder=2):
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                            mutation_scale=mscale, linewidth=lw, linestyle=ls,
                            color=color, zorder=zorder)
    ax.add_patch(arr)
    return arr


def elbow(ax, pts, color=GRAY, lw=1.4, ls=(0, (4, 2.5)), zorder=2):
    """Rectilinear connector through pts; final leg gets an arrowhead."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                 color=color, linewidth=lw, linestyle=ls, zorder=zorder,
                 solid_capstyle="butt")
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                   mutation_scale=12, linewidth=lw,
                                   linestyle=ls, color=color, zorder=zorder))


def main():
    apply_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 9.3)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Band backgrounds + labels
    # ------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.18, 7.75), 13.24, 1.40,
                                  boxstyle="round,pad=0.02,rounding_size=0.10",
                                  linewidth=0, facecolor=BLUE_TINT, zorder=0))
    ax.text(0.42, 8.95, "PIPELINE A  —  binary segmentation of SEM images",
             ha="left", va="center", fontsize=10, fontweight="bold", color=BLUE, zorder=1)

    ax.add_patch(FancyBboxPatch((0.18, 0.15), 13.24, 7.40,
                                  boxstyle="round,pad=0.02,rounding_size=0.10",
                                  linewidth=0, facecolor=GREEN_TINT, zorder=0))
    ax.text(0.42, 7.30, "PIPELINE B  —  instance segmentation of dense filament networks",
             ha="left", va="center", fontsize=10, fontweight="bold", color=GREEN, zorder=1)

    # ------------------------------------------------------------------
    # Pipeline A row (y-center 8.15)
    # ------------------------------------------------------------------
    yA = 8.15
    hA = 0.75
    boxesA = [
        (1.55, 2.70, "Procedural binary masks\n+ real SEM images", GRAY, "white", False, (0, (4, 2)), 7.0),
        (4.35, 2.05, "CycleGAN\nimage translation", BLUE, BLUE_TINT, True, "solid", 7.4),
        (6.85, 2.35, "SEM-like synthetic\ntraining images", GRAY, "white", False, (0, (4, 2)), 7.4),
        (9.35, 2.05, "U-Net\nsegmentation", BLUE, BLUE_TINT, True, "solid", 7.4),
        (11.85, 2.35, "Binary CNT\nbundle-axis mask", DARK, "white", True, "solid", 7.4),
    ]
    for xc, w, title, edge, face, bold, ls, fs in boxesA:
        lw = 2.0 if edge == DARK else 1.5
        rbox(ax, xc, yA, w, hA, title, edge, face=face, fontsize=fs, bold=bold, ls=ls, lw=lw)
    for (x0, w0), (x1, w1) in zip([(b[0], b[1]) for b in boxesA[:-1]], [(b[0], b[1]) for b in boxesA[1:]]):
        harrow(ax, x0 + w0 / 2 + 0.04, yA, x1 - w1 / 2 - 0.04, yA, color=GRAY, lw=1.4)

    # ------------------------------------------------------------------
    # A -> B hand-off: routed down the right margin and across a gutter
    # well below the Pipeline-B band label, into the OR junction
    # (relationship 1: Pipeline A's output can feed Pipeline B).
    # ------------------------------------------------------------------
    junc_x, junc_y = 3.10, 5.55
    elbow(ax, [(11.85, yA - hA / 2 - 0.03), (11.85, 6.90), (junc_x, 6.90), (junc_x, junc_y + 0.20)],
          color=DARK, lw=1.6, ls="solid")
    # Sits just BELOW the horizontal run of the elbow and to the right of its
    # vertical drop, so the connector cannot pass through the text. A
    # right-aligned label ending past x=11.85 would be crossed by the drop.
    ax.text(11.78, 6.74, "SEM-derived mask", ha="left", va="top",
             fontsize=7.2, color=DARK, style="italic", zorder=4)

    # ------------------------------------------------------------------
    # Pipeline B, top tier: two alternative inputs -> OR junction ->
    # Stages 1-3 panel.
    # ------------------------------------------------------------------
    b_in = rbox(ax, 1.55, junc_y, 2.35, 0.85,
                 "Procedural binary\naxis mask", GRAY, face="white",
                 fontsize=7.4, fontcolor=DARK, ls=(0, (4, 2)), lw=1.5, bold=False)
    ax.text(1.55, junc_y - 0.425 - 0.20, "(development / evaluation path)",
             ha="center", va="top", fontsize=6.8, color=GRAY, style="italic", zorder=4)

    harrow(ax, 1.55 + 2.35 / 2 + 0.04, junc_y, junc_x - 0.21, junc_y, color=GRAY, lw=1.4)

    ax.add_patch(Circle((junc_x, junc_y), 0.19, facecolor="white", edgecolor=DARK, linewidth=1.5, zorder=5))
    ax.text(junc_x, junc_y, "OR", ha="center", va="center", fontsize=7.0, fontweight="bold", color=DARK, zorder=6)

    panel_left = 3.45
    stage_w, stage_gap, panel_pad = 2.30, 0.15, 0.15
    panel_right = panel_left + 2 * panel_pad + 3 * stage_w + 2 * stage_gap
    stage_y = 5.40
    stage_h = 1.30
    panel_top = stage_y + stage_h / 2 + 0.32
    panel_bot = stage_y - stage_h / 2 - 0.20
    ax.add_patch(FancyBboxPatch((panel_left, panel_bot), panel_right - panel_left, panel_top - panel_bot,
                                  boxstyle="round,pad=0.02,rounding_size=0.14",
                                  linewidth=1.3, linestyle=(0, (5, 3)), edgecolor=LIGHT_GRAY,
                                  facecolor="none", zorder=1))
    ax.text(panel_left + 0.18, panel_top - 0.08, "Stages 1–3  —  operate on the binary axis mask only",
             ha="left", va="top", fontsize=7.6, fontweight="bold", style="italic", color=GRAY, zorder=4)

    harrow(ax, junc_x + 0.21, junc_y, panel_left - 0.03, junc_y, color=GRAY, lw=1.4)

    s_centers = [panel_left + panel_pad + stage_w / 2 + i * (stage_w + stage_gap) for i in range(3)]
    stage_names = [
        ("STAGE 1", "Tiled residual\nHough\nvectorization"),
        ("STAGE 2", "Orientation-\naware fragment\ncleaning"),
        ("STAGE 3", "Geometry-based\ninstance\nreconnection"),
    ]
    for xc, (tag, name) in zip(s_centers, stage_names):
        stage_box(ax, xc, stage_y, stage_w, stage_h, tag, name, GREEN, fontsize=7.8)
    harrow(ax, s_centers[0] + stage_w / 2 + 0.04, stage_y, s_centers[1] - stage_w / 2 - 0.04, stage_y, color=GREEN, lw=1.3)
    harrow(ax, s_centers[1] + stage_w / 2 + 0.04, stage_y, s_centers[2] - stage_w / 2 - 0.04, stage_y, color=GREEN, lw=1.3)

    # ------------------------------------------------------------------
    # Pipeline B, bottom tier: Stage 4, set apart from the panel, fed by
    # centreline instances (from Stage 3, top-right entry) and by the
    # paired grayscale image (from above-left entry) -- the ONLY point in
    # the whole diagram where a grayscale image is consumed.
    # ------------------------------------------------------------------
    s4_x, s4_y = 7.55, 2.15
    s4_w, s4_h = 3.35, 1.30

    entry_inst_x = s4_x + 0.95
    elbow_y = panel_bot - 0.75
    elbow(ax, [(s_centers[2], panel_bot - 0.03), (s_centers[2], elbow_y),
               (entry_inst_x, elbow_y), (entry_inst_x, s4_y + s4_h / 2 + 0.03)],
          color=DARK, lw=1.5, ls="solid")
    ax.text(s_centers[2] + 0.14, elbow_y + 0.17, "individual centreline\ninstances",
             ha="left", va="bottom", fontsize=6.8, color=DARK, style="italic",
             zorder=4, linespacing=1.3)

    gray_x = s4_x - 1.05
    gray_y = panel_bot - 0.70
    rbox(ax, gray_x, gray_y, 2.30, 0.78,
          "Paired grayscale\nSEM image", ORANGE, face=ORANGE_TINT,
          fontsize=7.4, lw=1.6, ls=(0, (4, 2)))
    harrow(ax, gray_x, gray_y - 0.39 - 0.03, gray_x, s4_y + s4_h / 2 + 0.03, color=ORANGE, lw=1.7)

    stage_box(ax, s4_x, s4_y, s4_w, s4_h, "STAGE 4",
               "SEM-assisted\napparent-width\nestimation & rendering", ORANGE, fontsize=7.4)

    out_x = s4_x + s4_w / 2 + 0.35 + 1.45
    out_w = 2.90
    rbox(ax, out_x, s4_y, out_w, s4_h,
          "Instance-labelled\nmask + per-instance\ngeometry", DARK, face="white", fontsize=7.6, lw=1.7)
    harrow(ax, s4_x + s4_w / 2 + 0.04, s4_y, out_x - out_w / 2 - 0.04, s4_y, color=GRAY, lw=1.5)

    # ------------------------------------------------------------------
    # Legend (colour semantics)
    # ------------------------------------------------------------------
    leg_items = [
        (BLUE, "Learned component"),
        (GREEN, "Geometric FilaSeg stage"),
        (ORANGE, "Grayscale-image input"),
        (DARK, "Data artifact"),
    ]
    ly = 0.58
    lx_positions = [0.55, 3.55, 6.85, 9.85]
    for (c, label), lx in zip(leg_items, lx_positions):
        ax.add_patch(FancyBboxPatch((lx, ly - 0.09), 0.34, 0.18,
                                      boxstyle="round,pad=0.01,rounding_size=0.03",
                                      linewidth=1.4, edgecolor=c, facecolor="white", zorder=4))
        ax.text(lx + 0.46, ly, label, ha="left", va="center", fontsize=7.2, color=DARK, zorder=4)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fig(fig, "fig_filaseg_scope", bbox_inches="tight")
    print("saved fig_filaseg_scope")


if __name__ == "__main__":
    main()
