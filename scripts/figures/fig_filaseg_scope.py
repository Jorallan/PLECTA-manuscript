"""FIG: FilaSeg scope schematic.

Vertical flow diagram showing exactly what FilaSeg covers: a binary
centreline mask plus paired SEM image go in, four processing stages run
(vectorization, fragment cleaning, reconnection, width estimation +
rendering), and an instance-labelled mask with per-instance geometry comes
out. A light enclosing frame marks the boundary of FilaSeg's scope around
the four numbered stages; a dashed side arrow shows the SEM image also
feeding stage 4 directly. Pure vector schematic (no external raster),
theme-neutral, publication style, consistent with the other paper figures.
"""
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, save_fig, BLUE, ORANGE, GREEN, GRAY, LIGHT_GRAY, DARK

# (kind, number, title, subtitle, right-hand annotation, color)
STAGES = [
    ("io", None, "INPUT", "Binary centreline mask\n+ paired SEM image", None, GRAY),
    ("stage", "1", "Hough-based\nvectorization", None, "oriented line primitives", BLUE),
    ("stage", "2", "Orientation-aware\nfragment cleaning", None, "smoothed two-tip fragments", ORANGE),
    ("stage", "3", "Geometry-based\ninstance reconnection", None, "grouped instances", GREEN),
    ("stage", "4", "SEM-gradient-based\nwidth estimation\nand rendering", None, "measured, rendered\ninstances", DARK),
    ("io", None, "OUTPUT", "Instance-labelled mask +\nper-instance geometry", None, GRAY),
]


def main():
    apply_style()

    box_w = 3.5
    io_h = 1.6
    x_left = 1.5
    x_right = x_left + box_w
    x_center = x_left + box_w / 2
    gap = 0.42  # edge-to-edge vertical gap between boxes (for arrows)

    # Per-box height: IO boxes get a fixed height; stage boxes are sized to
    # their (centred) text so there is no leftover empty band.
    heights = []
    for kind, num, title, sub, annot, color in STAGES:
        if kind == "io":
            heights.append(io_h)
        else:
            n_lines = title.count("\n") + 1
            heights.append(0.62 + 0.34 * n_lines)

    y_top = 11.4
    centers = [y_top - heights[0] / 2]
    for i in range(1, len(STAGES)):
        centers.append(centers[-1] - heights[i - 1] / 2 - gap - heights[i] / 2)

    fig_h_top = centers[0] + heights[0] / 2 + 0.6
    fig_h_bot = centers[-1] - heights[-1] / 2 - 0.55
    fig, ax = plt.subplots(figsize=(5.7, 6.7))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(fig_h_bot, fig_h_top)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Enclosing "FilaSeg scope" frame around the four numbered stages
    # ------------------------------------------------------------------
    stage_idx = [i for i, s in enumerate(STAGES) if s[0] == "stage"]
    frame_top = centers[stage_idx[0]] + heights[stage_idx[0]] / 2 + 0.34
    frame_bot = centers[stage_idx[-1]] - heights[stage_idx[-1]] / 2 - 0.34
    frame_pad = 0.32
    frame = FancyBboxPatch(
        (x_left - frame_pad, frame_bot),
        box_w + 2 * frame_pad,
        frame_top - frame_bot,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.3,
        linestyle=(0, (5, 3)),
        edgecolor=LIGHT_GRAY,
        facecolor="none",
        zorder=1,
    )
    ax.add_patch(frame)
    ax.text(
        x_left - frame_pad + 0.16,
        frame_top - 0.05,
        "FilaSeg",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=GRAY,
        style="italic",
        zorder=1,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
    )

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------
    for i, (kind, num, title, sub, annot, color) in enumerate(STAGES):
        yc = centers[i]
        box_h = heights[i]
        y0 = yc - box_h / 2
        is_io = kind == "io"
        lw = 1.3 if is_io else 1.9
        face = "#f3f4f6" if is_io else "white"
        ls = (0, (4, 2)) if is_io else "solid"
        box = FancyBboxPatch(
            (x_left, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=lw,
            linestyle=ls,
            edgecolor=color if not is_io else DARK,
            facecolor=face,
            zorder=3,
        )
        ax.add_patch(box)

        if is_io:
            ax.text(
                x_center, yc + 0.38, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=DARK, zorder=4,
            )
            ax.text(
                x_center, yc - 0.28, sub, ha="center", va="center",
                fontsize=8.6, color="#333333", zorder=4, linespacing=1.5,
            )
        else:
            # number badge, top-left corner of the box
            badge_x = x_left + 0.30
            badge_y = yc + box_h / 2 - 0.24
            ax.add_patch(plt.Circle((badge_x, badge_y), 0.165,
                                     color=color, zorder=4))
            ax.text(badge_x, badge_y, num, color="white",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    zorder=5)
            # title is vertically centred in the box (equal padding above
            # and below), independent of the badge, which sits in its own
            # corner and never overlaps the centred text block
            ax.text(
                x_center, yc, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=DARK, zorder=4,
                linespacing=1.45,
            )

        # main downward flow arrow to next box
        if i < len(STAGES) - 1:
            y_next_top = centers[i + 1] + heights[i + 1] / 2
            arr = FancyArrowPatch(
                (x_center, y0 - 0.03),
                (x_center, y_next_top + 0.03),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color=GRAY,
                zorder=2,
            )
            ax.add_patch(arr)

        # right-hand "produces" annotation
        if annot is not None:
            ax.text(
                x_right + 0.28,
                yc,
                annot,
                ha="left",
                va="center",
                fontsize=8.6,
                color="#333333",
                style="italic",
                zorder=4,
                linespacing=1.45,
            )

    # ------------------------------------------------------------------
    # Dashed side arrow: SEM image (INPUT) feeds stage 4 directly.
    # Routed as a rectilinear elbow entirely within the left gutter
    # (canvas edge <-> FilaSeg frame) so it never leaves the axes and
    # never crosses the frame or its label.
    # ------------------------------------------------------------------
    io_input_idx = 0
    stage4_idx = stage_idx[-1]
    y_input = centers[io_input_idx]
    y_stage4 = centers[stage4_idx]
    y_leave = y_input - heights[io_input_idx] / 2 + 0.16  # exit low on INPUT's left edge

    frame_left = x_left - frame_pad
    gutter_x = 0.62  # vertical run of the elbow, well left of the frame
    assert gutter_x < frame_left - 0.35, "gutter arrow would touch the FilaSeg frame"

    elbow_color = GRAY
    elbow_kwargs = dict(linewidth=1.3, linestyle=(0, (4, 2.5)), color=elbow_color,
                         zorder=2, solid_capstyle="butt")
    # leg 1: leave INPUT box leftward into the gutter
    ax.plot([x_left, gutter_x], [y_leave, y_leave], **elbow_kwargs)
    # leg 2: drop straight down through the gutter to stage 4's level
    ax.plot([gutter_x, gutter_x], [y_leave, y_stage4], **elbow_kwargs)
    # leg 3: turn right into stage 4's left edge (arrowhead)
    side_arrow = FancyArrowPatch(
        (gutter_x, y_stage4),
        (x_left - 0.02, y_stage4),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        linestyle=(0, (4, 2.5)),
        color=elbow_color,
        zorder=2,
    )
    ax.add_patch(side_arrow)

    # label beside the vertical run, rotated so it needs little horizontal
    # room and cannot collide with the arrow or the FilaSeg frame/label
    label_y = (y_leave + y_stage4) / 2
    ax.text(
        gutter_x - 0.22,
        label_y,
        "SEM intensity evidence",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.4,
        color=GRAY,
        style="italic",
        zorder=4,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fig(fig, "fig_filaseg_scope", bbox_inches="tight")
    print("saved fig_filaseg_scope")


if __name__ == "__main__":
    main()
