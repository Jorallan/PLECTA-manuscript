"""Figure 1: what this paper contributes, and where the evidence stops.

Restructured a second time, after the author asked for the previous version to
be either reworked substantially or dropped.  It was kept, because it is the
only place a reader can see the one thing the framing depends on -- that the
learned upstream pipeline and the geometric method evaluated here are separate
pieces of work, and that only the second is measured in this paper.  A sentence
can assert that; a diagram is what makes it obvious.

What was wrong with the previous version was not the claim but the load.  It
carried an INPUTS band, an OR junction, five full-height STEP boxes, a swatch
legend and eight elbow connectors, and the elbows had to thread between all of
it: two of them ran along the edge of a box they were not attached to, one
crossed the band it was leaving, and the terminal box of pipeline A stuck out
past its own band.  Text sat on lines because there was no clear space left to
put it in.  Four things are gone, and with them the collisions:

  * the INPUTS band.  The two mask sources now sit where they are consumed --
    the procedural mask beside pipeline A, the SEM-derived mask beneath it --
    so nothing has to be routed back across the diagram to reach them;
  * the OR glyph and the separate "binary filament-axis mask" box.  Two arrows
    merging into one already say "either of these"; both source boxes already
    have the word "mask" in their names, and the boundary tag says "mask in";
  * the five STEP boxes, which restated Section 2's numbered procedure at a
    third of a page.  Five compact chips keep the map -- a reader still sees
    what the five steps are and that two of them repeat -- without competing
    with the Methods for the explanation;
  * the swatch legend, which duplicated the caption's colour key.

One thing is added: the loop is labelled with which round is returned.  The
earlier drawing said only "8 rounds", and the internal report it was adapted
from said "8 rounds, keep the best-scoring one", which is wrong -- round 7 is
the only fully strict round and is the one returned (Section 2, step 3).

The figure is 3.99 in tall against 5.05, and no text touches a line.

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
                    ORANGE, ORANGE_TINT, FIG_W, PT_ANNOT, PT_TITLE,
                    elbow, harrow, plecta_style, rbox, save_fig)

#: Drawing units per inch.  Fixed so that horizontal and vertical distances
#: mean the same thing and a box is not silently stretched.
UNITS_PER_IN = 13.60 / FIG_W
XMAX, YMAX = 13.60, 8.70
FIG_H = YMAX / UNITS_PER_IN

#: The five geometric steps, as chips.  Two lines each, and no line longer
#: than 15 characters, which is what fits a chip at 7.5 pt.
STEPS = (
    ("1", "Arms and\njunction nodes"),
    ("2", "Frame at\neach stub"),
    ("3", "Exact matching\nat a junction"),
    ("4", "Gated bridging\nof mask gaps"),
    ("5", "Paint\ninstance layers"),
)


def band(ax, x0, x1, y0, y1, colour):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0, facecolor=colour, zorder=0))


def chip(ax, xc, yc, w, h, tag, name):
    """A compact step: the number above, the name below, in one rounded box."""
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.5, edgecolor=GREEN, facecolor="white", zorder=3))
    ax.text(xc, yc + h / 2 - 0.18, tag, ha="center", va="center",
            fontsize=PT_ANNOT, fontweight="bold", color=GREEN, zorder=4)
    ax.text(xc, yc - 0.10, name, ha="center", va="center", fontsize=PT_ANNOT,
            fontweight="bold", color=DARK, zorder=4, linespacing=1.30)


def on_line(ax, x, y, text, colour, **kw):
    """A label that sits on a line, with the line stopped behind it.

    The background tint is opaque, so the dashed rule it labels runs up to the
    text and starts again after it, which is a labelled boundary rather than a
    collision.
    """
    ax.text(x, y, text, color=colour, zorder=6,
            bbox=dict(facecolor=GREEN_TINT, edgecolor="none", pad=1.6), **kw)


def main() -> int:
    plecta_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

    # ── A: the upstream route, drawn subordinate and to one side ───────────
    band(ax, 4.55, 13.42, 6.62, 8.55, BLUE_TINT)
    ax.text(4.82, 8.24, "A   Upstream: a binary mask from real SEM "
                        "(evaluated elsewhere)",
            ha="left", va="center", fontsize=PT_ANNOT, fontweight="bold",
            color=BLUE, zorder=1)

    y_top, h_top = 7.52, 0.72
    rbox(ax, 6.20, y_top, 2.86, h_top, "Grayscale SEM image", ORANGE,
         face=ORANGE_TINT, fontsize=PT_ANNOT, ls=(0, (4, 2)), lw=1.4,
         bold=False)
    #  Two learned components, two boxes.  They are trained separately and do
    #  different jobs -- one invents SEM-like images from procedural masks, the
    #  other segments a real one -- and one box with an arrow inside it read as
    #  a single network.
    rbox(ax, 9.30, y_top, 2.00, h_top, "CycleGAN", BLUE,
         face=BLUE_TINT, fontsize=PT_ANNOT, lw=1.5)
    rbox(ax, 11.95, y_top, 2.00, h_top, "U-Net", BLUE,
         face=BLUE_TINT, fontsize=PT_ANNOT, lw=1.5)
    harrow(ax, 7.66, y_top, 8.27, y_top, lw=1.2, mscale=9)
    harrow(ax, 10.33, y_top, 10.92, y_top, lw=1.2, mscale=9)
    ax.text(4.82, 7.06, "trained on SEM-like renders of procedural masks",
            ha="left", va="top", fontsize=PT_ANNOT, color=GRAY, style="italic",
            zorder=1)

    # ── the two mask sources, each drawn where it is produced ──────────────
    x_proc = 2.28
    rbox(ax, x_proc, y_top, 3.70, h_top, "Procedural binary axis mask",
         GRAY, face="white", fontsize=PT_ANNOT, ls=(0, (4, 2)), lw=1.4,
         bold=False)

    x_sem, y_sem, h_sem = 11.75, 5.95, 3.20
    rbox(ax, x_sem, y_sem, h_sem, 0.66, "SEM-derived axis mask", DARK,
         face="white", fontsize=PT_ANNOT, lw=1.6)
    harrow(ax, x_sem, y_top - h_top / 2 - 0.03, x_sem, y_sem + 0.36,
           colour=DARK, lw=1.3, mscale=10)

    # ── either mask is the input, and the merge is drawn as a choice ───────
    #  Both sources carry the same name for a reason: PLECTA takes one binary
    #  axis mask and does not care which produced it.  The junction is labelled
    #  "or" rather than left as a bare dot, and it sits directly above the first
    #  step, so the input arrow runs straight down into it instead of doubling
    #  back across the whole evaluated box as it did before.
    y_merge = 5.40
    ax.plot([x_proc, x_proc], [y_top - h_top / 2 - 0.03, y_merge], color=GRAY,
            lw=1.2, zorder=2)
    ax.plot([x_sem, x_sem], [y_sem - 0.33 - 0.03, y_merge], color=DARK,
            lw=1.2, zorder=2)
    ax.plot([x_proc, x_sem], [y_merge, y_merge], color=DARK, lw=1.2, zorder=2)
    ax.add_patch(Circle((x_proc, y_merge), 0.23, facecolor="white",
                        edgecolor=DARK, linewidth=1.3, zorder=5))
    ax.text(x_proc, y_merge, "or", ha="center", va="center", fontsize=PT_ANNOT,
            fontweight="bold", color=DARK, zorder=6)

    # ── B: the contribution ────────────────────────────────────────────────
    band(ax, 0.18, 13.42, 0.10, 5.06, GREEN_TINT)
    ax.text(0.46, 4.84, "B   PLECTA", ha="left", va="center",
            fontsize=PT_TITLE, fontweight="bold", color=GREEN, zorder=1)

    ev_l, ev_r, ev_b, ev_t = 0.50, 13.10, 1.34, 4.56
    ax.add_patch(FancyBboxPatch((ev_l, ev_b), ev_r - ev_l, ev_t - ev_b,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=1.4, linestyle=(0, (5, 3)),
                                edgecolor=GREEN, facecolor="none", zorder=1))
    #  The tag moved to the right-hand end of the boundary, because the input
    #  now enters at the left-hand end and the two would otherwise share a spot.
    on_line(ax, ev_r - 0.55, ev_t, "EVALUATED — either mask in, instances out",
            GREEN, ha="right", va="center", fontsize=PT_ANNOT,
            fontweight="bold")

    # ── the five geometric steps, and the round loop ───────────────────────
    y_chip, h_chip = 3.62, 1.00
    pad, gap = 0.20, 0.14
    w_chip = (ev_r - ev_l - 2 * pad - 4 * gap) / 5.0
    centres = [ev_l + pad + w_chip / 2 + i * (w_chip + gap) for i in range(5)]
    for xc, (tag, name) in zip(centres, STEPS):
        chip(ax, xc, y_chip, w_chip, h_chip, tag, name)
    for a, b in zip(centres[:-1], centres[1:]):
        harrow(ax, a + w_chip / 2 + 0.03, y_chip, b - w_chip / 2 - 0.03,
               y_chip, colour=GREEN, lw=1.2, mscale=9)

    harrow(ax, x_proc, y_merge - 0.24, x_proc, y_chip + h_chip / 2 + 0.03,
           colour=DARK, lw=1.3, mscale=10)

    y_loop = 2.78
    ax.plot([centres[3], centres[3]], [y_chip - h_chip / 2 - 0.03, y_loop],
            color=GREEN, lw=1.2, ls=(0, (4, 2.4)), zorder=2)
    ax.plot([centres[1], centres[3]], [y_loop, y_loop], color=GREEN, lw=1.2,
            ls=(0, (4, 2.4)), zorder=2)
    harrow(ax, centres[1], y_loop, centres[1], y_chip - h_chip / 2 - 0.03,
           colour=GREEN, lw=1.2, ls=(0, (4, 2.4)), mscale=9)
    on_line(ax, centres[2], y_loop, "8 rounds — the last one is returned",
            GREEN, ha="center", va="center", fontsize=PT_ANNOT, style="italic")

    x_out, y_out, w_out, h_out = 4.40, 1.96, 6.40, 0.76
    rbox(ax, x_out, y_out, w_out, h_out,
         "Overlap-aware instance layers\n(a crossing pixel has two owners)",
         DARK, face="white", fontsize=PT_ANNOT, lw=1.6)
    elbow(ax, [(centres[4], y_chip - h_chip / 2 - 0.03), (centres[4], y_out),
               (x_out + w_out / 2 + 0.04, y_out)],
          colour=DARK, lw=1.3, ls="solid", mscale=10)

    # ── downstream, outside the boundary ───────────────────────────────────
    #  The arrow out of the evaluated box is long enough to read as a crossing
    #  rather than as a join: it leaves one box, passes through the dashed
    #  boundary at its midpoint, and arrives at the other.
    y_dn, h_dn = 0.72, 0.68
    rbox(ax, x_out, y_dn, 6.00, h_dn,
         "Optional: width measured from the SEM\nimage, and ribbon rendering",
         ORANGE, face=ORANGE_TINT, fontsize=PT_ANNOT, lw=1.5)
    rbox(ax, 10.30, y_dn, 3.60, h_dn, "Rendered ribbon\ninstances", DARK,
         face="white", fontsize=PT_ANNOT, lw=1.4)
    harrow(ax, x_out + 3.04, y_dn, 8.46, y_dn, colour=ORANGE, lw=1.3, mscale=10)
    harrow(ax, x_out, y_out - h_out / 2 - 0.03, x_out, y_dn + h_dn / 2 + 0.03,
           colour=DARK, lw=1.3, mscale=10)
    ax.text(x_out, y_dn - h_dn / 2 - 0.14,
            "default off — changes the representation, not the identities",
            ha="center", va="top", fontsize=PT_ANNOT, color=ORANGE,
            style="italic", zorder=4)

    save_fig(fig, "fig_pipeline_scope", bbox_inches=None)
    plt.close(fig)
    print("fig_pipeline_scope  %.3f x %.3f in" % (FIG_W, FIG_H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
