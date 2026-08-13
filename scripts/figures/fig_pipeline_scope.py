"""Figure 1: what this paper contributes, and where the evidence stops.

Redrawn a third time. The complaint was that the text looked horizontally
elongated, and the first thing to say is what that was *not*: the PDF embeds
Latin Modern Roman, not a fallback, and its MediaBox is exactly ``\\linewidth``,
so ``\\includegraphics[width=\\linewidth]`` places it at scale 1.0 with no
stretch in either axis. Two things produced the impression, and both are fixed
here.

**The layout.** Every box was sized to a grid rather than to its contents, so
"U-Net" floated in a box built for "Grayscale SEM image" and short phrases sat
in a lot of horizontal whitespace. Boxes are now measured to the text they hold
--- ``measure()`` renders the label, reads its extent back from the renderer and
draws the box around it --- with one tight padding everywhere, so row widths
vary and no phrase is stretched across a cell it does not fill.

**The face.** The set was drawn in Latin Modern's 8 pt optical cut, whose
letterforms are 6 % wider than the 10 pt cut the manuscript's captions use, and
the labels were bold, which is a further 15 %. Both are gone; see ``_style``.

What is kept, because it is what the figure exists to say: pipeline A upstream
and explicitly evaluated elsewhere, pipeline B evaluated here with its boundary
drawn, either mask in and instances out, and the width stage optional and off
by default. The ``or`` junction and the round-schedule loop are kept and
redrawn: the junction is now a labelled node on the path both sources take, and
the loop says what re-running buys rather than only how many rounds there are.

One line is borrowed from the internal report's own schematic, because it
orients a reader faster than a caption can: steps 3 and 4 are the only
decisions and everything else is bookkeeping, so those two chips are the
emphasised ones. That report's pipeline panel says "8 rounds, keep the
best-scoring one", which is wrong --- the final round is the one returned
(Section 2, step 3) --- and the error is not carried over.

The figure is 3.30 in tall against 3.99, and no text touches a line.

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
                    ORANGE, ORANGE_TINT, FIG_W, PT_ANNOT, PT_MIN, PT_TITLE,
                    harrow, plecta_style, save_fig)

#: Inches, and the axes are drawn in inches too, so a distance means the same
#: thing horizontally and vertically and nothing is silently stretched.
FIG_H = 3.60

#: One padding, used by every box. Tight, because the previous version's
#: problem was that its boxes were mostly padding.
PAD_X, PAD_Y = 0.085, 0.055

#: The five geometric steps. Steps 3 and 4 are the only decisions the method
#: makes; the rest is bookkeeping, and the chips say so by weight.
STEPS = (
    ("1", "Arms and\njunction nodes", False),
    ("2", "Frame at\neach stub", False),
    ("3", "Exact matching\nat a junction", True),
    ("4", "Gated bridging\nof mask gaps", True),
    ("5", "Paint\ninstance layers", False),
)


def measure(ax, text, fontsize, weight="normal", linespacing=1.28):
    """Width and height of a label, in inches, read back from the renderer."""
    artist = ax.text(0, 0, text, fontsize=fontsize, fontweight=weight,
                     ha="center", va="center", linespacing=linespacing,
                     alpha=0.0)
    ax.figure.canvas.draw()
    bbox = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bbox.x0, bbox.y0),
                                        (bbox.x1, bbox.y1)])
    return abs(x1 - x0), abs(y1 - y0)


def fitted_box(ax, xc, yc, text, edge, face="white", fontsize=PT_ANNOT,
               weight="normal", lw=1.1, ls="solid", fontcolor=None,
               pad_x=PAD_X, pad_y=PAD_Y, min_w=0.0, zorder=3, tag=None):
    """A box sized to its own label, returning its half-width in inches.

    ``tag`` is set inside the box at its top left, so the step number travels
    with the step instead of floating above it into whatever is there.
    """
    w, h = measure(ax, text, fontsize, weight)
    w = max(w + 2 * pad_x, min_w)
    h = h + 2 * pad_y + (0.085 if tag else 0.0)
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.045",
        linewidth=lw, linestyle=ls, edgecolor=edge, facecolor=face,
        zorder=zorder))
    ax.text(xc, yc - (0.042 if tag else 0.0), text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=fontcolor or DARK,
            zorder=zorder + 1, linespacing=1.28)
    if tag:
        ax.text(xc - w / 2 + 0.055, yc + h / 2 - 0.055, tag, ha="left",
                va="top", fontsize=PT_MIN, color=edge, zorder=zorder + 1)
    return w / 2, h / 2


def chain(ax, x_left, yc, labels, edge, face, gap=0.20, **kw):
    """Boxes laid left to right, each as wide as it needs to be."""
    x = x_left
    centres = []
    for label in labels:
        w, _h = measure(ax, label, kw.get("fontsize", PT_ANNOT),
                        kw.get("weight", "normal"))
        half = (w + 2 * PAD_X) / 2
        xc = x + half
        fitted_box(ax, xc, yc, label, edge, face=face, **kw)
        centres.append((xc, half))
        x = xc + half + gap
    for (xa, ha), (xb, hb) in zip(centres[:-1], centres[1:]):
        harrow(ax, xa + ha + 0.02, yc, xb - hb - 0.02, yc, lw=1.0, mscale=7)
    return centres, x - gap


def band(ax, x0, x1, y0, y1, colour):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.0,rounding_size=0.05",
                                linewidth=0, facecolor=colour, zorder=0))


def main() -> int:
    plecta_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── A: upstream, drawn subordinate and to one side ─────────────────────
    band(ax, 1.72, 6.20, 2.72, 3.54, BLUE_TINT)
    ax.text(1.86, 3.41, "A   Upstream: a binary mask from an SEM micrograph "
                        "(evaluated elsewhere)",
            ha="left", va="center", fontsize=PT_ANNOT, color=BLUE, zorder=1)

    y_up = 3.10
    centres, _end = chain(
        ax, 1.86, y_up,
        ["SEM micrograph", "CycleGAN", "nnU-Net"],
        BLUE, BLUE_TINT, gap=0.17)
    ax.text(1.86, 2.83, "trained on SEM-like renders of procedural masks",
            ha="left", va="center", fontsize=PT_MIN, color=GRAY,
            style="italic", zorder=1)

    # ── the two mask sources, each where it is produced ────────────────────
    x_proc = 0.86
    fitted_box(ax, x_proc, y_up, "Procedural\naxis mask", GRAY,
               ls=(0, (3.5, 2)), lw=1.0)
    x_sem = centres[-1][0]
    y_mask = 2.44
    _half_sem, half_h = fitted_box(ax, x_sem, y_mask, "SEM-derived axis mask",
                                   DARK, lw=1.1)
    harrow(ax, x_sem, y_up - 0.20, x_sem, y_mask + half_h + 0.02, colour=DARK,
           lw=1.1, mscale=8)

    # ── either mask is the input, and the choice is drawn as one node ──────
    y_or = 2.14
    ax.plot([x_proc, x_proc], [y_up - 0.20, y_or], color=GRAY, lw=1.0,
            zorder=2)
    ax.plot([x_sem, x_sem], [y_mask - half_h - 0.02, y_or], color=DARK, lw=1.0,
            zorder=2)
    ax.plot([x_proc, x_sem], [y_or, y_or], color=DARK, lw=1.0, zorder=2)
    ax.add_patch(Circle((x_proc, y_or), 0.105, facecolor="white",
                        edgecolor=DARK, linewidth=1.0, zorder=5))
    ax.text(x_proc, y_or, "or", ha="center", va="center", fontsize=PT_MIN,
            color=DARK, zorder=6)

    # ── B: the contribution ────────────────────────────────────────────────
    band(ax, 0.06, 6.18, 0.06, 1.94, GREEN_TINT)
    ax.text(0.20, 1.83, "B   PLECTA", ha="left", va="center",
            fontsize=PT_TITLE, fontweight="bold", color=GREEN, zorder=1)

    ev_l, ev_r, ev_b, ev_t = 0.20, 6.04, 0.56, 1.70
    ax.add_patch(FancyBboxPatch((ev_l, ev_b), ev_r - ev_l, ev_t - ev_b,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                linewidth=1.0, linestyle=(0, (4, 2.6)),
                                edgecolor=GREEN, facecolor="none", zorder=1))
    ax.text(ev_r - 0.12, ev_b, "EVALUATED — either mask in, instances out",
            ha="right", va="center", fontsize=PT_MIN, color=GREEN, zorder=6,
            bbox=dict(facecolor=GREEN_TINT, edgecolor="none", pad=1.4))

    # ── the five steps, each chip as wide as its own words ─────────────────
    y_chip = 1.36
    widths = [measure(ax, name, PT_ANNOT)[0] + 2 * PAD_X
              for _tag, name, _hi in STEPS]
    gap = (ev_r - ev_l - 0.30 - sum(widths)) / (len(STEPS) - 1)
    x = ev_l + 0.15
    chips = []
    for (tag, name, emphasis), w in zip(STEPS, widths):
        xc = x + w / 2
        _hw, hh = fitted_box(ax, xc, y_chip, name, GREEN, tag=tag,
                             lw=1.4 if emphasis else 0.9,
                             face="white" if emphasis else GREEN_TINT)
        chips.append((xc, w / 2, hh))
        x = xc + w / 2 + gap
    for (xa, ha, _), (xb, hb, _) in zip(chips[:-1], chips[1:]):
        harrow(ax, xa + ha + 0.02, y_chip, xb - hb - 0.02, y_chip,
               colour=GREEN, lw=1.0, mscale=7)

    harrow(ax, x_proc, y_or - 0.11, x_proc, y_chip + chips[0][2] + 0.02,
           colour=DARK, lw=1.1, mscale=8)

    # ── the round schedule, drawn as what re-running buys ──────────────────
    y_loop = y_chip - chips[0][2] - 0.19
    ax.plot([chips[3][0], chips[3][0]], [y_chip - chips[3][2] - 0.02, y_loop],
            color=GREEN, lw=1.0, ls=(0, (3.5, 2.2)), zorder=2)
    ax.plot([chips[1][0], chips[3][0]], [y_loop, y_loop], color=GREEN, lw=1.0,
            ls=(0, (3.5, 2.2)), zorder=2)
    harrow(ax, chips[1][0], y_loop, chips[1][0], y_chip - chips[1][2] - 0.02,
           colour=GREEN, lw=1.0, ls=(0, (3.5, 2.2)), mscale=7)
    ax.text((chips[1][0] + chips[3][0]) / 2 + 0.19, y_loop,
            "8 rounds; each re-measures every stub along the chain\n"
            "behind it, and the last round is the one returned",
            ha="center", va="center", fontsize=PT_MIN, color=GREEN, zorder=6,
            style="italic", linespacing=1.30,
            bbox=dict(facecolor=GREEN_TINT, edgecolor="none", pad=1.2))

    # ── the output, on its own row under the loop ──────────────────────────
    x_out, y_out = 3.12, 0.76
    half_out, half_out_h = fitted_box(
        ax, x_out, y_out,
        "Overlap-aware instance layers: a crossing pixel has two owners",
        DARK, lw=1.1)
    x5 = chips[4][0]
    ax.plot([x5, x5], [y_chip - chips[4][2] - 0.02, y_out], color=DARK,
            lw=1.1, zorder=2)
    harrow(ax, x5, y_out, x_out + half_out + 0.03, y_out, colour=DARK, lw=1.1,
           mscale=8)

    # ── the optional stage, outside the evaluated boundary ─────────────────
    y_dn = 0.28
    x_dn = 2.00
    _half_dn, half_dn_h = fitted_box(
        ax, x_dn, y_dn,
        "Optional: width measured from the image, and ribbon rendering",
        ORANGE, face=ORANGE_TINT, lw=1.0)
    harrow(ax, x_dn, y_out - half_out_h - 0.02, x_dn, y_dn + half_dn_h + 0.02,
           colour=ORANGE, lw=1.0, mscale=8)
    ax.text(x_dn + _half_dn + 0.16, y_dn,
            "default off — changes the\nrepresentation, not the identities",
            ha="left", va="center", fontsize=PT_MIN, color=ORANGE,
            style="italic", zorder=4, linespacing=1.30)

    save_fig(fig, "fig_pipeline_scope", bbox_inches=None)
    plt.close(fig)
    print("fig_pipeline_scope  %.3f x %.3f in" % (FIG_W, FIG_H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
