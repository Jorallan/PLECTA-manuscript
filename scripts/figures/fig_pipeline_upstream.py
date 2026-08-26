"""Figure: where the real-SEM axis mask comes from, and why it is not ours.

Section 3.5 puts PLECTA on a real micrograph, and to do that it needs a binary
strand-axis mask.  One is produced by Pipeline A --- procedural mask ->
CycleGAN -> paired synthetic corpus -> nnU-Net -> binary strand-axis mask ---
and the paragraph that names those five stages is easy to read as a claim that
they belong to this paper.  They do not.  Pipeline A is upstream of everything
measured here, it is not a contribution of this manuscript, and it carries its
own separate evaluation; the caption says so and the drawing has to say it
too.

So the figure is built the way Figure 1 is built, and for the same reason: a
dashed boundary is drawn around the one stage this paper actually measures,
and everything else sits visibly outside it, in its own tinted band with its
own heading.

  TRAINING (top row) a generator draws a binary mask; a CycleGAN translates
      that mask into an SEM-like micrograph.  Both come together into a paired
      corpus, and the point of drawing the join rather than a single arrow is
      that the pairing costs nothing: the mask that produced the image is
      already the image's label, so no micrograph is ever hand-annotated.
      That corpus trains the nnU-Net.

  INFERENCE (bottom row) a real SEM micrograph enters, the trained nnU-Net
      segments it, and the output is a binary strand-axis mask.

  EVALUATED (green band) that mask is PLECTA's input, and PLECTA --- Pipeline
      B --- is the only stage on the page this paper scores.  The single arrow
      that crosses the dashed boundary is the whole interface between the two
      pipelines: one binary mask, nothing else.

Learned components are blue, following the palette note in ``_style``; the
data that passes between them is drawn in the ink colour, and only PLECTA is
green.

Fully self-contained; no data file is read.

    python scripts/figures/fig_pipeline_upstream.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, BLUE_TINT, DARK, GRAY, GREEN, GREEN_TINT, FIG_W,
                    PT_ANNOT, PT_MIN, PT_TITLE, harrow, plecta_style, save_fig)

#: Inches.  The axes are drawn in inches too, so a distance means the same
#: thing horizontally and vertically and nothing is silently stretched.
FIG_H = 3.02

#: One padding, used by every box, as in fig_pipeline_scope.
PAD_X, PAD_Y = 0.085, 0.055


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
               zorder=3):
    """A box sized to its own label; returns its half-width and half-height."""
    w, h = measure(ax, text, fontsize, weight)
    w, h = w + 2 * PAD_X, h + 2 * PAD_Y
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.045",
        linewidth=lw, linestyle=ls, edgecolor=edge, facecolor=face,
        zorder=zorder))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=fontcolor or DARK, zorder=zorder + 1,
            linespacing=1.28)
    return w / 2, h / 2


def chain(ax, x_left, yc, items, gap=0.30, fontsize=PT_ANNOT):
    """Boxes laid left to right, each as wide as it needs to be.

    ``items`` is a sequence of ``(label, edge_colour)``, so a learned stage can
    take the learned-component hue while the data around it stays ink.
    """
    x = x_left
    out = []
    for label, edge in items:
        w, _h = measure(ax, label, fontsize)
        half = (w + 2 * PAD_X) / 2
        xc = x + half
        _hw, hh = fitted_box(ax, xc, yc, label, edge, fontsize=fontsize)
        out.append((xc, half, hh))
        x = xc + half + gap
    for (xa, ha, _), (xb, hb, _) in zip(out[:-1], out[1:]):
        harrow(ax, xa + ha + 0.02, yc, xb - hb - 0.02, yc, colour=DARK,
               lw=1.0, mscale=7)
    return out, x - gap


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

    x_l, x_r = 0.06, FIG_W - 0.06

    # ── A: upstream, and explicitly not this paper's ──────────────────────
    band(ax, x_l, x_r, 1.14, 2.96, BLUE_TINT)
    head = "Pipeline A"
    ax.text(0.20, 2.85, head, ha="left", va="center",
            fontsize=PT_TITLE, fontweight="bold", color=BLUE, zorder=1)
    ax.text(0.20 + measure(ax, head, PT_TITLE, "bold")[0] + 0.10, 2.85,
            "makes the mask. Upstream of this paper, not a contribution of "
            "it, and evaluated separately.",
            ha="left", va="center", fontsize=PT_MIN, color=GRAY, zorder=1)

    # ── training: the corpus is manufactured, so nothing is annotated ─────
    y_tr = 2.56
    train, _end_tr = chain(ax, 0.34, y_tr, [
        ("Procedural\nbinary mask", DARK),
        ("CycleGAN", BLUE),
        ("SEM-like\nmicrograph", DARK),
    ], gap=0.26)
    (x_mask, hw_mask, hh_tr), _cyc, (x_sem, hw_sem, _) = train

    # ── inference: a real micrograph in, a binary axis mask out ───────────
    y_inf = 1.44
    infer, _end_inf = chain(ax, 0.34, y_inf, [
        ("Real SEM\nmicrograph", DARK),
        ("Trained\nnnU-Net", BLUE),
        ("Binary strand-\naxis mask", DARK),
    ], gap=0.30)
    (_x_real, _, _), (x_net, _hw_net, hh_net), (x_out, _hw_out, hh_out) = infer

    # ── the pairing, drawn as a join rather than an arrow ──────────────────
    #    Two things arrive at the corpus and neither of them is a person: that
    #    is the whole argument for the CycleGAN, so the mask and the image are
    #    drawn meeting, not one flowing into the other.
    y_join = y_tr - hh_tr - 0.14
    for x in (x_mask, x_sem):
        ax.plot([x, x], [y_tr - hh_tr - 0.02, y_join], color=DARK, lw=1.0,
                zorder=2)
    ax.plot([x_mask, x_sem], [y_join, y_join], color=DARK, lw=1.0, zorder=2)

    y_corp = 2.02
    hw_corp, hh_corp = fitted_box(
        ax, x_net, y_corp,
        "Paired synthetic corpus:\nimage and mask, no annotator",
        DARK, lw=1.1)
    harrow(ax, x_net, y_join, x_net, y_corp + hh_corp + 0.02, colour=DARK,
           lw=1.0, mscale=7)

    harrow(ax, x_net, y_corp - hh_corp - 0.02, x_net, y_inf + hh_net + 0.02,
           colour=BLUE, lw=1.0, mscale=7)
    ax.text(x_net + 0.09, (y_corp - hh_corp + y_inf + hh_net) / 2, "trains",
            ha="left", va="center", fontsize=PT_MIN, color=BLUE, zorder=4)

    x_note = max(x_sem + hw_sem, x_out + 0.62) + 0.34
    ax.text(x_note, y_tr,
            "TRAINING\nthe pair is manufactured: the mask\n"
            "that made the image is its label",
            ha="left", va="center", fontsize=PT_MIN, color=GRAY, zorder=4,
            linespacing=1.34)
    ax.text(x_note, y_inf,
            "INFERENCE\nthe segmenter reads the micrograph,\nnever an annotation",
            ha="left", va="center", fontsize=PT_MIN, color=GRAY, zorder=4,
            linespacing=1.34)

    # ── B: the one stage this paper measures ──────────────────────────────
    band(ax, x_l, x_r, 0.06, 1.02, GREEN_TINT)
    head_b = "Pipeline B"
    ax.text(0.20, 0.91, head_b, ha="left", va="center",
            fontsize=PT_TITLE, fontweight="bold", color=GREEN, zorder=1)
    ax.text(0.20 + measure(ax, head_b, PT_TITLE, "bold")[0] + 0.10, 0.91,
            "reads that mask and nothing else.",
            ha="left", va="center", fontsize=PT_MIN, color=GRAY, zorder=1)

    ev_l, ev_r, ev_b, ev_t = 0.20, x_r - 0.16, 0.20, 0.78
    ax.add_patch(FancyBboxPatch((ev_l, ev_b), ev_r - ev_l, ev_t - ev_b,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                linewidth=1.0, linestyle=(0, (4, 2.6)),
                                edgecolor=GREEN, facecolor="none", zorder=1))
    ax.text(ev_r - 0.34, ev_b, "EVALUATED HERE",
            ha="right", va="center", fontsize=PT_MIN, color=GREEN, zorder=6,
            bbox=dict(facecolor=GREEN_TINT, edgecolor="none", pad=1.4))

    y_pl = 0.49
    hw_pl, hh_pl = fitted_box(ax, x_out, y_pl, "PLECTA", GREEN,
                              face="white", lw=1.4, fontcolor=GREEN)
    harrow(ax, x_out, y_inf - hh_out - 0.02, x_out, y_pl + hh_pl + 0.02,
           colour=DARK, lw=1.1, mscale=8)
    ax.text(x_note, y_pl,
            "the grouping stage: one binary mask in,\n"
            "overlap-aware instance layers out",
            ha="left", va="center", fontsize=PT_MIN, color=GREEN, zorder=6,
            linespacing=1.34)

    save_fig(fig, "fig_pipeline_upstream", bbox_inches=None)
    plt.close(fig)
    print("fig_pipeline_upstream  %.3f x %.3f in" % (FIG_W, FIG_H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
