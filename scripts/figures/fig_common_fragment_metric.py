"""The scorer: how the common-fragment metric is built and what it can see.

Every number in the paper is measured with this metric and until now it was
explained only in prose.  Row 1 is its construction, on real pixels; row 2 is
the consequence that the manuscript states explicitly and that a reader has no
other way to check.

The construction (``eval/core/common_metric.py::common_fragments``) reads the
INPUT MASK and nothing else -- not the reference, not any method's output:

    skeletonise -> prune spurs (<= 3 px) -> mark skeleton pixels with three or
    more 8-neighbours and delete them -> label the remaining 8-connected
    pieces -> discard pieces shorter than 6 px.

Because the fragment set is a function of the input alone, the reference and
every method being compared are cut into the *same* fragments.  Each fragment is
then assigned to one reference instance and one predicted instance by majority
overlap against each instance's full mask (so an overlapping instance keeps its
pixels), and the score is counted over fragment PAIRS: a pair agrees when the
two fragments are together on both sides, or apart on both sides.

Row 2 is the point.  No fragment contains a junction pixel, so the metric never
counts the crossing raster itself.  It scores how a crossing is RESOLVED --
which arm continues into which, visible in whether the correct arms end up
paired -- and not how it is DRAWN, that is, whether the emitted raster marks a
crossing pixel as owned by one filament or by two.  On the displayed crossing,
redrawing changes 0 of the 6 fragment pairs and re-resolving changes 4 of 6.
Over the 50 comparator scenes the same thing is measured rather than argued:
4.8 % to 19.9 % of PLECTA's pixels are claimed by two instances, and flattening
every one of them changes F₁ by 0.000 in all five coverage strata.

Data, all committed: results/plecta_figure_data.json ("scorer": a 60 x 60 crop
of cov20/synth_0001 around a crossing) and results/plecta_graft_comparison.json
("overlap_cost").

    python scripts/figures/fig_common_fragment_metric.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from _style import (CHORD, FIG_W, FIL_A, FIL_B, FONE, GRAY, INK,
                    INSTANCE_CYCLE,
                    JUNCTION, PT_ANNOT, PT_MIN, PT_TITLE, blank_rgb,
                    load_figure_data, paint, plecta_style, save_fig)

REPO = Path(__file__).resolve().parents[2]

AGREE_C = "#2f855a"     # a pair whose verdict is unchanged
CHANGE_C = "#C2185B"    # a pair whose verdict flips

ROW1_IN = 0.92          # side of a construction panel, inches
ROW2_IN = 0.92          # side of a consequence panel, inches
ZOOM = 17               # half-side of the row-2 crop, pixels


def unpack(payload):
    out = np.zeros(payload["h"] * payload["w"], dtype=bool)
    out[payload["idx"]] = True
    return out.reshape(payload["h"], payload["w"])


def unpack_labels(payload):
    out = np.zeros(payload["h"] * payload["w"], dtype=int)
    out[payload["idx"]] = payload["lab"]
    return out.reshape(payload["h"], payload["w"])


def pixel_panel(fig, rect, n):
    ax = fig.add_axes(rect)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(CHORD)
        s.set_linewidth(0.5)
    return ax


def show(ax, layers_colours, shape):
    """Paint boolean layers into one raster; later layers win."""
    img = blank_rgb(shape)
    for layer, colour in layers_colours:
        paint(img, layer, colour)
    ax.imshow(img, interpolation="nearest", zorder=2)
    return ax


def opposite_pairs(frag, junction):
    """Pair each fragment with the one leaving the node in the opposite sense.

    Derived rather than typed in: the direction of a fragment is the bearing of
    its node-most pixel from the node centroid, and the reference pairing is the
    perfect matching that maximises how antipodal the paired bearings are.
    """
    centre = np.array(np.nonzero(junction)).mean(axis=1)
    ids = [k for k in np.unique(frag) if k != 0]
    bearing = {}
    for k in ids:
        pts = np.array(np.nonzero(frag == k)).T.astype(float)
        near = pts[np.argmin(np.linalg.norm(pts - centre, axis=1))]
        v = near - centre
        bearing[k] = np.arctan2(-v[0], v[1])

    def turn(a, b):
        d = abs(bearing[a] - bearing[b]) % (2 * np.pi)
        return abs(np.pi - min(d, 2 * np.pi - d))       # 0 == perfectly opposite

    best, best_cost = None, None
    first = ids[0]
    for partner in ids[1:]:
        rest = [k for k in ids if k not in (first, partner)]
        matching = [(first, partner), tuple(rest)]
        cost = sum(turn(a, b) for a, b in matching)
        if best_cost is None or cost < best_cost:
            best, best_cost = matching, cost
    return best


def pair_verdicts(ids, grouping):
    """{(i, j): True if i and j are in the same group} over all fragment pairs."""
    where = {k: g for g, group in enumerate(grouping) for k in group}
    return {p: where[p[0]] == where[p[1]]
            for p in itertools.combinations(sorted(ids), 2)}


def verdict_strip(fig, x, y_top, reference, other, cell_in=0.105):
    """One cell per fragment pair, filled when that pair's verdict flips.

    ``x`` and ``y_top`` are figure fractions; the cells are drawn square in
    *inches*, which on a non-square canvas means the height fraction is not the
    width fraction -- getting that wrong is what made them the size of panels.
    """
    fig_h = fig.get_figheight()
    cw, ch = cell_in / FIG_W, cell_in / fig_h
    gap = cw * 0.30
    changed = 0
    for i, key in enumerate(sorted(reference)):
        flip = reference[key] != other[key]
        changed += flip
        fig.patches.append(Rectangle(
            (x + i * (cw + gap), y_top - ch), cw, ch,
            transform=fig.transFigure, linewidth=0.8,
            edgecolor=CHANGE_C if flip else AGREE_C,
            facecolor=CHANGE_C if flip else "white", zorder=6))
    n = len(reference)
    fig.text(x + n * (cw + gap) + 0.010, y_top - ch / 2.0,
             "%d of %d pairs change" % (changed, n),
             fontsize=PT_ANNOT, color=CHANGE_C if changed else AGREE_C,
             ha="left", va="center")
    return changed


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="-|>,head_length=0.45,head_width=0.25", mutation_scale=11,
        color=CHORD, lw=1.0, shrinkA=0, shrinkB=0, zorder=10))


def main() -> int:
    plecta_style()
    data = load_figure_data()
    s = data["scorer"]
    overlap = json.loads((REPO / "results" / "plecta_graft_comparison.json")
                         .read_text(encoding="utf-8"))["overlap_cost"]

    n = s["size"]
    mask = unpack(s["mask"])
    skel = unpack(s["skeleton"])
    junc = unpack(s["junction_px"])
    frag = unpack_labels(s["fragments"])
    dropped = unpack(s["dropped"])
    ids = [int(k) for k in np.unique(frag) if k != 0]
    if len(ids) != 4:
        raise SystemExit("this crop no longer has four fragments: %r" % (ids,))

    reference = opposite_pairs(frag, junc)
    swapped = [(reference[0][0], reference[1][0]),
               (reference[0][1], reference[1][1])]
    v_ref = pair_verdicts(ids, reference)
    v_swapped = pair_verdicts(ids, swapped)
    v_redrawn = dict(v_ref)      # redrawing cannot move a fragment: identical

    frag_colour = {k: INSTANCE_CYCLE[i] for i, k in enumerate(ids)}
    inst_colour = {}
    for group, colour in zip(reference, (FIL_A, FIL_B)):
        for k in group:
            inst_colour[k] = colour
    swap_colour = {}
    for group, colour in zip(swapped, (FIL_A, FIL_B)):
        for k in group:
            swap_colour[k] = colour

    # ── page ───────────────────────────────────────────────────────────
    #  Laid out top-down in INCHES and converted once.  The previous attempt
    #  mixed inches and figure fractions and put half-inch cells where 0.1 in
    #  ones were meant.
    BAND = (("top", 0.14), ("tag1", 0.20), ("row1", ROW1_IN), ("name1", 0.34),
            ("gap", 0.20), ("tag2", 0.20), ("sub2", 0.16), ("row2", ROW2_IN),
            ("strip", 0.26), ("prose", 0.20), ("bottom", 0.08))
    fig_h = sum(v for _k, v in BAND)
    at, run = {}, 0.0
    for key, height in BAND:                # distance from the top, inches
        at[key] = run
        run += height

    fig = plt.figure(figsize=(FIG_W, fig_h))
    IN = 1.0 / fig_h                       # inches -> figure fraction, vertical
    w1, h1 = ROW1_IN / FIG_W, ROW1_IN * IN
    w2, h2 = ROW2_IN / FIG_W, ROW2_IN * IN

    row1_y = 1.0 - (at["row1"] + ROW1_IN) * IN
    left, right = 0.030, 0.984
    step = (right - left - w1) / 4.0
    col = [left + i * step for i in range(5)]

    # ── row 1: the construction, from the input mask alone ─────────────
    ax = pixel_panel(fig, [col[0], row1_y, w1, h1], n)
    show(ax, [(mask, INK)], mask.shape)

    ax = pixel_panel(fig, [col[1], row1_y, w1, h1], n)
    show(ax, [(mask, "#aeb9c6"), (skel, INK)], mask.shape)

    ax = pixel_panel(fig, [col[2], row1_y, w1, h1], n)
    show(ax, [(skel & ~junc, INK), (junc, JUNCTION)], mask.shape)

    ax = pixel_panel(fig, [col[3], row1_y, w1, h1], n)
    layers = [(frag == k, frag_colour[k]) for k in ids]
    show(ax, layers + [(dropped, GRAY)], mask.shape)

    ax = pixel_panel(fig, [col[4], row1_y, w1, h1], n)
    show(ax, [(frag == k, inst_colour[k]) for k in ids], mask.shape)

    y_mid = row1_y + h1 / 2.0
    for i in range(4):
        arrow(fig, col[i] + w1 + 0.004, col[i + 1] - 0.004, y_mid)

    # The tag goes above the panel, the step name below it: a two-line name is
    # what collided with the tag when both sat on the same baseline.
    names = ("input mask", "skeletonise,\nprune spurs", "delete pixels with\n" r"$\geq$" " 3 neighbours",
             "label 8-connected\npieces", "assign each to\none instance")
    for x, letter, name in zip(col, "abcde", names):
        fig.text(x, row1_y + h1 + 0.010, "(%s)" % letter, fontsize=PT_TITLE,
                 fontweight="bold", color=INK, ha="left", va="bottom")
        fig.text(x + w1 / 2.0, row1_y - 0.014, name, fontsize=PT_ANNOT,
                 color=INK, ha="center", va="top", linespacing=1.30)

    # ── row 2: what the metric can and cannot see ──────────────────────
    c0, c1 = 30, 30
    sl = (slice(c0 - ZOOM, c0 + ZOOM + 1), slice(c1 - ZOOM, c1 + ZOOM + 1))
    zn = 2 * ZOOM + 1
    z_frag, z_junc = frag[sl], junc[sl]

    row2_y = 1.0 - (at["row2"] + ROW2_IN) * IN
    gap_in = 0.34 / FIG_W
    block = 2 * w2 + gap_in
    bx = [left + 0.004, right - block]

    def crossing(rect, colour_of, paint_junction):
        ax = pixel_panel(fig, rect, zn)
        show(ax, [(z_frag == k, colour_of[k]) for k in np.unique(z_frag)
                  if k != 0], z_frag.shape)
        if paint_junction == "both":
            for r, c in zip(*np.nonzero(z_junc)):
                x0, y0 = c - 0.5, r - 0.5
                ax.add_patch(Polygon([(x0, y0), (x0 + 1, y0), (x0, y0 + 1)],
                                     facecolor=FIL_A, edgecolor="none", zorder=5))
                ax.add_patch(Polygon([(x0 + 1, y0), (x0 + 1, y0 + 1),
                                      (x0, y0 + 1)], facecolor=FIL_B,
                                     edgecolor="none", zorder=5))
        elif paint_junction == "one":
            for r, c in zip(*np.nonzero(z_junc)):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=FIL_A, edgecolor="none",
                                       zorder=5))
        if paint_junction is not None:
            ax.add_patch(plt.Circle((ZOOM, ZOOM), 4.6, fill=False,
                                    edgecolor=JUNCTION, lw=0.8, zorder=7))
        return ax

    crossing([bx[0], row2_y, w2, h2], inst_colour, "both")
    crossing([bx[0] + w2 + gap_in, row2_y, w2, h2], inst_colour, "one")
    crossing([bx[1], row2_y, w2, h2], inst_colour, None)
    crossing([bx[1] + w2 + gap_in, row2_y, w2, h2], swap_colour, None)

    for x, sign in ((bx[0] + w2 + gap_in / 2.0, r"$=$"),
                    (bx[1] + w2 + gap_in / 2.0, r"$\neq$")):
        # 8.5 bold, not larger: the two-size scale holds for symbols too, and
        # an 11 pt "=" was the only glyph in the set outside it.
        fig.text(x, row2_y + h2 / 2.0, sign, fontsize=PT_TITLE,
                 fontweight="bold",
                 color=AGREE_C if sign == "=" else CHANGE_C,
                 ha="center", va="center")

    for x, letter, name, sub in (
            (bx[0], "f", "How the crossing is drawn",
             "one owner at the junction pixels, or two"),
            (bx[1], "g", "How the crossing is resolved",
             "which arm continues into which")):
        fig.text(x, 1.0 - (at["tag2"] + 0.16) * IN, "(%s)" % letter,
                 fontsize=PT_TITLE, fontweight="bold", color=INK,
                 ha="left", va="baseline")
        fig.text(x + 0.038, 1.0 - (at["tag2"] + 0.16) * IN, name,
                 fontsize=PT_TITLE, fontweight="bold", color=INK,
                 ha="left", va="baseline")
        fig.text(x, 1.0 - (at["sub2"] + 0.135) * IN, sub, fontsize=PT_ANNOT,
                 color=GRAY, ha="left", va="baseline")

    strip_y = 1.0 - (at["strip"] + 0.100) * IN
    n_drawn = verdict_strip(fig, bx[0], strip_y, v_ref, v_redrawn)
    n_res = verdict_strip(fig, bx[1], strip_y, v_ref, v_swapped)

    shared = [overlap["by_density"][k]["plecta_shared_fraction"]
              for k in sorted(overlap["by_density"], key=int)]
    fig.text(left, 1.0 - (at["prose"] + 0.045) * IN,
             ("Measured: %.1f–%.1f %% of PLECTA's pixels have two owners; "
              "flattening them all moves " + FONE + " by %.3f.")
             % (100 * min(shared), 100 * max(shared),
                overlap["pooled"]["plecta_f1_cost_of_flattening_max"]),
             fontsize=PT_ANNOT, color=INK, ha="left", va="top")

    save_fig(fig, "fig_common_fragment_metric", bbox_inches=None)
    plt.close(fig)

    print("crop %s %dx%d  mask %d px  skeleton %d px  junction %d px"
          % (s["scene"], n, n, mask.sum(), skel.sum(), junc.sum()))
    print("fragments:", ids, "sizes",
          [int((frag == k).sum()) for k in ids], "dropped px", int(dropped.sum()))
    print("reference pairing:", reference, " swapped:", swapped)
    print("pair verdicts changed -- redrawn: %d/6, re-resolved: %d/6"
          % (n_drawn, n_res))
    print("shared fraction %.3f-%.3f, max flattening cost %.4f"
          % (min(shared), max(shared),
             overlap["pooled"]["plecta_f1_cost_of_flattening_max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
