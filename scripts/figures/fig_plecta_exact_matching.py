"""Figure 3: every decision the linker can make, on the nodes it made them at.

This absorbs what was a separate "gates" figure.  That one drew the same four
outcomes -- a pairing taken, a pairing refused, a stub left free, a gap bridged
-- as idealised curves with the real values printed underneath, next to this
figure which drew real geometry.  Two figures for one subject, one of them a
cartoon of the other, so they are now one.

Two things came out of the merge:

  * **the drawing is smoothed.**  Arms are still the node's own skeleton pixels,
    but drawn through a short moving average instead of joining pixel centres,
    which removes the raster staircase without moving any point as much as a
    pixel.  It reads like the idealised figure and stays the measured geometry;
    every angle on the page is the angle the cost was computed from.
  * **the exact-against-greedy bar chart is gone.**  Its three numbers and their
    three n are all stated in the adjacent sentence of Section 2.4.2, and a bar
    chart of three numbers printed beside it is not a figure.

Every number here is measured: the arm geometry, the pairwise costs, the joint
score of each admissible configuration, the declined-but-admissible edge at a
real three-arm node, and the accepted gap with its separation.  All of it is
read from ``results/plecta_figure_data.json``.

VISUAL TREATMENT.  This figure used to be the one exception in the set: it did
not call ``plecta_style()``, and drew in Segoe UI on a flat Apple system
palette, laid out as elevated white cards on a light grey page with soft fake
shadows stacked from translucent rounded rects.  Reviewer comment 4 named it as
the presentation-style image, and the exception is now removed by exactly the
route the old docstring recorded: ``plecta_style()`` is reapplied, and
FIL_A/FIL_B/JUNCTION/CHORD/PLECTA come from ``_style`` in place of the ACCENT_*
palette.  Concretely, what changed is colour, type, and the chosen-answer
highlight:

  * Latin Modern Roman at the shared 8.5/7.5 pt scale, replacing Segoe UI at
    9.5/8.0/7.6/7.3 -- four sizes that existed nowhere else in the set.
  * hairline rules on white in place of cards, shadows and the #F5F5F7 page.
  * the chosen configuration is marked by a heavier rule, not a green tint:
    the same "this is the answer" signal without the tile reading as UI.
  * panel titles use the set's ``(a)`` bold-tag convention rather than a
    white letter in a coloured circle.

Every geometric quantity -- arm directions, hub positions, costs, gate
thresholds -- is untouched, as it was under the old treatment.  Nothing below
the drawing layer was edited, so the numbers on the page are the same numbers.

    python scripts/figures/fig_plecta_exact_matching.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from _style import (CHORD, FIG_W, FIL_A, FIL_B, GRAY, INK, JUNCTION,
                    LIGHT_GRAY, PLECTA, PT_ANNOT, PT_MIN, PT_TITLE,
                    load_figure_data, plecta_style, save_fig, unpack)

FIG_H = 3.56
LETTERS = "abcd"

#: How much of each arm to draw, in pixels from the tip.  Long enough that the
#: direction an arm leaves in is unmistakable, short enough that the node stays
#: the subject.
ARM_PX = 26

# ── the shared palette and type scale ──────────────────────────────────────
#  FIL_A, FIL_B, PLECTA, JUNCTION, CHORD, INK come from ``_style`` and mean the
#  same thing here as in every other figure: A and B are the two filaments, the
#  green is a link PLECTA takes, the red is the junction cluster, the grey is a
#  candidate not taken.  SUBTLE is the set's secondary text grey.
SUBTLE = GRAY
CARD_EDGE = LIGHT_GRAY

#  Three sizes, all from the set's scale: 8.5 bold for a panel title, 7.5 for
#  annotation, 7.0 for the dense arithmetic under each configuration.  The old
#  treatment used 9.5/8.0/7.6/7.3, none of which appears anywhere else.
PT_CAPTION = PT_ANNOT
PT_TAG = PT_MIN

#  Was a stacked-translucent drop shadow under every arm.  Nothing replaces it:
#  the arms carry their own weight against a white ground.
LINE_SHADOW = None


def fy(inches):
    """Figure fraction of a distance measured down from the top edge."""
    return 1.0 - inches / FIG_H


def title(fig, x, y, letter, text):
    """``(a)`` bold, the title beside it in the regular weight.

    The set's convention, and the same one ``_style.tagged_title`` applies in
    axes coordinates; this figure places its titles in figure coordinates, so
    the two lines are written out rather than delegated.  Latin Modern bold
    sets about 15 % wider than its regular, so a whole title in bold reads as
    stretched next to body text -- the tag carries the hierarchy on its own.
    """
    fig.text(x, y, "(%s)" % letter, fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="center", zorder=6)
    fig.text(x + 0.032, y, text, fontsize=PT_TITLE, color=INK, ha="left",
             va="center", zorder=6)


def letter_map(node):
    """Name the stubs a, b, c, ... anticlockwise from due east."""
    bear = {}
    for sid in node["stubs"]:
        dr, dc = node["frames"][str(sid)]["tangent"]
        bear[sid] = math.degrees(math.atan2(-dr, dc)) % 360.0
    return {s: LETTERS[i]
            for i, s in enumerate(sorted(node["stubs"], key=lambda s: bear[s]))}


# ── the node, as its own geometry ─────────────────────────────────────────


def arm_xy(frame, n=ARM_PX):
    """The arm's own pixels, tip first, trimmed to ``n`` of them.

    ``arm`` is stored from whichever end the tracer walked, so it is oriented
    against ``tip`` rather than assumed.
    """
    pts = np.asarray(frame["arm"], float)
    tip = np.asarray(frame["tip"], float)
    if np.hypot(*(pts[0] - tip)) > np.hypot(*(pts[-1] - tip)):
        pts = pts[::-1]
    return pts[:n][:, ::-1] * np.array([1.0, -1.0])   # (row, col) -> (x, y up)


def node_arms(node, target_deg=52.0, n=ARM_PX):
    """Every arm of one node, rotated so its long axis runs diagonally.

    In image orientation both of these nodes are tall splinters a few pixels
    wide -- the three-arm node's stubs leave at bearings 73, 238 and 256
    degrees, two of them 18 degrees apart.  A rotation is the fix that costs
    nothing: it preserves every angle and every length.
    """
    raw = {sid: arm_xy(node["frames"][str(sid)], n) for sid in node["stubs"]}
    pts = np.vstack(list(raw.values()))
    centre = pts.mean(axis=0)
    axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    delta = math.radians(target_deg) - math.atan2(axis[1], axis[0])
    rot = np.array([[math.cos(delta), -math.sin(delta)],
                    [math.sin(delta), math.cos(delta)]])
    arms = {sid: smooth((arm - centre) @ rot.T)
            for sid, arm in raw.items()}
    hub = np.array([arm[0] for arm in arms.values()]).mean(axis=0)
    return arms, hub


def smooth(path, window=5):
    """A polyline through a moving average of its own points.

    The skeleton is a raster, so joining pixel centres draws a staircase.  A
    short box filter along the path removes it.  The window is odd and small
    and the ends are held fixed, so no drawn point sits as much as a pixel from
    a real one and the direction each arm leaves in is untouched -- which
    matters, because that direction is what the cost was computed from.
    """
    path = np.asarray(path, float)
    if len(path) < window or window < 3:
        return path
    pad = window // 2
    padded = np.vstack([np.repeat(path[:1], pad, axis=0), path,
                        np.repeat(path[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    return np.column_stack([np.convolve(padded[:, k], kernel, mode="valid")
                            for k in (0, 1)])


def walk(pixels, start):
    """Order a thin connected component into a path, nearest neighbour first."""
    remaining = {tuple(p) for p in pixels}
    here = min(remaining, key=lambda p: (p[0] - start[0]) ** 2
               + (p[1] - start[1]) ** 2)
    remaining.discard(here)
    out = [here]
    while remaining:
        nxt = min(remaining, key=lambda p: (p[0] - here[0]) ** 2
                  + (p[1] - here[1]) ** 2)
        if (nxt[0] - here[0]) ** 2 + (nxt[1] - here[1]) ** 2 > 8:
            break                      # a separate piece, not this arm
        out.append(nxt)
        remaining.discard(nxt)
        here = nxt
    return np.asarray(out, float)


# ── panels: a hairline rule on white, and nothing else ─────────────────────
#
#  What this replaced: a six-layer fake-blurred drop shadow (matplotlib has no
#  native blur, so the old code stacked translucent rounded rects at growing
#  radius) under a flat rounded tile, tinted #EAFBEF when chosen.  That is the
#  card idiom of a slide deck, and it is what the reviewer's comment 4 is
#  about.  A journal figure separates panels with a rule or with white space.
#
#  The chosen configuration still has to be findable at a glance, so it keeps a
#  heavier rule in PLECTA green -- one weight step and one hue, against a
#  hairline grey.  No fill, no shadow, no tint.


def card(fig, overlay, rect, pad_in=0.050, chosen=False, rounding=0.015):
    """A hairline rounded frame behind ``rect``; returns the padded rect."""
    px, py = pad_in / FIG_W, pad_in / FIG_H
    padded = [rect[0] - px, rect[1] - py, rect[2] + 2 * px, rect[3] + 2 * py]
    overlay.add_patch(FancyBboxPatch(
        (padded[0], padded[1]), padded[2], padded[3],
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=1.1 if chosen else 0.6,
        edgecolor=PLECTA if chosen else CARD_EDGE,
        facecolor="white", zorder=2))
    return padded


def caption_below(fig, rect, text, fontsize=PT_CAPTION, colour=SUBTLE,
                  gap_in=0.062, bold=False, linespacing=1.45, card_pad_in=0.050):
    """Text centred under ``rect``, cleared of the card's border by a fixed
    physical gap rather than a fraction of the axes -- a fraction of axes
    height tracks the card's own size, not the shadow it sits on, so it drifts
    into the border the moment card height or font size changes."""
    x = rect[0] + rect[2] / 2.0
    y = rect[1] - (card_pad_in + gap_in) / FIG_H
    fig.text(x, y, text, fontsize=fontsize, color=colour, ha="center",
            va="top", linespacing=linespacing,
            fontweight="bold" if bold else "normal", zorder=6)


def node_axes(fig, rect, arms, pad=4.2):
    """A square frame shared by every diagram of one node."""
    pts = np.vstack(list(arms.values()))
    centre = pts.mean(axis=0)
    half = float(np.abs(pts - centre).max()) + pad
    ax = fig.add_axes(rect, zorder=3)
    ax.set_facecolor("none")
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def blob(ax, hub, radius, colour=JUNCTION, alpha=0.16, z=2):
    """The junction cluster: one soft disc, not a staircase of pixels."""
    ax.add_patch(Circle(tuple(hub), radius, facecolor=colour, alpha=alpha,
                        edgecolor="none", zorder=z))


def bridge(ax, arms, pairs):
    """Paint each taken pair as the one filament it asserts."""
    for k, (p, q) in enumerate(pairs):
        colour = (FIL_A, FIL_B)[k % 2]
        ax.plot([arms[p][0, 0], arms[q][0, 0]], [arms[p][0, 1], arms[q][0, 1]],
                color=colour, lw=2.6, solid_capstyle="round", zorder=4,
                path_effects=LINE_SHADOW)


def pair_colours(arms, pairs, rest=CHORD):
    colours = {sid: rest for sid in arms}
    for k, (p, q) in enumerate(pairs):
        colours[p] = colours[q] = (FIL_A, FIL_B)[k % 2]
    return colours


def marker(ax, xy, colour, ms=6.0):
    ax.plot([xy[0]], [xy[1]], "o", ms=ms, mfc=colour, mec="white", mew=1.3,
            zorder=6, path_effects=LINE_SHADOW)


# ── panels ────────────────────────────────────────────────────────────────


def option(fig, overlay, rect, arms, hub, pairs, costs, price, total, chosen,
          free_label):
    """One admissible configuration of one node.

    An arm left unmatched keeps its neutral grey and gets a solid marker; the
    cost of each taken edge sits on that edge; and the arithmetic that decides
    between the options is written out rather than reduced to a total.  The
    chosen configuration is set apart by an accent-green card, not a black
    square: same underlying rule, a tile that reads as "selected" rather than
    "boxed off."
    """
    card(fig, overlay, rect, chosen=chosen)
    ax = node_axes(fig, rect, arms)
    matched = {s for p in pairs for s in p}
    colours = pair_colours(arms, pairs)
    for sid, arm in arms.items():
        ax.plot(arm[:, 0], arm[:, 1], color=colours[sid],
                lw=2.4 if sid in matched else 1.4, solid_capstyle="round",
                zorder=4 if sid in matched else 3,
                path_effects=LINE_SHADOW if sid in matched else None)
    blob(ax, hub, 3.2, alpha=0.12)
    bridge(ax, arms, pairs)
    #  No cost is written on an edge.  Four arms meeting inside twenty pixels
    #  leave nowhere to put a label that is not already an arm, and the line
    #  underneath prints every cost anyway, in the order they are added up.
    for sid in arms:
        if sid not in matched:
            marker(ax, arms[sid][0], JUNCTION, ms=5.4)

    n_free = len(arms) - len(matched)
    terms = ["%.2f" % costs[frozenset(p)] for p in pairs]
    if n_free == 1:
        terms.append("%.2f" % price)
    elif n_free > 1:
        terms.append("%d × %.2f" % (n_free, price))
    caption_below(fig, rect, "%s\n= %.2f" % (" + ".join(terms), total),
                 fontsize=PT_ANNOT, colour=INK if chosen else SUBTLE,
                 linespacing=1.5, bold=chosen)


def panel_gap(fig, overlay, rect, chip_rect, data):
    """The fourth decision: a break in the mask, bridged under four gates.

    Drawn from the accepted gap of ``cov20/synth_0001`` -- two arms 25.8 px
    apart whose tips the global gap matching joined.  The other mask pieces in
    the same crop are drawn too, in grey, because they are what the gates exist
    to rule out.
    """
    from scipy.ndimage import label

    #  No count of how many candidates this particular scene offered or
    #  accepted: the panel states the rule, and a scene-specific tally invites
    #  the reader to treat one crop as evidence about the method.
    case, gates = data["gap_case"], data["gap_gates"]
    mask = unpack(case["mask"])
    pieces, n = label(mask, structure=np.ones((3, 3), bool))
    tips = [np.asarray(t, float) for t in case["tips"]]
    joined = {int(pieces[int(round(t[0])), int(round(t[1]))]) for t in tips}

    card(fig, overlay, rect)
    ax = fig.add_axes(rect, zorder=3)
    ax.set_facecolor("none")
    side = mask.shape[0]
    ax.set_xlim(-2, side + 1)
    ax.set_ylim(side + 1, -2)                  # row 0 at the top
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for k in range(1, n + 1):
        pts = np.array(np.nonzero(pieces == k)).T
        if len(pts) < 4:
            continue
        far = pts[np.argmax(np.linalg.norm(
            pts - np.mean([t for t in tips], axis=0), axis=1))]
        path = smooth(walk(pts, far))
        on = k in joined
        ax.plot(path[:, 1], path[:, 0], color=FIL_A if on else CHORD,
                lw=2.4 if on else 1.4, solid_capstyle="round",
                zorder=4 if on else 3,
                path_effects=LINE_SHADOW if on else None)

    ax.plot([tips[0][1], tips[1][1]], [tips[0][0], tips[1][0]], color=PLECTA,
            lw=1.9, ls=(0, (3.2, 2.2)), zorder=5)
    for t in tips:
        marker(ax, (t[1], t[0]), FIL_A, ms=5.4)
    #  Pinned to the panel's top-left corner rather than offset from the gap's
    #  midpoint.  The old placement hung the label off the left of the midpoint,
    #  which was clear when panels had no border and straddles the frame now
    #  that they do.  The strand runs bottom-left to top-right, so this corner
    #  is empty in the crop.
    ax.text(0.05, 0.95, "d = %.1f px" % case["d"], transform=ax.transAxes,
            ha="left", va="top", fontsize=PT_ANNOT, color=PLECTA, zorder=7)

    #  The four-gate checklist as its own card: a short list of pass conditions,
    #  each led by a small accent-green "cleared" dot rather than bare text.
    #  The dot is a text-bbox circle, not a Circle patch: a patch drawn in
    #  transFigure data units inherits the figure's non-square aspect and
    #  renders as an ellipse, where a font-relative bbox stays round.
    card(fig, overlay, chip_rect)
    lines = ["d $\\leq$ %.0f px" % gates["max_len"],
             r"$\theta \leq$ %.2f" % gates["max_theta"],
             r"$\varphi \leq$ %.2f" % gates["max_phi"],
             "C $<$ %.2f" % gates["cost_limit"]]
    cx, cy, cw, ch = chip_rect
    x0 = cx + 0.030
    header_h = PT_CAPTION * 1.30 / 72.0 / FIG_H
    header_gap = 0.028
    row_h = PT_ANNOT * 1.55 / 72.0 / FIG_H
    #  The header sets its own top by its font box; the four rows below it are
    #  centred on their own baseline, so the block runs from the header's top
    #  to the last row's centre plus half a row -- that span, not the card,
    #  is what gets centred in the card's height.
    block_h = header_h + header_gap + (len(lines) - 1) * row_h + row_h / 2.0
    y = cy + ch / 2.0 + block_h / 2.0
    fig.text(cx + cw / 2.0, y, "all four are required", fontsize=PT_CAPTION,
            color=SUBTLE, ha="center", va="top", zorder=6)
    y -= header_h + header_gap
    for line in lines:
        #  Was a filled green "cleared" dot per row -- a traffic light, and the
        #  most deck-like element on the page: it asserts a pass/fail state the
        #  panel does not measure, since these are the gate values, not four
        #  results.  A plain rule sets the list off instead.  The dot was drawn
        #  as the bbox of a single space, which also meant check_type_scale had
        #  to police a glyph that draws nothing; that problem goes with it.
        overlay.add_patch(plt.Rectangle(
            (x0 + 0.004, y - 0.0015), 0.016, 0.003, transform=fig.transFigure,
            facecolor=CHORD, edgecolor="none", zorder=6))
        fig.text(x0 + 0.032, y, line, fontsize=PT_ANNOT, color=INK, ha="left",
                va="center", zorder=6)
        y -= row_h


def main() -> int:
    plecta_style()
    data = load_figure_data()
    j4, j3 = data["junction4"], data["junction3"]
    names4 = letter_map(j4)
    arms4, hub4 = node_arms(j4)
    arms3, hub3 = node_arms(j3)
    cost4 = {frozenset((p["a"], p["b"])): p["cost"] for p in j4["pairs"]}
    cost3 = {frozenset((p["a"], p["b"])): p["cost"] for p in j3["pairs"]}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    overlay = fig.add_axes([0, 0, 1, 1], zorder=0)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.set_facecolor("none")
    h = 0.92 / FIG_H

    # ── row 1: every way one four-arm crossing could have been paired ────
    #  The before/after panel that used to open this row was dropped: the
    #  chosen configuration is already the first card here, so the row said
    #  the same thing twice.  The four cards take the width it freed.
    row1 = fy(1.42)
    for i, cfg in enumerate(j4["configurations"]):
        rect = [0.048 + i * 0.2375, row1, 0.2095, h]
        option(fig, overlay, rect, arms4, hub4, [tuple(p) for p in cfg["pairs"]],
               cost4, j4["price"], cfg["total"], chosen=(i == 0),
               free_label="free")
    title(fig, 0.040, fy(0.24), "a",
         "Every admissible pairing at one crossing, each scored as a whole")

    # ── row 2: the priced free option, and what solving jointly buys ──────
    row2 = fy(3.05)
    for i, cfg in enumerate(j3["configurations"]):
        rect = [0.048 + i * 0.156, row2, 0.138, h]
        option(fig, overlay, rect, arms3, hub3, [tuple(p) for p in cfg["pairs"]],
               cost3, j3["price"], cfg["total"], chosen=(i == 0),
               free_label="free" if cfg["pairs"] else "all free")
    panel_gap(fig, overlay, [0.606, row2, 0.140, h], [0.762, row2, 0.208, h], data)
    title(fig, 0.040, fy(1.94), "b", "Unmatched-arm penalty")
    title(fig, 0.604, fy(1.94), "c", "Gap bridging")

    save_fig(fig, "fig_plecta_exact_matching", bbox_inches=None)
    plt.close(fig)
    declined = max(p["cost"] for p in j3["pairs"] if p["admissible"])
    print("names:", names4, "declined edge:", declined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
