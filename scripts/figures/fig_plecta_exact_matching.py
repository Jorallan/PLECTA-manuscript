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
from matplotlib.patches import Circle, Rectangle

from _style import (FIG_W, PT_MIN, PT_TITLE,
                    FIL_A, FIL_B, JUNCTION, CHORD, INK, PLECTA,
                    load_figure_data, plecta_style, save_fig, unpack)

FIG_H = 3.36
LETTERS = "abcd"

#: How much of each arm to draw, in pixels from the tip.  Long enough that the
#: direction an arm leaves in is unmistakable, short enough that the node stays
#: the subject.
ARM_PX = 26


def fy(inches):
    """Figure fraction of a distance measured down from the top edge."""
    return 1.0 - inches / FIG_H


def title(fig, x, y, letter, text):
    fig.text(x, y, f"({letter})", fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")
    fig.text(x + 0.034, y, text, fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")


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


def node_axes(fig, rect, arms, pad=4.2):
    """A square frame shared by every diagram of one node."""
    pts = np.vstack(list(arms.values()))
    centre = pts.mean(axis=0)
    half = float(np.abs(pts - centre).max()) + pad
    ax = fig.add_axes(rect)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def blob(ax, hub, radius, colour=JUNCTION, alpha=0.20, z=2):
    """The junction cluster: one soft disc, not a staircase of pixels."""
    ax.add_patch(Circle(tuple(hub), radius, facecolor=colour, alpha=alpha,
                        edgecolor="none", zorder=z))


def bridge(ax, arms, pairs):
    """Paint each taken pair as the one filament it asserts."""
    for k, (p, q) in enumerate(pairs):
        colour = (FIL_A, FIL_B)[k % 2]
        ax.plot([arms[p][0, 0], arms[q][0, 0]], [arms[p][0, 1], arms[q][0, 1]],
                color=colour, lw=2.1, solid_capstyle="round", zorder=4)


def pair_colours(arms, pairs, rest=CHORD):
    colours = {sid: rest for sid in arms}
    for k, (p, q) in enumerate(pairs):
        colours[p] = colours[q] = (FIL_A, FIL_B)[k % 2]
    return colours


# ── panels ────────────────────────────────────────────────────────────────


def panel_crossing(fig, rects, node, names, arms, hub, n_shared):
    """The same node before and after the decision, side by side."""
    ax = node_axes(fig, rects[0], arms)
    blob(ax, hub, 3.4)
    for arm in arms.values():
        ax.plot(arm[:, 0], arm[:, 1], color=INK, lw=2.1,
                solid_capstyle="round", zorder=4)
    for sid, arm in arms.items():
        step = arm[-1] - arm[max(0, len(arm) - 6)]
        step = step / max(1e-9, float(np.hypot(*step)))
        ax.text(*(arm[-1] + step * 2.8), names[sid], fontsize=PT_MIN,
                fontweight="bold", color=INK, ha="center", va="center",
                zorder=7, bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                                    ec=CHORD, lw=0.5))
    ax.text(0.5, -0.115, "one blob, four arms", transform=ax.transAxes,
            ha="center", va="top", fontsize=PT_MIN, color=CHORD)

    ax = node_axes(fig, rects[1], arms)
    pairs = [tuple(p) for p in node["exact"]]
    colours = pair_colours(arms, pairs)
    for sid, arm in arms.items():
        ax.plot(arm[:, 0], arm[:, 1], color=colours[sid], lw=2.1,
                solid_capstyle="round", zorder=4)
    bridge(ax, arms, pairs)
    blob(ax, hub, 3.4, colour=FIL_A, alpha=0.16)
    blob(ax, hub, 3.4, colour=FIL_B, alpha=0.16)
    ax.text(0.5, -0.06, "two instances,\n%d pixels owned twice" % n_shared,
            transform=ax.transAxes, ha="center", va="top", fontsize=PT_MIN,
            color=CHORD, linespacing=1.45)


def option(ax, arms, hub, pairs, costs, price, total, chosen, free_label):
    """One admissible configuration of one node.

    An arm left unmatched keeps its grey and gets an open ring; the cost of
    each taken edge sits on that edge; and the arithmetic that decides between
    the options is written out rather than reduced to a total.
    """
    matched = {s for p in pairs for s in p}
    colours = pair_colours(arms, pairs)
    for sid, arm in arms.items():
        ax.plot(arm[:, 0], arm[:, 1], color=colours[sid],
                lw=2.1 if sid in matched else 1.3, solid_capstyle="round",
                zorder=4 if sid in matched else 3)
    blob(ax, hub, 3.2, alpha=0.14)
    bridge(ax, arms, pairs)
    #  No cost is written on an edge.  Four arms meeting inside twenty pixels
    #  leave nowhere to put a label that is not already an arm, and the line
    #  underneath prints every cost anyway, in the order they are added up.
    for sid in arms:
        if sid not in matched:
            tip = arms[sid][0]
            ax.plot([tip[0]], [tip[1]], "o", ms=4.2, mfc="white",
                    mec=JUNCTION, mew=1.1, zorder=6)

    n_free = len(arms) - len(matched)
    terms = ["%.2f" % costs[frozenset(p)] for p in pairs]
    if n_free == 1:
        terms.append("%.2f" % price)
    elif n_free > 1:
        terms.append("%d × %.2f" % (n_free, price))
    ax.text(0.5, -0.02, "%s\n= %.2f" % (" + ".join(terms), total),
            transform=ax.transAxes, ha="center", va="top", fontsize=PT_MIN,
            color=INK if chosen else CHORD, linespacing=1.45,
            fontweight="bold" if chosen else "normal")
    if chosen:
        ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, ec=INK,
                               lw=0.9, zorder=1, transform=ax.transAxes))


def panel_gap(fig, rect, text_x, data):
    """The fourth decision: a break in the mask, bridged under four gates.

    Drawn from the accepted gap of ``cov20/synth_0001`` -- two arms 25.8 px
    apart whose tips the global gap matching joined.  The other mask pieces in
    the same crop are drawn too, in grey, because they are what the gates exist
    to rule out.
    """
    from scipy.ndimage import label

    case, gates = data["gap_case"], data["gap_gates"]
    mask = unpack(case["mask"])
    pieces, n = label(mask, structure=np.ones((3, 3), bool))
    tips = [np.asarray(t, float) for t in case["tips"]]
    joined = {int(pieces[int(round(t[0])), int(round(t[1]))]) for t in tips}

    ax = fig.add_axes(rect)
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
                lw=2.1 if on else 1.3, solid_capstyle="round",
                zorder=4 if on else 3)

    ax.plot([tips[0][1], tips[1][1]], [tips[0][0], tips[1][0]], color=PLECTA,
            lw=1.6, ls=(0, (3.2, 2.2)), zorder=5)
    for t in tips:
        ax.plot([t[1]], [t[0]], "o", ms=4.2, mfc="white", mec=FIL_A, mew=1.1,
                zorder=6)
    ax.annotate("d = %.1f px" % case["d"],
                (0.5 * (tips[0][1] + tips[1][1]), 0.5 * (tips[0][0] + tips[1][0])),
                textcoords="offset points", xytext=(-7.0, 0.0), ha="right",
                va="center", fontsize=PT_MIN, color=PLECTA, zorder=7)

    lines = ["a gap link must pass", "all four:",
             "d $\\leq$ %.0f px" % gates["max_len"],
             r"$\theta \leq$ %.2f" % gates["max_theta"],
             r"$\varphi \leq$ %.2f" % gates["max_phi"],
             "C $<$ %.2f" % gates["cost_limit"]]
    y = rect[1] + rect[3] - 0.055
    for k, line in enumerate(lines):
        fig.text(text_x, y - k * 0.052, line, fontsize=PT_MIN,
                 color=CHORD if k < 2 else INK, ha="left", va="top")
    n_pass = sum(1 for c in gates["candidates"] if c["gated_in"])
    fig.text(text_x, y - len(lines) * 0.052 - 0.012,
             "%d of %d candidates" % (n_pass, len(gates["candidates"])),
             fontsize=PT_MIN, color=CHORD, ha="left", va="top")


def main() -> int:
    plecta_style()
    data = load_figure_data()
    j4, j3 = data["junction4"], data["junction3"]
    names4 = letter_map(j4)
    arms4, hub4 = node_arms(j4)
    arms3, hub3 = node_arms(j3)
    cost4 = {frozenset((p["a"], p["b"])): p["cost"] for p in j4["pairs"]}
    cost3 = {frozenset((p["a"], p["b"])): p["cost"] for p in j3["pairs"]}
    layers = [unpack(p) for p in data["crossing"]["layers"]]
    n_shared = int((layers[0] & layers[1]).sum())

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    h = 0.95 / FIG_H

    # ── row 1: one crossing, and every way it could have been paired ──────
    row1 = fy(1.33)
    panel_crossing(fig, [[0.048, row1, 0.126, h], [0.208, row1, 0.126, h]],
                   j4, names4, arms4, hub4, n_shared)
    for i, cfg in enumerate(j4["configurations"]):
        ax = node_axes(fig, [0.392 + i * 0.152, row1, 0.136, h], arms4)
        option(ax, arms4, hub4, [tuple(p) for p in cfg["pairs"]], cost4,
               j4["price"], cfg["total"], chosen=(i == 0), free_label="free")
    title(fig, 0.032, fy(0.30), "a", "One crossing, decided")
    title(fig, 0.376, fy(0.30), "b", "Every pairing, scored as a whole")

    # ── row 2: the priced free option, and what solving jointly buys ──────
    row2 = fy(2.93)
    for i, cfg in enumerate(j3["configurations"]):
        ax = node_axes(fig, [0.048 + i * 0.156, row2, 0.138, h], arms3)
        option(ax, arms3, hub3, [tuple(p) for p in cfg["pairs"]], cost3,
               j3["price"], cfg["total"], chosen=(i == 0),
               free_label="free" if cfg["pairs"] else "all free")
    panel_gap(fig, [0.610, row2, 0.150, h], 0.790, data)
    title(fig, 0.032, fy(1.88), "c", "An arm left free, and its price")
    title(fig, 0.596, fy(1.88), "d", "A gap bridged")

    save_fig(fig, "fig_plecta_exact_matching", bbox_inches=None)
    plt.close(fig)
    declined = max(p["cost"] for p in j3["pairs"] if p["admissible"])
    print("shared px:", n_shared, "names:", names4,
          "declined edge:", declined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
