"""Figure 4: the four decisions the linker can make, and the gate on each.

Redrawn after review.  Two things were wrong with the earlier version:

  * the strokes were 5.4 pt wide in a 1.4 inch panel, so the arms merged into
    each other and into the junction disc;
  * each arm was drawn *backwards through* its own junction, so at a crossing
    the two arms genuinely overlapped and the link between their tips was
    buried underneath them.

Now an arm is drawn where it belongs: outward from its stub tip, away from the
junction, at 2.4 pt.  The tip sits clear of the node disc and the link the
method draws between two tips is the most visible thing in the panel.  The
small arrow at a tip is that stub's outward tangent -- the direction the
method asks the arm to continue in -- so it points into the junction.

Idealised drawings; the measured cases are Figs. 3 and 5.  Self-contained.

    python scripts/figures/fig_plecta_gates.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from _style import (BLUE, DARK, FIG_W, GRAY, GREEN, JUNCTION, ORANGE,
                    PT_ANNOT, PT_LEGEND, PT_TITLE, plecta_style, save_fig)

FIG_H = 2.62
DEG = math.pi / 180.0
ARM_LW = 2.4
R_TIP = 0.092          # tip sits this far from the node centre
R_NODE = 0.050
HEAD = dict(arrowstyle="-|>,head_length=0.50,head_width=0.30",
            mutation_scale=8, shrinkA=0, shrinkB=0)


def unit(deg):
    return np.array([math.cos(deg * DEG), math.sin(deg * DEG)])


def arm_outward(ax, tip, bearing, length, colour, bend=0.0, z=3):
    """The arm body: from its stub tip, outward, away from the junction."""
    t = unit(bearing)
    nrm = np.array([-t[1], t[0]])
    s = np.linspace(0, 1, 80)[:, None]
    pts = tip + t * length * s + nrm * bend * length * (s ** 2)
    ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=ARM_LW,
            solid_capstyle="round", zorder=z)


def tip_dot(ax, p, colour):
    ax.add_patch(Circle(tuple(p), 0.023, facecolor="white", edgecolor=colour,
                        linewidth=1.2, zorder=8))


def tangent(ax, tip, bearing, colour=DARK):
    """The stub's outward tangent, drawn on the arm and pointing at the tip.

    Off the tip rather than through it, so it never sits on top of the link
    the panel exists to show.
    """
    u = unit(bearing)
    ax.add_patch(FancyArrowPatch(tuple(tip + u * 0.105), tuple(tip + u * 0.030),
                                 color=colour, lw=0.9, zorder=7, **HEAD))


def accepted(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREEN, lw=1.8,
            solid_capstyle="round", zorder=6)


def refused(ax, p0, p1, at=0.50):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRAY, lw=1.2,
            ls=(0, (3.0, 2.2)), zorder=5)
    cross = p0 + (p1 - p0) * at
    ax.plot([cross[0]], [cross[1]], marker="x", ms=6.0, mew=1.6, color=GRAY,
            zorder=9)


def junction(ax, centre, spokes):
    """A node disc and its stubs.  ``spokes`` is (bearing, colour, length, bend).

    Every stub tip sits ``R_TIP`` from the node centre, just outside the disc,
    and its arm runs outward from there -- so the disc is the thing all the
    tips touch and the link drawn between two tips is never buried under an
    arm.
    """
    ax.add_patch(Circle(tuple(centre), R_NODE, facecolor=JUNCTION,
                        edgecolor="none", alpha=0.80, zorder=2))
    tips = []
    for bearing, colour, length, bend in spokes:
        tip = np.asarray(centre) + unit(bearing) * R_TIP
        arm_outward(ax, tip, bearing, length, colour, bend=bend)
        tip_dot(ax, tip, colour)
        tips.append(tip)
    return tips


def stage(fig, rect, xspan=1.0):
    ax = fig.add_axes(rect)
    ratio = (rect[3] * FIG_H) / (rect[2] * FIG_W)
    ax.set_xlim(0.0, xspan)
    ax.set_ylim(0.0, xspan * ratio)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.yspan = xspan * ratio
    return ax


# ── the four outcomes ─────────────────────────────────────────────────────


def panel_pair_taken(ax):
    node = np.array([0.50, 0.52 * ax.yspan])
    ta, tb = junction(ax, node, [(202, BLUE, 0.34, 0.12),
                                 (18, ORANGE, 0.34, -0.12)])
    accepted(ax, ta, tb)
    tangent(ax, ta, 202)
    tangent(ax, tb, 18)


def panel_pair_refused(ax):
    node = np.array([0.50, 0.46 * ax.yspan])
    ta, tb = junction(ax, node, [(204, BLUE, 0.34, 0.10),
                                 (92, ORANGE, 0.26, -0.10)])
    refused(ax, ta, tb)
    tangent(ax, ta, 204)
    tangent(ax, tb, 92)


def panel_unmatched(ax):
    node = np.array([0.50, 0.52 * ax.yspan])
    ta, tb, tc = junction(ax, node, [(198, BLUE, 0.32, 0.07),
                                     (14, BLUE, 0.32, -0.07),
                                     (274, ORANGE, 0.26, 0.06)])
    accepted(ax, ta, tb)
    ax.add_patch(Circle(tuple(tc), 0.042, facecolor="none", edgecolor=JUNCTION,
                        linewidth=1.3, zorder=9))
    ax.text(tc[0] - 0.068, tc[1] - 0.030, "free", ha="right", va="center",
            fontsize=PT_ANNOT, color=JUNCTION)


def panel_gap(ax):
    Y = ax.yspan
    ta = np.array([0.335, 0.415 * Y])
    tb = np.array([0.665, 0.505 * Y])
    arm_outward(ax, ta, 194, 0.30, BLUE, bend=0.07)
    arm_outward(ax, tb, 14, 0.30, ORANGE, bend=-0.07)
    tip_dot(ax, ta, BLUE)
    tip_dot(ax, tb, ORANGE)
    accepted(ax, ta, tb)
    tangent(ax, ta, 194)
    tangent(ax, tb, 14)
    off = np.array([0.0, 0.125])
    ax.annotate("", xy=tuple(tb + off), xytext=tuple(ta + off),
                arrowprops=dict(arrowstyle="<|-|>,head_length=0.52,head_width=0.30",
                                mutation_scale=8, color=GRAY, lw=0.8,
                                shrinkA=0, shrinkB=0))
    mid = (ta + tb) / 2.0 + off
    ax.text(mid[0], mid[1] + 0.024, r"$d$", ha="center", va="bottom",
            fontsize=PT_ANNOT, color=GRAY)


def main() -> int:
    plecta_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    panels = [
        (panel_pair_taken, "a", "Pairing taken",
         r"$C_{ab} = 0.15 < 1.24$", GREEN),
        (panel_pair_refused, "b", "Pairing refused",
         r"$C_{ab} = 1.72 > 1.24$", GRAY),
        (panel_unmatched, "c", "Stub left free", r"$p_i = 0.62$", JUNCTION),
        (panel_gap, "d", "Gap bridged",
         r"$d \leq 85$ px,  $\theta \leq 0.40$"
         "\n" r"$\varphi \leq 0.28$,  $C_{ab} < 0.60$", GREEN),
    ]
    width, left, gap = 0.234, 0.014, 0.014
    for i, (fn, letter, name, gate, colour) in enumerate(panels):
        x0 = left + i * (width + gap)
        fn(stage(fig, [x0, 0.330, width, 0.470]))
        centre = x0 + width / 2.0
        fig.text(centre, 0.868, f"({letter})  {name}", fontsize=PT_TITLE,
                 fontweight="bold", color=DARK, ha="center", va="bottom")
        fig.text(centre, 0.302, gate, fontsize=PT_ANNOT, color=colour,
                 ha="center", va="top", linespacing=1.55)

    handles = [
        Line2D([], [], color=BLUE, lw=ARM_LW, solid_capstyle="round",
               label="Arm of filament A"),
        Line2D([], [], color=ORANGE, lw=ARM_LW, solid_capstyle="round",
               label="Arm of filament B"),
        Line2D([], [], color=GREEN, lw=1.8, label="Link accepted"),
        Line2D([], [], color=GRAY, lw=1.2, ls=(0, (3.0, 2.2)), marker="x",
               ms=5.5, mew=1.5, label="Link refused"),
        Line2D([], [], color=JUNCTION, lw=0, marker="o", ms=6.0, alpha=0.8,
               label="Junction cluster"),
    ]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.010), fontsize=PT_LEGEND,
               handlelength=2.0, columnspacing=1.5, handletextpad=0.6)

    save_fig(fig, "fig_plecta_gates", bbox_inches=None)
    plt.close(fig)
    print("fig_plecta_gates  %.3f x %.3f in, arm lw %.1f pt"
          % (FIG_W, FIG_H, ARM_LW))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
