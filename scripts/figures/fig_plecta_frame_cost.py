"""Figure 3: the geometry the pairing cost is built from.

Stripped after review.  The figure now *shows* the four quantities and names
nothing else: no cost expression, no weight table, no prose.  Those live in
the Methods, where an equation can be read at reading size instead of being
squeezed under a drawing.

What remains is the drawing plus its symbols:

  (a) the frame fitted at a stub -- tangent t, curvature kappa, fit window w;
  (b) the tangent-reversal angle theta, read where it is defined: between the
      outgoing tangent of one arm and the *reversed* tangent of the other,
      brought to a common origin at the first arm's tip;
  (c) the two chord turns phi_a, phi_b onto the tip-to-tip chord u, and the
      separation d;
  (d) the fade q(d) that removes the chord terms when d is only a few pixels,
      with the real tip separations at the crossing of Fig. 5 as ticks.

Every angle is drawn at the size it is stated at: the three bearings below fix
theta, phi_a and phi_b exactly, and each arc is drawn from those same numbers.

Panel (d) reads ``results/plecta_figure_data.json``; everything else is
analytic.

    python scripts/figures/fig_plecta_frame_cost.py
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
from matplotlib.patches import Arc, FancyArrowPatch

from _style import (FIG_W, PT_ANNOT, PT_AXIS, PT_TICK, PT_TITLE,
                    FIL_A, FIL_B, GEOM, CHORD, INK,
                    load_figure_data, plecta_style, save_fig)

FIG_H = 2.85
DEG = math.pi / 180.0
HEAD = dict(arrowstyle="-|>,head_length=0.46,head_width=0.26",
            mutation_scale=8.5, shrinkA=0, shrinkB=0)

#: alpha is the bearing of t_a, beta that of -t_b, gamma that of the chord.
#: theta = |alpha - beta|, phi_a = |alpha - gamma|, phi_b = |beta - gamma|.
A_DEG, B_DEG, U_DEG = 38.0, 186.0, 20.0


def unit(deg):
    return np.array([math.cos(deg * DEG), math.sin(deg * DEG)])


def arrow(ax, p0, p1, colour, lw=1.2, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), lw=lw, ls=ls,
                                 color=colour, zorder=z, **HEAD))


def angle_arc(ax, vertex, a_deg, b_deg, radius, label, colour=GEOM,
              lab_r=None):
    lo, hi = sorted((a_deg % 360.0, b_deg % 360.0))
    if hi - lo > 180.0:
        lo, hi = hi - 360.0, lo
    ax.add_patch(Arc(tuple(vertex), 2 * radius, 2 * radius, angle=0.0,
                     theta1=lo, theta2=hi, color=colour, lw=1.1, zorder=7))
    pos = np.asarray(vertex) + (lab_r or radius * 1.60) * unit((lo + hi) / 2.0)
    ax.text(pos[0], pos[1], label, ha="center", va="center", color=colour,
            fontsize=PT_ANNOT, zorder=8)


def bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def arm_to_tip(tip, tangent, length, bend=0.0):
    """A smooth arm arriving at ``tip`` with outward direction ``tangent``."""
    tip = np.asarray(tip, float)
    t = np.asarray(tangent, float) / np.hypot(*tangent)
    nrm = np.array([-t[1], t[0]])
    return bezier(tip, tip - t * length * 0.34,
                  tip - t * length * 0.70 + nrm * bend * length * 0.5,
                  tip - t * length + nrm * bend * length)


def stage(fig, rect, xspan=1.0):
    """A drawing panel whose data limits already match its printed aspect."""
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


def title(fig, x, y, letter, text):
    fig.text(x, y, f"({letter})", fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")
    fig.text(x + 0.040, y, text, fontsize=PT_TITLE,
             color=INK, ha="left", va="bottom")


# ── panels ────────────────────────────────────────────────────────────────


def panel_frame(ax):
    Y = ax.yspan
    tip = np.array([0.615, 0.585 * Y])
    t_dir = unit(32)
    arm = arm_to_tip(tip, t_dir, 0.62, bend=0.34)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arm, axis=0).T))])
    inside = s <= 0.30

    ax.plot(arm[inside, 0], arm[inside, 1], color=GEOM, lw=6.0, alpha=0.25,
            solid_capstyle="round", zorder=2)
    ax.plot(arm[:, 0], arm[:, 1], color=FIL_A, lw=2.1, solid_capstyle="round",
            zorder=3)

    ss = np.linspace(0, 0.30, 60)[:, None]
    nrm = np.array([-t_dir[1], t_dir[0]])
    ax.plot(*(tip + ss * t_dir + (ss ** 2) * 2.1 * nrm).T, color=GEOM, lw=1.0,
            ls=(0, (3.0, 2.0)), zorder=4)

    arrow(ax, tip, tip + t_dir * 0.24, FIL_A, lw=1.2)
    ax.plot([tip[0]], [tip[1]], "o", ms=3.8, mfc="white", mec=FIL_A, mew=1.2,
            zorder=9)

    ax.text(*(tip + t_dir * 0.275 + np.array([0.020, -0.010])),
            r"$\mathbf{t}$", color=FIL_A, fontsize=PT_ANNOT, ha="left",
            va="top", fontweight="bold")
    ax.text(0.755, 0.945 * Y, r"$\kappa$", color=GEOM, fontsize=PT_ANNOT,
            ha="left", va="top")
    band = arm[inside]
    mid = band[len(band) // 2]
    ax.text(mid[0], mid[1] - 0.055, r"$w$", color=GEOM, fontsize=PT_ANNOT,
            ha="center", va="top")


def panel_theta(ax):
    """theta, drawn as the two-step construction it is.

    The previous version put both steps in one sketch: the reversed tangent
    appeared once where it lies and once again at the common origin, and the
    reader had to work out that the two arrows were the same vector.  Splitting
    it makes each move a picture -- here are the two outward tangents, and here
    they are with one of them reversed and both read from a single point --
    which is the construction the definition describes, in the order it
    describes it.
    """
    left, right = 0.205, 0.790

    # step 1: the two arms as they lie, each with its own outward tangent
    #  The two arrows point at each other along nearly one line -- that is what
    #  a continuation *is* -- so they are drawn short against a wide tip
    #  separation, and their labels are stacked off the line, above one head
    #  and below the other.  Long arrows here meet in the middle and read as a
    #  single double-headed one.
    tip_a = np.array([0.120, 0.110])
    tip_b = tip_a + unit(U_DEG) * 0.190
    ta, tb = unit(A_DEG), unit(B_DEG)
    for tip, t, col, bend in ((tip_a, ta, FIL_A, 0.22),
                              (tip_b, tb, FIL_B, -0.22)):
        arm = arm_to_tip(tip, t, 0.115, bend=bend)
        ax.plot(arm[:, 0], arm[:, 1], color=col, lw=2.1,
                solid_capstyle="round", zorder=3)
        ax.plot([tip[0]], [tip[1]], "o", ms=3.4, mfc="white", mec=col,
                mew=1.2, zorder=9)
    arrow(ax, tip_a, tip_a + ta * 0.060, FIL_A, lw=1.2, z=8)
    arrow(ax, tip_b, tip_b + tb * 0.060, FIL_B, lw=1.2, z=8)
    ax.text(*(tip_a + ta * 0.060 + np.array([0.0, 0.014])),
            r"$\mathbf{t}_a$", color=FIL_A, fontsize=PT_ANNOT, ha="center",
            va="bottom", zorder=9)
    ax.text(*(tip_b + tb * 0.060 + np.array([0.0, -0.012])),
            r"$\mathbf{t}_b$", color=FIL_B, fontsize=PT_ANNOT, ha="center",
            va="top", zorder=9)

    # the move
    arrow(ax, np.array([0.435, 0.150]), np.array([0.545, 0.150]), CHORD,
          lw=1.0, z=4)

    # step 2: one origin, one tangent reversed, and the angle between them
    origin = np.array([right - 0.100, 0.112])
    stub = arm_to_tip(origin, ta, 0.095, bend=0.22)
    ax.plot(stub[:, 0], stub[:, 1], color=FIL_A, lw=2.1,
            solid_capstyle="round", zorder=3)
    R = 0.185
    arrow(ax, origin, origin + ta * R, FIL_A, lw=1.2, z=8)
    arrow(ax, origin, origin + unit(B_DEG - 180.0) * R, FIL_B, lw=1.1,
          ls=(0, (2.6, 1.8)), z=8)
    ax.plot([origin[0]], [origin[1]], "o", ms=3.4, mfc="white", mec=FIL_A,
            mew=1.2, zorder=9)
    angle_arc(ax, origin, B_DEG - 180.0, A_DEG, R * 0.58, r"$\theta$",
              lab_r=R * 0.94)
    ax.text(*(origin + ta * R + np.array([0.010, 0.006])), r"$\mathbf{t}_a$",
            color=FIL_A, fontsize=PT_ANNOT, ha="left", va="bottom", zorder=9)
    ax.text(*(origin + unit(B_DEG - 180.0) * R + np.array([0.012, -0.004])),
            r"$-\mathbf{t}_b$", color=FIL_B, fontsize=PT_ANNOT, ha="left",
            va="top", zorder=9)

    for x, text in ((left, "as they lie"), (right, "reversed, one origin")):
        ax.text(x, 0.322, text, ha="center", va="top", fontsize=PT_ANNOT,
                color=CHORD, zorder=9)


def panel_phi(ax):
    Y = ax.yspan
    u = unit(U_DEG)
    nrm = np.array([-u[1], u[0]])
    tip_a = np.array([0.185, 0.360 * Y])
    tip_b = tip_a + u * 0.52
    ta, tb = unit(A_DEG), unit(B_DEG)

    for tip, t, col, bend in ((tip_a, ta, FIL_A, 0.26), (tip_b, tb, FIL_B, -0.26)):
        arm = arm_to_tip(tip, t, 0.24, bend=bend)
        ax.plot(arm[:, 0], arm[:, 1], color=col, lw=2.1,
                solid_capstyle="round", zorder=3)
        ax.plot([tip[0]], [tip[1]], "o", ms=3.4, mfc="white", mec=col,
                mew=1.2, zorder=9)

    ax.plot(*np.array([tip_a, tip_b]).T, color=CHORD, lw=1.0,
            ls=(0, (3.4, 2.2)), zorder=2)
    arrow(ax, tip_a + u * 0.140, tip_a + u * 0.240, CHORD, lw=0.9, z=3)
    ax.text(*(tip_a + u * 0.192 - nrm * 0.048), r"$\mathbf{u}$",
            color=CHORD, fontsize=PT_ANNOT, ha="center", va="top")

    arrow(ax, tip_a, tip_a + ta * 0.19, FIL_A, lw=1.2)
    arrow(ax, tip_b, tip_b + tb * 0.19, FIL_B, lw=1.2)
    arrow(ax, tip_b, tip_b - u * 0.15, CHORD, lw=0.9, ls=(0, (2.4, 1.7)), z=3)
    ax.text(*(tip_a + ta * 0.215 + np.array([0.008, 0.004])),
            r"$\mathbf{t}_a$", color=FIL_A, fontsize=PT_ANNOT, ha="left",
            va="bottom")
    ax.text(*(tip_b + tb * 0.205 + nrm * 0.052), r"$\mathbf{t}_b$",
            color=FIL_B, fontsize=PT_ANNOT, ha="center", va="bottom")

    angle_arc(ax, tip_a, U_DEG, A_DEG, 0.130, r"$\varphi_a$", lab_r=0.215)
    angle_arc(ax, tip_b, U_DEG + 180.0, B_DEG, 0.130, r"$\varphi_b$",
              lab_r=0.235)

    off = -nrm * 0.120
    ax.annotate("", xy=tuple(tip_b + off), xytext=tuple(tip_a + off),
                arrowprops=dict(arrowstyle="<|-|>,head_length=0.46,head_width=0.26",
                                mutation_scale=8.5, color=CHORD, lw=0.8,
                                shrinkA=0, shrinkB=0))
    for tip in (tip_a, tip_b):
        ax.plot(*np.array([tip, tip + off * 1.12]).T, color=CHORD, lw=0.5,
                ls=(0, (1.6, 1.6)), zorder=1)
    mid = (tip_a + tip_b) / 2.0 + off
    ax.text(mid[0], mid[1] - 0.020, r"$d$", color=CHORD, fontsize=PT_ANNOT,
            ha="center", va="top")


def panel_q(ax, separations):
    d0 = 4.0
    d = np.linspace(0, 12, 400)
    ax.plot(d, np.minimum(1.0, d / d0), color=GEOM, lw=1.7, zorder=4)
    ax.axvline(d0, color=GEOM, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    for value in sorted(set(round(v, 2) for v in separations)):
        ax.plot([value], [0.0], marker="^", ms=3.2, color=INK, clip_on=False,
                zorder=6)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel(r"$d$ (px)", fontsize=PT_AXIS, labelpad=1.0)
    ax.set_ylabel(r"$q(d)$", fontsize=PT_AXIS, labelpad=1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 4, 8, 12])
    ax.tick_params(labelsize=PT_TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> int:
    plecta_style()
    separations = [row["d"] for row in load_figure_data()["junction4"]["pairs"]]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = stage(fig, [0.045, 0.560, 0.400, 0.330])
    ax_b = stage(fig, [0.505, 0.555, 0.470, 0.340])
    ax_c = stage(fig, [0.045, 0.080, 0.415, 0.350])
    ax_d = fig.add_axes([0.640, 0.155, 0.320, 0.275])

    panel_frame(ax_a)
    panel_theta(ax_b)
    panel_phi(ax_c)
    panel_q(ax_d, separations)

    #  Lower case, and the title itself in the regular weight: the bold tag
    #  carries the hierarchy, and Latin Modern bold sets about 15 % wider than
    #  its regular, so a fully bold title reads as stretched beside body text.
    title(fig, 0.030, 0.918, "a", "the frame at a stub")
    title(fig, 0.490, 0.918, "b", r"tangent reversal  $\theta$")
    title(fig, 0.030, 0.452, "c", r"chord turns  $\varphi_a,\ \varphi_b$")
    title(fig, 0.515, 0.452, "d", r"chord fade  $q(d)$")

    save_fig(fig, "fig_plecta_frame_cost", bbox_inches=None)
    plt.close(fig)
    print("tip separations at the crossing:",
          sorted(round(v, 2) for v in separations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
