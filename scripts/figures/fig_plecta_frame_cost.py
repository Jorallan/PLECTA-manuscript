"""Figure: the stub frame and the four terms of the pairing cost.

Every angle arc is drawn at the vertex the angle is actually measured at, and
the reversed tangent that theta is measured against is drawn explicitly, at the
common origin where the angle is read.  The drawn configuration is stated
numerically in the panel titles so a reader can check the arcs against the
numbers.

Panel (d) marks the real tip separations measured at the crossing in
``results/plecta_figure_data.json``; everything else is analytic.

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

from _style import (FIG_W, PT_BODY, PT_LABEL, PT_TINY, PT_TITLE,
                    FIL_A, FIL_B, GEOM, CHORD, INK,
                    load_figure_data, plecta_style, save_fig)

FIG_H = 5.85
DEG = math.pi / 180.0
HEAD = dict(arrowstyle="-|>,head_length=0.46,head_width=0.26",
            mutation_scale=8.5, shrinkA=0, shrinkB=0)

# The drawn configuration.  With alpha the bearing of t_a, beta that of -t_b
# and gamma that of the chord, theta = |alpha - beta|, phi_a = |alpha - gamma|
# and phi_b = |beta - gamma|; fixing three bearings therefore fixes all three
# angles exactly, and the arcs below are drawn from the same three numbers.
A_DEG, B_DEG, U_DEG = 38.0, 186.0, 20.0
THETA = abs(A_DEG - (B_DEG - 180.0))        # 32 deg
PHI_A = abs(A_DEG - U_DEG)                  # 18 deg
PHI_B = abs((B_DEG - 180.0) - U_DEG)        # 14 deg


def unit(deg):
    return np.array([math.cos(deg * DEG), math.sin(deg * DEG)])


def arrow(ax, p0, p1, colour, lw=1.2, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), lw=lw, ls=ls,
                                 color=colour, zorder=z, **HEAD))


def angle_arc(ax, vertex, a_deg, b_deg, radius, label, colour=GEOM,
              lab_r=None, fs=PT_BODY):
    lo, hi = sorted((a_deg % 360.0, b_deg % 360.0))
    if hi - lo > 180.0:
        lo, hi = hi - 360.0, lo
    ax.add_patch(Arc(tuple(vertex), 2 * radius, 2 * radius, angle=0.0,
                     theta1=lo, theta2=hi, color=colour, lw=1.2, zorder=7))
    pos = np.asarray(vertex) + (lab_r or radius * 1.55) * unit((lo + hi) / 2.0)
    ax.text(pos[0], pos[1], label, ha="center", va="center", color=colour,
            fontsize=fs, zorder=8)


def bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def arm_to_tip(tip, tangent, length, bend=0.0):
    """A smooth arm arriving at ``tip`` with outward direction ``tangent``.

    Built backwards from the tip, so the tangent the annotation names is
    exactly the tangent the drawing has.
    """
    tip = np.asarray(tip, float)
    t = np.asarray(tangent, float) / np.hypot(*tangent)
    nrm = np.array([-t[1], t[0]])
    return bezier(tip, tip - t * length * 0.34,
                  tip - t * length * 0.70 + nrm * bend * length * 0.5,
                  tip - t * length + nrm * bend * length)


def stage(fig, rect, xspan=1.0):
    """A drawing panel whose data limits already match its printed aspect.

    Setting the limits to the box ratio means ``aspect='equal'`` never has to
    shrink the axes or widen the limits, so an angle drawn at n degrees is
    printed at n degrees and the composition is not silently rescaled.
    """
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
    fig.text(x + 0.042, y, text, fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")


def note(fig, x, y, text):
    fig.text(x, y, text, fontsize=PT_TINY, color=INK, ha="left", va="top",
             linespacing=1.55)


# ── panels ────────────────────────────────────────────────────────────────


def panel_frame(ax):
    Y = ax.yspan
    tip = np.array([0.60, 0.60 * Y])
    t_dir = unit(32)
    arm = arm_to_tip(tip, t_dir, 0.62, bend=0.34)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arm, axis=0).T))])
    inside = s <= 0.30

    ax.plot(arm[inside, 0], arm[inside, 1], color=GEOM, lw=6.5, alpha=0.25,
            solid_capstyle="round", zorder=2)
    ax.plot(arm[:, 0], arm[:, 1], color=FIL_A, lw=2.3, solid_capstyle="round",
            zorder=3)

    ss = np.linspace(0, 0.30, 60)[:, None]
    nrm = np.array([-t_dir[1], t_dir[0]])
    ax.plot(*(tip + ss * t_dir + (ss ** 2) * 2.1 * nrm).T, color=GEOM, lw=1.1,
            ls=(0, (3.0, 2.0)), zorder=4)

    arrow(ax, tip, tip + t_dir * 0.24, FIL_A, lw=1.3)
    ax.plot([tip[0]], [tip[1]], "o", ms=4.0, mfc="white", mec=FIL_A, mew=1.3,
            zorder=9)

    ax.text(*(tip + t_dir * 0.27 + np.array([0.02, -0.01])), r"$\mathbf{t}$",
            color=FIL_A, fontsize=PT_BODY, ha="left", va="top",
            fontweight="bold")
    ax.text(0.735, 0.955 * Y, r"$\kappa$", color=GEOM, fontsize=PT_BODY,
            ha="left", va="top")
    ax.text(0.245, 0.175 * Y, "fit window $w$", color=GEOM, fontsize=PT_LABEL,
            ha="center", va="top")
    ax.text(tip[0] + 0.035, tip[1] - 0.015, "tip, $s=0$", color=INK,
            fontsize=PT_TINY, ha="left", va="top")


def panel_theta(ax):
    Y = ax.yspan
    # left: the crossing, with each arm's own outward tangent
    tip_a = np.array([0.180, 0.40 * Y])
    tip_b = np.array([0.255, 0.50 * Y])
    ta, tb = unit(A_DEG), unit(B_DEG)
    for tip, t, col, bend in ((tip_a, ta, FIL_A, 0.20), (tip_b, tb, FIL_B, -0.20)):
        arm = arm_to_tip(tip, t, 0.23, bend=bend)
        ax.plot(arm[:, 0], arm[:, 1], color=col, lw=2.1,
                solid_capstyle="round", zorder=3)
        ax.plot([tip[0]], [tip[1]], "o", ms=3.4, mfc="white", mec=col,
                mew=1.2, zorder=9)
    arrow(ax, tip_a, tip_a + ta * 0.15, FIL_A, lw=1.2)
    arrow(ax, tip_b, tip_b + tb * 0.15, FIL_B, lw=1.2)
    ax.text(*(tip_a + ta * 0.18 + np.array([0.004, 0.008])), r"$\mathbf{t}_a$",
            color=FIL_A, fontsize=PT_LABEL, ha="left", va="bottom")
    ax.text(*(tip_b + tb * 0.18 + np.array([0.0, -0.012])), r"$\mathbf{t}_b$",
            color=FIL_B, fontsize=PT_LABEL, ha="center", va="top")
    ax.text(0.200, 0.035 * Y, "at the crossing", color=CHORD,
            fontsize=PT_TINY, ha="center", va="bottom")

    ax.plot([0.430, 0.430], [0.10 * Y, 0.92 * Y], color=CHORD, lw=0.6,
            ls=(0, (2.2, 2.0)), zorder=1)

    # right: the same two directions from one origin, which is where theta
    # is read.  No enclosing circle: it only clipped and added no information.
    dial = np.array([0.660, 0.46 * Y])
    R = 0.20
    ax.plot([dial[0]], [dial[1]], "o", ms=2.8, color=CHORD, zorder=9)
    arrow(ax, dial, dial + unit(A_DEG) * R, FIL_A, lw=1.2, z=8)
    arrow(ax, dial, dial + unit(B_DEG) * R * 0.62, FIL_B, lw=1.2, z=8)
    arrow(ax, dial, dial + unit(B_DEG - 180.0) * R, FIL_B, lw=1.1,
          ls=(0, (2.6, 1.8)), z=8)
    angle_arc(ax, dial, B_DEG - 180.0, A_DEG, R * 0.52, r"$\theta$",
              lab_r=R * 0.82)
    ax.text(*(dial + unit(A_DEG) * R + np.array([0.014, 0.006])),
            r"$\mathbf{t}_a$", color=FIL_A, fontsize=PT_LABEL, ha="left",
            va="bottom", zorder=9)
    ax.text(*(dial + unit(B_DEG) * R * 0.62 + np.array([-0.014, -0.004])),
            r"$\mathbf{t}_b$", color=FIL_B, fontsize=PT_LABEL, ha="right",
            va="top", zorder=9)
    ax.text(*(dial + unit(B_DEG - 180.0) * R + np.array([0.016, -0.004])),
            r"$-\mathbf{t}_b$", color=FIL_B, fontsize=PT_LABEL, ha="left",
            va="top", zorder=9)
    ax.text(0.705, 0.035 * Y, "directions, one origin", color=CHORD,
            fontsize=PT_TINY, ha="center", va="bottom")


def panel_phi(ax):
    Y = ax.yspan
    u = unit(U_DEG)
    nrm = np.array([-u[1], u[0]])
    tip_a = np.array([0.175, 0.42 * Y])
    tip_b = tip_a + u * 0.60
    ta, tb = unit(A_DEG), unit(B_DEG)

    for tip, t, col, bend in ((tip_a, ta, FIL_A, 0.26), (tip_b, tb, FIL_B, -0.26)):
        arm = arm_to_tip(tip, t, 0.30, bend=bend)
        ax.plot(arm[:, 0], arm[:, 1], color=col, lw=2.1,
                solid_capstyle="round", zorder=3)
        ax.plot([tip[0]], [tip[1]], "o", ms=3.4, mfc="white", mec=col,
                mew=1.2, zorder=9)

    ax.plot(*np.array([tip_a, tip_b]).T, color=CHORD, lw=1.0,
            ls=(0, (3.4, 2.2)), zorder=2)
    arrow(ax, tip_a + u * 0.155, tip_a + u * 0.265, CHORD, lw=0.9, z=3)
    ax.text(*(tip_a + u * 0.215 - nrm * 0.052), r"$\mathbf{u}$",
            color=CHORD, fontsize=PT_LABEL, ha="center", va="top")

    arrow(ax, tip_a, tip_a + ta * 0.21, FIL_A, lw=1.2)
    arrow(ax, tip_b, tip_b + tb * 0.21, FIL_B, lw=1.2)
    arrow(ax, tip_b, tip_b - u * 0.17, CHORD, lw=0.9, ls=(0, (2.4, 1.7)), z=3)
    ax.text(*(tip_a + ta * 0.235 + np.array([0.006, 0.006])),
            r"$\mathbf{t}_a$", color=FIL_A, fontsize=PT_LABEL, ha="left",
            va="bottom")
    ax.text(*(tip_b + tb * 0.235 + nrm * 0.038), r"$\mathbf{t}_b$",
            color=FIL_B, fontsize=PT_LABEL, ha="center", va="bottom")
    ax.text(*(tip_b - u * 0.150 - nrm * 0.052), r"$-\mathbf{u}$",
            color=CHORD, fontsize=PT_LABEL, ha="center", va="top")

    angle_arc(ax, tip_a, U_DEG, A_DEG, 0.135, r"$\varphi_a$", lab_r=0.215)
    angle_arc(ax, tip_b, U_DEG + 180.0, B_DEG, 0.135, r"$\varphi_b$",
              lab_r=0.235)

    off = -nrm * 0.125
    ax.annotate("", xy=tuple(tip_b + off), xytext=tuple(tip_a + off),
                arrowprops=dict(arrowstyle="<|-|>,head_length=0.46,head_width=0.26",
                                mutation_scale=8.5, color=CHORD, lw=0.8,
                                shrinkA=0, shrinkB=0))
    for tip in (tip_a, tip_b):
        ax.plot(*np.array([tip, tip + off * 1.12]).T, color=CHORD, lw=0.5,
                ls=(0, (1.6, 1.6)), zorder=1)
    mid = (tip_a + tip_b) / 2.0 + off
    ax.text(mid[0], mid[1] - 0.020, r"$d$", color=CHORD, fontsize=PT_BODY,
            ha="center", va="top")


def panel_q(ax, separations):
    d0 = 4.0
    d = np.linspace(0, 12, 400)
    ax.plot(d, np.minimum(1.0, d / d0), color=GEOM, lw=1.8, zorder=4)
    ax.axvline(d0, color=GEOM, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax.text(d0 + 0.35, 0.13, r"$d_0 = 4$ px", color=GEOM, fontsize=PT_TINY,
            ha="left", va="center")
    for value in sorted(set(round(v, 2) for v in separations)):
        ax.plot([value], [0.0], marker="^", ms=3.4, color=INK, clip_on=False,
                zorder=6)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("tip separation $d$ (px)", fontsize=PT_LABEL, labelpad=1.0)
    ax.set_ylabel(r"$q(d)$", fontsize=PT_BODY, labelpad=1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 4, 8, 12])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> int:
    plecta_style()
    separations = [row["d"] for row in load_figure_data()["junction4"]["pairs"]]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = stage(fig, [0.048, 0.735, 0.400, 0.200])
    ax_b = stage(fig, [0.535, 0.735, 0.430, 0.200])
    ax_c = stage(fig, [0.048, 0.352, 0.415, 0.205])
    ax_d = fig.add_axes([0.625, 0.393, 0.330, 0.160])

    panel_frame(ax_a)
    panel_theta(ax_b)
    panel_phi(ax_c)
    panel_q(ax_d, separations)

    title(fig, 0.030, 0.952, "a", "Frame at a stub")
    title(fig, 0.517, 0.952, "b",
          r"Direct term:  $\theta = %.0f^\circ$ here" % THETA)
    title(fig, 0.030, 0.585, "c",
          r"Chord terms:  $\varphi_a = %.0f^\circ$, $\varphi_b = %.0f^\circ$ here"
          % (PHI_A, PHI_B))
    title(fig, 0.517, 0.585, "d", r"Chord confidence  $q(d)$")

    note(fig, 0.048, 0.722,
         "Each coordinate is fitted by least squares in\n"
         "arclength, tip at $s=0$, then read outward.\n"
         "Quadratic if the span is $\\geq 18$ px, else linear.\n"
         "$w=24$ px locally, $55$ px along a chain.")
    note(fig, 0.535, 0.722,
         r"$\theta=\arccos(\mathbf{t}_a\cdot-\mathbf{t}_b)$ compares two"
         "\ndirections and never uses the chord, so it\n"
         "still means something when the tips are\n"
         "three pixels apart.")
    note(fig, 0.048, 0.338,
         "The two turns onto the chord $\\mathbf{u}$ joining the\n"
         "tips. Unlike $\\theta$ this sees lateral offset, so\n"
         "parallel arms that never meet score well on\n"
         "$\\theta$ and badly here.")
    note(fig, 0.535, 0.290,
         "Inside a crossing the chord spans a few\n"
         "pixels and is badly quantised, so $q$ fades the\n"
         "$\\varphi$ term out. Triangles: the six real tip\n"
         "separations at the crossing of Fig. 5.")

    fig.text(0.5, 0.152,
             r"$C_{ab}\;=\;w_d\,\theta\;+\;w_t\,q(d)\,(\varphi_a+\varphi_b)"
             r"\;+\;w_\ell\,d/\ell_0\;+\;30\,w_\kappa\,|\kappa_a+\kappa_b|$",
             ha="center", va="center", fontsize=PT_BODY + 1.5, color=INK)
    fig.text(0.5, 0.086,
             "junction:  $w_d\\!=\\!0.6$,  $w_t\\!=\\!1.2$,  "
             "$w_\\kappa\\!=\\!1.0$,  $d_0\\!=\\!4$ px,  $w_\\ell\\!=\\!0$",
             ha="center", va="center", fontsize=PT_LABEL, color=INK)
    fig.text(0.5, 0.034,
             "gap:  $w_d\\!=\\!0.2$,  $w_t\\!=\\!1.0$,  $w_\\kappa\\!=\\!0.5$,  "
             "$d_0\\!=\\!3$ px,  $w_\\ell\\!=\\!0.15$,  $\\ell_0\\!=\\!60$ px",
             ha="center", va="center", fontsize=PT_LABEL, color=INK)

    save_fig(fig, "fig_plecta_frame_cost", bbox_inches=None)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
