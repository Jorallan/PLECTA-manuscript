"""Figure: the four decisions PLECTA can make, and the gate on each.

Built in the same idiom as the earlier reconnection-gate figure in this
repository -- thick round-capped arms, tip markers, direction arrows, an
accepted link drawn solid and a refused one dashed and crossed out, and one
legend for the whole row.  The four panels are the four outcomes the linker
can produce, with the numerical gate printed under each.

Idealised drawings.  The real numbers behind them are in Figs. "frame_cost"
and "exact_matching", which use measured values throughout; this figure is
the vocabulary, not the evidence.  Self-contained; no data file is read.

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
from matplotlib.patches import Arc, Circle, FancyArrowPatch

from _style import (BLUE, DARK, FIG_W, GRAY, GREEN, JUNCTION, ORANGE,
                    PT_BODY, PT_LABEL, PT_TINY, plecta_style, save_fig)

FIG_H = 2.55
DEG = math.pi / 180.0
ARM_LW = 5.4
HEAD = dict(arrowstyle="-|>,head_length=0.48,head_width=0.28",
            mutation_scale=10, shrinkA=0, shrinkB=0)


def unit(deg):
    return np.array([math.cos(deg * DEG), math.sin(deg * DEG)])


def arm(ax, tip, bearing, length, colour, bend=0.0, z=3):
    """A fragment drawn *backwards* from its tip, so the tip is exact."""
    t = unit(bearing)
    nrm = np.array([-t[1], t[0]])
    s = np.linspace(0, 1, 60)[:, None]
    pts = tip - t * length * s + nrm * bend * length * (s ** 2)
    ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=ARM_LW,
            solid_capstyle="round", zorder=z, alpha=0.95)
    return np.asarray(tip, float)


def tip_dot(ax, p, colour):
    ax.add_patch(Circle(tuple(p), 0.030, facecolor="white", edgecolor=colour,
                        linewidth=1.5, zorder=8))


def dir_arrow(ax, p, bearing, length=0.20, colour=DARK):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + unit(bearing) * length),
                                 color=colour, lw=1.3, zorder=7, **HEAD))


def accepted(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREEN, lw=2.2,
            solid_capstyle="round", zorder=6)


def refused(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRAY, lw=1.6,
            ls=(0, (3.4, 2.4)), zorder=5)
    mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
    ax.plot([mid[0]], [mid[1]], marker="x", ms=7.5, mew=2.0, color=GRAY,
            zorder=9)


def node_disc(ax, centre, r=0.075):
    ax.add_patch(Circle(tuple(centre), r, facecolor=JUNCTION,
                        edgecolor="none", alpha=0.85, zorder=4))


def angle_arc(ax, vertex, a_deg, b_deg, radius, label):
    lo, hi = sorted((a_deg % 360.0, b_deg % 360.0))
    if hi - lo > 180.0:
        lo, hi = hi - 360.0, lo
    ax.add_patch(Arc(tuple(vertex), 2 * radius, 2 * radius, theta1=lo,
                     theta2=hi, color=GREEN, lw=1.3, zorder=8))
    pos = np.asarray(vertex) + radius * 1.62 * unit((lo + hi) / 2.0)
    ax.text(pos[0], pos[1], label, ha="center", va="center", color=GREEN,
            fontsize=PT_LABEL, zorder=9)


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
#
#  In every panel the stub tips sit outside the junction disc, so the link
#  the method draws between them is visible rather than buried under it.

R_TIP = 0.135
R_NODE = 0.070


def junction(ax, centre, bearings, colours, lengths, bends):
    """A junction disc with its stubs, and the tip of each stub."""
    ax.add_patch(Circle(tuple(centre), R_NODE, facecolor=JUNCTION,
                        edgecolor="none", alpha=0.80, zorder=2))
    tips = []
    for bearing, colour, length, bend in zip(bearings, colours, lengths, bends):
        tip = np.asarray(centre) + unit(bearing) * R_TIP
        arm(ax, tip, bearing, length, colour, bend=bend)
        tip_dot(ax, tip, colour)
        tips.append(tip)
    return tips


def panel_pair_taken(ax):
    Y = ax.yspan
    node = np.array([0.50, 0.52 * Y])
    ta, tb = junction(ax, node, [196, 16], [BLUE, ORANGE], [0.30, 0.30],
                      [0.10, -0.10])
    accepted(ax, ta, tb)
    dir_arrow(ax, ta, 196, 0.15)
    dir_arrow(ax, tb, 16, 0.15)


def panel_pair_refused(ax):
    Y = ax.yspan
    node = np.array([0.50, 0.52 * Y])
    ta, tb = junction(ax, node, [200, 104], [BLUE, ORANGE], [0.28, 0.26],
                      [0.10, -0.08])
    refused(ax, ta, tb)
    dir_arrow(ax, ta, 200, 0.15)
    dir_arrow(ax, tb, 104, 0.15)


def panel_unmatched(ax):
    Y = ax.yspan
    node = np.array([0.52, 0.56 * Y])
    ta, tb, tc = junction(ax, node, [192, 12, 278], [BLUE, BLUE, ORANGE],
                          [0.28, 0.28, 0.22], [0.05, -0.05, 0.08])
    accepted(ax, ta, tb)
    ax.add_patch(Circle(tuple(tc), 0.058, facecolor="none", edgecolor=JUNCTION,
                        linewidth=1.7, zorder=9))
    ax.text(tc[0] - 0.085, tc[1] - 0.01, r"free", ha="right", va="center",
            fontsize=PT_TINY, color=JUNCTION)


def panel_gap(ax):
    Y = ax.yspan
    ta = np.array([0.28, 0.44 * Y])
    tb = np.array([0.62, 0.54 * Y])
    arm(ax, ta, 15, 0.26, BLUE, bend=0.06)
    arm(ax, tb, 195, 0.26, ORANGE, bend=-0.06)
    tip_dot(ax, ta, BLUE)
    tip_dot(ax, tb, ORANGE)
    accepted(ax, ta, tb)
    dir_arrow(ax, ta, 15, 0.13)
    dir_arrow(ax, tb, 195, 0.13)
    off = np.array([0.0, 0.115])
    ax.annotate("", xy=tuple(tb + off), xytext=tuple(ta + off),
                arrowprops=dict(arrowstyle="<|-|>,head_length=0.5,head_width=0.28",
                                mutation_scale=9, color=GRAY, lw=0.9,
                                shrinkA=0, shrinkB=0))
    mid = (ta + tb) / 2.0 + off
    ax.text(mid[0], mid[1] + 0.030, r"$d$", ha="center", va="bottom",
            fontsize=PT_LABEL, color=GRAY)


def main() -> int:
    plecta_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    panels = [
        (panel_pair_taken, "a", "Pairing taken",
         r"$C_{ab} = 0.15 < 1.24$", GREEN),
        (panel_pair_refused, "b", "Pairing refused",
         r"$C_{ab} = 1.72 \geq 2p_i = 1.24$", GRAY),
        (panel_unmatched, "c", "Stub left free",
         "$p_i = 0.62$ beats every\nedge this stub has", JUNCTION),
        (panel_gap, "d", "Gap bridged",
         r"$d \leq 85$ px,  $\theta \leq 0.40$ rad,"
         "\n" r"$\varphi \leq 0.28$ rad,  $C_{ab} < 0.60$", GREEN),
    ]
    width, left, gap = 0.232, 0.016, 0.016
    for i, (fn, letter, title, gate, colour) in enumerate(panels):
        x0 = left + i * (width + gap)
        fn(stage(fig, [x0, 0.400, width, 0.400]))
        centre = x0 + width / 2.0
        fig.text(centre, 0.865, f"({letter})  {title}", fontsize=PT_BODY,
                 fontweight="bold", color=DARK, ha="center", va="bottom")
        fig.text(centre, 0.372, gate, fontsize=PT_LABEL, color=colour,
                 ha="center", va="top", linespacing=1.5)

    handles = [
        Line2D([], [], color=BLUE, lw=ARM_LW, solid_capstyle="round",
               label="Arm of filament A"),
        Line2D([], [], color=ORANGE, lw=ARM_LW, solid_capstyle="round",
               label="Arm of filament B"),
        Line2D([], [], color=GREEN, lw=2.2, label="Link accepted"),
        Line2D([], [], color=GRAY, lw=1.6, ls=(0, (3.4, 2.4)), marker="x",
               ms=6.5, mew=1.8, label="Link refused"),
        Line2D([], [], color=JUNCTION, lw=0, marker="o", ms=7.0,
               label="Junction cluster"),
    ]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), fontsize=PT_LABEL,
               handlelength=2.0, columnspacing=1.5, handletextpad=0.6)

    save_fig(fig, "fig_plecta_gates", bbox_inches=None)
    plt.close(fig)
    print("saved fig_plecta_gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

