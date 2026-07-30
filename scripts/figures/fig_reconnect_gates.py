"""FIG: schematic of stage-3 reconnection tip-to-tip join geometry and gates.

Three panels, each showing two filament fragments and a candidate join
between their facing tips:
  (a) short gap, accepted    -- near-collinear, tiny gap (short-gap pass)
  (b) extended gap, accepted -- collinear, longer gap, Hermite bridge drawn
      explicitly (extended-gap pass)
  (c) crossing, rejected     -- tips close but local orientations disagree
      strongly (steep crossing; vetoed by the orientation gate)

Replaces the old docs/figures/09_reconnect_gates.png dev schematic, which
used code identifiers (stage_clear/stage_strict) as labels, baked in
numeric thresholds, and encoded accept/reject by red/green text alone.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, save_fig, BLUE, ORANGE, GREEN, GRAY, DARK  # noqa: E402


FRAG_LW = 8.5
BRIDGE_LW = 2.6
TIP_MS = 7.0

XLIM = (0, 100)
YLIM = (0, 46)


def draw_fragment(ax, pts, color):
    """Thick rounded stroke through pts (Nx2 array)."""
    xs, ys = pts[:, 0], pts[:, 1]
    ax.plot(xs, ys, color=color, linewidth=FRAG_LW, solid_capstyle="round",
             solid_joinstyle="round", zorder=2)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def draw_tangent_arrow(ax, tip, inward_dir, color, length=13.0):
    """Arrow from the tip pointing inward into the fragment body.

    Drawn in DARK (not the fragment's own color) so it is visible on top
    of the thick fragment stroke -- an arrow the same color as the stroke
    it overlays would be invisible.
    """
    end = tip + inward_dir * length
    arr = FancyArrowPatch(tip, end, arrowstyle="-|>", mutation_scale=13,
                           color=DARK, linewidth=2.0, zorder=5,
                           shrinkA=0, shrinkB=0, capstyle="round")
    ax.add_patch(arr)
    return end


def draw_gap_arrow(ax, p_b, p_t):
    """Thin double-headed arrow spanning the gap between the two tips."""
    arr = FancyArrowPatch(p_b, p_t, arrowstyle="<->", mutation_scale=7,
                           color=DARK, linewidth=1.0, zorder=4)
    ax.add_patch(arr)


def hermite_bridge(p_b, d_b_in, p_t, d_t_in, n=60):
    """Cubic Hermite curve from p_b to p_t with inward tangents at each end,
    matching the manuscript's bridge construction (tangents -d_b, d_t)."""
    p0, p1 = np.asarray(p_b, float), np.asarray(p_t, float)
    scale = np.linalg.norm(p1 - p0)
    m0 = -d_b_in * scale
    m1 = d_t_in * scale
    t = np.linspace(0, 1, n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def panel_common(ax, title):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)


def tip_marker(ax, pt, color):
    ax.plot([pt[0]], [pt[1]], marker="o", markersize=TIP_MS,
             markerfacecolor=color, markeredgecolor=DARK,
             markeredgewidth=0.8, zorder=6)


def label(ax, pt, text, dx=0, dy=0, fontsize=9.5, ha="center", va="bottom",
          color=DARK, weight="normal"):
    ax.text(pt[0] + dx, pt[1] + dy, text, fontsize=fontsize, ha=ha,
            va=va, color=color, fontweight=weight, zorder=7)


def panel_a(ax):
    """Short gap, accepted: nearly collinear, tiny gap."""
    panel_common(ax, "(a) short gap, accepted")
    p_b_far = np.array([5, 11])
    p_b = np.array([44, 13])
    p_t = np.array([56, 13])
    p_t_far = np.array([95, 15])

    draw_fragment(ax, np.array([p_b_far, p_b]), BLUE)
    draw_fragment(ax, np.array([p_t, p_t_far]), ORANGE)

    d_b = unit(p_b_far - p_b)
    d_t = unit(p_t_far - p_t)

    # solid bridge = accepted (drawn first, under tangent arrows/tips)
    ax.plot([p_b[0], p_t[0]], [p_b[1], p_t[1]], color=GREEN,
             linewidth=BRIDGE_LW, solid_capstyle="round", zorder=3)

    end_b = draw_tangent_arrow(ax, p_b, d_b, BLUE)
    end_t = draw_tangent_arrow(ax, p_t, d_t, ORANGE)

    draw_gap_arrow(ax, p_b + np.array([0, 8]), p_t + np.array([0, 8]))
    label(ax, ((p_b + p_t) / 2), "dist", dy=10, fontsize=8.8)

    tip_marker(ax, p_b, BLUE)
    tip_marker(ax, p_t, ORANGE)
    label(ax, p_b, r"$\mathbf{p}_b$", dx=-4, dy=-2, va="top", fontsize=9.5)
    label(ax, p_t, r"$\mathbf{p}_t$", dx=4, dy=-2, va="top", fontsize=9.5)
    label(ax, end_b, r"$\hat{\mathbf{d}}_b$", dx=-2, dy=2, fontsize=9.5)
    label(ax, end_t, r"$\hat{\mathbf{d}}_t$", dx=2, dy=2, fontsize=9.5)


def panel_b(ax):
    """Extended gap, accepted: collinear, longer gap, explicit Hermite bridge."""
    panel_common(ax, "(b) extended gap, accepted")
    p_b_far = np.array([4, 9])
    p_b = np.array([32, 12])
    p_t = np.array([66, 16])
    p_t_far = np.array([96, 19])

    draw_fragment(ax, np.array([p_b_far, p_b]), BLUE)
    draw_fragment(ax, np.array([p_t, p_t_far]), ORANGE)

    d_b = unit(p_b_far - p_b)
    d_t = unit(p_t_far - p_t)

    bridge = hermite_bridge(p_b, d_b, p_t, d_t)
    ax.plot(bridge[:, 0], bridge[:, 1], color=GREEN, linewidth=BRIDGE_LW,
             solid_capstyle="round", zorder=3)

    end_b = draw_tangent_arrow(ax, p_b, d_b, BLUE)
    end_t = draw_tangent_arrow(ax, p_t, d_t, ORANGE)

    draw_gap_arrow(ax, p_b + np.array([0, 10]), p_t + np.array([0, 10]))
    label(ax, ((p_b + p_t) / 2), "dist", dy=12, fontsize=8.8)

    tip_marker(ax, p_b, BLUE)
    tip_marker(ax, p_t, ORANGE)
    label(ax, p_b, r"$\mathbf{p}_b$", dx=-3, dy=-2, va="top", fontsize=9.5)
    label(ax, p_t, r"$\mathbf{p}_t$", dx=3, dy=-2, va="top", fontsize=9.5)
    label(ax, end_b, r"$\hat{\mathbf{d}}_b$", dx=-2, dy=2, fontsize=9.5)
    label(ax, end_t, r"$\hat{\mathbf{d}}_t$", dx=2, dy=2, fontsize=9.5)


def panel_c(ax):
    """Crossing, rejected: steep angle, tips close, orientations disagree."""
    panel_common(ax, "(c) crossing, rejected")
    p_b_far = np.array([4, 9])
    p_b = np.array([38, 12])
    p_t = np.array([55, 23])
    p_t_far = np.array([63, 43])
    join = (p_b + p_t) / 2

    # light circular window around the join (drawn first, under everything)
    win = Circle(join, 16, facecolor="none", edgecolor=GRAY, linewidth=1.1,
                 linestyle=(0, (3, 2)), zorder=1)
    ax.add_patch(win)

    draw_fragment(ax, np.array([p_b_far, p_b]), BLUE)
    draw_fragment(ax, np.array([p_t, p_t_far]), ORANGE)

    d_b = unit(p_b_far - p_b)
    d_t = unit(p_t_far - p_t)

    # dashed, crossed-out bridge = rejected
    ax.plot([p_b[0], p_t[0]], [p_b[1], p_t[1]], color=GRAY,
             linewidth=BRIDGE_LW, linestyle=(0, (2.2, 2.2)),
             solid_capstyle="round", zorder=3)
    mid = (p_b + p_t) / 2
    for dsign in (1, -1):
        ax.plot([mid[0] - 2.0, mid[0] + 2.0],
                 [mid[1] + dsign * 2.0, mid[1] - dsign * 2.0],
                 color=GRAY, linewidth=2.0, zorder=4, solid_capstyle="round")

    end_b = draw_tangent_arrow(ax, p_b, d_b, BLUE)
    end_t = draw_tangent_arrow(ax, p_t, d_t, ORANGE)

    # gap arrow offset well to the lower-right of the p_b-p_t segment -- the
    # orientation-mismatch arc/label (below) sit on the opposite,
    # upper-left side, so the two annotations stay clearly separated
    gap_vec = p_t - p_b
    normal = unit(np.array([-gap_vec[1], gap_vec[0]]))
    if normal[1] < 0:
        normal = -normal
    off = -normal * 9.0
    draw_gap_arrow(ax, p_b + off, p_t + off)
    dist_mid = (p_b + p_t) / 2 + off
    label(ax, dist_mid, "dist", dx=0, dy=-2, va="top", ha="center",
          fontsize=8.8)

    # angle arc showing local-orientation mismatch (draw the minor arc
    # between the two inward tangent directions, not the reflex one),
    # kept well inside the local window circle so it cannot touch the
    # circle's stroke; the label sits clearly outside the circle, joined
    # to the arc by a short leader line so it reads as one annotation
    ang_b = np.degrees(np.arctan2(d_b[1], d_b[0]))
    ang_t = np.degrees(np.arctan2(d_t[1], d_t[0]))
    ang_t = ang_b + ((ang_t - ang_b + 180) % 360 - 180)
    arc_r = 7
    arc = Arc(join, 2 * arc_r, 2 * arc_r, angle=0, theta1=min(ang_b, ang_t),
              theta2=max(ang_b, ang_t), color=DARK, linewidth=1.3, zorder=5)
    ax.add_patch(arc)
    theta_mid = np.radians((ang_b + ang_t) / 2)
    dirn = np.array([np.cos(theta_mid), np.sin(theta_mid)])
    circle_r = 16
    leader_start = join + (circle_r + 1.0) * dirn
    leader_end = join + (circle_r + 7.0) * dirn
    ax.plot([leader_start[0], leader_end[0]], [leader_start[1], leader_end[1]],
             color=GRAY, linewidth=0.9, zorder=5)
    label_pt = join + (circle_r + 9.5) * dirn
    label(ax, label_pt, r"$\Delta\theta_{\mathrm{loc}}$", dx=0, dy=0,
          ha="center", va="center", fontsize=9.3)

    tip_marker(ax, p_b, BLUE)
    tip_marker(ax, p_t, ORANGE)
    label(ax, p_b, r"$\mathbf{p}_b$", dx=-3, dy=-8, va="top", ha="right",
          fontsize=9.5)
    label(ax, p_t, r"$\mathbf{p}_t$", dx=8, dy=-2, ha="left", va="top",
          fontsize=9.5)
    label(ax, end_b, r"$\hat{\mathbf{d}}_b$", dx=-2, dy=2, fontsize=9.5)
    label(ax, end_t, r"$\hat{\mathbf{d}}_t$", dx=3, dy=1, ha="left", fontsize=9.5)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    legend_elems = [
        Line2D([0], [0], color=BLUE, lw=FRAG_LW * 0.5, solid_capstyle="round",
               label="fragment (base tip)"),
        Line2D([0], [0], color=ORANGE, lw=FRAG_LW * 0.5, solid_capstyle="round",
               label="fragment (target tip)"),
        Line2D([0], [0], color=GREEN, lw=BRIDGE_LW, label="accepted join (solid bridge)"),
        Line2D([0], [0], color=GRAY, lw=BRIDGE_LW, linestyle=(0, (2.2, 2.2)),
               label="rejected join (dashed, crossed out)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=8.8,
               handlelength=2.0, columnspacing=1.4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.20, wspace=0.10)
    save_fig(fig, "fig_reconnect_gates", bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    main()
