"""Figure: what the depth stage measures at one crossing.

Section 2.5.2 says the stage reads a *core* arc shared by two strands and each
strand's own *flanks*, kept clear of the other's stroke, and asks which
strand's flank brightness the core matches.  This draws that, on a real
crossing rather than on two ruled lines:

  (a) the greyscale around a real 90-degree crossing, with both centrelines,
      the core arc, each strand's flank window, and a dot at every position
      actually sampled -- all of it read back from ``plecta.depth`` itself, so
      the dots sit where the measurement happened and nowhere else;
  (b) the grey levels those dots returned, against signed arclength, with the
      two flank medians, the pooled core median and the noise band that
      floors the denominator.

Panel (b) is the argument: the core median lands on the upper strand's flank
level and away from the lower one's, which is the occlusion cue, and the gap
between them is what the noise floor is compared against.

The crossing is chosen once and named in CROSSING below: confident but not
saturated, and near-perpendicular so the two windows are legible.  Nothing is
tuned here; the figure reports a measurement made with the shipped parameters.

    python scripts/figures/fig_depth_evidence.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import (FIG_W, INK, FIL_A, FIL_B, GEOM, GRAY, LIGHT_GRAY,
                    PT_ANNOT, PT_AXIS, PT_TITLE, plecta_style, save_fig,
                    thin_spines)

PLECTA_ROOT = r"C:\Repos\PLECTA"
SCENE = os.path.join(r"C:\Repos\filaments_quantification\input",
                     "synthetic_depth_dev", "cov35", "synth_0002")
CROSSING = (24, 38)         # instance ids. Chosen for legibility, not at
                            #  random: 83.8 deg so both windows are visible,
                            #  flank levels 23x their own spread apart so the
                            #  medians separate on the page, and a decisive
                            #  score that is still short of saturation. The
                            #  caption says it is an example.
HALF = 46                   # px half-width of the image crop drawn in (a)


def measure():
    """Core and flank samples at one crossing, from plecta.depth itself."""
    sys.path.insert(0, PLECTA_ROOT)
    from PIL import Image
    import plecta.depth as D
    from plecta.parameters import build

    params = build(D.DepthParams)
    gt = json.load(open(os.path.join(SCENE, "gt_depth.json")))
    image = np.asarray(Image.open(os.path.join(SCENE, "sem.png")).convert("L"),
                       dtype=float) / 255.0
    lines = {int(a["id"]): np.asarray(a["centerline_xy"], float)
             for a in gt["instances"]}

    i, j = CROSSING
    truth = next(c for c in gt["crossings"]
                 if {int(c["i"]), int(c["j"])} == {i, j})
    x, y = float(truth["x"]), float(truth["y"])

    crossing = D.PredCrossing(i=i, j=j, x=x, y=y)
    D.crossing_evidence(image, None, lines, crossing, params)

    #  The same split the evidence path used, re-run so the drawn dots are the
    #  sampled positions rather than a redrawing of them.
    window = max(params.window_px, 3.0 * params.core_px)
    out = {"x": x, "y": y, "core_px": params.core_px, "window": window,
           "score": crossing.score, "features": dict(crossing.features),
           "over_truth": int(truth["over"]), "i": i, "j": j,
           "angle": float(truth["angle_deg"]), "image": image}
    for key, sid in (("i", i), ("j", j)):
        dense = D._densify(lines[sid]) if hasattr(D, "_densify") else lines[sid]
        core_pts, flank_pts, _ = D._arc_window(dense, x, y, window,
                                               params.core_px)
        out[key] = {
            "dense": dense,
            "core": core_pts, "flank": flank_pts,
            "core_grey": D._sample_image(image, core_pts),
            "flank_grey": D._sample_image(image, flank_pts),
        }
    return out


def signed_arclength(dense, pts, x, y):
    """Arclength of `pts` along `dense`, signed about the point nearest (x,y)."""
    step = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(dense, axis=0).T))])
    k0 = int(np.argmin(np.hypot(dense[:, 0] - x, dense[:, 1] - y)))
    out = []
    for p in pts:
        k = int(np.argmin(np.hypot(dense[:, 0] - p[0], dense[:, 1] - p[1])))
        out.append(step[k] - step[k0])
    return np.asarray(out)


def panel_geometry(ax, d):
    x, y = d["x"], d["y"]
    ax.imshow(d["image"], cmap="gray", vmin=0, vmax=1, origin="upper",
              interpolation="nearest", zorder=0)
    ax.set_xlim(x - HALF, x + HALF)
    ax.set_ylim(y + HALF, y - HALF)

    for key, colour, label in (("i", FIL_A, "strand $i$"),
                               ("j", FIL_B, "strand $j$")):
        s = d[key]
        ax.plot(s["dense"][:, 0], s["dense"][:, 1], color=colour, lw=0.9,
                ls=(0, (3, 3)), zorder=3)
        ax.plot(s["flank"][:, 0], s["flank"][:, 1], "o", ms=2.2, mfc=colour,
                mec="white", mew=0.35, zorder=5, label=label)
        ax.plot(s["core"][:, 0], s["core"][:, 1], "o", ms=2.2, mfc="white",
                mec=INK, mew=0.6, zorder=6)

    #  The core is an arclength about the intersection, so it draws as a disc
    #  of that radius: every sampled point inside it is a core point.
    ax.add_patch(plt.Circle((x, y), d["core_px"], fill=False, ec=INK, lw=0.9,
                            ls=(0, (2, 2)), zorder=4))
    ax.plot([x], [y], "+", color=INK, ms=5, mew=0.9, zorder=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LIGHT_GRAY)
        sp.set_linewidth(0.6)


def panel_levels(ax, d):
    """Every sampled grey against where along the strand it was read."""
    sigma = float(d["features"]["sigma_hat"])
    m = {}
    for key, colour in (("i", FIL_A), ("j", FIL_B)):
        s = d[key]
        t = signed_arclength(s["dense"], s["flank"], d["x"], d["y"])
        ax.plot(t, s["flank_grey"], "o", ms=2.4, mfc=colour, mec="white",
                mew=0.35, zorder=4)
        m[key] = float(np.median(s["flank_grey"]))
        ax.axhline(m[key], color=colour, lw=0.9, zorder=3)

    core_grey = np.concatenate([d["i"]["core_grey"], d["j"]["core_grey"]])
    core_t = np.concatenate([
        signed_arclength(d[k]["dense"], d[k]["core"], d["x"], d["y"])
        for k in ("i", "j")])
    m_c = float(np.median(core_grey))
    ax.plot(core_t, core_grey, "o", ms=2.4, mfc="white", mec=INK, mew=0.6,
            zorder=5)
    ax.axhline(m_c, color=INK, lw=1.1, zorder=6)

    ax.axvspan(-d["core_px"], d["core_px"], color=GRAY, alpha=0.10, lw=0,
               zorder=0)
    ax.annotate("core", (0, 0.0), xytext=(0, 3), xycoords=("data",
                "axes fraction"), textcoords="offset points",
                fontsize=PT_ANNOT, color=GRAY, ha="center", va="bottom")

    ax.set_xlim(-d["window"], d["window"])
    ax.set_xlabel("arclength from the crossing (px)", fontsize=PT_AXIS)
    ax.set_ylabel("grey level", fontsize=PT_AXIS)
    ax.tick_params(labelsize=PT_AXIS)
    thin_spines(ax)

    #  Each label sits at its own line. Only when two of them are within a
    #  noise width -- which is the case the stage abstains on -- are they
    #  nudged apart, and then by shifting them off the line, never by
    #  reordering, so a label always names the line it touches.
    right = d["window"]
    levels = [("$m_j$", m["j"], FIL_B), ("$m_i$", m["i"], FIL_A),
              ("$m_c$", m_c, INK)]
    span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
    levels.sort(key=lambda r: r[1])
    shift = [0.0] * len(levels)
    for k in range(1, len(levels)):
        if levels[k][1] - levels[k - 1][1] < 0.045 * span:
            shift[k - 1], shift[k] = -0.022 * span, 0.022 * span
    for (text, value, colour), dy in zip(levels, shift):
        ax.annotate(text, (right, value + dy), xytext=(3, 0),
                    textcoords="offset points", fontsize=PT_ANNOT,
                    color=colour, va="center", annotation_clip=False)
    return m, m_c, sigma


def panel_decision(ax, d, m, m_c, sigma):
    """The two distances the score compares, against the floor it divides by.

    Drawn because the arithmetic is the whole rule and, at this crossing, the
    denominator is the noise floor rather than the contrast -- the case the
    floor exists for, and one a reader would otherwise have to take on trust.
    """
    di, dj = abs(m_c - m["i"]), abs(m_c - m["j"])
    contrast, floor = abs(m["i"] - m["j"]), 2.0 * sigma
    denom = max(contrast, floor)

    ax.barh([1], [dj], height=0.42, color=FIL_B, zorder=3)
    ax.barh([0], [di], height=0.42, color=FIL_A, zorder=3)
    for yy, v in ((1, dj), (0, di)):
        ax.annotate("%.3f" % v, (v, yy), xytext=(3, 0),
                    textcoords="offset points", fontsize=PT_ANNOT, color=INK,
                    va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["$|m_c-m_i|$", "$|m_c-m_j|$"], fontsize=PT_ANNOT)

    ax.axvline(contrast, color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.axvline(floor, color=GEOM, lw=1.1, zorder=2)
    lo, hi = (contrast, floor) if contrast < floor else (floor, contrast)
    lo_c, hi_c = ((GRAY, GEOM) if contrast < floor else (GEOM, GRAY))
    lo_t, hi_t = (("$|m_i-m_j|$", r"$2\hat{\sigma}$") if contrast < floor
                  else (r"$2\hat{\sigma}$", "$|m_i-m_j|$"))
    ax.annotate(lo_t, (lo, 1.0), xytext=(3, -2), xycoords=("data",
                "axes fraction"), textcoords="offset points",
                fontsize=PT_ANNOT, color=lo_c, ha="left", va="top")
    ax.annotate(hi_t, (hi, 1.0), xytext=(-3, -12), xycoords=("data",
                "axes fraction"), textcoords="offset points",
                fontsize=PT_ANNOT, color=hi_c, ha="right", va="top")

    ax.set_ylim(-0.45, 2.05)
    ax.set_xlim(0, denom * 1.42)
    ax.set_xlabel("grey level", fontsize=PT_AXIS)
    ax.tick_params(labelsize=PT_AXIS)
    thin_spines(ax)
    ax.annotate("$s=%+.2f$" % d["score"], (0.5, 0.0), xytext=(0, -26),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=PT_ANNOT, color=INK, ha="center", va="top",
                annotation_clip=False)
    return di, dj, contrast, floor


def main() -> int:
    plecta_style()
    d = measure()

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.335))
    ax_a = fig.add_axes([0.018, 0.125, 0.250, 0.735])
    ax_b = fig.add_axes([0.372, 0.215, 0.300, 0.645])
    ax_c = fig.add_axes([0.815, 0.215, 0.150, 0.645])
    panel_geometry(ax_a, d)
    m, m_c, sigma = panel_levels(ax_b, d)
    di, dj, contrast, floor = panel_decision(ax_c, d, m, m_c, sigma)

    #  Panel tags are parenthesised, matching the rest of the set and the
    #  (a)/(b) the caption uses.  This figure and fig_depth_order were the
    #  only two that set a bare letter.  Titles are de-idiomed with the prose
    #  (reviewer comment 5): "what it returns", "what decides".
    for x0, letter, text in ((0.018, "a", "where the greyscale is sampled"),
                             (0.372, "b", "the levels it returns"),
                             (0.760, "c", "the decision rule")):
        fig.text(x0, 0.935, "(%s)" % letter, fontsize=PT_TITLE,
                 fontweight="bold", color=INK)
        fig.text(x0 + 0.034, 0.935, text, fontsize=PT_TITLE, color=INK)

    save_fig(fig, "fig_depth_evidence", bbox_inches=None)
    plt.close(fig)
    print("crossing %s angle %.1f deg  score %+.3f  (truth: %d over)" %
          (CROSSING, d["angle"], d["score"], d["over_truth"]))
    print("m_i %.3f  m_j %.3f  m_c %.3f  sigma %.4f" % (m["i"], m["j"], m_c,
                                                        sigma))
    print("|m_c-m_i| %.4f  |m_c-m_j| %.4f  contrast %.4f  2*sigma %.4f  "
          "-> denominator is the %s"
          % (di, dj, contrast, floor,
             "noise floor" if floor > contrast else "contrast"))
    print("core samples %d/%d  flank samples %d/%d"
          % (len(d["i"]["core"]), len(d["j"]["core"]),
             len(d["i"]["flank"]), len(d["j"]["flank"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
