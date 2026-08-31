"""Figure: why one order has to be solved for the whole field.

Section 2.5.2 says local over/under calls need not be mutually consistent, and
that the stage keeps the heaviest set of them that can all hold at once.  This
shows that happening, on a contradiction the stage really met rather than an
invented one: four strands in a development scene whose four local calls close
a cycle.

  (a) the four strands where they cross, each crossing marked with the call
      the local evidence made and the weight it carries;
  (b) the same four calls as a directed graph, which has no consistent
      reading: follow the arrows and they return to where they started;
  (c) what the global solve keeps.  The two heaviest calls stand, the two
      lightest are reversed, and the result reads as an order.

Everything is measured: the strands are their own centrelines, the calls and
weights come from ``plecta.depth`` under the fixed configuration, and which
calls get reversed is the solver's answer, not a choice made here.

Panel (c) shows the order among these four only.  Layer numbers are not drawn,
because the layer a strand lands in is fixed by its whole component -- these
four sit in a seven-layer scene -- and a two-layer picture of the four would
say something the stage does not.

    python scripts/figures/fig_depth_order.py
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
from matplotlib.patches import FancyArrowPatch

from _style import (FIG_W, INK, GRAY, LIGHT_GRAY, FIL_A, FIL_B, PLECTA,
                    JUNCTION, MISSED_INST, PT_ANNOT, PT_TITLE, plecta_style,
                    save_fig)

PLECTA_ROOT = r"C:\Repos\PLECTA"
SCENE = os.path.join(r"C:\Repos\filaments_quantification\input",
                     "synthetic_depth_dev", "cov35", "synth_0003")
GROUP = [4, 13, 21, 31]     # the strands of one real contradiction
PAD = 34                    # px around the four crossings in (a)

#  Four strands need four tellable-apart strokes. Colour alone is not enough
#  -- two of these read alike in greyscale and to a colour-blind reader -- so
#  each strand also carries its own dash pattern, and its id travels with it
#  into the graph panels, where the node is filled in the same colour.
STRAND_STYLE = {
    4:  (FIL_A, (0, (5, 1.6))),
    13: (FIL_B, (0, (1.4, 1.4))),
    21: (PLECTA, (0, (5, 1.5, 1.2, 1.5))),
    31: (MISSED_INST, "solid"),
}


def measure():
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

    crossings = []
    for c in gt["crossings"]:
        pc = D.PredCrossing(i=int(c["i"]), j=int(c["j"]),
                            x=float(c["x"]), y=float(c["y"]))
        try:
            D.crossing_evidence(image, None, lines, pc, params)
        except Exception:
            continue
        crossings.append(pc)
    D.apply_abstention(crossings, params=params)
    summary = D.resolve_global_order(sorted(lines), crossings, params)

    calls = []
    for c in crossings:
        if c.i in GROUP and c.j in GROUP and not c.abstain and c.raw_over >= 0:
            calls.append({
                "x": c.x, "y": c.y, "weight": abs(c.score),
                "local_over": c.raw_over,
                "local_under": c.j if c.raw_over == c.i else c.i,
                "final_over": c.over,
                "final_under": c.j if c.over == c.i else c.i,
                "flipped": bool(c.flipped),
            })
    return {"image": image, "lines": lines, "calls": calls,
            "n_layers_scene": summary}


def node_places(calls):
    """A square layout for the four strands, cycle-order round the square."""
    #  Walk the local calls as a directed graph so adjacent nodes on the
    #  square are adjacent in the cycle: the contradiction then reads as a
    #  loop round the square rather than as crossing diagonals.
    nxt = {c["local_over"]: c["local_under"] for c in calls}
    walk, node = [], GROUP[0]
    while node not in walk:
        walk.append(node)
        node = nxt.get(node)
        if node is None:
            break
    for g in GROUP:
        if g not in walk:
            walk.append(g)
    corners = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    return {sid: corners[k] for k, sid in enumerate(walk[:4])}


def arrow(ax, p, q, colour, lw=1.2, shrink=13, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=colour,
                                 lw=lw, shrinkA=shrink, shrinkB=shrink,
                                 mutation_scale=7, zorder=3))


def _badge_text_colour(fill):
    """Dark digits on a light fill, white on a dark one.

    White on the amber of strand 31 (MISSED_INST) was below 2:1 contrast,
    which is the kind of thing a reviewer notices in print.  The 0.6 luma
    threshold sends amber to ink and the blue, orange and green to white.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(fill)
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def draw_nodes(ax, places):
    for sid, (x, y) in places.items():
        colour = STRAND_STYLE[sid][0]
        ax.plot([x], [y], "o", ms=15, mfc=colour, mec=INK, mew=0.8,
                alpha=0.85, zorder=4)
        ax.text(x, y, str(sid), fontsize=PT_ANNOT,
                color=_badge_text_colour(colour),
                fontweight="bold", ha="center", va="center", zorder=5)
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.30, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_scene(ax, d):
    xs = [c["x"] for c in d["calls"]]
    ys = [c["y"] for c in d["calls"]]
    ax.imshow(d["image"], cmap="gray", vmin=0, vmax=1, origin="upper",
              interpolation="nearest", zorder=0)
    ax.set_xlim(min(xs) - PAD, max(xs) + PAD)
    ax.set_ylim(max(ys) + PAD, min(ys) - PAD)

    for sid in GROUP:
        line = d["lines"][sid]
        colour, dash = STRAND_STYLE[sid]
        ax.plot(line[:, 0], line[:, 1], color=colour, lw=1.3, ls=dash,
                zorder=2)
        #  Name each strand where it leaves the crop, so the label never sits
        #  on a crossing the panel is about.
        inside = [(x, y) for x, y in line
                  if min(xs) - PAD < x < max(xs) + PAD
                  and min(ys) - PAD < y < max(ys) + PAD]
        if inside:
            #  A little way in from the frame, and a different distance for
            #  each strand, so two entering the crop side by side do not put
            #  their labels in the same place.
            frac = (0.10, 0.26, 0.10, 0.52)[GROUP.index(sid)]
            at = inside[min(len(inside) - 1, max(1, int(frac * len(inside))))]
            ax.annotate(str(sid), at, fontsize=PT_ANNOT,
                        color=_badge_text_colour(STRAND_STYLE[sid][0]),
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.16",
                                  fc=STRAND_STYLE[sid][0], ec="white",
                                  lw=0.7), zorder=6)

    for c in d["calls"]:
        ax.plot([c["x"]], [c["y"]], "o", ms=4.6, mfc="white", mec=JUNCTION,
                mew=1.1, zorder=5)
        ax.annotate("%d over %d" % (c["local_over"], c["local_under"]),
                    (c["x"], c["y"]), xytext=(0, -11),
                    textcoords="offset points", fontsize=PT_ANNOT,
                    color=INK, ha="center", va="top", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.82))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LIGHT_GRAY)
        sp.set_linewidth(0.6)


def panel_cycle(ax, d, places):
    draw_nodes(ax, places)
    for c in d["calls"]:
        arrow(ax, places[c["local_over"]], places[c["local_under"]], JUNCTION)
        p, q = places[c["local_over"]], places[c["local_under"]]
        ax.annotate("%.2f" % c["weight"],
                    ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2),
                    fontsize=PT_ANNOT, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white",
                              ec="none", alpha=0.9), zorder=6)
    ax.annotate("the arrows form a cycle", (0.5, -0.30),
                fontsize=PT_ANNOT, color=JUNCTION, ha="center", va="top")


def panel_resolved(ax, d, places):
    draw_nodes(ax, places)
    for c in d["calls"]:
        colour = GRAY if c["flipped"] else PLECTA
        arrow(ax, places[c["final_over"]], places[c["final_under"]], colour,
              lw=1.4 if not c["flipped"] else 1.0)
        if c["flipped"]:
            p, q = places[c["final_over"]], places[c["final_under"]]
            ax.annotate("reversed", ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2),
                        fontsize=PT_ANNOT, color=GRAY, ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.14", fc="white",
                                  ec="none", alpha=0.9), zorder=6)
    kept = sum(c["weight"] for c in d["calls"] if not c["flipped"])
    lost = sum(c["weight"] for c in d["calls"] if c["flipped"])
    ax.annotate("retains %.2f of %.2f,\nand is now acyclic"
                % (kept, kept + lost), (0.5, -0.30), fontsize=PT_ANNOT,
                color=INK, ha="center", va="top")


def main() -> int:
    plecta_style()
    d = measure()
    if len(d["calls"]) != 4:
        raise SystemExit("expected 4 calls among %s, got %d"
                         % (GROUP, len(d["calls"])))
    places = node_places(d["calls"])

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.335))
    ax_a = fig.add_axes([0.015, 0.100, 0.300, 0.760])
    ax_b = fig.add_axes([0.395, 0.100, 0.230, 0.760])
    ax_c = fig.add_axes([0.740, 0.100, 0.230, 0.760])
    panel_scene(ax_a, d)
    panel_cycle(ax_b, d, places)
    panel_resolved(ax_c, d, places)

    for x0, letter, text in ((0.015, "a", "Local estimates"),
                             (0.360, "b", "Cyclic relation graph"),
                             (0.705, "c", "Reconciled acyclic order")):
        fig.text(x0, 0.940, "(%s)" % letter, fontsize=PT_TITLE,
                 fontweight="bold", color=INK)
        fig.text(x0 + 0.034, 0.940, text, fontsize=PT_TITLE, color=INK)

    save_fig(fig, "fig_depth_order", bbox_inches=None)
    plt.close(fig)
    for c in d["calls"]:
        print("  %2d over %2d  w=%.2f  ->  %2d over %2d %s"
              % (c["local_over"], c["local_under"], c["weight"],
                 c["final_over"], c["final_under"],
                 "REVERSED" if c["flipped"] else ""))
    print("scene-wide: %d components, %d raw cycles, %d relations flipped"
          % (d["n_layers_scene"]["n_components"],
             d["n_layers_scene"]["n_precedence_cycles_raw"],
             d["n_layers_scene"]["n_relations_flipped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
