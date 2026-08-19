"""A contradictory local crossing-order cycle and its global correction.

Local over/under evidence is estimated per crossing, so nothing stops it
from asserting A > B, B > C, C > A -- impossible when every instance has one
z. This figure shows the first such cycle in the held-out predicted-instance
run (first scene, in scene order, whose raw precedence graph is cyclic; no
other selection): the raw relations with their local probabilities, and the
globally corrected relations after the maximum-weight acyclic orientation,
with the reversed relation highlighted.

Writes to figures/archive/ (no `\\includegraphics` yet).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _style import (CHORD, DIFFER, FIG_W, GREEN, INK, PT_ANNOT, PT_TITLE,  # noqa: E402
                    plecta_style, save_fig, tagged_title)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(REPO, "exploration", "depth_25d", "runs",
                   "heldout__pred_meas")


def find_cycle_scene():
    """First scene with a cyclic raw graph; prefer a cycle of 3+ instances.

    Scenes are visited in sorted order; among them the first whose largest
    strongly connected component has >= 3 nodes wins, otherwise the first
    with any cycle (a 2-cycle: the same pair crossing twice with opposite
    local orders). No result-based selection beyond cycle size.
    """
    fallback = None
    for pd_path in sorted(Path(RUN).rglob("pred_depth.json")):
        pred = json.loads(pd_path.read_text(encoding="utf-8"))
        if pred["solver_report"].get("n_precedence_cycles_raw", 0) < 1 \
                or not any(c["flipped"] for c in pred["crossings"]):
            continue
        if len(cyclic_nodes(pred, largest=True)) >= 3:
            return pd_path, pred
        if fallback is None:
            fallback = (pd_path, pred)
    if fallback:
        return fallback
    raise SystemExit(f"no cyclic scene found under {RUN}")


def cyclic_nodes(pred, largest=False):
    """Nodes of a strongly connected raw-relation component (>1 node)."""
    edges = {}
    for c in pred["crossings"]:
        if c["raw_over"] is None:
            continue
        lo = c["j"] if c["raw_over"] == c["i"] else c["i"]
        edges.setdefault(c["raw_over"], set()).add(lo)
    # brute SCC via reachability (graphs are small)
    nodes = sorted({n for k, vs in edges.items() for n in (k, *vs)})

    def reach(src):
        seen, work = set(), [src]
        while work:
            n = work.pop()
            for m in edges.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    work.append(m)
        return seen

    sccs = []
    for n in nodes:
        comp = {n} | {m for m in reach(n) if n in reach(m)}
        if len(comp) > 1 and comp not in sccs:
            sccs.append(comp)
    if not sccs:
        return []
    return sorted((max if largest else min)(sccs, key=len))


def draw_graph(ax, nodes, relations, title_letter, title, show_flip):
    pos = {}
    n = len(nodes)
    for k, node in enumerate(nodes):
        ang = np.pi / 2 + 2 * np.pi * k / n
        pos[node] = (math.cos(ang), math.sin(ang))
    for node, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.16, facecolor="white",
                                edgecolor=INK, linewidth=1.0, zorder=3))
        ax.text(x, y, str(node), ha="center", va="center",
                fontsize=PT_ANNOT, color=INK, zorder=4)
    for (hi, lo, p, flipped) in relations:
        x0, y0 = pos[hi]
        x1, y1 = pos[lo]
        d = np.array([x1 - x0, y1 - y0])
        d = d / np.linalg.norm(d)
        start = (x0 + 0.19 * d[0], y0 + 0.19 * d[1])
        end = (x1 - 0.19 * d[0], y1 - 0.19 * d[1])
        colour = DIFFER if (flipped and show_flip) else GREEN
        rad = 0.25
        # curved so antiparallel relations between one pair stay separate.
        # The reversed edge is distinguished by colour and width alone, not
        # a dash pattern: FancyArrowPatch does not space dashes relative to
        # the arrowhead, so the final dash segment routinely bunches up at
        # the tip into a solid clump rather than reading as "dashed".
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=2.0 if (flipped and show_flip) else 1.2,
            color=colour, zorder=2))
        # The label has to sit clear of the arc itself, not of the straight
        # chord between the nodes: arc3 bows the path out by ~rad * |chord|
        # at the midpoint, and placing the label from the CHORD's normal (as
        # a straight-line offset would) put it back on top of that bow --
        # which is what made "p = 0.88" read as overlapping double digits.
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        normal = np.array([d[1], -d[0]])
        bow = rad * np.linalg.norm(np.array(end) - np.array(start))
        off = normal * (bow + 0.15)
        ax.text(mx + off[0], my + off[1], f"p = {p:.2f}", ha="center",
                va="center", fontsize=PT_ANNOT, color=CHORD)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    tagged_title(ax, title_letter, title, dy=1.05, gap=0.135)


def main():
    plecta_style()
    pd_path, pred = find_cycle_scene()
    nodes = cyclic_nodes(pred)
    node_set = set(nodes)
    raw, corrected = [], []
    for c in pred["crossings"]:
        if c["raw_over"] is None:
            continue
        if not ({c["i"], c["j"]} <= node_set):
            continue
        raw_lo = c["j"] if c["raw_over"] == c["i"] else c["i"]
        p_raw = c["p_over"] if c["raw_over"] == c["i"] else 1 - c["p_over"]
        raw.append((c["raw_over"], raw_lo, p_raw, c["flipped"]))
        over = c["over"]
        lo = c["j"] if over == c["i"] else c["i"]
        corrected.append((over, lo, p_raw, c["flipped"]))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 0.72, FIG_W * 0.40))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04,
                        wspace=0.10)
    draw_graph(axes[0], nodes, raw, "a",
               "raw local relations (cyclic)", show_flip=True)
    draw_graph(axes[1], nodes, corrected, "b",
               "after global correction (acyclic)", show_flip=True)
    scene = pd_path.parent.name.replace("__", "/")
    fig.text(0.02, 0.035,
             f"scene {scene}; the heavier red relation is the weakest in "
             "the cycle and is",
             fontsize=PT_ANNOT, color=INK, ha="left")
    fig.text(0.02, 0.005,
             "reversed by the maximum-weight acyclic orientation; its "
             "local probability is kept in the record.",
             fontsize=PT_ANNOT, color=INK, ha="left")
    save_fig(fig, "fig_depth_cycle", subdir="archive")


if __name__ == "__main__":
    main()
