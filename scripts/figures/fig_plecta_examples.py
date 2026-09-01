"""Figure 8: qualitative examples of the reconstruction.

Changed after review, twice.  The greedy-continuation column went first; then
the second PLECTA column went too.  The figure used to show PLECTA's output
twice -- the one-pixel centreline instances that are scored, and the same
instances after the downstream stack -- under headings reading "(scored)" and
"(not scored)".

One output column is enough.  At this panel size a one-pixel centreline is
barely visible against the reference beside it, so the column that carried the
figure's qualitative argument was always the rendered one; the other column
spent a quarter of the width restating in faint pixels what the F1 beside each
row already states as a number.  The distinction it was drawn to make is a
sentence, not a column: rendering is downstream of the grouping and cannot move
a pixel from one instance to another, so the identities shown are exactly the
identities scored.  The caption says that in words.

Rows are one scene per coverage level, each the median scene of its own
stratum, so the rows span the difficulty range and none is chosen by outcome.

Data: results/plecta_examples_full.json (committed), written by
scripts/figures/extract_plecta_examples.py.

    python scripts/figures/fig_plecta_examples.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import (DARK, FIG_W, FONE, GRAY, INSTANCE_CYCLE, PT_ANNOT,
                    PT_TITLE,
                    plecta_style, save_fig, tagged_title, unpack,
                    unpack_labels)

REPO = Path(__file__).resolve().parents[2]
BG = "#101216"

#: Vertical fraction of each square scene that is drawn, centred.
CROP = 2.0 / 3.0

#  Column headings carry a panel tag like every other multi-panel figure in
#  the set, and are lower case in the regular weight.  Without the tags this
#  was the only figure a caption could not refer to by letter.
COLUMNS = (
    ("mask", "Input mask"),
    ("reference", "Reference"),
    ("plecta_rendered", "PLECTA reconstruction"),
)


def label_rgb(payload):
    """A label image as RGB on a dark ground."""
    import matplotlib.colors as mcolors
    lab = unpack_labels(payload)
    img = np.zeros((lab.shape[0], lab.shape[1], 3), float)
    img[:] = mcolors.to_rgb(BG)
    for value in np.unique(lab):
        if value == 0:
            continue
        img[lab == value] = mcolors.to_rgb(
            INSTANCE_CYCLE[(int(value) - 1) % len(INSTANCE_CYCLE)])
    return img


def mask_rgb(payload):
    import matplotlib.colors as mcolors
    m = unpack(payload)
    img = np.zeros((m.shape[0], m.shape[1], 3), float)
    img[:] = mcolors.to_rgb(BG)
    img[m] = (1.0, 1.0, 1.0)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="plecta_examples_full.json")
    args = ap.parse_args(argv)

    plecta_style()
    data = json.loads((REPO / "results" / args.data).read_text(encoding="utf-8"))
    panels = data["panels"]
    keys = [k for k, _ in COLUMNS if k in panels[0]]
    heads = [t for k, t in COLUMNS if k in panels[0]]

    #  The rotated per-row labels of an earlier round were hard to scan, per
    #  the figure review; each row now carries a horizontal one-line heading
    #  above it, and the left margin the rotated labels occupied returns to
    #  the panels.
    left, right = 0.014, 0.984
    cell = (right - left) / len(keys)
    side_in = cell * FIG_W
    head_in, row_head_in, foot_in = 0.22, 0.17, 0.05
    # Each scene is square, so nine square panels at full text width force a
    # figure taller than it is wide and the page it lands on carries almost no
    # text. A horizontal window of each scene shows the same thing -- these are
    # qualitative examples, not measurements -- at two thirds the height, and
    # keeps all three densities and all three columns. The caption says it is
    # a crop.
    panel_in = side_in * CROP
    fig_h = head_in + len(panels) * (row_head_in + panel_in) + foot_in
    fig = plt.figure(figsize=(FIG_W, fig_h))
    cell_h = panel_in / fig_h

    #  Column headings once, at the top, in the set's tag convention: the
    #  bold ``(a)`` and the heading in the regular weight beside it.
    for c, head in enumerate(heads):
        x0 = left + c * cell + 0.003
        y = 1.0 - (head_in - 0.05) / fig_h
        fig.text(x0, y, "(%s)" % "abc"[c], fontsize=PT_TITLE,
                 fontweight="bold", color=DARK, ha="left", va="bottom")
        fig.text(x0 + 0.037, y, head, fontsize=PT_TITLE, color=DARK,
                 ha="left", va="bottom")

    for r, panel in enumerate(panels):
        y0 = 1.0 - (head_in + (r + 1) * (row_head_in + panel_in)) / fig_h
        for c, key in enumerate(keys):
            img = mask_rgb(panel[key]) if key == "mask" \
                else label_rgb(panel[key])
            ax = fig.add_axes([left + c * cell + 0.003, y0 + 0.0022,
                               cell - 0.006, cell_h - 0.0044])
            #  Equal-aspect images letterbox inside a rect whose aspect is
            #  not exactly theirs, and matplotlib centres the shrunken box,
            #  which pushed every panel right of the headings and row labels
            #  that share the rect's own left edge.  Anchoring south-west
            #  pins the panel to that edge instead.
            ax.set_anchor("SW")
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRAY)
                s.set_linewidth(0.5)
        label = ("{0}% coverage,  ".format(panel["density"])
                 + FONE + " = " + "{:.3f}".format(panel["plecta_f1"])
                 + "   (scene {0})".format(panel["scene"]))
        fig.text(left + 0.003, y0 + cell_h + 0.030 / fig_h, label,
                 ha="left", va="bottom", fontsize=PT_ANNOT, color=DARK)

    save_fig(fig, "fig_plecta_examples", bbox_inches=None)
    plt.close(fig)
    print("columns:", keys)
    print("selection rule:", data["selection_rule"])
    for panel in panels:
        print(" ", panel["key"], panel["scene"], panel["density"],
              "F1", panel["plecta_f1"],
              "instances scored/rendered",
              panel["n_plecta"], panel.get("n_plecta_rendered", "-"),
              "absorbed", panel.get("n_absorbed_by_rendering", "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
