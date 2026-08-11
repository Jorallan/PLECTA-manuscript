"""Figure 8: qualitative examples, and what the downstream rendering adds.

Changed after review.  The greedy-continuation column is gone.  In its place
the figure now shows PLECTA twice: the grouping the paper evaluates, which is
a set of one-pixel centreline instances, and the same instances after the full
downstream stack -- width measured from the paired SEM image, gaps
interpolated, centreline resampled and smoothed, each chain drawn at its
fitted width.

The two PLECTA columns contain the *same instances*.  Rendering redraws them;
it cannot move a pixel from one instance to another, and the reported metrics
score identity, so every number in this paper comes from the "scored" column.
The column headings say so, and the caption says so again.

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

from _style import (DARK, FIG_W, GRAY, INSTANCE_CYCLE, PT_ANNOT, PT_TITLE,
                    plecta_style, save_fig, unpack, unpack_labels)

REPO = Path(__file__).resolve().parents[2]
BG = "#101216"

COLUMNS = (
    ("mask", "Input mask"),
    ("reference", "Reference"),
    ("plecta", "PLECTA\n(scored)"),
    ("plecta_rendered", "PLECTA + rendering\n(not scored)"),
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

    # right < 1 by enough that the widest column heading, which is centred on
    # its column, still lands inside the canvas: bbox_inches is None, so
    # anything past the figure edge is simply cut off.
    left, right = 0.082, 0.984
    cell = (right - left) / len(keys)
    side_in = cell * FIG_W
    head_in, foot_in = 0.34, 0.05
    fig_h = len(panels) * side_in + head_in + foot_in
    fig = plt.figure(figsize=(FIG_W, fig_h))
    cell_h = side_in / fig_h
    top = 1.0 - head_in / fig_h

    for r, panel in enumerate(panels):
        y0 = top - (r + 1) * cell_h
        for c, key in enumerate(keys):
            img = mask_rgb(panel[key]) if key == "mask" \
                else label_rgb(panel[key])
            ax = fig.add_axes([left + c * cell + 0.003, y0 + 0.0022,
                               cell - 0.006, cell_h - 0.0044])
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRAY)
                s.set_linewidth(0.5)
            if r == 0:
                ax.set_title(heads[c], fontsize=PT_TITLE, fontweight="bold",
                             color=DARK, pad=3.5, linespacing=1.35)
        fig.text(left - 0.010, y0 + cell_h / 2.0,
                 "{0}\ncoverage {1}%\n$F_1 = {2:.3f}$"
                 .format(panel["scene"], panel["density"],
                         panel["plecta_f1"]),
                 rotation=90, ha="right", va="center", fontsize=PT_ANNOT,
                 color=DARK, linespacing=1.45)

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
