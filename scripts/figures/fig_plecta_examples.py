"""Figure: paired qualitative examples, PLECTA against one comparator.

Same four-column structure as the earlier paired-examples figure in this
repository -- input mask, overlap-aware reference, comparator, method -- with
the row label carrying the scene, its target coverage and the paired
difference in F1.

The comparator is a parameter, not a constant: point ``--data`` at any
``results/plecta_examples_<name>.json`` written by
``extract_plecta_examples.py`` and the columns relabel themselves.

    python scripts/figures/fig_plecta_examples.py --data plecta_examples_greedy.json
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

from _style import (DARK, FIG_W, GRAY, INSTANCE_CYCLE, PT_BODY, PT_LABEL,
                    PT_TINY, plecta_style, save_fig, unpack, unpack_labels)

REPO = Path(__file__).resolve().parents[2]
BG = "#101216"


def label_rgb(payload):
    """A label image as RGB on the dark ground the earlier figure used."""
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
    ap.add_argument("--data", default="plecta_examples_greedy.json")
    args = ap.parse_args(argv)

    plecta_style()
    data = json.loads((REPO / "results" / args.data).read_text(encoding="utf-8"))
    panels = data["panels"]
    columns = ("Input mask", "Reference",
               data["comparator_label"], "PLECTA")

    left, right, top = 0.115, 0.995, 0.925
    cell = (right - left) / 4.0
    side_in = cell * FIG_W
    fig_h = len(panels) * side_in + 0.62
    fig = plt.figure(figsize=(FIG_W, fig_h))
    cell_h = side_in / fig_h

    for r, panel in enumerate(panels):
        y0 = top - (r + 1) * cell_h
        images = (mask_rgb(panel["mask"]), label_rgb(panel["reference"]),
                  label_rgb(panel["comparator"]), label_rgb(panel["plecta"]))
        for c, img in enumerate(images):
            ax = fig.add_axes([left + c * cell + 0.004, y0 + 0.004,
                               cell - 0.008, cell_h - 0.008])
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRAY)
                s.set_linewidth(0.5)
            if r == 0:
                ax.set_title(columns[c], fontsize=PT_LABEL, fontweight="bold",
                             color=DARK, pad=3.5)
        fig.text(left - 0.012, y0 + cell_h / 2.0,
                 "{0}  {1}\ncov {2}%\n$\\Delta F_1 = {3:+.3f}$"
                 .format(panel["scene"], "", panel["density"],
                         panel["delta_f1"]).replace("  \n", "\n"),
                 rotation=90, ha="right", va="center", fontsize=PT_TINY,
                 color=DARK, linespacing=1.5)

    fig.text(left, 0.012,
             "Reference instances are the generator's rendered ribbons; the two "
             "method columns are centreline instances, hence the thinner stroke.",
             fontsize=PT_TINY, color=GRAY, ha="left", va="bottom")

    save_fig(fig, "fig_plecta_examples", bbox_inches=None)
    plt.close(fig)
    for panel in panels:
        print(panel["key"], panel["scene"], panel["density"],
              panel["delta_f1"], panel["rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
