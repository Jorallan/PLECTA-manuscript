"""The qualitative real-SEM figure: one crop, three instance labellings.

Four panels across ``\\linewidth``: the SEM micrograph, the manual annotation,
PLECTA on the manual-derived axis, and PLECTA on the nnU-Net axis.  Instance
colours correspond -- a predicted instance is drawn in the colour of the
annotated filament it was matched to -- because three separately coloured label
images ask the reader to do the matching by eye, which is exactly the judgement
the picture is meant to supply.

Everything drawn comes from ``results/figure_assets/real_sem_crop.npz``, written
by ``extract_real_sem_crop.py``, which runs PLECTA on the whole field and cuts
the crop afterwards.  Nothing is recomputed here.

    python scripts/figures/fig_real_sem.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _style import (FIG_W, INSTANCE_CYCLE, FALSE_INST,            # noqa: E402
                    blank_rgb, framed, paint, plecta_style, save_fig,
                    tagged_title)

ASSET = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "figure_assets",
    "real_sem_crop.npz"))

#: Instances are one pixel wide.  At the ~1.5 in each panel gets on the page a
#: one-pixel centreline is invisible, so every instance is dilated by one pixel
#: for *rendering only*; no geometry is changed and the caption says so.
DRAW_DILATION = 1
#: Two instances closer than this must not be given the same colour.
COLOUR_SEPARATION = 6

#: The identity palette, minus the two entries close enough to ``FALSE_INST``
#: (#C2185B) to be confused with it.  An unmatched instance is the one thing in
#: this figure the reader must be able to pick out at a glance, and with
#: #B03A6A and #D45087 in the cycle it could not be: legitimate filaments were
#: drawn in a crimson a shade off the alarm colour.
IDENTITY_CYCLE = tuple(c for c in INSTANCE_CYCLE
                       if c not in ("#B03A6A", "#D45087"))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    if radius <= 0:
        return np.asarray(mask, bool)
    size = 2 * radius + 1
    return binary_dilation(np.asarray(mask, bool),
                           structure=np.ones((size, size), bool))


def colour_of_reference(reference: np.ndarray) -> dict:
    """Greedy graph colouring, so no two nearby filaments share a colour.

    A cyclic palette assigned by instance id puts the same colour on adjacent
    filaments often enough to be read as one object.  Colouring the adjacency
    graph instead costs nothing here (fewer than a hundred instances in the
    crop) and removes that reading.  Instances are coloured largest first, which
    makes the assignment deterministic.
    """
    ids = [int(i) for i in np.unique(reference) if i > 0]
    masks = {i: reference == i for i in ids}
    grown = {i: dilate(masks[i], COLOUR_SEPARATION) for i in ids}
    neighbours = {i: set() for i in ids}
    for i in ids:
        touching = {int(v) for v in np.unique(reference[grown[i]]) if v > 0}
        touching.discard(i)
        neighbours[i] |= touching
        for j in touching:
            neighbours[j].add(i)

    order = sorted(ids, key=lambda i: (-int(masks[i].sum()), i))
    assigned: dict[int, int] = {}
    for i in order:
        taken = {assigned[j] for j in neighbours[i] if j in assigned}
        slot = next((k for k in range(len(IDENTITY_CYCLE)) if k not in taken), 0)
        assigned[i] = slot
    return {i: IDENTITY_CYCLE[k] for i, k in assigned.items()}


def draw_instances(ax, labels: np.ndarray, colours: dict, unmatched: int) -> None:
    """Paint a label image, one colour per instance, unmatched in the alarm hue."""
    image = blank_rgb(labels.shape)
    for ident in (int(v) for v in np.unique(labels)):
        if ident == 0:
            continue
        if ident == unmatched:
            colour = FALSE_INST
        else:
            #  A prediction can be matched to an annotated filament whose own
            #  pixels all lie outside the crop.  It is matched, so it must not
            #  be drawn in the unmatched colour; it just has no neighbour-aware
            #  slot, so it takes a deterministic one.
            colour = colours.get(ident,
                                 IDENTITY_CYCLE[ident % len(IDENTITY_CYCLE)])
        paint(image, dilate(labels == ident, DRAW_DILATION), colour)
    ax.imshow(image, interpolation="nearest")


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    unmatched = int(meta["unmatched_label"])
    reference, sem = data["reference"], data["sem"]
    colours = colour_of_reference(reference)

    #  Four square panels on one row.  The page must come out exactly FIG_W
    #  wide: ``check_type_scale`` reads /MediaBox and fails the figure if the
    #  include scale is not 1.0, because at any other scale a label specified at
    #  n pt does not print at n pt.  So no ``bbox_inches="tight"`` here, and
    #  every title has to fit inside its own panel rather than being allowed to
    #  overhang and grow the canvas.
    pad, gap = 0.005, 0.020
    panel = (1.0 - 2 * pad - 3 * gap) / 4.0
    panel_in = panel * FIG_W
    title_in = 0.19
    fig_h = panel_in + title_in
    fig = plt.figure(figsize=(FIG_W, fig_h))

    #  Panels (c) and (d) are both PLECTA and differ only in the axis they were
    #  handed, so the titles name the input.  Short enough to sit inside a
    #  1.5 in panel: at 8.5 pt the longest of them sets about 1.3 in with its
    #  tag, and a title that overruns its panel reads as a title of the next.
    titles = (("a", "SEM micrograph"),
              ("b", "Manual annotation"),
              ("c", "Manual-derived axis"),
              ("d", "nnU-Net axis"))
    for column, (tag, title) in enumerate(titles):
        left = pad + column * (panel + gap)
        ax = fig.add_axes([left, 0.0, panel, panel_in / fig_h])
        framed(ax)
        if column == 0:
            ax.imshow(sem, cmap="gray", interpolation="nearest")
        elif column == 1:
            draw_instances(ax, reference, colours, unmatched)
        else:
            draw_instances(ax, data["manual" if column == 2 else "unet"],
                           colours, unmatched)
        #  ``gap`` is in axes fractions, and these panels are a quarter the
        #  width the shared default was set for.  A bold "(a)" at 8.5 pt is
        #  about 0.21 in, which is 0.145 of a 1.45 in panel, so anything under
        #  that prints the tag on top of the first word.
        tagged_title(ax, tag, title, dy=1.02, gap=0.165)

    #  No legend: the one colour that needs explaining is named in the caption,
    #  which costs no vertical space on a page this tight.
    save_fig(fig, "fig_real_sem")
    plt.close(fig)
    print("[fig_real_sem] %s crop at row %d col %d, %d px; local F1 %s"
          % (meta["field"], meta["crop_row"], meta["crop_col"],
             meta["crop_px"],
             ", ".join("%s %.3f" % kv for kv in sorted(meta["local_f1"].items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
