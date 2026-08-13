"""The qualitative real-SEM figure: reconstructed bundles, at width, on the image.

Three panels across ``\\linewidth``: the manual annotation, PLECTA from the
manual-derived axis, and PLECTA from the nnU-Net axis, each drawn over the
micrograph it came from and each at the width its filaments were measured to
have. The same method and the same parameters produced the two reconstructions;
only the mask differs.

Drawing all three at width is what makes them the same kind of object. The
annotation is stored painted at the width the annotator drew, and PLECTA's own
optional image layer (Section 2.5) measures a width per instance by FWHM across
the centreline and stamps a ribbon, so the comparison is like with like rather
than a region against a hairline. That layer runs *after* the grouping is fixed
and cannot move a pixel from one instance to another, which is why no score in
the paper depends on it -- the caption says so, and the centreline form the
scores are actually computed on is ``fig_real_sem_axis``.

Everything drawn comes from ``results/figure_assets/real_sem_crop.npz``, written
by ``extract_real_sem_crop.py``, which runs PLECTA and the width rendering on
the whole field and cuts the crop afterwards. Nothing is recomputed here.

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
                    framed, paint, plecta_style, save_fig, tagged_title)

ASSET = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "figure_assets",
    "real_sem_crop.npz"))

#: Instances are one pixel wide.  Dilated by one for *rendering only*, so that a
#: centreline stays visible over the texture beneath it; no geometry is changed
#: and the caption says so.
DRAW_DILATION = 1
#: Two instances closer than this must not be given the same colour.
COLOUR_SEPARATION = 6

#: The identity palette, minus the three entries close enough to ``FALSE_INST``
#: (#C2185B) to be confused with it: two pinks and a brick red.  An unmatched
#: instance is the one thing in these figures the reader must be able to pick
#: out at a glance, and it cannot be if a legitimate filament beside it is drawn
#: a shade away from the alarm colour.  Nine colours are still more than the
#: greedy colouring needs.
IDENTITY_CYCLE = tuple(c for c in INSTANCE_CYCLE
                       if c not in ("#B03A6A", "#D45087", "#C0392B"))

#: The micrograph keeps its natural polarity -- bright filaments on a dark
#: ground -- because inverting it does not work here: the dark background
#: between filaments becomes white blobs that read as the objects, and the
#: filaments become a grey web between them, which is exactly backwards. Shown
#: as it is, the ridge under each instance is the thing the instance claims to
#: be, and the eye checks the two against each other directly. The range is
#: compressed away from both ends so that neither the background nor a bright
#: ridge competes with the saturated instance colours laid over them.
DARKEST, BRIGHTEST = 0.10, 0.82


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
    crop) and removes that reading.  Largest instance first, so the assignment
    is deterministic.
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


def micrograph_ground(sem: np.ndarray) -> np.ndarray:
    """The SEM crop as a pale grey backdrop, as one RGB image.

    Contrast is set on the 2nd and 98th percentiles rather than on the extremes:
    a handful of saturated specks otherwise take the whole range and the ridges
    collapse into an even haze, which is what a plain min-max stretch did here.
    """
    norm = np.asarray(sem, float)
    lo, hi = (float(v) for v in np.percentile(norm, (2.0, 98.0)))
    norm = np.clip((norm - lo) / (hi - lo), 0.0, 1.0) if hi > lo \
        else np.zeros_like(norm)
    grey = DARKEST + (BRIGHTEST - DARKEST) * norm
    return np.dstack([grey, grey, grey])


def overlay(ax, sem: np.ndarray, labels: np.ndarray, colours: dict,
            unmatched: int, dilation: int = DRAW_DILATION) -> None:
    """Draw one instance set over the micrograph, in correspondence colours.

    ``dilation`` is 1 for centreline panels, where a one-pixel instance would
    otherwise be invisible, and 0 for width-rendered ones, which already carry
    the width the image was measured to give them.
    """
    image = micrograph_ground(sem)
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
        paint(image, dilate(labels == ident, dilation), colour)
    ax.imshow(image, interpolation="nearest")


def panel_row(fig, sem, panels, colours, unmatched, dilation, gap_frac):
    """Lay a row of equal panels across the page and draw each one.

    The page must come out exactly FIG_W wide: ``check_type_scale`` reads
    /MediaBox and fails the figure if the include scale is not 1.0, because at
    any other scale a label specified at n pt does not print at n pt.  So no
    ``bbox_inches="tight"`` anywhere in this set, and every title has to fit
    inside its own panel rather than overhang and grow the canvas.
    """
    pad, gap = 0.005, 0.020
    n = len(panels)
    panel = (1.0 - 2 * pad - (n - 1) * gap) / n
    panel_in = panel * FIG_W
    crop_h, crop_w = sem.shape[:2]
    panel_h_in = panel_in * crop_h / crop_w
    fig.set_size_inches(FIG_W, panel_h_in + 0.19)
    fig_h = float(fig.get_size_inches()[1])
    for column, (tag, title, labels) in enumerate(panels):
        ax = fig.add_axes([pad + column * (panel + gap), 0.0,
                           panel, panel_h_in / fig_h])
        framed(ax)
        overlay(ax, sem, labels, colours, unmatched, dilation)
        tagged_title(ax, tag, title, dy=1.02, gap=gap_frac)


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    unmatched = int(meta["unmatched_label"])
    sem = data["sem"]
    colours = colour_of_reference(data["reference"])

    #  Three panels, drawn at the width each instance was measured to have, so
    #  the annotation and the two reconstructions are compared as the same kind
    #  of object.  Dilation is 0 here: these already carry their width.
    fig = plt.figure()
    panel_row(fig, sem,
              (("a", "Manual annotation", data["reference_width"]),
               ("b", "PLECTA, manual-derived axis", data["manual_width"]),
               ("c", "PLECTA, nnU-Net axis", data["unet_width"])),
              colours, unmatched, dilation=0, gap_frac=0.118)
    save_fig(fig, "fig_real_sem")
    plt.close(fig)

    widths = meta.get("width_render", {})
    print("[fig_real_sem] %s crop row %d col %d, %dx%d px; "
          "width measured on %s"
          % (meta["field"], meta["crop_row"], meta["crop_col"],
             meta["crop_w"], meta["crop_h"],
             ", ".join("%s %d/%d at median %.1f px"
                       % (k, v["n_measured"], v["n_instances"],
                          v["scene_median_width_px"])
                       for k, v in sorted(widths.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
