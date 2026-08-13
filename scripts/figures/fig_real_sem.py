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

from _style import (FIG_W, FALSE_INST, INK, PT_TITLE,             # noqa: E402
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

#: How many identity colours to generate.  The shared ``INSTANCE_CYCLE`` has
#: twelve, which is the right size for a panel holding a handful of instances
#: and far too few here: a crop of this field carries sixty to ninety, and at
#: twelve the eye starts reading the repeat as a relationship.  A wider ramp is
#: generated instead, and the greedy colouring below still guarantees that no
#: two *adjacent* filaments share one.
N_IDENTITY_COLOURS = 24

#: Hues within this band are reserved for ``FALSE_INST`` (#C2185B, hue 0.93).
#: An unmatched instance is the one thing in these figures the reader must be
#: able to pick out at a glance, and it cannot be if a legitimate filament
#: beside it is drawn a shade away from the alarm colour.  The band is wide:
#: a first attempt reserved 0.86-0.99 and still put a magenta filament in the
#: annotation panel that read as an unmatched one.
RESERVED_HUES = (0.80, 1.00)


def identity_palette(n: int = N_IDENTITY_COLOURS) -> tuple:
    """``n`` well-separated colours, avoiding the reserved band.

    Hues advance by the golden angle, which spreads any prefix of the sequence
    about as evenly round the wheel as a set of that size can be, so the palette
    does not need re-tuning when ``n`` changes.  Saturation and value cycle
    through three settings as well, so two filaments that happen to land on
    close hues still differ in tone rather than only in colour, which is what
    keeps them apart in greyscale print.
    """
    import colorsys
    lo, hi = RESERVED_HUES
    tones = ((0.78, 0.72), (0.58, 0.90), (0.92, 0.52))
    out, hue, k = [], 0.11, 0
    while len(out) < n:
        hue = (hue + 0.6180339887498949) % 1.0
        if lo <= hue <= hi:
            continue
        s, v = tones[k % len(tones)]
        out.append("#%02X%02X%02X" % tuple(
            round(255 * c) for c in colorsys.hsv_to_rgb(hue, s, v)))
        k += 1
    return tuple(out)


IDENTITY_CYCLE = identity_palette()

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


def unpack(data, field: str, layer: str) -> tuple:
    """``(colour_ids, [pixel index array per instance])`` for one layer."""
    prefix = f"{field}__{layer}__"
    ids = data[prefix + "colour_ids"]
    indptr, indices = data[prefix + "indptr"], data[prefix + "indices"]
    return ids, [indices[indptr[k]:indptr[k + 1]] for k in range(len(ids))]


def label_image(shape, ids, runs) -> np.ndarray:
    """A label image, for the colouring pass only.

    Overlap is irrelevant to choosing colours -- two filaments that cross are
    neighbours either way -- so the adjacency graph is built on a flattened
    view.  Nothing drawn comes from this.
    """
    out = np.zeros(int(np.prod(shape)), np.int32)
    for ident, run in zip(ids, runs):
        out[run] = ident
    return out.reshape(shape)


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
    used = [0] * len(IDENTITY_CYCLE)
    for i in order:
        taken = {assigned[j] for j in neighbours[i] if j in assigned}
        #  Least-used admissible colour, not the lowest-numbered one.  Taking
        #  the lowest free slot is the textbook greedy rule and it wastes the
        #  palette: it packs everything into the first handful of colours and
        #  a twenty-four colour ramp comes out looking like a six colour one.
        slot = min((k for k in range(len(IDENTITY_CYCLE)) if k not in taken),
                   key=lambda k: (used[k], k), default=0)
        assigned[i] = slot
        used[slot] += 1
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


def overlay(ax, sem: np.ndarray, ids, runs, colours: dict,
            unmatched: int, dilation: int = DRAW_DILATION) -> None:
    """Draw one instance set over the micrograph, in correspondence colours.

    Instances are accumulated and averaged rather than painted one over another,
    so a pixel two filaments both own is drawn as the mean of their two colours.
    That is not decoration: PLECTA emits overlapping layers, a crossing pixel
    genuinely belongs to both filaments, and painting in sequence would show an
    arbitrary winner at exactly the place the method is doing its work.

    ``dilation`` is 1 for centreline panels, where a one-pixel instance would
    otherwise be invisible, and 0 for width-rendered ones, which already carry
    the width the image was measured to give them.
    """
    import matplotlib.colors as mcolors

    shape = sem.shape[:2]
    total = np.zeros(shape + (3,), dtype=float)
    count = np.zeros(shape, dtype=float)
    scratch = np.zeros(int(np.prod(shape)), dtype=bool)

    for ident, run in zip(ids, runs):
        if ident == unmatched:
            colour = FALSE_INST
        else:
            #  A prediction can be matched to an annotated filament whose own
            #  pixels all lie outside the crop.  It is matched, so it must not
            #  be drawn in the unmatched colour; it just has no neighbour-aware
            #  slot, so it takes a deterministic one.
            colour = colours.get(int(ident),
                                 IDENTITY_CYCLE[int(ident) % len(IDENTITY_CYCLE)])
        scratch[run] = True
        #  At dilation 0 ``dilate`` hands back a view of ``scratch`` itself, so
        #  the accumulation has to happen before the buffer is cleared.  Doing
        #  it the other way round zeroes the mask under the reader's feet and
        #  every width-rendered panel comes out empty.
        mask = dilate(scratch.reshape(shape), dilation)
        total[mask] += np.asarray(mcolors.to_rgb(colour))
        count[mask] += 1.0
        scratch[run] = False

    image = micrograph_ground(sem)
    hit = count > 0
    image[hit] = total[hit] / count[hit, None]
    ax.imshow(image, interpolation="nearest")


def panel_grid(fig, rows, unmatched, dilation, gap_frac, label_frac=0.0):
    """Lay a grid of equal panels across the page and draw each one.

    ``rows`` is a sequence of ``(row_label, sem, colours, panels)``, where
    ``panels`` is ``[(tag, title, labels), ...]``.  Column titles are written
    once, above the top row, because every row shows the same three things for a
    different field; the field name runs up the left edge instead, which costs a
    strip of width rather than a line of height per row.  Colours travel with
    the row: each field has its own annotation and so its own assignment.

    The page must come out exactly FIG_W wide: ``check_type_scale`` reads
    /MediaBox and fails the figure if the include scale is not 1.0, because at
    any other scale a label specified at n pt does not print at n pt.  So no
    ``bbox_inches="tight"`` anywhere in this set, and every title has to fit
    inside its own panel rather than overhang and grow the canvas.
    """
    pad, gap, row_gap_in = 0.005, 0.020, 0.075
    n_cols = len(rows[0][3])
    left = pad + label_frac
    panel = (1.0 - pad - left - (n_cols - 1) * gap) / n_cols
    panel_in = panel * FIG_W
    crop_h, crop_w = rows[0][1].shape[:2]
    panel_h_in = panel_in * crop_h / crop_w
    fig_h = len(rows) * panel_h_in + (len(rows) - 1) * row_gap_in + 0.19
    fig.set_size_inches(FIG_W, fig_h)

    for r, (row_label, sem, colours, panels) in enumerate(rows):
        bottom_in = (len(rows) - 1 - r) * (panel_h_in + row_gap_in)
        for column, (tag, title, (ids, runs)) in enumerate(panels):
            ax = fig.add_axes([left + column * (panel + gap),
                               bottom_in / fig_h, panel, panel_h_in / fig_h])
            framed(ax)
            overlay(ax, sem, ids, runs, colours, unmatched, dilation)
            if r == 0:
                tagged_title(ax, tag, title, dy=1.02, gap=gap_frac)
        if row_label:
            fig.text(left - 0.008, (bottom_in + panel_h_in / 2) / fig_h,
                     row_label, rotation=90, ha="right", va="center",
                     fontsize=PT_TITLE, color=INK)


def field_label(field: str) -> str:
    """``b58_110`` as it is written in the text.

    No LaTeX escaping: these labels are drawn by matplotlib, not typeset by
    TeX, so a backslash before the underscore prints as a backslash.
    """
    return field.upper()


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    unmatched = int(meta["unmatched_label"])
    fields = list(meta["fields"])

    #  One row per field, three panels each, drawn at the width every instance
    #  was measured to have, so the annotation and the two reconstructions are
    #  compared as the same kind of object.  Dilation is 0: these carry width.
    rows = []
    for field in fields:
        sem = data[f"{field}__sem"]
        reference = unpack(data, field, "reference")
        rows.append((
            field_label(field), sem,
            colour_of_reference(label_image(sem.shape[:2], *reference)),
            (("a", "Manual annotation", unpack(data, field, "reference_width")),
             ("b", "PLECTA, manual-derived axis",
              unpack(data, field, "manual_width")),
             ("c", "PLECTA, nnU-Net axis", unpack(data, field, "unet_width"))),
        ))

    fig = plt.figure()
    panel_grid(fig, rows, unmatched, dilation=0, gap_frac=0.118,
               label_frac=0.016)
    save_fig(fig, "fig_real_sem")
    plt.close(fig)

    for field in fields:
        block = meta["per_field"][field]
        print("[fig_real_sem] %s crop row %d col %d, %dx%d px; width on %s"
              % (field, block["crop_row"], block["crop_col"],
                 block["crop_w"], block["crop_h"],
                 ", ".join("%s %d/%d at median %.1f px"
                           % (k, v["n_measured"], v["n_instances"],
                              v["scene_median_width_px"])
                           for k, v in sorted(block["width_render"].items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
