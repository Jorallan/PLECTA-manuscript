"""The qualitative real-SEM figure: one field, five panels, whole frame.

The annotation on top, then a row per mask source: the binary axis PLECTA was
handed, and what PLECTA made of it. The same method and the same parameters
produced the two reconstructions; only the mask differs, and the two axis panels
are what that difference looks like.

    (a) annotator A's annotation, at the width they drew
    (b) the manual-derived axis      (c) PLECTA from (b)
    (d) the nnU-Net axis             (e) PLECTA from (d)

This replaces a pair of figures -- the two input axes, and the annotation beside
the two reconstructions -- which between them showed the same five things across
two floats and two crops, with the reader carrying panel (b) of one over to
panel (c) of the other. Merged, an axis sits beside the reconstruction it
produced.

**Whole field, not a crop.** The earlier pair drew a 512x344 window chosen as
the one whose local F1 was jointly closest to the whole-field value on both
axes. That rule was defensible and the caption had to spend a sentence
defending it; the frame the number was actually computed on needs no defence.

**All five are the same size.** (a) is the reference the other four are read
against and it has no counterpart beside it, so it is centred rather than
enlarged; drawing it larger cost the figure a page of its own and showed
nothing the same panel does not show at this size.

Drawing (a), (c) and (e) at width is what makes them the same kind of object.
The annotation is stored painted at the width the annotator drew, and PLECTA's
own optional image layer (Section 2.5) measures a width per instance by FWHM
across the centreline and stamps a ribbon, so the comparison is like with like
rather than a region against a hairline. That layer runs *after* the grouping is
fixed and cannot move a pixel from one instance to another, which is why no
score in the paper depends on it. (b) and (d) are the binary masks themselves,
in one colour, because within an axis panel there is nothing to tell apart.

Everything drawn comes from ``results/figure_assets/real_sem_field.npz``,
written by ``extract_real_sem_field.py``, which runs PLECTA and the width
rendering on the field and stores the result whole. Nothing is recomputed here.

    python scripts/figures/fig_real_sem.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _style import (FIG_W, FALSE_INST, PLECTA, framed,            # noqa: E402
                    paint, plecta_style, save_fig, tagged_title)

ASSET = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "figure_assets",
    "real_sem_field.npz"))

#: A binary axis is 2 px thick (manual) or 4 px (nnU-Net), and the whole field
#: is reduced by about a factor of two into a half-width panel.  Dilated by one
#: for *rendering only*, so an axis stays visible over the texture beneath it;
#: no geometry is changed and the caption says so.
DRAW_DILATION = 1
#: Two instances closer than this must not be given the same colour.
COLOUR_SEPARATION = 6

#: Width of panel (a) as a multiple of the four panels below it.  1.0, so all
#: five are the same size and (a) is simply the one that sits alone, centred.
#:
#: An earlier round drew (a) larger, on the argument that the reference the
#: other four are read against deserves the room.  It does not need it: at this
#: raster a strand is several pixels wide in every panel, so nothing was gained
#: that (a) did not already show, and the cost was the whole float.  At 0.87 of
#: \\linewidth the figure was 8.4 in before its caption, which is a text block,
#: so it had to be a [p] float on a page of its own; at 1.0 it is 6.8 in and
#: goes back to being an ordinary [tbp] float on a page with text.  Raising it
#: again means measuring the ceiling from LaTeX's "Float too large for page by
#: N pt" warning rather than guessing -- against this caption that ceiling was
#: 0.905 of \\linewidth -- and putting the figure back on its own page.
HERO_SCALE = 1.0

#: The raster resolution, in dpi, of the images written into the PDF.  Every
#: other figure in the set draws vectors and does not care; this one is five
#: photographs, and at the shared default of 150 a 1536 px field lands in a
#: half-width panel as 459 px -- a 3.3x reduction, which drops a two-pixel axis
#: stroke to two thirds of a pixel and breaks it into dashes.  At 220 the same
#: panel is 673 px, a 2.3x reduction that an antialiased filter carries as a
#: continuous line.  The PNG is written at 300 dpi by ``save_fig`` either way.
#:
#: It has to be given to ``plt.figure``, not to ``fig.set_dpi`` afterwards:
#: ``savefig`` resolves the default ``savefig.dpi="figure"`` against the dpi the
#: figure was *constructed* with, so a later ``set_dpi`` changes the figure and
#: not one pixel of the file.  The page is FIG_W inches wide either way, so the
#: include scale stays 1.0 and ``check_type_scale`` is satisfied.
RASTER_DPI = 220

#: How many identity colours to generate.  The shared ``INSTANCE_CYCLE`` has
#: twelve, which is the right size for a panel holding a handful of instances
#: and far too few here: the field carries over four hundred, and at twelve the
#: eye starts reading the repeat as a relationship.  A wider ramp is generated
#: instead, and the greedy colouring below still guarantees that no two
#: *adjacent* filaments share one.
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


def unpack(data, field: str, layer: str) -> tuple:
    """``(colour_ids, [flat pixel index array per instance])`` for one layer."""
    prefix = f"{field}__{layer}__"
    ids = data[prefix + "colour_ids"]
    indptr, indices = data[prefix + "indptr"], data[prefix + "indices"]
    return ids, [indices[indptr[k]:indptr[k + 1]] for k in range(len(ids))]


def grown_indices(run: np.ndarray, shape, radius: int) -> np.ndarray:
    """``run`` dilated by a square of ``radius``, still as flat indices.

    The dilation is done inside the run's own bounding box rather than over the
    frame.  On a 512x344 crop the difference did not matter; on a 1536x1024
    field with four hundred instances, dilating each one over the whole frame is
    several minutes, and this is a couple of seconds.  ``maximum_filter`` on a
    boolean array is the separable form of a square structuring element, so the
    result is what ``binary_dilation`` with ``ones((2r+1, 2r+1))`` gives.
    """
    if radius <= 0:
        return run
    from scipy.ndimage import maximum_filter
    height, width = shape
    rows, cols = np.divmod(run, width)
    r0 = max(int(rows.min()) - radius, 0)
    r1 = min(int(rows.max()) + radius + 1, height)
    c0 = max(int(cols.min()) - radius, 0)
    c1 = min(int(cols.max()) + radius + 1, width)
    local = np.zeros((r1 - r0, c1 - c0), bool)
    local[rows - r0, cols - c0] = True
    lr, lc = np.nonzero(maximum_filter(local, size=2 * radius + 1))
    return (lr + r0) * width + (lc + c0)


def colour_of_reference(shape, ids, runs) -> dict:
    """Greedy graph colouring, so no two nearby filaments share a colour.

    A cyclic palette assigned by instance id puts the same colour on adjacent
    filaments often enough to be read as one object.  Colouring the adjacency
    graph instead removes that reading.  Largest instance first, so the
    assignment is deterministic.

    Adjacency is read off a label image, on which overlap is irrelevant -- two
    filaments that cross are neighbours whichever of them a shared pixel is
    recorded under -- so the last one written wins there and nothing drawn comes
    from it.
    """
    height, width = shape
    label = np.zeros(height * width, np.int32)
    for ident, run in zip(ids, runs):
        label[run] = int(ident)
    label = label.reshape(shape)

    ids = [int(i) for i in ids]
    size = {int(i): len(r) for i, r in zip(ids, runs)}
    neighbours = {i: set() for i in ids}
    for ident, run in zip(ids, runs):
        grown = grown_indices(run, shape, COLOUR_SEPARATION)
        touching = {int(v) for v in np.unique(label.ravel()[grown]) if v > 0}
        touching.discard(ident)
        neighbours[ident] |= touching
        for j in touching:
            neighbours[j].add(ident)

    order = sorted(ids, key=lambda i: (-size[i], i))
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
    """The SEM field as a pale grey backdrop, as one RGB image.

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


def show(ax, image: np.ndarray) -> None:
    """Put one finished RGB panel on the page.

    ``interpolation="antialiased"`` rather than ``"nearest"``: the field is
    reduced by a factor of two or more into every panel, and nearest-neighbour
    sampling at that reduction keeps one pixel in four or five and throws the
    rest, which turns a continuous two-pixel axis into a dotted line.  The
    antialiased filter averages instead, so a stroke narrower than the output
    pixel survives as a lighter continuous one.
    """
    ax.imshow(np.clip(image, 0.0, 1.0), interpolation="antialiased")


def instances(sem: np.ndarray, ids, runs, colours: dict, unmatched: int,
              dilation: int = 0) -> np.ndarray:
    """One instance set over the micrograph, in correspondence colours.

    Instances are accumulated and averaged rather than painted one over another,
    so a pixel two filaments both own is drawn as the mean of their two colours.
    That is not decoration: PLECTA emits overlapping layers, a crossing pixel
    genuinely belongs to both filaments, and painting in sequence would show an
    arbitrary winner at exactly the place the method is doing its work.

    ``dilation`` is 0 for the width-rendered panels, which already carry the
    width the image was measured to give them, and 1 for a centreline panel,
    where a one-pixel instance would otherwise be invisible.
    """
    import matplotlib.colors as mcolors

    shape = sem.shape[:2]
    n = int(np.prod(shape))
    total = np.zeros((n, 3), dtype=float)
    count = np.zeros(n, dtype=float)

    for ident, run in zip(ids, runs):
        if int(ident) == unmatched:
            colour = FALSE_INST
        else:
            #  A prediction can be matched to an annotated filament that is
            #  itself unusual; it is matched, so it must not be drawn in the
            #  unmatched colour, and it takes a deterministic slot if the
            #  neighbour-aware pass never saw it.
            colour = colours.get(int(ident),
                                 IDENTITY_CYCLE[int(ident) % len(IDENTITY_CYCLE)])
        #  Indices within one instance are unique, so a plain fancy-indexed
        #  ``+=`` accumulates each of its pixels exactly once.
        idx = grown_indices(run, shape, dilation)
        total[idx] += np.asarray(mcolors.to_rgb(colour))
        count[idx] += 1.0

    image = micrograph_ground(sem).reshape(n, 3)
    hit = count > 0
    image[hit] = total[hit] / count[hit, None]
    return image.reshape(shape + (3,))


def axis_mask(sem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One binary axis over the micrograph, in a single colour."""
    from scipy.ndimage import maximum_filter
    image = micrograph_ground(sem)
    grown = maximum_filter(np.asarray(mask, bool),
                           size=2 * DRAW_DILATION + 1) if DRAW_DILATION \
        else np.asarray(mask, bool)
    return paint(image, grown, PLECTA)


def field_label(field: str) -> str:
    """``b58_110`` as the manuscript writes it, for the run log.

    The field name is no longer drawn: with one field it belongs in the caption,
    which is also the only place it can carry the ``B58\\_110`` escaping TeX
    wants and matplotlib would print literally.
    """
    return field.upper()


#  ── page layout ────────────────────────────────────────────────────────────
#
#  Laid out top-down in inches and converted to figure fractions once, at the
#  end.  The page must come out exactly FIG_W wide: ``check_type_scale`` reads
#  /MediaBox and fails the figure if the include scale is not 1.0, because at
#  any other scale a label specified at n pt does not print at n pt.  So no
#  ``bbox_inches="tight"``, and every title has to fit inside its own panel
#  rather than overhang and grow the canvas.

PAD = 0.005 * FIG_W          # side margin, inches
COL_GAP = 0.020 * FIG_W      # between the two panels of a row
ROW_GAP = 0.075              # between rows
TITLE_H = 0.19               # the strip a row's titles are set in
TITLE_LIFT = 0.045           # baseline clearance above a panel's top edge
TAG_GAP = 0.23               # from "(a)" to the first word of the title


def add_panel(fig, fig_h: float, left: float, top: float, width: float,
              height: float, tag: str, title: str):
    """One framed image panel, placed from its top-left corner in inches."""
    ax = fig.add_axes([left / FIG_W, (fig_h - top - height) / fig_h,
                       width / FIG_W, height / fig_h])
    framed(ax)
    tagged_title(ax, tag, title, dy=1.0 + TITLE_LIFT / height,
                 gap=TAG_GAP / width)
    return ax


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    unmatched = int(meta["unmatched_label"])
    field = list(meta["fields"])[0]

    sem = data[f"{field}__sem"]
    shape = sem.shape[:2]
    colours = colour_of_reference(shape, *unpack(data, field, "reference"))

    aspect = shape[0] / shape[1]
    panel_w = (FIG_W - 2 * PAD - COL_GAP) / 2.0
    panel_h = panel_w * aspect
    hero_w = HERO_SCALE * panel_w
    hero_h = hero_w * aspect
    fig_h = 3 * TITLE_H + hero_h + 2 * panel_h + 2 * ROW_GAP
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=RASTER_DPI)

    #  (a), centred: the reference the four panels below are read against.
    ax = add_panel(fig, fig_h, (FIG_W - hero_w) / 2.0, TITLE_H, hero_w, hero_h,
                   "a", "Annotator A's annotation")
    show(ax, instances(sem, *unpack(data, field, "reference_width"),
                       colours, unmatched))

    #  One row per mask source: the axis PLECTA was handed, then what it made
    #  of it.  Reading across a row is the whole point of the arrangement.
    rows = ((("b", "Manual-derived axis", "mask_manual"),
             ("c", "PLECTA, manual-derived axis", "manual_width")),
            (("d", "nnU-Net axis", "mask_unet"),
             ("e", "PLECTA, nnU-Net axis", "unet_width")))
    for r, ((atag, atitle, akey), (ptag, ptitle, pkey)) in enumerate(rows):
        top = 2 * TITLE_H + hero_h + ROW_GAP + r * (TITLE_H + panel_h + ROW_GAP)
        ax = add_panel(fig, fig_h, PAD, top, panel_w, panel_h, atag, atitle)
        show(ax, axis_mask(sem, data[f"{field}__{akey}"]))
        ax = add_panel(fig, fig_h, PAD + panel_w + COL_GAP, top,
                       panel_w, panel_h, ptag, ptitle)
        show(ax, instances(sem, *unpack(data, field, pkey), colours, unmatched))

    save_fig(fig, "fig_real_sem")
    plt.close(fig)

    block = meta["per_field"][field]
    print("[fig_real_sem] %s whole field %dx%d px, %d annotated filaments; %s"
          % (field_label(field), shape[1], shape[0], block["n_reference"],
             ", ".join("%s F1 %.4f, %d instances, %d matched, width on %d/%d "
                       "at median %.1f px"
                       % (k, block["field_f1"][k], block["n_instances"][k],
                          block["n_matched"][k],
                          block["width_render"][k]["n_measured"],
                          block["width_render"][k]["n_instances"],
                          block["width_render"][k]["scene_median_width_px"])
                       for k in sorted(block["field_f1"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
