"""Figure 1 / the graphical abstract: a micrograph becomes individual strands.

Four panels, all measured rather than idealised: one fixed central crop of one
CNT SEM field, its nnU-Net strand-axis mask, the PLECTA reconstruction of that
mask, and the depth stage's layered arrangement of those same instances.

Panels (c) and (d) are drawn at the strands' own widths.  The width comes from
PLECTA's optional image layer, driven through the ``width_render`` of
``real_sem_study`` exactly as ``extract_real_sem_field.py`` drives it: one
width per instance, the median of its valid FWHM cuts across the centreline,
the scene median where an instance has too few cuts.  That layer runs after the
grouping is fixed, so it cannot move a pixel from one instance to another.

The field is ``b58_110``, which is held out of the nnU-Net run that produced
the mask.  ``b58_100`` is *not* drawn: ``results/figure_assets/real_sem_field.npz``
records that it "is contaminated under every available real-trained run".

The crop is the geometric centre of the field.  No crop search, no parameter
search, no scoring and no reference labels are involved.
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle       # noqa: E402
from PIL import Image                                           # noqa: E402

import _style                                                   # noqa: E402
from _style import (plecta_style, save_fig, FIG_W, framed,      # noqa: E402
                    INSTANCE_CYCLE, INK, GRAY, GREEN, PT_TITLE, PT_MIN)
from _tube3d import draw_tubes_shaded, tube_mesh                # noqa: E402

#  The width layer, and the alias shim it needs, are set up by the repository's
#  own extractor; importing it is how that bootstrap gets reused rather than
#  copied.
import extract_real_sem_field as _EX                            # noqa: E402
WR = _EX.WR

DEFAULT_SCENE = r"C:\Repos\comparisons\real_sem_study\scenes_v2\b58_110\nnunet"
DEFAULT_APP = r"C:\Repos\PLECTA_APP"
DEFAULT_PLECTA = r"C:\Repos\PLECTA"
CROP = 512

#: The crops the engine reads.  Derived from the scene on every run, so they
#: are scratch rather than data and do not belong beside the generator.
WORK = os.path.join(tempfile.gettempdir(), "plecta_fig_graphical_abstract")


def crops(scene):
    """The fixed central crop of the field, written where the engine can read it."""
    os.makedirs(WORK, exist_ok=True)
    sem = np.asarray(Image.open(os.path.join(scene, "sem.png")).convert("L"))
    mask = np.asarray(Image.open(os.path.join(scene, "mask.png")).convert("L"))
    y0, x0 = (sem.shape[0] - CROP) // 2, (sem.shape[1] - CROP) // 2
    box = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    Image.fromarray(sem[box]).save(os.path.join(WORK, "sem.png"))
    Image.fromarray(mask[box]).save(os.path.join(WORK, "mask.png"))
    return sem[box], mask[box], (y0, x0)


def reconstruct(app, plecta):
    """Run the frozen grouping configuration once on the fixed crop."""
    os.environ["PLECTA_SOURCE"] = plecta
    sys.path[:0] = [app, plecta]
    from plecta_app.pipeline import Engine
    from plecta_app.settings import Settings
    return Engine().run(Settings(
        scene_path=os.path.join(WORK, "mask.png"),
        image_path=os.path.join(WORK, "sem.png"),
        radius_source="fixed", fixed_radius_px=3.0,
        undecided_order="id_based"))


def instance_colour(inst):
    """The one place a strand's hue is decided, so (c) and (d) cannot drift."""
    return INSTANCE_CYCLE[inst["id"] % len(INSTANCE_CYCLE)]


def rasterise(xy, shape):
    """A one-pixel centreline raster of one instance's polyline.

    ``width_render`` skeletonises whatever mask it is handed, so a polyline
    stamped one pixel wide arrives at the same centreline the grouping
    produced.  Sampled at half-pixel steps, which cannot leave a gap.
    """
    m = np.zeros(shape, bool)
    d = np.hypot(*np.diff(xy, axis=0).T)
    n = np.maximum((d / 0.5).astype(int) + 1, 2)
    for (x0, y0), (x1, y1), k in zip(xy[:-1], xy[1:], n):
        t = np.linspace(0.0, 1.0, k)
        r = np.rint(y0 + t * (y1 - y0)).astype(int)
        c = np.rint(x0 + t * (x1 - x0)).astype(int)
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        m[r[ok], c[ok]] = True
    return m


def ribbons(scene_dir, mask, instances):
    """Width-render every instance, and recover the width each was drawn at.

    ``render_all`` returns the stamped masks and scene-level statistics, but
    keeps its per-instance ``width_rendered_px`` to itself.  Each ribbon is
    stamped at one constant width along its own centreline, so dividing a
    ribbon's area by that centreline's length returns the width it was stamped
    at -- arithmetic on the layer's own output, not a second measurement.
    """
    masks, record = WR.render_all(scene_dir, mask, instances)
    widths = {}
    for k, ribbon in masks.items():
        length = float(instances[k].sum())
        widths[k] = float(ribbon.sum()) / length if length else np.nan
    return masks, widths, record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--app", default=DEFAULT_APP)
    ap.add_argument("--plecta", default=DEFAULT_PLECTA)
    args = ap.parse_args(argv)

    from pathlib import Path
    sem, mask, origin = crops(args.scene)
    result = reconstruct(args.app, args.plecta)
    instances = [i for i in result["instances"] if len(i["xy"]) > 1]

    #  Width-render through PLECTA's own image layer, on the same crop.
    axis = np.asarray(mask > 0)
    raster = {i["id"]: rasterise(np.asarray(i["xy"], float), axis.shape)
              for i in instances}
    ribbon, width_px, record = ribbons(Path(WORK), axis, raster)

    plecta_style()
    if not _style._FONT_STATE["resolved"]:
        raise SystemExit("body face did not resolve; refusing to draw the "
                         "graphical abstract in the fallback face")

    FIG_H = 1.68
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    #  Laid out in inches and converted once: a title placed in axes fractions
    #  moves when the panel width does.
    fx = lambda v: v / FIG_W                                    # noqa: E731
    fy = lambda v: v / FIG_H                                    # noqa: E731

    left_in, gap_in = 0.09, 0.30
    w_in = (FIG_W - 2 * left_in - 3 * gap_in) / 4.0
    top_in = 0.30
    xs = [fx(left_in + k * (w_in + gap_in)) for k in range(4)]
    w, h = fx(w_in), fy(w_in)                                   # square
    top = 1.0 - fy(top_in)

    titles = [("a", "Micrograph"), ("b", "Strand-axis mask"),
              ("c", "Strand instances"), ("d", "Layered arrangement")]

    axes = []
    for k in range(3):
        ax = fig.add_axes([xs[k], top - h, w, h])
        framed(ax, GRAY, 0.5)
        ax.set_xlim(0, CROP)
        ax.set_ylim(CROP, 0)
        axes.append(ax)

    axes[0].imshow(sem, cmap="gray", interpolation="nearest",
                   extent=(0, CROP, CROP, 0))
    axes[1].imshow(mask > 0, cmap="gray_r", interpolation="nearest",
                   extent=(0, CROP, CROP, 0), vmin=0, vmax=1)

    #  (c) the instances at the width the SEM says each strand has, painted as
    #  an RGB image so a crossing shows the strand drawn over it rather than a
    #  blend of the two.
    rgb = np.ones(axis.shape + (3,), float)
    for inst in instances:
        sel = ribbon[inst["id"]]
        rgb[sel] = matplotlib.colors.to_rgb(instance_colour(inst))
    axes[2].imshow(rgb, interpolation="nearest", extent=(0, CROP, CROP, 0))

    #  A scale bar in pixels: the scene carries no calibrated pixel size
    #  (``scale.measured`` is false), so the bar is not labelled in nm.  It
    #  sits on its own dark plate, having been unreadable against the bare
    #  micrograph wherever the crop happened to be bright.
    bar, pad = 100.0, 9.0
    axes[0].add_patch(Rectangle((CROP - bar - 2 * pad, CROP - 40),
                                bar + 2 * pad, 40, fc="black", ec="none",
                                alpha=0.62, zorder=6))
    axes[0].add_patch(Rectangle((CROP - bar - pad, CROP - 15), bar, 4.0,
                                fc="white", ec="none", zorder=7))
    axes[0].text(CROP - bar / 2.0 - pad, CROP - 20, "100 px", color="white",
                 fontsize=PT_MIN, ha="center", va="bottom", zorder=7)

    #  (d) the same instances as solid tubes at their estimated heights,
    #  carrying the hues of (c) so a reader can see that the layered objects
    #  are the instances recovered one panel earlier.  The colours are passed
    #  through rather than left to the 3-D module's own ramp, which is a
    #  different palette; ``colours`` is aligned with ``tubes`` by
    #  construction.
    tubes, tube_cols, zs = [], [], []
    for inst in instances:
        xy = np.asarray(inst["xy"], float)
        z = float(inst["z"])
        #  The measured half-width, so a tube is as thick as its strand is
        #  wide; the engine's own radius here is the fixed placeholder that
        #  only sets the contact separation for the depth solve.
        r = 0.5 * float(width_px[inst["id"]])
        v, f = tube_mesh(xy, z, r, n_theta=12)
        tubes.append((v, f, z))
        tube_cols.append(instance_colour(inst))
        zs.append(z)
    z_lo, z_hi = min(zs), max(zs)
    span = max(z_hi - z_lo, 1.0)
    ax3 = fig.add_axes([xs[3], top - h, w, h], projection="3d")
    ax3.set_facecolor("none")
    draw_tubes_shaded(ax3, tubes, z_lo, span, extent=CROP, zoom=1.70,
                      elev=24, azim=-62, colours=tube_cols,
                      z_stretch=span / CROP)
    axes.append(ax3)

    #  Titles: tag bold, words regular, offset by a fixed inch gap.
    y_title = top + fy(0.045)
    for k, (tag, text) in enumerate(titles):
        fig.text(xs[k], y_title, f"({tag})", fontsize=PT_TITLE,
                 fontweight="bold", color=INK, ha="left", va="bottom")
        fig.text(xs[k] + fx(0.235), y_title, text, fontsize=PT_TITLE,
                 color=INK, ha="left", va="bottom")

    y_arrow = top - h / 2.0
    for k in range(3):
        fig.patches.append(FancyArrowPatch(
            (xs[k] + w + fx(0.045), y_arrow), (xs[k + 1] - fx(0.045), y_arrow),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7.5,
            lw=0.9, color=GREEN, shrinkA=0, shrinkB=0, zorder=5))

    save_fig(fig, "fig_graphical_abstract", pdf_dpi=300, bbox_inches=None)
    print("field              :", os.path.basename(os.path.dirname(args.scene)))
    print("crop origin (y, x) :", origin)
    print("instances drawn    :", len(tubes))
    print("z range (px)       : %.1f .. %.1f" % (z_lo, z_hi))
    print("width measured     : %d/%d instances, scene median %s px"
          % (record["n_measured"], record["n_instances"],
             record["scene_median_width_px"]))
    ws = np.array([width_px[i["id"]] for i in instances], float)
    print("width drawn (px)   : min %.1f  med %.1f  max %.1f"
          % (ws.min(), np.median(ws), ws.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
