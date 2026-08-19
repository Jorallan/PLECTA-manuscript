"""The depth stage on one real, manually-annotated SEM field.

Runtime and plumbing demonstration only -- see
`scripts/results/make_real_depth_demo.py` for why nothing here is scored.
No depth ground truth exists for real SEM data, so this figure shows what the
stage produces and how long it takes, not whether the depth is correct.

Panels: (a) the raw SEM; (b) the thin skeleton mask PLECTA actually reads
(the same "b58_skel" mask used for every other real-data PLECTA number in
this manuscript); (c) predicted 2-D instances, unchanged core; (d) crossing
decisions this scene produced (decided / abstained / reversed by the global
solver -- no correct/wrong colouring, because there is no truth to grade
against); (e) the shaded 3-D tube reconstruction.

Scene selection: not a selection -- there is exactly one real field with a
registered image at manuscript scale, and it is the one every other real-data
number already uses.

Writes to figures/archive/ (no `\\includegraphics` reaches it yet).

Usage:
    python scripts/figures/fig_depth_real.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _style import (CHORD, FIG_W, INK, JUNCTION, PT_ANNOT, PT_TITLE,  # noqa: E402
                    bare, framed, plecta_style, save_fig, tagged_title)
from _tube3d import draw_tubes_shaded, tube_mesh  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SCENE = r"C:\Repos\comparisons\real_sem_study\scenes_v2\b58_100\skel"
SUMMARY = os.path.join(REPO, "results", "plecta_real_depth_demo.json")
ASSETS = os.path.join(REPO, "results", "figure_assets", "real_depth_demo.npz")

DEPTH_CMAP = "viridis"


def main():
    plecta_style()
    from skimage import io as skio

    summary = json.load(open(SUMMARY, encoding="utf-8"))
    npz = np.load(ASSETS)
    sem = skio.imread(os.path.join(SCENE, "sem.png"))
    mask = skio.imread(os.path.join(SCENE, "mask.png")) > 0
    H, W = mask.shape

    ids = npz["ids"]
    z = npz["z"]
    d = npz["d"]
    off = npz["centreline_offsets"]
    coords = npz["centreline_coords"]
    finite_z = z[np.isfinite(z)]
    z_lo, z_hi = float(finite_z.min()), float(finite_z.max())
    span = max(1e-6, z_hi - z_lo)

    import matplotlib
    cmap = matplotlib.colormaps["viridis"]

    def z_colour(t):
        return cmap(0.10 + 0.85 * float(np.clip(t, 0, 1)))

    fig = plt.figure(figsize=(FIG_W, FIG_W * 1.16))
    gs = fig.add_gridspec(3, 2, left=0.02, right=0.98, top=0.965, bottom=0.058,
                          wspace=0.08, hspace=0.28,
                          height_ratios=(1.0, 1.0, 1.35))
    TAG_GAP = 0.10

    # (a) raw SEM ----------------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    framed(axa)
    axa.imshow(sem, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    tagged_title(axa, "a", "raw SEM (real field)", dy=1.02, gap=TAG_GAP)

    # (b) skeleton mask ------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    framed(axb)
    axb.imshow(~mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    tagged_title(axb, "b", "skeleton mask (PLECTA's only input)", dy=1.02,
                gap=TAG_GAP)

    # (c) predicted 2-D instances ---------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    framed(axc)
    axc.imshow(0.25 + 0.5 * (sem.astype(float) / 255.0), cmap="gray",
              vmin=0, vmax=1, interpolation="nearest")
    for k, iid in enumerate(ids):
        s, e = off[k], off[k + 1]
        pts = coords[s:e]
        if len(pts) < 2:
            continue
        h = (0.13 + k * 0.61803398875) % 1.0
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.9)
        axc.plot(pts[:, 0], pts[:, 1], color=(r, g, b), linewidth=0.5,
                 solid_capstyle="round")
    axc.set_xlim(0, W)
    axc.set_ylim(H, 0)
    tagged_title(axc, "c", f"predicted 2-D instances ({len(ids)})", dy=1.02,
                gap=TAG_GAP)

    # (d) crossing decisions ---------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    framed(axd)
    axd.imshow(0.3 + 0.5 * (sem.astype(float) / 255.0), cmap="gray",
              vmin=0, vmax=1, interpolation="nearest")
    id_to_z = {int(i): zz for i, zz in zip(ids, z)}
    n_decided = n_abstained = n_flipped = 0
    for i, j, x, y, over, abst, flip in zip(
            npz["cross_i"], npz["cross_j"], npz["cross_x"], npz["cross_y"],
            npz["cross_over"], npz["cross_abstain"], npz["cross_flipped"]):
        if abst:
            axd.plot(x, y, "o", ms=4.2, mfc="none", mec="#e0a10a", mew=1.2,
                     zorder=6)
            n_abstained += 1
            continue
        n_decided += 1
        zz = id_to_z.get(int(over))
        col = z_colour((zz - z_lo) / span) if zz is not None and np.isfinite(zz) \
            else CHORD
        axd.plot(x, y, "o", ms=2.6, mfc=col, mec="white", mew=0.35, zorder=7)
        if flip:
            axd.plot(x, y, "o", ms=6.4, mfc="none", mec=JUNCTION, mew=1.0,
                     zorder=6)
            n_flipped += 1
    axd.set_xlim(0, W)
    axd.set_ylim(H, 0)
    tagged_title(axd, "d", "crossing decisions (this scene, ungraded)",
                dy=1.02, gap=TAG_GAP)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.2, mfc=z_colour(0.8),
              mec="white", label="decided (colour = inferred z)"),
        Line2D([], [], marker="o", ls="none", ms=4.6, mfc="none",
              mec="#e0a10a", mew=1.2, label="abstained"),
        Line2D([], [], marker="o", ls="none", ms=6.6, mfc="none",
              mec=JUNCTION, mew=1.0, label="reversed by global solver"),
    ]
    # The shared style sets legend.frameon = False (bare markers read fine
    # against a plain panel), but here the legend sits over a dense scatter,
    # so it needs its own opaque frame -- frameon=True overrides the rcParam
    # and actually draws the facecolor/framealpha below rather than leaving
    # only the text and handles floating over the data.
    leg = axd.legend(handles=handles, loc="lower right", fontsize=6.6,
                     borderaxespad=0.3, handletextpad=0.35,
                     frameon=True, framealpha=1.0, facecolor="white",
                     edgecolor=CHORD)
    leg.set_zorder(50)

    # (e) shaded 3-D reconstruction ---------------------------------------
    # Title lives as a figure-level line ABOVE the 3-D axes, not as text2D on
    # the 3-D axes itself: text2D on a 3-D Axes is transformed through the
    # same projection as the data and can run past the figure's right edge
    # for a title this long, which is what produced the clipped caption.
    row_top = gs[2, :].get_position(fig).y1
    fig.text(0.015, row_top + 0.012, "(e)", fontsize=PT_TITLE,
             fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.015 + TAG_GAP * 0.5, row_top + 0.012,
             f"3-D reconstruction ({len(ids)} tubes, K = "
             f"{summary['depth']['n_layers']} layers) \u2014 unvalidated: "
             "no real-SEM depth ground truth exists",
             fontsize=PT_TITLE, color=INK, ha="left", va="bottom")

    axe = fig.add_subplot(gs[2, :], projection="3d")
    tubes = []
    for k, iid in enumerate(ids):
        s, e = off[k], off[k + 1]
        pts = coords[s:e]
        zz = z[k]
        if len(pts) < 2 or not np.isfinite(zz):
            continue
        radius = float(d[k]) / 2.0 if np.isfinite(d[k]) else 5.0
        verts, faces = tube_mesh(pts.astype(np.float64), float(zz), radius,
                                 n_theta=20, step=4.0)
        tubes.append((verts, faces, float(zz)))
    draw_tubes_shaded(axe, tubes, z_lo, span, extent=max(H, W), zoom=1.4)

    t2d = summary["timing_s"]["predict_2d"]
    tdepth = summary["timing_s"]["depth_stage"]
    fig.text(0.015, 0.020,
             f"{summary['shape'][0]}\u00d7{summary['shape'][1]} px real field, "
             f"{summary['n_instances_2d']} instances, "
             f"{summary['depth']['n_crossings']} crossings "
             f"({n_abstained} abstained, {n_flipped} reversed by the "
             "global solver).",
             fontsize=PT_ANNOT, color=INK, ha="left")
    fig.text(0.015, 0.002,
             f"Runtime: {t2d:.1f} s 2-D grouping + {tdepth:.1f} s depth "
             f"stage = {t2d + tdepth:.1f} s total, one field, "
             "single-threaded.",
             fontsize=PT_ANNOT, color=INK, ha="left")

    save_fig(fig, "fig_depth_real", subdir="archive")


if __name__ == "__main__":
    main()
