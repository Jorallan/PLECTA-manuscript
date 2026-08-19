"""One depth-resolved synthetic scene: latent truth and every observation.

Sanity figure for the 2.5-D generator (`eval/generators/synth_depth.py` in
filaments_quantification). Six panels:

    (a) latent 3-D geometry -- planar rods at their true z, line width
        proportional to physical diameter, coloured by depth;
    (b) top-down projection with every projected crossing marked: the marker
        takes the colour of the rod on top, a heavy ring flags physical
        contact (|z_i - z_j| = r_i + r_j);
    (c) rendered grayscale observation (the image PLECTA's depth stage reads);
    (d) degraded thin axis mask, w = 2 (the only input the 2-D core reads);
    (e) depth map: z of the topmost instance per covered pixel;
    (f) minimal-K layer assignment derived from the crossing DAG.

No `\\includegraphics` reaches this figure yet, so it writes to
`figures/archive/` (`save_fig(..., subdir="archive")`) per repository rule.

Scene selection: the default scene is the first development scene of the
densest development coverage (cov35/synth_0000) -- first by index, not chosen
by result.

Usage:
    python scripts/figures/fig_depth_scene.py [--scene <scene_dir>]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _style import (CHORD, INK, JUNCTION, bare, framed, plecta_style,  # noqa: E402
                    save_fig, tagged_title, PT_ANNOT)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

DEFAULT_SCENE = os.path.normpath(
    "C:/Repos/filaments_quantification/input/synthetic_depth_dev/cov35/synth_0000")

DEPTH_CMAP = "viridis"


def load_scene(scene):
    from skimage import io as skio
    import tifffile

    with open(os.path.join(scene, "gt_depth.json"), encoding="utf-8") as fh:
        gt = json.load(fh)
    sem = skio.imread(os.path.join(scene, "sem.png"))
    mask = skio.imread(os.path.join(scene, "mask_w2.png")) > 0
    depth = tifffile.imread(os.path.join(scene, "depth_map.tif"))
    labels = tifffile.imread(os.path.join(scene, "gt_labels.tif"))
    return gt, sem, mask, depth, labels


def z_colour(norm_z):
    import matplotlib
    return matplotlib.colormaps[DEPTH_CMAP](0.10 + 0.85 * norm_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    args = ap.parse_args()

    plecta_style()
    gt, sem, mask, depth, labels = load_scene(args.scene)
    inst = gt["instances"]
    zs = np.array([r["z"] for r in inst])
    z_lo, z_hi = float(zs.min()), float(zs.max())
    span = max(1e-6, z_hi - z_lo)

    from _style import FIG_W
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.72))
    gs = fig.add_gridspec(2, 3, left=0.015, right=0.955, top=0.93, bottom=0.02,
                          wspace=0.10, hspace=0.16)
    TAG_GAP = 0.105   # axes-fraction gap between the (a) tag and the title

    # (a) latent 3-D geometry --------------------------------------------
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    H, W = np.asarray(sem).shape[:2]
    for r in inst:
        pts = np.asarray(r["centerline_xy"], dtype=float)
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        seg3 = np.concatenate(
            [seg, np.full((seg.shape[0], 2, 1), r["z"])], axis=2)
        col = z_colour((r["z"] - z_lo) / span)
        ax3.add_collection3d(Line3DCollection(
            seg3, colors=[col], linewidths=0.5 + 0.16 * r["d_mean"]))
    ax3.set_xlim(0, W)
    ax3.set_ylim(H, 0)
    ax3.set_zlim(z_lo - 8, z_hi + 8)
    ax3.set_box_aspect((1, 1, 0.42), zoom=1.45)
    ax3.view_init(elev=26, azim=-64)
    ax3.set_axis_off()
    ax3.text2D(0.5, 0.03,
               f"z span {span:.0f} px, K = {gt['n_layers']} layers",
               transform=ax3.transAxes, ha="center", va="top",
               fontsize=PT_ANNOT, color=INK)
    from _style import PT_TITLE
    ax3.text2D(0.0, 1.02, "(a)", transform=ax3.transAxes, ha="left",
               va="bottom", fontsize=PT_TITLE, fontweight="bold", color=INK)
    ax3.text2D(TAG_GAP, 1.02, "latent 3-D scene", transform=ax3.transAxes,
               ha="left", va="bottom", fontsize=PT_TITLE, color=INK)

    # (b) top-down projection with crossing order -------------------------
    axb = fig.add_subplot(gs[0, 1])
    framed(axb)
    for r in inst:
        pts = np.asarray(r["centerline_xy"], dtype=float)
        col = z_colour((r["z"] - z_lo) / span)
        axb.plot(pts[:, 0], pts[:, 1], color=col,
                 linewidth=0.35 + 0.10 * r["d_mean"], solid_capstyle="round")
    z_of = {r["id"]: r["z"] for r in inst}
    for c in gt["crossings"]:
        col = z_colour((z_of[c["over"]] - z_lo) / span)
        if c["contact"]:
            axb.plot(c["x"], c["y"], "o", ms=4.6, mfc="none", mec=JUNCTION,
                     mew=1.1, zorder=6)
        axb.plot(c["x"], c["y"], "o", ms=2.4, mfc=col, mec="white",
                 mew=0.4, zorder=7)
    axb.set_xlim(0, W)
    axb.set_ylim(H, 0)
    tagged_title(axb, "b", "projection + crossing order", dy=1.02, gap=TAG_GAP)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc=z_colour(0.9),
               mec="white", label="marker = rod on top"),
        Line2D([], [], marker="o", ls="none", ms=5.2, mfc="none",
               mec=JUNCTION, mew=1.1, label="ring = contact"),
    ]
    axb.legend(handles=handles, loc="lower left", fontsize=PT_ANNOT,
               borderaxespad=0.1, handletextpad=0.3)

    # (c) grayscale observation ------------------------------------------
    axc = fig.add_subplot(gs[0, 2])
    framed(axc)
    axc.imshow(sem, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    tagged_title(axc, "c", "rendered observation", dy=1.02, gap=TAG_GAP)

    # (d) thin axis mask --------------------------------------------------
    axd = fig.add_subplot(gs[1, 0])
    framed(axd)
    axd.imshow(~mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    tagged_title(axd, "d", "degraded axis mask (w = 2)", dy=1.02, gap=TAG_GAP)

    # (e) depth map -------------------------------------------------------
    axe = fig.add_subplot(gs[1, 1])
    framed(axe)
    dm = np.ma.masked_invalid(depth)
    im = axe.imshow(dm, cmap=DEPTH_CMAP, interpolation="nearest")
    im.get_cmap().set_bad("white")
    tagged_title(axe, "e", "depth map (topmost z)", dy=1.02, gap=TAG_GAP)
    cb = fig.colorbar(im, ax=axe, fraction=0.043, pad=0.02)
    cb.ax.tick_params(labelsize=PT_ANNOT)
    cb.set_label("z (px)", fontsize=PT_ANNOT)

    # (f) layers ----------------------------------------------------------
    axf = fig.add_subplot(gs[1, 2])
    framed(axf)
    layer_of = {r["id"]: r["layer"] for r in inst}
    n_layers = int(gt["n_layers"])
    layer_img = np.full(labels.shape, np.nan, dtype=float)
    for fid, lay in layer_of.items():
        layer_img[labels == fid] = lay
    lm = np.ma.masked_invalid(layer_img)
    imf = axf.imshow(lm, cmap=DEPTH_CMAP, vmin=0, vmax=max(1, n_layers - 1),
                     interpolation="nearest")
    imf.get_cmap().set_bad("white")
    tagged_title(axf, "f", f"layers (minimal K = {n_layers})", dy=1.02,
                 gap=TAG_GAP)
    cbf = fig.colorbar(imf, ax=axf, fraction=0.043, pad=0.02,
                       ticks=list(range(n_layers)))
    cbf.ax.tick_params(labelsize=PT_ANNOT)
    cbf.set_label("layer", fontsize=PT_ANNOT)

    save_fig(fig, "fig_depth_scene", subdir="archive")


if __name__ == "__main__":
    main()
