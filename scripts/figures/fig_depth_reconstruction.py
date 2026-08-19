"""Latent 3-D truth beside the full post-hoc 3-D reconstruction.

Left column: the generator's latent scene, rendered as shaded solid tubes at
the true diameters and depths. Right column: the reconstruction from the mask
and grayscale image alone (PLECTA 2-D instances, image-derived crossing
order, global consistency, measured diameters, compact-stack metric z),
rendered the same way. Two viewpoints per scene: oblique and near-side
elevation. Equal axes; both sides share one depth colour scale after removing
the free global z offset (median residual over matched instances), consistent
with the offset-free metric-z scores.

Scene selection: first held-out scene of the densest coverage
(cov35/synth_0000) -- first by index, not chosen by result.

Writes to figures/archive/ (no `\\includegraphics` reaches it yet).

Usage:
    python scripts/figures/fig_depth_reconstruction.py \
        [--run exploration/depth_25d/runs/heldout__pred_meas] \
        [--scenes C:/Repos/filaments_quantification/input/synthetic_depth_heldout] \
        [--scene cov35/synth_0000]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _style import (FIG_W, INK, PT_ANNOT, PT_TITLE, plecta_style,  # noqa: E402
                    save_fig)
from _tube3d import draw_tubes_shaded, tube_mesh  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_RUN = os.path.join(REPO, "exploration", "depth_25d", "runs",
                           "heldout__pred_meas")
DEFAULT_SCENES = r"C:\Repos\filaments_quantification\input\synthetic_depth_heldout"
DEFAULT_SCENE = "cov35/synth_0000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--scenes", default=DEFAULT_SCENES)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    args = ap.parse_args()

    plecta_style()
    cov, scene = args.scene.split("/")
    gt = json.load(open(os.path.join(args.scenes, cov, scene,
                                     "gt_depth.json"), encoding="utf-8"))
    pred = json.load(open(os.path.join(args.run, f"{cov}__{scene}",
                                       "pred_depth.json"), encoding="utf-8"))
    imap = {int(k): v for k, v in pred.get("instance_map", {}).items()}

    gt_z = {r["id"]: r["z"] for r in gt["instances"]}
    # free global offset: median z residual over mapped instances
    resid = [r["z"] - gt_z[imap[r["id"]]] for r in pred["instances"]
             if r["z"] is not None and imap.get(r["id"], 0) in gt_z]
    offset = float(np.median(resid)) if resid else 0.0

    def mesh(r, z, radius):
        verts, faces = tube_mesh(np.asarray(r["centerline_xy"], float), z,
                                 radius, n_theta=22, step=3.5)
        return verts, faces, z

    gt_tubes = [mesh(r, r["z"], r["d_mean"] / 2.0) for r in gt["instances"]]
    pr_tubes = [mesh(r, r["z"] - offset,
                     (r["d"] / 2.0) if r.get("d") else 5.0)
                for r in pred["instances"] if r["z"] is not None]

    zs = [t[2] for t in gt_tubes] + [t[2] for t in pr_tubes]
    z_lo, z_hi = min(zs), max(zs)
    span = max(1e-6, z_hi - z_lo)

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.62))
    # the raised, more top-down "oblique" view reads the layered stack far
    # better than a near-flat one: with the tubes packed close in z, a low
    # elevation foreshortens the very separation the figure exists to show.
    views = [dict(elev=34, azim=-56), dict(elev=12, azim=-90)]
    titles = ["ground truth", "reconstruction (mask + image only)"]
    for col, tubes in enumerate((gt_tubes, pr_tubes)):
        for row, view in enumerate(views):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            draw_tubes_shaded(ax, tubes, z_lo, span, zoom=1.4, **view)
            if row == 0:
                tag = chr(ord("a") + col)
                ax.text2D(0.02, 1.00, f"({tag})", transform=ax.transAxes,
                          ha="left", va="bottom", fontsize=PT_TITLE,
                          fontweight="bold", color=INK)
                # a fixed axes-fraction gap after a bold "(x)" tag is too
                # narrow once the tag itself is more than one digit wide;
                # anchoring the title's left edge, rather than guessing a
                # gap in axes fraction, is what keeps this from crowding it
                ax.text2D(0.145, 1.00, titles[col], transform=ax.transAxes,
                          ha="left", va="bottom", fontsize=PT_TITLE,
                          color=INK)
            label = "oblique" if row == 0 else "side elevation"
            ax.text2D(0.5, 0.0, label, transform=ax.transAxes, ha="center",
                      va="top", fontsize=PT_ANNOT, color=INK)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.03,
                        wspace=0.0, hspace=0.06)
    save_fig(fig, "fig_depth_reconstruction", subdir="archive")


if __name__ == "__main__":
    main()
