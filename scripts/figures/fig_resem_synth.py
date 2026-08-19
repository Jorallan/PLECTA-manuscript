"""Synthetic input, depth reconstruction, and the image it re-renders to.

Two rows, one per input condition, so the columns are directly comparable:

* **binary axis mask** -- the only input PLECTA's grouping ever reads. The 2-D
  instances are reconstructed from it, then the depth stage places them.
* **ground-truth instances** -- the same scene with the grouping given, so the
  row isolates what the depth stage does from what the grouping does.

Columns are: the input as the method sees it, the reconstructed film as shaded
solid tubes, and the reconstruction re-rendered through the forward SEM model.
The last column is the honest test of the whole chain: if the depth and radii
are wrong, the re-render does not look like the micrograph it came from, and
the reference image is drawn beside it so a reader can check rather than take
it on trust.

The re-render comes from PLECTA App's `render_sem`
(https://github.com/Jorallan/PLECTA_APP), a fourth source tree beyond the three
this repository normally depends on; `--app` points at it. The 3-D column uses
this repository's own `_tube3d`, not the app's WebGL view, so the panel is in
the manuscript's face and style rather than the app's dark UI theme.

    python fig_resem_synth.py [--scene cov20/synth_0000]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (FIG_W, PT_ANNOT, PT_TITLE, bare,  # noqa: E402
                    plecta_style, save_fig)
from _tube3d import draw_tubes_shaded, tube_mesh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENES = os.path.join(REPO, "exploration", "depth_order_eval", "scenes")
DEFAULT_APP = r"C:\Repos\PLECTA_APP"
DEFAULT_PLECTA = r"C:\Repos\stubmatch"

# (input file, row label). The NPZ is the reference instance set the scorer
# itself reads, so the second row is the same ground truth the numbers use.
ROWS = (
    ("mask_clean.png", "binary axis mask"),
    ("gt_multilabel.npz", "ground-truth instances"),
)


def _engine(app_dir, plecta_dir):
    sys.path.insert(0, app_dir)
    os.environ.setdefault("PLECTA_SOURCE", plecta_dir)
    from plecta_app import render_sem
    from plecta_app.pipeline import Engine
    from plecta_app.settings import Settings
    return Engine, Settings, render_sem


def _tubes(instances):
    """(verts, faces, z) per instance, ready for the shaded renderer.

    The app reports a centreline as ``xy`` and a radius directly, unlike the
    depth stage's own record which carries ``centerline_xy`` and a diameter.
    """
    out = []
    for inst in instances:
        xy = np.asarray(inst["xy"], float)
        if len(xy) < 2:
            continue
        z = float(inst["z"])
        verts, faces = tube_mesh(xy, z, float(inst["radius"]), n_theta=12)
        out.append((verts, faces, z))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", default=DEFAULT_SCENES)
    ap.add_argument("--scene", default="cov20/synth_0000")
    ap.add_argument("--app", default=DEFAULT_APP)
    ap.add_argument("--plecta", default=DEFAULT_PLECTA)
    args = ap.parse_args()

    Engine, Settings, render_sem = _engine(args.app, args.plecta)
    cov, name = args.scene.split("/")
    scene = os.path.join(args.scenes, cov, name)
    reference = np.asarray(Image.open(os.path.join(scene, "sem.png")).convert("L"),
                           float) / 255.0

    plecta_style()
    fig = plt.figure(figsize=(FIG_W, 0.62 * FIG_W))
    gs = fig.add_gridspec(2, 4, wspace=0.06, hspace=0.12)

    for r, (input_name, input_label) in enumerate(ROWS):
        result = Engine().run(Settings(
            scene_path=os.path.join(scene, input_name),
            image_path=os.path.join(scene, "sem.png")))

        if input_name.endswith(".npz"):
            # Draw the reference instances as their own centrelines, so the
            # panel shows what was handed in rather than a silhouette.
            shown = np.zeros_like(reference)
            for inst in result["instances"]:
                xy = np.asarray(inst["xy"], float).round().astype(int)
                ok = ((xy[:, 0] >= 0) & (xy[:, 0] < shown.shape[1])
                      & (xy[:, 1] >= 0) & (xy[:, 1] < shown.shape[0]))
                shown[xy[ok, 1], xy[ok, 0]] = 1.0
        else:
            shown = np.asarray(Image.open(os.path.join(scene, input_name))
                               .convert("L"), float) / 255.0

        tubes = _tubes(result["instances"])
        zs = [t[2] for t in tubes] or [0.0]
        z_lo, span = min(zs), max(1e-6, max(zs) - min(zs))

        ax = fig.add_subplot(gs[r, 0]); bare(ax)
        ax.imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_ylabel(input_label, fontsize=PT_ANNOT)

        ax3 = fig.add_subplot(gs[r, 1], projection="3d")
        draw_tubes_shaded(ax3, tubes, z_lo, span,
                          extent=reference.shape[0], zoom=1.55,
                          elev=20, azim=-62, colour="grey",
                          z_stretch=span / reference.shape[0])

        rendered = np.asarray(render_sem.render(result), float)
        ax = fig.add_subplot(gs[r, 2]); bare(ax)
        ax.imshow(rendered, cmap="gray", interpolation="nearest")

        ax = fig.add_subplot(gs[r, 3]); bare(ax)
        ax.imshow(reference, cmap="gray", interpolation="nearest")

        if r == 0:
            for c, title in enumerate(("input", "depth reconstruction",
                                       "re-rendered", "reference image")):
                fig.axes[-4 + c].set_title(title, fontsize=PT_TITLE, pad=3)

    save_fig(fig, "fig_resem_synth", bbox_inches="tight")
    print("wrote fig_resem_synth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
