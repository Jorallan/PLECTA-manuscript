"""A synthetic scene, its depth reconstruction, and the image it re-renders to.

One row: the image the scene was rendered from, that scene re-rendered from
PLECTA's reconstruction through a forward SEM model, and the film the
reconstruction implies. If the depths or radii were wrong the re-render would
not resemble the image it came from, and the two are set side by side so a
reader can check rather than take it on trust.

**Why only the mask condition is drawn.** An earlier version carried a second
row for the ground-truth-instance condition of Table~3. It was dropped as
uninformative, and deliberately rather than for space: from the non-degraded
axis the 2-D reconstruction recovers essentially the reference scene -- 22
instances against 22 and the same layer count at 20\% coverage -- so the two
rows differed by a re-render RMSE of 0.037 and looked identical. The gap the
table reports between those conditions is a gap in *crossing recovery*, and a
rendered image does not show which crossings were found, so this figure cannot
display that contrast even in principle. The table carries it.

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
DEFAULT_PLECTA = r"C:\Repos\PLECTA"

#: The binary axis mask is the only input PLECTA's grouping reads.
INPUT_NAME = "mask_clean.png"


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
    fig = plt.figure(figsize=(FIG_W, 0.38 * FIG_W))
    gs = fig.add_gridspec(1, 3, wspace=0.06)

    result = Engine().run(Settings(
        scene_path=os.path.join(scene, INPUT_NAME),
        image_path=os.path.join(scene, "sem.png")))
    tubes = _tubes(result["instances"])
    zs = [t[2] for t in tubes] or [0.0]
    z_lo, span = min(zs), max(1e-6, max(zs) - min(zs))
    rendered = np.asarray(render_sem.render(result), float)

    ax = fig.add_subplot(gs[0, 0]); bare(ax)
    ax.imshow(reference, cmap="gray", interpolation="nearest")
    ax.set_title("reference image", fontsize=PT_TITLE, pad=3)

    ax = fig.add_subplot(gs[0, 1]); bare(ax)
    ax.imshow(rendered, cmap="gray", interpolation="nearest")
    ax.set_title("re-rendered", fontsize=PT_TITLE, pad=3)

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    draw_tubes_shaded(ax3, tubes, z_lo, span,
                      extent=reference.shape[0], zoom=2.05,
                      elev=20, azim=-62, colour="grey",
                      z_stretch=span / reference.shape[0])
    ax3.set_title("depth reconstruction", fontsize=PT_TITLE, pad=3)

    save_fig(fig, "fig_resem_synth", bbox_inches="tight")
    print("wrote fig_resem_synth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
