"""A real CNT SEM field, its depth reconstruction, and the image it re-renders to.

One row over a crop of B58\\_100: the manual annotation's skeleton -- the
oracle-like axis condition of Section~\\ref{sec:res-real} -- then the film
PLECTA's depth stage builds from it, then that reconstruction pushed back
through the forward SEM model and set beside the micrograph it came from.

**Nothing here is scored, and the figure says so.** No depth ground truth
exists for real SEM: the manual annotation records which pixels belong to which
filament, not which filament passes in front. So this demonstrates that the
stage runs end to end on real data and that its output re-renders to something
recognisable; it is not evidence that the depths are right. The quantitative
depth result is Table~\\ref{tab:depth-order}, on synthetic scenes where the
order is known.

The re-render parameters are **fitted** to the micrograph by
``render_sem.tune`` rather than set by hand, and the residual is printed on the
panel, so a reader can see how well the forward model reproduces the real image
instead of judging a picture that was adjusted until it looked convincing.

Sources: PLECTA App (https://github.com/Jorallan/PLECTA_APP) for the forward
model, this repository's ``_tube3d`` for the solid-tube column.

    python fig_resem_real.py [--crop 512]
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

DEFAULT_SCENE = (r"C:\Repos\comparisons\real_sem_study\scenes_v2"
                 r"\b58_100\skel")
DEFAULT_APP = r"C:\Repos\PLECTA_APP"
DEFAULT_PLECTA = r"C:\Repos\stubmatch"


def _engine(app_dir, plecta_dir):
    sys.path.insert(0, app_dir)
    os.environ.setdefault("PLECTA_SOURCE", plecta_dir)
    from plecta_app import render_sem
    from plecta_app.pipeline import Engine
    from plecta_app.settings import Settings
    return Engine, Settings, render_sem


def _tubes(instances):
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
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--app", default=DEFAULT_APP)
    ap.add_argument("--plecta", default=DEFAULT_PLECTA)
    ap.add_argument("--crop", type=int, default=512,
                    help="side of the square crop drawn, in px")
    args = ap.parse_args()

    Engine, Settings, render_sem = _engine(args.app, args.plecta)
    result = Engine().run(Settings(
        scene_path=os.path.join(args.scene, "gt_multilabel.npz"),
        image_path=os.path.join(args.scene, "sem.png")))

    sem = np.asarray(Image.open(os.path.join(args.scene, "sem.png"))
                     .convert("L"), float) / 255.0
    mask = np.asarray(Image.open(os.path.join(args.scene, "mask.png"))
                      .convert("L"), float) / 255.0

    # Fit the forward model to this micrograph rather than setting it by eye,
    # then report the residual on the panel.
    # tune() returns (params, fit). Its own docstring is explicit that the
    # fit statistic is IN-SAMPLE and is not an independent measure of the
    # reconstruction, so the panel reports difference() instead, which nothing
    # is optimised against.
    fit = {}
    try:
        params, fit = render_sem.tune(result, sem)
    except Exception:                      # noqa: BLE001 - fall back, and say so
        params = render_sem.RenderParams()
    rendered = np.asarray(render_sem.render(result, params), float)
    try:
        residual = render_sem.difference(rendered, sem)
    except Exception:                      # noqa: BLE001
        residual = {}

    c = args.crop
    r0 = (sem.shape[0] - c) // 2
    c0 = (sem.shape[1] - c) // 2
    box = (slice(r0, r0 + c), slice(c0, c0 + c))

    tubes = _tubes(result["instances"])
    zs = [t[2] for t in tubes] or [0.0]
    z_lo, span = min(zs), max(1e-6, max(zs) - min(zs))

    plecta_style()
    fig = plt.figure(figsize=(FIG_W, 0.30 * FIG_W))
    gs = fig.add_gridspec(1, 4, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0]); bare(ax)
    ax.imshow(mask[box], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("manual axis", fontsize=PT_TITLE, pad=3)

    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    draw_tubes_shaded(ax3, tubes, z_lo, span,
                      extent=max(sem.shape), zoom=0.95)
    ax3.set_title("depth reconstruction", fontsize=PT_TITLE, pad=3)

    ax = fig.add_subplot(gs[0, 2]); bare(ax)
    ax.imshow(rendered[box], cmap="gray", interpolation="nearest")
    ax.set_title("re-rendered", fontsize=PT_TITLE, pad=3)
    if residual:
        key = next((k for k in ("rmse", "mae", "l1") if k in residual), None)
        if key:
            ax.set_xlabel(f"{key.upper()} {residual[key]:.3f}",
                          fontsize=PT_ANNOT)

    ax = fig.add_subplot(gs[0, 3]); bare(ax)
    ax.imshow(sem[box], cmap="gray", interpolation="nearest")
    ax.set_title("micrograph", fontsize=PT_TITLE, pad=3)

    save_fig(fig, "fig_resem_real", bbox_inches="tight")
    print("wrote fig_resem_real")
    print("  in-sample tune fit :", fit)
    print("  independent residual:", residual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
