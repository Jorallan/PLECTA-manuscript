"""Render the real development crop with correspondence-matched instance colours.

This figure is descriptive only.  It loads the stored locked real-development
run and the stored manual annotation without recomputing the pipeline.
Predicted instances are assigned one-to-one to manual instances by maximum
total binary-mask IoU.  A prediction inherits its matched manual colour;
unmatched predictions are magenta and missed manual instances are outlined
with dashed yellow contours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from skimage import io as skio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import instance_io as IIO  # noqa: E402

from _style import BLUE, apply_style, save_fig  # noqa: E402
from fig_paired_locked_examples import (  # noqa: E402
    FALSE_COLOR,
    MISSED_COLOR,
    _draw_instances,
    _gt_color,
    _rgb_instances,
    one_to_one_matches,
)


BASE = "sem_full_00000_1p66_crop512"
REPORT = ROOT / "output" / "final_real_ablation" / "real_ablation_results.json"
MANUAL = ROOT / f"{BASE}_manual" / f"{BASE}_manual_multilabel.npz"
SEM = ROOT / "input" / BASE / "sem.png"
MASK = ROOT / "input" / BASE / "mask.png"
# Matched predictions use the shared palette blue (_style.BLUE) rather than a
# one-off accent, so this figure carries no colour the sibling figures lack.
AGREEMENT_COLORS = ("#000000", BLUE, MISSED_COLOR, FALSE_COLOR)


def _locked_prediction() -> Path:
    if not REPORT.is_file():
        raise FileNotFoundError(f"no final real-ablation report found: {REPORT}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    locked = next(
        row for row in report["runs"]
        if row.get("variant") == "locked" and row.get("status") == "ok"
    )
    return Path(locked["prediction_artifact"])


def _read_gray(path: Path) -> np.ndarray:
    image = np.asarray(skio.imread(str(path)))
    return image[..., 0] if image.ndim == 3 else image


def _identity_agreement(
    ground_truth: IIO.InstanceSet,
    prediction: IIO.InstanceSet,
    match: dict[int, int],
) -> np.ndarray:
    """Return 0 background, 1 matched prediction, 2 missed GT, 3 false prediction."""
    classes = np.zeros(ground_truth.shape, dtype=np.uint8)
    for pid, mask in prediction.masks.items():
        classes[mask] = 1 if pid in match else 3
    for gid in set(ground_truth.masks).difference(match.values()):
        classes[ground_truth.masks[gid]] = 2
    return classes


def _draw_manual(ax: Any, ground_truth: IIO.InstanceSet) -> None:
    ax.imshow(_rgb_instances(ground_truth, {}, gt=True))


def main() -> int:
    apply_style()
    for path in (SEM, MASK, MANUAL):
        if not path.is_file():
            raise FileNotFoundError(path)

    prediction_path = _locked_prediction()
    ground_truth, _ = IIO.load_gt(MANUAL, shape=_read_gray(MASK).shape)
    prediction = IIO.load_pred_instances(prediction_path)
    match = one_to_one_matches(ground_truth, prediction)
    agreement = _identity_agreement(ground_truth, prediction, match)

    fig, axes = plt.subplots(1, 5, figsize=(12.4, 2.75))
    axes[0].imshow(_read_gray(SEM), cmap="gray")
    axes[1].imshow(_read_gray(MASK), cmap="gray", vmin=0, vmax=255)
    _draw_manual(axes[2], ground_truth)
    _draw_instances(axes[3], prediction, match, ground_truth)
    axes[4].imshow(
        agreement,
        cmap=ListedColormap(AGREEMENT_COLORS),
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    for gid in set(ground_truth.masks).difference(match.values()):
        axes[4].contour(
            ground_truth.masks[gid].astype(float),
            levels=[0.5],
            colors=MISSED_COLOR,
            linestyles="--",
            linewidths=0.7,
        )

    # Five panels across one 6.27 in text column leave ~1.2 in per panel on the
    # page, so the two long titles are wrapped rather than shrunk: kept on one
    # line they would have to drop to ~6 pt rendered to stop colliding with
    # their neighbours, well under the ~8.5 pt used by the other figures.
    # Wording is unchanged.
    titles = (
        "SEM input",
        "Centreline mask",
        "Manual instances",
        "FilaSeg final\ninstances",
        "Instance\ncorrespondence map",
    )
    for ax, title in zip(axes, titles):
        # 8.2 pt rendered at the 0.494 page scale this figure saves at.
        ax.set_title(title, fontsize=16.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend = [
        Line2D([], [], color=BLUE, lw=5, label="Matched prediction"),
        Line2D([], [], color=FALSE_COLOR, lw=5, label="Unmatched prediction"),
        Line2D([], [], color=MISSED_COLOR, lw=5, label="Missed manual pixels"),
        Line2D([], [], color=MISSED_COLOR, lw=1.2, ls="--", label="Missed manual instance"),
    ]
    # Anchored below the axes area rather than inside the margin reserved by
    # `tight_layout`: at this size the single legend row is taller than that
    # margin and would otherwise sit on top of the bottom of the panels.
    # `bbox_inches="tight"` grows the canvas to include it.
    fig.legend(
        handles=legend,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=14.6,  # 7.2 pt rendered at the 0.494 page scale
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_fig(fig, "fig_qualitative_real", bbox_inches="tight")
    plt.close(fig)

    print(f"[fig_qualitative_real] prediction: {prediction_path}")
    print(
        "[fig_qualitative_real] "
        f"{len(match)} matched, {len(prediction.masks) - len(match)} unmatched predictions, "
        f"{len(ground_truth.masks) - len(set(match.values()))} missed manual instances"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
