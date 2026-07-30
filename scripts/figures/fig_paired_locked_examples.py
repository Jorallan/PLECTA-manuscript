"""Render audit-selected, correspondence-coloured locked-evaluation panels.

The input audit is the sole source of scene selection.  Within a scene, each
prediction is matched independently to overlap-aware ground truth by a maximum
sum one-to-one assignment of binary-mask IoU.  Matches with zero IoU are
discarded.  Thus a matched prediction inherits its ground-truth colour, while
unmatched predictions are false instances and unmatched ground-truth instances
are shown as missed dashed outlines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io as skio
from skimage.morphology import binary_dilation, disk, footprint_rectangle, skeletonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import instance_io as IIO  # noqa: E402
from _style import apply_style  # noqa: E402


FALSE_COLOR = "#D81B60"
MISSED_COLOR = "#FDE725"
SELECTIONS = (
    ("three_largest_filaseg_wins", "wins", "Largest FilaSeg wins"),
    ("three_largest_minimum_turn_wins", "losses", "Largest minimum-turn wins"),
    ("two_near_ties", "ties", "Near ties"),
)

#: Display-only centreline thickening (see `_thicken_for_display`).  This
#: never touches results/ data, matching, or scoring -- it only changes the
#: array handed to `imshow` for the two centreline columns, immediately
#: before rendering.  2 px (diamond footprint) was compared against 3 px
#: (square footprint) on both figures.  Per-instance colours stayed crisp
#: and distinguishable at both widths, even in the densest crossing regions
#: (nearest-neighbour interpolation, no anti-aliased colour blending).  3 px
#: was chosen because most centreline segments run diagonally, where the
#: diamond footprint under-fills relative to the cardinal directions and
#: still reads as faint/broken at 2 px; the square footprint fills those
#: diagonal segments and turn points uniformly, which is the difference
#: that decides legibility in print.
DEFAULT_CENTRELINE_DILATION_PX = 3


def read_selection(audit: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return the report's stored selection, without recomputing or hand-picking."""
    selection = audit.get("qualitative_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("audit has no qualitative_selection")
    result: dict[str, list[dict[str, Any]]] = {}
    for key, _suffix, _title in SELECTIONS:
        rows = selection.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"audit qualitative_selection lacks {key}")
        result[key] = [dict(row) for row in rows]
    return result


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum()) / union if union else 0.0


def _gt_color(instance_id: int) -> tuple[float, float, float]:
    """Return a deterministic high-contrast colour without a short repeating cycle."""
    hue = (0.618033988749895 * int(instance_id)) % 1.0
    return tuple(matplotlib.colors.hsv_to_rgb((hue, 0.78, 0.92)))


def one_to_one_matches(
    ground_truth: IIO.InstanceSet, prediction: IIO.InstanceSet
) -> dict[int, int]:
    """Map prediction IDs to GT IDs by deterministic maximum-total-IoU matching."""
    gt_ids, pred_ids = sorted(ground_truth.masks), sorted(prediction.masks)
    if not gt_ids or not pred_ids:
        return {}
    scores = np.array([
        [_iou(ground_truth.masks[gid], prediction.masks[pid]) for pid in pred_ids]
        for gid in gt_ids
    ])
    rows, cols = linear_sum_assignment(-scores)
    return {
        pred_ids[col]: gt_ids[row]
        for row, col in zip(rows, cols)
        if scores[row, col] > 0.0
    }


def _centreline(instances: IIO.InstanceSet) -> IIO.InstanceSet:
    masks = {iid: skeletonize(mask) for iid, mask in instances.masks.items()}
    return IIO.InstanceSet(instances.shape, {iid: mask for iid, mask in masks.items()
                                             if mask.any()}, source=instances.source)


def _display_footprint(width_px: int) -> np.ndarray | None:
    """Return the structuring element used to thicken a 1 px centreline for display.

    2 px uses a diamond (4-connected) footprint; 3 px uses a full 3x3 square,
    which fills the corners a diamond leaves open and so reads as a uniform
    width regardless of the local line orientation.  Both footprints are
    odd-sized and centred, so dilation never shifts the rendered line.
    """
    if width_px <= 1:
        return None
    if width_px == 2:
        return disk(1)
    if width_px == 3:
        return footprint_rectangle((3, 3))
    return disk(max(1, width_px // 2))


def _thicken_for_display(instances: IIO.InstanceSet, width_px: int) -> IIO.InstanceSet:
    """Dilate each instance mask independently, for rendering only.

    This must never be used for matching, scoring, or anything that reads
    back into results/ -- it exists solely to make 1 px skeleton centrelines
    legible in print.  Dilating per-instance mask (rather than a combined
    label array) guarantees each instance keeps exactly its own colour: a
    shared-pixel tie between two nearby dilated instances is resolved the
    same deterministic way (later `iid` wins) as the undilated renderer
    already uses in `_rgb_instances`, so instance identity is never
    ambiguous or overwritten incorrectly.
    """
    footprint = _display_footprint(width_px)
    if footprint is None:
        return instances
    masks = {iid: binary_dilation(mask, footprint) for iid, mask in instances.masks.items()}
    return IIO.InstanceSet(instances.shape, masks, source=instances.source)


def _rgb_instances(instances: IIO.InstanceSet, match: Mapping[int, int], *, gt: bool = False) -> np.ndarray:
    image = np.zeros((*instances.shape, 3), dtype=float)
    for iid in sorted(instances.masks):
        gid = iid if gt else match.get(iid)
        color = _gt_color(int(gid)) if gid is not None else FALSE_COLOR
        image[instances.masks[iid]] = matplotlib.colors.to_rgb(color)
    return image


def _missed_ids(gt: IIO.InstanceSet, match: Mapping[int, int]) -> set[int]:
    return set(gt.masks).difference(match.values())


def _draw_instances(ax: Any, instances: IIO.InstanceSet, match: Mapping[int, int], gt: IIO.InstanceSet) -> None:
    ax.imshow(_rgb_instances(instances, match))
    for gid in _missed_ids(gt, match):
        ax.contour(gt.masks[gid].astype(float), levels=[0.5], colors=MISSED_COLOR,
                   linestyles="--", linewidths=0.7)


def _path(audit_row: Mapping[str, Any], name: str) -> Path:
    try:
        return Path(str(audit_row["artifacts"][name]["path"]))
    except KeyError as exc:
        raise ValueError(f"audit row has no {name} artifact") from exc


def _load_mask(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    return np.asarray(image[..., 0] if image.ndim == 3 else image) > 0


def _row_for_scene(audit: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    for row in audit.get("per_scene", []):
        if str(row.get("geometry_id")) == scene_id:
            return row
    raise ValueError(f"selected scene is absent from audit per_scene: {scene_id}")


def render_panel(
    audit: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    title: str,
    output_base: Path,
    *,
    centreline_dilation_px: int = DEFAULT_CENTRELINE_DILATION_PX,
) -> None:
    # Same shared rcParams (frameless legends, bold axes titles, 11 pt base)
    # as every other paper figure; must run before the figure is created.
    apply_style()
    n_cols = 4
    fig, axes = plt.subplots(
        len(selected), n_cols, figsize=(9.9, 2.35 * len(selected)), squeeze=False
    )
    # Method names are the manuscript's own prose names, identical to the
    # legends of Figures 4 and 8, so one name per method appears everywhere.
    headers = ("Degraded input mask", "Overlap-aware GT", "Minimum-turn tracer",
               "FilaSeg (Stage 3)")
    for row_index, chosen in enumerate(selected):
        row = _row_for_scene(audit, str(chosen["geometry_id"]))
        mask = _load_mask(_path(row, "input_mask"))
        gt, _ = IIO.load_gt(_path(row, "ground_truth_selected"), shape=mask.shape)
        minimum_turn = _centreline(IIO.load_pred_instances(_path(row, "minimum_turn")))
        filaseg = _centreline(IIO.load_pred_instances(_path(row, "filaseg_stage3")))
        # Matching/scoring runs on the true 1 px skeletons above; only the
        # arrays handed to imshow below are thickened, and only for display.
        mt_match, fs_match = one_to_one_matches(gt, minimum_turn), one_to_one_matches(gt, filaseg)
        minimum_turn_display = _thicken_for_display(minimum_turn, centreline_dilation_px)
        filaseg_display = _thicken_for_display(filaseg, centreline_dilation_px)
        panels = axes[row_index]
        panels[0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        panels[1].imshow(_rgb_instances(gt, {}, gt=True))
        _draw_instances(panels[2], minimum_turn_display, mt_match, gt)
        _draw_instances(panels[3], filaseg_display, fs_match, gt)
        delta = float(chosen["filaseg_minus_minimum_turn_f1"])
        # 8.0 pt rendered at 0.660 page scale.  Upright F, matching \Fone in main.tex.
        panels[0].set_ylabel(f"{chosen['geometry_id']}\n$\\Delta$F$_1$={delta:+.3f}",
                             fontsize=12.1)
        for column, ax in enumerate(panels):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                # 8.5 pt rendered at 0.660 page scale.
                ax.set_title(headers[column], fontsize=12.9, pad=6)
    handles = [
        Line2D([], [], color=FALSE_COLOR, lw=4, label="False instance"),
        Line2D([], [], color=MISSED_COLOR, lw=1.2, ls="--", label="Missed GT instance"),
    ]
    # 9.5 pt rendered at 0.660 page scale (one step above the column titles).
    fig.suptitle(title, x=0.02, ha="left", fontsize=14.4, fontweight="bold")
    # 7.5 pt rendered at 0.660 page scale.
    fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               fontsize=11.4)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95), w_pad=0.3, h_pad=0.3)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--centreline-dilation-px",
        type=int,
        default=DEFAULT_CENTRELINE_DILATION_PX,
        help=(
            "Display-only dilation width (pixels) applied to the rendered "
            "minimum-turn and FilaSeg centreline panels. Does not affect "
            "matching, scoring, or any stored result (default: "
            f"{DEFAULT_CENTRELINE_DILATION_PX})."
        ),
    )
    args = parser.parse_args(argv)
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    selected = read_selection(audit)
    for key, suffix, title in SELECTIONS:
        render_panel(
            audit, selected[key], title, args.output_dir / f"fig_paired_locked_{suffix}",
            centreline_dilation_px=args.centreline_dilation_px,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
