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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io as skio
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import common_metric as CM  # noqa: E402
import instance_io as IIO  # noqa: E402


FALSE_COLOR = "#D81B60"
MISSED_COLOR = "#FDE725"
OVERLAY_COLORS = ("#ffffff", "#3B7EA1", "#2A9D8F", "#F4A261", "#D1495B")
SELECTIONS = (
    ("three_largest_filaseg_wins", "wins", "Largest FilaSeg wins"),
    ("three_largest_minimum_turn_wins", "losses", "Largest minimum-turn wins"),
    ("two_near_ties", "ties", "Near ties"),
)


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


def common_fragment_classes(
    fragments: np.ndarray,
    gt_assignment: Mapping[int, int],
    minimum_turn_assignment: Mapping[int, int],
    filaseg_assignment: Mapping[int, int],
    minimum_turn_to_gt: Mapping[int, int],
    filaseg_to_gt: Mapping[int, int],
) -> np.ndarray:
    """Classify each common fragment as both, FilaSeg-only, min-turn-only, neither.

    A method is correct for a fragment when the fragment's predicted instance
    has the GT ID assigned by that method's one-to-one correspondence.  Codes
    are 0 background, 1 both, 2 FilaSeg only, 3 minimum-turn only, 4 neither.
    """
    classes = np.zeros(fragments.shape, dtype=np.uint8)
    for fragment_id in np.unique(fragments):
        if fragment_id == 0:
            continue
        fid = int(fragment_id)
        expected = gt_assignment.get(fid)
        mt_ok = (expected is not None and
                 minimum_turn_to_gt.get(minimum_turn_assignment.get(fid)) == expected)
        fs_ok = (expected is not None and
                 filaseg_to_gt.get(filaseg_assignment.get(fid)) == expected)
        code = 1 if mt_ok and fs_ok else 2 if fs_ok else 3 if mt_ok else 4
        classes[fragments == fid] = code
    return classes


def _centreline(instances: IIO.InstanceSet) -> IIO.InstanceSet:
    masks = {iid: skeletonize(mask) for iid, mask in instances.masks.items()}
    return IIO.InstanceSet(instances.shape, {iid: mask for iid, mask in masks.items()
                                             if mask.any()}, source=instances.source)


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


def render_panel(audit: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], title: str, output_base: Path) -> None:
    fig, axes = plt.subplots(len(selected), 5, figsize=(12.2, 2.35 * len(selected)), squeeze=False)
    headers = ("Degraded input mask", "Overlap-aware GT", "Minimum-turn centreline",
               "FilaSeg Stage-3 centreline", "Common-fragment outcome")
    for row_index, chosen in enumerate(selected):
        row = _row_for_scene(audit, str(chosen["geometry_id"]))
        mask = _load_mask(_path(row, "input_mask"))
        gt, _ = IIO.load_gt(_path(row, "ground_truth_selected"), shape=mask.shape)
        minimum_turn = _centreline(IIO.load_pred_instances(_path(row, "minimum_turn")))
        filaseg = _centreline(IIO.load_pred_instances(_path(row, "filaseg_stage3")))
        mt_match, fs_match = one_to_one_matches(gt, minimum_turn), one_to_one_matches(gt, filaseg)
        fragments = CM.common_fragments(mask)
        classes = common_fragment_classes(
            fragments, CM.assign_fragments(fragments, gt),
            CM.assign_fragments(fragments, minimum_turn), CM.assign_fragments(fragments, filaseg),
            mt_match, fs_match,
        )
        panels = axes[row_index]
        panels[0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        panels[1].imshow(_rgb_instances(gt, {}, gt=True))
        _draw_instances(panels[2], minimum_turn, mt_match, gt)
        _draw_instances(panels[3], filaseg, fs_match, gt)
        panels[4].imshow(classes, cmap=ListedColormap(OVERLAY_COLORS), vmin=0, vmax=4)
        delta = float(chosen["filaseg_minus_minimum_turn_f1"])
        panels[0].set_ylabel(f"{chosen['geometry_id']}\n$\\Delta F_1$={delta:+.3f}", fontsize=8.5)
        for column, ax in enumerate(panels):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(headers[column], fontsize=9.2, pad=6)
    handles = [
        Line2D([], [], color=OVERLAY_COLORS[1], lw=5, label="Correct for both"),
        Line2D([], [], color=OVERLAY_COLORS[2], lw=5, label="FilaSeg only"),
        Line2D([], [], color=OVERLAY_COLORS[3], lw=5, label="Minimum-turn only"),
        Line2D([], [], color=OVERLAY_COLORS[4], lw=5, label="Neither"),
        Line2D([], [], color=FALSE_COLOR, lw=4, label="False instance"),
        Line2D([], [], color=MISSED_COLOR, lw=1.2, ls="--", label="Missed GT instance"),
    ]
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    selected = read_selection(audit)
    for key, suffix, title in SELECTIONS:
        render_panel(audit, selected[key], title, args.output_dir / f"fig_paired_locked_{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
