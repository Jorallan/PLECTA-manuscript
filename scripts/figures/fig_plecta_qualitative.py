"""Render rule-selected held-out PLECTA examples with fragment-level errors.

The examples are not score-selected: for each requested held-out coverage, the
script takes the lexicographically first scene listed in the fixed evaluation
record. Ground truth remains multilabel, so crossing pixels can belong to more
than one filament. A dark centre with a yellow halo marks those shared pixels.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from skimage import io as skio
from skimage.morphology import skeletonize


PAPER_ROOT = Path(__file__).resolve().parents[2]
PLECTA_ROOT = Path(os.environ.get("PLECTA_ROOT", r"C:\Repos\stubmatch"))
EVALUATION_ROOT = Path(os.environ.get(
    "PLECTA_EVALUATION_ROOT", PLECTA_ROOT / ".local" / "evaluation"
))
METRIC_ROOT = Path(os.environ.get(
    "PLECTA_EVAL_ROOT", r"C:\Repos\filaments_quantification\eval\core"
))

for path in (PLECTA_ROOT, METRIC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_metric as metric  # noqa: E402
from plecta.linking import Params  # noqa: E402
from plecta.predict import predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import DARK, GRAY, GREEN, ORANGE, apply_style  # noqa: E402


INPUT_NAME = "mask_w1.png"
SELECTION_RULE = (
    "Lexicographically first scene ID in the fixed held-out per-scene record "
    "at each requested coverage; no metric or visual-quality selection."
)


def load_multilabel(path: Path) -> dict[int, np.ndarray]:
    """Load the project's sparse, overlap-aware multilabel format."""
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["shape"])
        ids = [int(v) for v in data["ids"]]
        indptr = np.asarray(data["indptr"], dtype=np.int64)
        indices = np.asarray(data["indices"], dtype=np.int64)
    masks = {}
    for k, instance_id in enumerate(ids):
        mask = np.zeros(shape[0] * shape[1], dtype=bool)
        mask[indices[indptr[k]:indptr[k + 1]]] = True
        masks[instance_id] = mask.reshape(shape)
    return masks


def colour(instance_id: int) -> tuple[float, float, float]:
    """Stable, high-contrast colour keyed by the ground-truth ID."""
    hue = (0.11 + 0.61803398875 * int(instance_id)) % 1.0
    saturation = 0.62 + 0.16 * (int(instance_id) % 3) / 2
    value = 0.58 + 0.16 * (int(instance_id) % 2)
    return colorsys.hsv_to_rgb(hue, saturation, value)


def optimal_group_map(gt_assign: dict[int, int], pred_assign: dict[int, int]):
    """Match predicted groups to GT groups by maximum shared fragments."""
    gt_ids = sorted({int(v) for v in gt_assign.values() if int(v) > 0})
    pred_ids = sorted({int(v) for v in pred_assign.values() if int(v) > 0})
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)
    gi = {v: i for i, v in enumerate(gt_ids)}
    pi = {v: i for i, v in enumerate(pred_ids)}
    for fragment_id in sorted(set(gt_assign) | set(pred_assign)):
        g = int(gt_assign.get(fragment_id, 0))
        p = int(pred_assign.get(fragment_id, 0))
        if g > 0 and p > 0:
            counts[gi[g], pi[p]] += 1
    mapping: dict[int, int] = {}
    if counts.size:
        rows, cols = linear_sum_assignment(-counts)
        mapping = {pred_ids[c]: gt_ids[r] for r, c in zip(rows, cols)
                   if counts[r, c] > 0}
    return mapping


def render_instances(
    masks: dict[int, np.ndarray],
    shape: tuple[int, int],
    id_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, int]:
    """Colour layered instances without flattening shared pixels away."""
    rgb = np.ones((*shape, 3), dtype=float)
    occupancy = np.zeros(shape, dtype=np.uint16)
    for instance_id in sorted(masks):
        mask = np.asarray(masks[instance_id], dtype=bool)
        mapped = (id_map or {}).get(instance_id, instance_id)
        rgb[mask] = colour(mapped)
        occupancy += mask
    overlap = occupancy > 1
    halo = binary_dilation(overlap, iterations=1) & ~overlap
    rgb[halo] = (1.0, 0.82, 0.18)
    rgb[overlap] = (0.07, 0.09, 0.12)
    return rgb, int(overlap.sum())


def render_input(mask: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=float)
    rgb[mask] = (0.08, 0.10, 0.13)
    return rgb


def overlap_pixels(masks: dict[int, np.ndarray], shape: tuple[int, int]) -> int:
    occupancy = np.zeros(shape, dtype=np.uint16)
    for mask in masks.values():
        occupancy += np.asarray(mask, dtype=bool)
    return int((occupancy > 1).sum())


def render_fragment_errors(
    mask: np.ndarray,
    fragments: np.ndarray,
    gt_assign: dict[int, int],
    pred_assign: dict[int, int],
    pred_to_gt: dict[int, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Paint fragments correct only under the optimal one-to-one group map."""
    rgb = np.ones((*mask.shape, 3), dtype=float)
    rgb[mask] = (0.88, 0.89, 0.91)
    counts = {"matched": 0, "regrouped": 0, "gt_unassigned": 0}
    colours = {
        "matched": (0.18, 0.52, 0.35),
        "regrouped": (0.84, 0.29, 0.25),
        "gt_unassigned": (0.45, 0.34, 0.68),
    }
    for fragment_id in range(1, int(fragments.max()) + 1):
        g = int(gt_assign.get(fragment_id, 0))
        p = int(pred_assign.get(fragment_id, 0))
        if g == 0:
            state = "gt_unassigned"
        elif p > 0 and pred_to_gt.get(p) == g:
            state = "matched"
        else:
            state = "regrouped"
        counts[state] += 1
        pixels = binary_dilation(fragments == fragment_id, iterations=1)
        rgb[pixels] = colours[state]
    return rgb, counts


def row_for_scene(report: dict, scene_id: str) -> dict:
    matches = [row for row in report["per_scene"] if row["scene"] == scene_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one held-out row for {scene_id}, got {len(matches)}")
    return matches[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverages", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--results", type=Path,
                        default=EVALUATION_ROOT / "results" / "heldout.json")
    parser.add_argument("--data", type=Path, default=EVALUATION_ROOT / "data")
    parser.add_argument("--params", type=Path,
                        default=PLECTA_ROOT / "plecta" / "params.json")
    parser.add_argument("--output", type=Path,
                        default=PAPER_ROOT / "figures" / "fig_plecta_qualitative")
    parser.add_argument("--manifest", type=Path,
                        default=PAPER_ROOT / "results" / "plecta_qualitative_manifest.json")
    args = parser.parse_args(argv)

    report = json.loads(args.results.read_text(encoding="utf-8"))
    params_bytes = args.params.read_bytes()
    params_hash = hashlib.sha256(params_bytes).hexdigest()
    params = Params(**json.loads(args.params.read_text(encoding="utf-8")))

    selected = []
    for coverage in args.coverages:
        prefix = f"val/cov{coverage}/"
        candidates = sorted(row["scene"] for row in report["per_scene"]
                            if row["scene"].startswith(prefix))
        if not candidates:
            raise RuntimeError(f"no held-out scenes at {coverage}% coverage")
        selected.append((coverage, candidates[0], row_for_scene(report, candidates[0])))

    apply_style()
    fig, axes = plt.subplots(len(selected), 4, figsize=(12.4, 6.55), squeeze=False)
    headings = (
        "Degraded input mask",
        "Overlap-aware GT axes",
        "PLECTA instances",
        "Fragment grouping",
    )
    for ax, heading in zip(axes[0], headings):
        ax.set_title(heading, fontsize=10.2, pad=8)

    manifest_scenes = []
    for row_index, (coverage, scene_id, recorded) in enumerate(selected):
        scene_dir = args.data / Path(scene_id)
        meta = json.loads((scene_dir / "gt_meta.json").read_text(encoding="utf-8"))
        mask = np.asarray(skio.imread(scene_dir / INPUT_NAME)) > 0
        gt_masks = load_multilabel(scene_dir / "gt_multilabel.npz")
        fragments = metric.common_fragments(mask)
        gt_assign = metric.assign_fragments(fragments, gt_masks)
        pred_masks = predict(mask, params)
        pred_assign = metric.assign_fragments(fragments, pred_masks)
        recomputed = metric.pairwise_scores(
            dict(gt_assign), dict(pred_assign), gt_unassigned_policy="singleton"
        )
        for key in ("precision", "recall", "f1", "adjusted_rand_index",
                    "vi_split_bits", "vi_merge_bits"):
            if not np.isclose(float(recomputed[key]), float(recorded[key]), atol=1e-12):
                raise RuntimeError(
                    f"{scene_id}: recomputed {key}={recomputed[key]} differs from "
                    f"held-out record {recorded[key]}"
                )

        pred_to_gt = optimal_group_map(gt_assign, pred_assign)
        # Each overlap-aware GT layer is skeletonised independently for this
        # axis-level visualization. Its full-width overlap count is retained
        # in the manifest; prediction and scoring always use the original GT.
        gt_display = {key: skeletonize(value) for key, value in gt_masks.items()}
        gt_full_overlap = overlap_pixels(gt_masks, mask.shape)
        gt_rgb, gt_overlap = render_instances(gt_display, mask.shape)
        pred_rgb, pred_overlap = render_instances(pred_masks, mask.shape, pred_to_gt)
        error_rgb, error_counts = render_fragment_errors(
            mask, fragments, gt_assign, pred_assign, pred_to_gt
        )
        panels = (render_input(mask), gt_rgb, pred_rgb, error_rgb)
        for ax, panel in zip(axes[row_index], panels):
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#cbd5e0")
                spine.set_linewidth(0.7)

        axes[row_index, 0].set_ylabel(
            f"{coverage}% coverage\nseed {meta['seed']}\n"
            f"F1 = {recorded['f1']:.3f}\nrecovery = "
            f"{recorded['fragment_recovery_recovery_rate']:.3f}",
            rotation=0, ha="right", va="center", labelpad=13, fontsize=8.7,
        )
        axes[row_index, 1].text(
            0.02, 0.02, f"{len(gt_masks)} GT axes · {gt_overlap:,} shared axis px",
            transform=axes[row_index, 1].transAxes, fontsize=7.1, color=DARK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0),
        )
        axes[row_index, 2].text(
            0.02, 0.02, f"{len(pred_masks)} predicted · {pred_overlap:,} shared px",
            transform=axes[row_index, 2].transAxes, fontsize=7.1, color=DARK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0),
        )
        axes[row_index, 3].text(
            0.02, 0.02,
            f"{error_counts['matched']}/{int(fragments.max())} fragments matched",
            transform=axes[row_index, 3].transAxes, fontsize=7.1, color=DARK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0),
        )

        manifest_scenes.append({
            "scene_id": scene_id,
            "coverage_percent": coverage,
            "seed": int(meta["seed"]),
            "input_filename": INPUT_NAME,
            "measured_coverage": float(meta["measured_coverage"]),
            "n_gt_instances": len(gt_masks),
            "n_pred_instances": len(pred_masks),
            "n_common_fragments": int(fragments.max()),
            "fragment_display_counts": error_counts,
            "overlap_pixels": {
                "ground_truth_full_layers": gt_full_overlap,
                "ground_truth_display_axes": gt_overlap,
                "prediction_axes": pred_overlap,
            },
            "metrics": {
                key: recorded[key] for key in (
                    "precision", "recall", "f1", "adjusted_rand_index",
                    "vi_split_bits", "vi_merge_bits",
                    "fragment_recovery_recovery_rate",
                )
            },
        })

    legend = [
        Patch(facecolor=GREEN, label="fragment in one-to-one matched group"),
        Patch(facecolor="#d64a40", label="split, merged, or unassigned by prediction"),
        Patch(facecolor="#7357ad", label="fragment unassigned in ground truth"),
        Patch(facecolor="#12171f", edgecolor="#f2c94c", linewidth=2,
              label="shared instance pixels (yellow halo)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.54, 0.012))
    fig.text(
        0.54, 0.965,
        "Held-out examples selected by scene name, not performance · instances shown by axis",
        ha="center", va="top", fontsize=8.2, color=GRAY,
    )
    fig.subplots_adjust(left=0.115, right=0.995, top=0.91, bottom=0.105,
                        wspace=0.055, hspace=0.10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": "fig_plecta_qualitative",
        "selection_rule": SELECTION_RULE,
        "heldout_record": ".local/evaluation/results/heldout.json (PLECTA repository)",
        "input_root": ".local/evaluation/data (PLECTA repository)",
        "plecta_params_file": "plecta/params.json (PLECTA repository)",
        "plecta_params_sha256": params_hash,
        "scenes": manifest_scenes,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"selected: {', '.join(scene_id for _, scene_id, _ in selected)}")
    print(f"parameters sha256: {params_hash}")
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.svg')}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
