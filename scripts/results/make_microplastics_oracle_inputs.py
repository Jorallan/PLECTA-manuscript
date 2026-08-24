"""Run frozen PLECTA on two GT-derived Microplastics SEM inputs.

NOT A MANUSCRIPT BUILD INPUT. This is a comparator-study driver: it writes to
``C:/Repos/comparisons/Microplastics_SEM/PLECTA_comparison/oracle_inputs`` and
produces no results/ record, no macro and no figure, and nothing in sections/,
results/ or the other scripts references it. By CLAUDE.md's own convention
comparator studies live in ``C:/Repos/comparisons/``, so this belongs beside
its output; it is kept here only for provenance until someone moves it. Do not
add it to any build sequence.

Conditions:

* ``gt_centerline``: skeleton of the thresholded manual ``label_gray`` mask;
* ``label_gray_50``: manual ``label_gray`` thresholded at 50%, with PLECTA
  performing its normal internal skeletonisation.

This is an instance-reconstruction/oracle-mask experiment. Released Prediction
1/2 outputs are not re-scored as same-input comparators because they were
generated from learned distance/embedding predictions, not these GT masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


COMPARISON = Path(r"C:\Repos\comparisons\Microplastics_SEM\PLECTA_comparison")
DATASET = Path(
    r"C:\Repos\comparisons\Microplastics_SEM\dataset2\[fibre_1crop]_Exp3"
)
PAPER = Path(
    r"C:\Repos\Paper-marked----FilaSeg-Geometry-Driven-Instance-Extraction-of-thin-filaments"
)
OUT = COMPARISON / "oracle_inputs"

CONDITIONS = ("gt_centerline", "label_gray_50")
LABELS = {
    "released_distance_support": "Released distance\nsupport",
    "gt_centerline": "Pre-skeletonized\ncontrol",
    "label_gray_50": "GT label_gray\n50% threshold",
}
COLORS = {
    "released_distance_support": "#8C8C8C",
    "gt_centerline": "#4C78A8",
    "label_gray_50": "#59A14F",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def finite_mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def load_helpers(comparison: Path):
    scripts = comparison / "scripts"
    sys.path.insert(0, str(scripts))
    import make_microplastics_sem_comparison as E
    import make_microplastics_sem_figures as F

    return E, F


def make_input(condition: str, gray_path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(gray_path), dtype=np.float32)
    binary = gray > (0.5 * float(gray.max()))
    if condition == "gt_centerline":
        # The colour-instance raster is only the identity reference: skeletonising
        # its layers or union creates occlusion gaps or raster junction knots.
        # label_gray is the manual semantic foreground. PLECTA applies its frozen
        # 3-px spur pruning to this centreline inside build_graph().
        return skeletonize(binary)
    if condition == "label_gray_50":
        return binary
    raise KeyError(condition)


def aggregate(E, rows: list[dict], condition: str) -> dict:
    block = E.aggregate_method([row for row in rows if row["method"] == condition])
    block["label"] = LABELS[condition].replace("\n", " ")
    return block


def write_tables(summary: dict, rows: list[dict], out: Path):
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    order = ("released_distance_support", *CONDITIONS)
    columns = (
        "condition",
        "identity_f1_macro",
        "adjusted_rand_index_macro",
        "crossing_fidelity_macro",
        "crossing_fidelity_pooled",
        "count_mae",
        "count_bias",
        "n_identity_f1_scenes",
        "n_crossing_fidelity_scenes",
        "n_crossings",
    )
    records = []
    for condition in order:
        block = summary["conditions"][condition]
        records.append({"condition": LABELS[condition].replace("\n", " "),
                        **{key: block[key] for key in columns[1:]}})
    with (table_dir / "oracle_input_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    md = [
        "| PLECTA input | Identity F1 | ARI | Crossing Fidelity macro | Crossing Fidelity pooled | Crossings | Count MAE | Count bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        md.append(
            "| {condition} | {identity_f1_macro:.3f} | {adjusted_rand_index_macro:.3f} | "
            "{crossing_fidelity_macro:.3f} | {crossing_fidelity_pooled:.3f} | "
            "{n_crossings:d} | {count_mae:.3f} | {count_bias:+.3f} |".format(**row)
        )
    md += [
        "",
        "The two GT-derived rows share the same centreline, fragments, and junctions: "
        "PLECTA skeletonizes the thick `label_gray` mask internally. The released-distance "
        "row has a different input-derived population and is context only.",
        "",
    ]
    (table_dir / "oracle_input_summary.md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "% Generated by make_microplastics_oracle_inputs.py; do not edit.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"PLECTA input & Identity $F_1$ & ARI & Crossing Fidelity & Crossings & Count MAE & Count bias \\",
        r"\midrule",
    ]
    for row in records:
        tex.append(
            f"{row['condition']} & {row['identity_f1_macro']:.3f} & "
            f"{row['adjusted_rand_index_macro']:.3f} & "
            f"{row['crossing_fidelity_macro']:.3f} & {row['n_crossings']} & "
            f"{row['count_mae']:.3f} & "
            f"{row['count_bias']:+.3f} " + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    (table_dir / "oracle_input_summary.tex").write_text("\n".join(tex), encoding="utf-8")

    fields = ("condition", *[key for key in E_COLUMNS if key not in {"method", "method_label"}])
    with (table_dir / "per_scene_oracle_inputs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"condition": row["method"], **{
                key: row[key] for key in fields[1:]
            }})


E_COLUMNS = (
    "scene", "method", "method_label", "n_gt_instances", "n_pred_instances",
    "count_error", "count_abs_error", "identity_precision", "identity_recall",
    "identity_f1", "adjusted_rand_index", "crossing_fidelity",
    "crossing_pair_accuracy", "crossing_unpaired_fidelity", "n_crossings",
    "n_exact_crossings", "n_unpaired_exact_crossings", "n_common_fragments",
    "n_gt_assigned_fragments", "n_gt_unassigned_fragments",
    "n_pred_assigned_fragments", "n_pred_unassigned_fragments", "n_gt_pairs",
    "n_pred_pairs", "tp", "fp", "fn", "input_foreground_pixels",
    "input_vs_gt_iou", "runtime_seconds",
)


def metric_figure(summary: dict, rows: list[dict], out: Path):
    order = ("released_distance_support", *CONDITIONS)
    specs = (
        ("identity_f1_macro", "Identity F$_1$ ↑", (0, 1.08)),
        ("adjusted_rand_index_macro", "Adjusted Rand index ↑", (-0.1, 1.08)),
        ("crossing_fidelity_macro", "Crossing Fidelity ↑", (0, 1.08)),
        ("count_mae", "Count MAE ↓", None),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.7), layout="constrained")
    for ax, (key, title, ylim), letter in zip(axes.ravel(), specs, "abcd"):
        values = [summary["conditions"][condition][key] for condition in order]
        if ylim is None:
            ylim = (0, max(values) * 1.22 if max(values) else 1)
        bars = ax.bar(range(3), values, color=[COLORS[c] for c in order], width=0.68)
        for bar, value, condition in zip(bars, values, order):
            label = f"{value:.3f}"
            if key == "crossing_fidelity_macro":
                label += f"\n(n={summary['conditions'][condition]['n_crossings']})"
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035 * (ylim[1] - ylim[0]),
                    label, ha="center", va="bottom", fontsize=8.3)
        ax.set_xticks(range(3), [LABELS[c] for c in order])
        ax.tick_params(axis="x", labelsize=8.2)
        ax.set_ylim(*ylim)
        ax.set_title(f"({letter}) {title}", loc="left")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen PLECTA: input-condition results",
                 fontsize=11, fontweight="bold")
    save_figure(fig, out / "figures" / "fig_oracle_input_metrics")


def count_figure(summary: dict, rows: list[dict], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), sharex=True, sharey=True,
                             layout="constrained")
    max_pred = max(int(r["n_pred_instances"]) for r in rows)
    max_gt = max(int(r["n_gt_instances"]) for r in rows)
    for ax, condition, color, letter in zip(
        axes, CONDITIONS, (COLORS[c] for c in CONDITIONS), "ab"
    ):
        selected = [r for r in rows if r["method"] == condition]
        gt = np.asarray([int(r["n_gt_instances"]) for r in selected])
        pred = np.asarray([int(r["n_pred_instances"]) for r in selected])
        ax.plot([0, max_gt + 1], [0, max_gt + 1], "--", lw=1, color="#777777")
        ax.scatter(gt, pred, s=34, color=color, edgecolor="white", linewidth=0.5)
        block = summary["conditions"][condition]
        ax.text(0.04, 0.96, f"MAE {block['count_mae']:.3f}\nbias {block['count_bias']:+.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_title(f"({letter}) {LABELS[condition].replace(chr(10), ' ')}", loc="left")
        ax.set_xlabel("GT fibre count")
        ax.grid(color="#e2e2e2", lw=0.6)
    axes[0].set_ylabel("Predicted fibre count")
    axes[0].set_xlim(0, max_gt + 1)
    axes[0].set_ylim(0, max_pred + 2)
    save_figure(fig, out / "figures" / "fig_oracle_input_counts")


def qualitative_figure(F, payloads: dict, input_masks: dict, rows: list[dict], out: Path):
    gt_counts = {r["scene"]: int(r["n_gt_instances"])
                 for r in rows if r["method"] == "gt_centerline"}
    ordered = sorted(payloads, key=lambda stem: (gt_counts[stem], stem))
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    headings = (
        "SEM", "GT instances\n(reference)",
        "Manual semantic-mask\ninput (`label_gray`)", "PLECTA instances",
    )
    fig, axes = plt.subplots(3, 4, figsize=(7.7, 5.7), layout="constrained")
    for row_index, stem in enumerate(selected):
        payload = payloads[stem]
        images = (
            payload["sem"], F.colour_instances(payload["gt"]),
            input_masks[stem]["label_gray_50"],
            F.colour_instances(payload["label_gray_50"]),
        )
        for ax, image, heading in zip(axes[row_index], images, headings):
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if row_index == 0: ax.set_title(heading, fontsize=8.7)
        axes[row_index, 0].set_ylabel(f"{stem}\nGT n={gt_counts[stem]}", fontsize=8.3)
    fig.suptitle("PLECTA oracle instance reconstruction", fontsize=11, fontweight="bold")
    save_figure(fig, out / "figures" / "fig_oracle_input_qualitative")
    return selected


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison-root", type=Path, default=COMPARISON)
    ap.add_argument("--dataset-root", type=Path, default=DATASET)
    ap.add_argument("--paper-repo", type=Path, default=PAPER)
    ap.add_argument("--out", type=Path, default=OUT)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    comparison = args.comparison_root.resolve()
    dataset = args.dataset_root.resolve()
    paper = args.paper_repo.resolve()
    out = args.out.resolve()
    E, F = load_helpers(comparison)
    F.apply_style()
    common_metric, crossing_metric, common_path = E.load_canonical_metrics(
        E.DEFAULT_EVALUATOR, E.DEFAULT_CROSSING_METRIC
    )
    sys.path.insert(0, str(paper / "software"))
    from plecta.indep.predictor import predict, save_multilabel_npz
    from plecta.predict import PARAMS_PATH, build_params

    params = build_params()
    label_dir = dataset / "label_index" / "test" / "label"
    image_dir = dataset / "label_index" / "test" / "image"
    gray_dir = dataset / "label_gray"
    rows = []
    payloads = {}
    input_masks = {}
    for label_path in sorted(label_dir.glob("*.png")):
        stem = label_path.stem
        gt_masks = E.masks_from_image(Image.open(label_path))
        gt = E.skeletonise_instances(gt_masks)
        sem = np.asarray(Image.open(image_dir / f"{stem}.png"))
        payloads[stem] = {"sem": sem, "gt": gt}
        input_masks[stem] = {}
        for condition in CONDITIONS:
            input_mask = make_input(condition, gray_dir / f"{stem}.png")
            input_masks[stem][condition] = input_mask
            if condition == "gt_centerline":
                input_path = (
                    out / "diagnostics" / "raw_label_gray_skeleton" / f"{stem}.png"
                )
            else:
                input_path = out / "inputs" / "label_gray_50" / f"{stem}.png"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(input_mask, dtype=np.uint8) * 255).save(input_path)
            fragments = common_metric.common_fragments(
                input_mask,
                min_len_px=E.SCORER_CONSTANTS["min_len_px"],
                prune_spur_px=E.SCORER_CONSTANTS["prune_spur_px"],
            )
            null = crossing_metric.junction_resolution(
                input_mask, gt, {},
                prune_spur_px=E.SCORER_CONSTANTS["prune_spur_px"],
                min_len_px=E.SCORER_CONSTANTS["min_len_px"],
                min_overlap_frac=E.SCORER_CONSTANTS["min_overlap_frac"],
                max_dist_px=E.SCORER_CONSTANTS["max_dist_px"],
            )
            control = E.score_partition(
                common_metric, crossing_metric, input_mask, fragments, gt, gt, null
            )
            for metric in ("f1", "adjusted_rand_index"):
                value = float(control["pair"][metric])
                if math.isfinite(value) and not math.isclose(value, 1.0, abs_tol=1e-12):
                    raise AssertionError(
                        f"{stem}/{condition}: GT-self {metric} is {value}, not 1"
                    )
            crossing_value = float(control["crossing"]["exact_rate"])
            if math.isfinite(crossing_value) and not math.isclose(
                crossing_value, 1.0, abs_tol=1e-12
            ):
                raise AssertionError(
                    f"{stem}/{condition}: GT-self Crossing Fidelity is "
                    f"{crossing_value}, not 1"
                )
            t0 = time.perf_counter()
            pred = E.skeletonise_instances(predict(input_mask, params))
            runtime_seconds = time.perf_counter() - t0
            scores = E.score_partition(
                common_metric, crossing_metric, input_mask, fragments, gt, pred, null
            )
            row = E.row_from_scores(
                stem, condition, LABELS[condition].replace("\n", " "), len(gt), len(pred),
                scores, null, int(fragments.max()), int(input_mask.sum()), float("nan"),
                runtime_seconds,
            )
            rows.append(row)
            payloads[stem][condition] = pred
            pred_path = out / "predictions" / condition / f"{stem}.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            save_multilabel_npz(pred_path, pred, input_mask.shape)
        centreline_pred = payloads[stem]["gt_centerline"]
        gray_pred = payloads[stem]["label_gray_50"]
        if set(centreline_pred) != set(gray_pred) or any(
            not np.array_equal(centreline_pred[key], gray_pred[key])
            for key in centreline_pred
        ):
            raise AssertionError(
                f"{stem}: pre-skeletonized and thick label_gray predictions differ"
            )
        print(stem, *(f"{c}={len(payloads[stem][c])}" for c in CONDITIONS), flush=True)

    prior = json.loads((comparison / "metrics" / "summary.json").read_text(encoding="utf-8"))
    conditions = {
        "released_distance_support": {
            **prior["methods"]["plecta"],
            "label": LABELS["released_distance_support"].replace("\n", " "),
        },
        **{condition: aggregate(E, rows, condition) for condition in CONDITIONS},
    }
    summary = {
        "study": "frozen PLECTA on GT-derived Microplastics SEM inputs",
        "interpretation": "oracle-mask instance-reconstruction experiment",
        "conditions": conditions,
        "protocol": {
            "gt_centerline": "skeleton of manual label_gray > 0.5 * its maximum; PLECTA then applies frozen spur_px pruning",
            "label_gray_50": "manual grayscale mask > 0.5 * its maximum; normal PLECTA internal skeletonisation",
            "ground_truth": "independently skeletonised colour instances",
            "n_scenes": 21,
            "parameters": "frozen paper params; no tuning",
            "released_prediction_1_2": "not same-input comparators and therefore not included as oracle arms",
            "oracle_prediction_equality": "asserted pixel-identical instance masks for every scene",
        },
        "plecta_parameters": asdict(params),
        "provenance": {
            "script": Path(__file__).resolve().as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "common_metric": common_path.as_posix(),
            "common_metric_sha256": sha256(common_path),
            "plecta_params": PARAMS_PATH.as_posix(),
            "plecta_params_sha256": sha256(PARAMS_PATH),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=E.CSV_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps(E.json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    write_tables(summary, rows, out)
    metric_figure(summary, rows, out)
    count_figure(summary, rows, out)
    selected = qualitative_figure(F, payloads, input_masks, rows, out)
    readme = (
        "# Oracle-input PLECTA experiment\n\n"
        "Frozen PLECTA was run on (1) the skeleton of `label_gray` thresholded at "
        "50% and (2) that thick thresholded mask itself. This isolates instance "
        "reconstruction from learned semantic-mask prediction. Released Prediction "
        "1/2 outputs are not same-input baselines. The pre-skeletonized input has "
        f"{conditions['gt_centerline']['n_crossings']} detected junctions and the "
        "thresholded `label_gray` inputs have "
        f"{conditions['label_gray_50']['n_crossings']}; the Crossing Fidelity rows "
        "therefore have the same evaluation population. PLECTA produced "
        "pixel-identical instances for both representations in every scene. "
        "The thick thresholded `label_gray` mask is the canonical PLECTA input. "
        "Raw one-pixel skeleton PNGs are retained only under `diagnostics/` as a "
        "representation control; their cap branches are medial-axis artifacts and "
        "are not used in the qualitative figure. "
        "The coloured GT-instance panel is the identity reference, not a PLECTA "
        "input; its apparent crossing gaps come from the single-layer colour raster.\n\n"
        f"Qualitative scenes, selected only by GT count: {', '.join(selected)}.\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote oracle experiment to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
