"""Extract the qualitative example scenes for Fig. 8.

Two selection modes, both stated rules rather than searches.

``--select plecta`` (the mode Fig. 8 now uses).  The comparator column was
dropped on review, so the rule can no longer be a difference against it.  The
scenes are PLECTA's own best, median and *worst* F1 over the comparison set,
so the figure is obliged to show a failure case.

``--select comparator`` (the earlier mode, kept working).  The scene where
PLECTA beats the comparator by most, the scene closest to the median
delta-F1, and the scene where the comparator beats PLECTA by most.

``--render`` additionally runs the full downstream stack -- width measurement
from the paired SEM image, then ribbon rendering -- and packs those masks
alongside the evaluated centreline instances.  The two are stored separately
and never mixed: the grouping that the paper scores is ``plecta``, and
``plecta_rendered`` is a redrawing of exactly the same instances.  Rendering
cannot move a pixel from one instance to another, only change how thickly each
one is drawn.

Writes ``results/plecta_examples_<name>.json``.

    python scripts/figures/extract_plecta_examples.py \
        --select plecta --render --name full \
        --record plecta_dnai_comparison.json \
        --scenes C:/Repos/comparisons/filaseg_stubmatch_strandface/data/testcmp10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COVERAGE_THR = 0.8
PURITY_THR = 0.8


def pack(mask):
    mask = np.asarray(mask, dtype=bool)
    return {"h": int(mask.shape[0]), "w": int(mask.shape[1]),
            "idx": np.flatnonzero(mask.ravel()).astype(int).tolist()}


def load_layers(npz_path):
    data = np.load(npz_path)
    shape = tuple(int(v) for v in data["shape"])
    ids, indptr, indices = data["ids"], data["indptr"], data["indices"]
    out = []
    for k in range(len(ids)):
        flat = indices[indptr[k]:indptr[k + 1]]
        m = np.zeros(shape[0] * shape[1], bool)
        m[flat] = True
        out.append(m.reshape(shape))
    return out, shape


def load_label_tif(path, shape):
    from skimage import io as skio
    lab = np.asarray(skio.imread(str(path)))
    return [lab == i for i in np.unique(lab) if int(i) != 0]


def pack_labels(layers):
    """Layers as one int label image, largest first so small ones stay visible.

    A label image cannot express the crossing pixels that belong to two
    instances, but at whole-scene zoom that overlap is a pixel wide and
    invisible; Figs. 1 and 5 carry the overlap where it can actually be seen.
    """
    import numpy as np
    if not layers:
        return {"h": 1, "w": 1, "idx": [], "lab": []}
    order = sorted(layers, key=lambda m: -int(m.sum()))
    lab = np.zeros(order[0].shape, np.int32)
    for k, m in enumerate(order, start=1):
        lab[m] = k
    flat = lab.ravel()
    nz = np.flatnonzero(flat)
    return {"h": int(lab.shape[0]), "w": int(lab.shape[1]),
            "idx": nz.astype(int).tolist(),
            "lab": flat[nz].astype(int).tolist(),
            "n": len(order)}


def _unused_recovery(gt_layers, pred_layers):
    """Kept for reference: instance-level recovery at the scorer's thresholds.

    Not used by the figure.  At these thresholds a correct but pixel-wise
    fragmented instance counts as a miss, so on PLECTA's core output -- which
    is deliberately not a single connected run per filament -- the marking
    labels most instances and carries no information.
    """
    gt_area = np.array([m.sum() for m in gt_layers], float)
    pred_area = np.array([m.sum() for m in pred_layers], float)
    recovered = np.zeros(len(gt_layers), bool)
    useful = np.zeros(len(pred_layers), bool)
    for j, pred in enumerate(pred_layers):
        if not pred_area[j]:
            continue
        for i, gt in enumerate(gt_layers):
            if not gt_area[i]:
                continue
            inter = float(np.count_nonzero(gt & pred))
            if not inter:
                continue
            if inter / gt_area[i] >= COVERAGE_THR and \
                    inter / pred_area[j] >= PURITY_THR:
                recovered[i] = True
                useful[j] = True
    return recovered, useful


def union(layers, keep):
    if not len(layers):
        return np.zeros((1, 1), bool)
    out = np.zeros(layers[0].shape, bool)
    for m, flag in zip(layers, keep):
        if flag:
            out |= m
    return out


def plecta_rows(record, condition):
    """(scene, density, f1) per scene for PLECTA, from the DNAi record."""
    rows = [r for r in record["per_scene"]
            if r["method"] == "plecta" and r["mask_variant"] == condition
            and r["status"] == "ok"]
    return [{"scene": r["scene"], "density": int(r["density"]),
             "plecta_f1": float(r["f1"])} for r in rows]


def comparator_rows(record, condition):
    rows = record["conditions"][condition]["per_scene"]
    return [{"scene": r["scene"], "density": int(r["density"]),
             "plecta_f1": float(r["plecta_f1"]),
             "comparator_f1": float(r["greedy_f1"])} for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plecta", default=r"C:/Repos/PLECTA")
    ap.add_argument("--select", choices=("stratum", "plecta", "comparator"),
                    default="stratum")
    ap.add_argument("--strata", default="20,40,60",
                    help="--select stratum: the coverage levels to show")
    ap.add_argument("--name", default="full",
                    help="output goes to results/plecta_examples_<name>.json")
    ap.add_argument("--render", action="store_true",
                    help="also emit the full downstream stack: width "
                         "measurement from the paired SEM image, then ribbon "
                         "rendering of the same instances")
    ap.add_argument("--comparator-label", default="Greedy continuation")
    ap.add_argument("--record", default="plecta_dnai_comparison.json")
    ap.add_argument("--condition", default="degraded")
    ap.add_argument(
        "--predictions",
        default=r"C:/Repos/comparisons/filaseg_stubmatch_strandface/predictions/testcmp10/baseline_skeleton")
    ap.add_argument(
        "--scenes",
        default=r"C:/Repos/comparisons/filaseg_stubmatch_strandface/data/testcmp10")
    ap.add_argument("--mask-name", default="mask_w1.png")
    ap.add_argument("--sem-name", default="sem.png")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.plecta)))
    from plecta.graph import read_mask                   # noqa: E402
    from plecta.predict import load_params, predict      # noqa: E402

    record = json.loads((REPO / "results" / args.record).read_text(encoding="utf-8"))
    if args.select == "stratum":
        # One row per coverage level, each the *median* scene of its own
        # stratum.  This spans the whole difficulty range -- the last row is
        # the hardest density by construction -- and no row is chosen by how
        # well or badly the method did.
        rows = plecta_rows(record, args.condition)
        score = np.array([r["plecta_f1"] for r in rows])
        picks = []
        for level in (int(v) for v in args.strata.split(",")):
            idx = [i for i, r in enumerate(rows) if r["density"] == level]
            median = float(np.median(score[idx]))
            best = min(idx, key=lambda i: abs(score[i] - median))
            picks.append((f"cov{level}", best,
                          "median F1 of the %d scenes at %d%% coverage"
                          % (len(idx), level)))
        rule_text = ("One scene per row: the median-F1 scene at each of the "
                     "coverage levels shown. The rows span the whole "
                     "difficulty range and none is chosen by outcome.")
    elif args.select == "plecta":
        rows = plecta_rows(record, args.condition)
        score = np.array([r["plecta_f1"] for r in rows])
        order = np.argsort(score)
        picks = [
            ("best", int(order[-1]), "highest PLECTA F1 of the %d scenes"
             % len(rows)),
            ("median", int(np.argmin(np.abs(score - np.median(score)))),
             "F1 closest to the median over all %d scenes" % len(rows)),
            ("worst", int(order[0]), "lowest PLECTA F1 of the %d scenes"
             % len(rows)),
        ]
        rule_text = ("One scene per row, chosen by a stated rule and not by "
                     "inspection: PLECTA's highest, median and lowest F1 over "
                     "the whole comparison set. The last row is therefore the "
                     "worst case, not a favourable one.")
    else:
        rows = comparator_rows(record, args.condition)
        score = np.array([r["plecta_f1"] - r["comparator_f1"] for r in rows])
        order = np.argsort(score)
        picks = [
            ("plecta_win", int(order[-1]), "largest PLECTA advantage"),
            ("typical", int(np.argmin(np.abs(score - np.median(score)))),
             "delta-F1 closest to the median over all %d scenes" % len(rows)),
            ("comparator_win", int(order[0]), "largest comparator advantage"),
        ]
        rule_text = ("One scene per row: the largest PLECTA advantage, the "
                     "scene closest to the median delta-F1, and the largest "
                     "comparator advantage.")

    params = load_params()
    scenes_root, pred_root = Path(args.scenes), Path(args.predictions)
    panels = []
    for key, index, rule in picks:
        row = rows[index]
        scene = row["scene"]
        scene_dir = scenes_root / f"cov{row['density']}" / scene
        mask = read_mask(scene_dir / args.mask_name)
        gt_layers, shape = load_layers(scene_dir / "gt_multilabel.npz")
        plecta_layers = list(predict(mask, params).values())

        entry = {"key": key, "rule": rule, "scene": scene,
                 "density": row["density"],
                 "plecta_f1": round(row["plecta_f1"], 4),
                 "shape": list(shape),
                 "mask": pack(mask),
                 "reference": pack_labels(gt_layers),
                 "plecta": pack_labels(plecta_layers),
                 "n_reference": len(gt_layers),
                 "n_plecta": len(plecta_layers)}
        if "comparator_f1" in row:
            entry["comparator_f1"] = round(row["comparator_f1"], 4)
            entry["delta_f1"] = round(float(score[index]), 4)
            comp_layers = load_label_tif(
                pred_root / f"cov{row['density']}" / scene / "labels.tif",
                shape)
            entry["comparator"] = pack_labels(comp_layers)
            entry["n_comparator"] = len(comp_layers)
        if args.render:
            from plecta.image.pipeline import measure_scene
            from plecta.image.refine import RefineParams, refine_scene
            res = measure_scene(scene_dir, mask_name=args.mask_name,
                                sem_name=args.sem_name, params=params)
            ribbons, _poly, log = refine_scene(res, RefineParams())
            entry["plecta_rendered"] = pack_labels(list(ribbons.values()))
            entry["n_plecta_rendered"] = len(ribbons)
            entry["n_absorbed_by_rendering"] = len(log) if log else 0
        panels.append(entry)
        print(key, scene, "cov", entry["density"], "F1", entry["plecta_f1"],
              "instances", entry["n_reference"], entry["n_plecta"],
              entry.get("n_plecta_rendered", "-"), flush=True)

    out = {
        "schema": 2,
        "select": args.select,
        "comparator_label": args.comparator_label,
        "condition": args.condition,
        "record": args.record,
        "rendered": bool(args.render),
        "n_scenes_in_comparison": len(rows),
        "selection_rule": rule_text,
        "rendering_note": "plecta_rendered redraws the instances in 'plecta' "
                          "at their measured widths. It is downstream of the "
                          "evaluated grouping and is not what the reported "
                          "metrics score.",
        "colour_rule": "Instance colours are arbitrary and carry identity "
                       "only within one panel; they are not matched between "
                       "columns.",
        "panels": panels,
    }
    path = REPO / "results" / f"plecta_examples_{args.name}.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("[wrote]", path, round(path.stat().st_size / 1e6, 2), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
