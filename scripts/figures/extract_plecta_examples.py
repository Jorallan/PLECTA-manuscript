"""Extract the paired qualitative examples: PLECTA against one comparator.

Deliberately parameterised.  The comparator's name, its per-scene record and
the directory holding its predictions are all arguments, so a different
comparator can be dropped in without touching the figure generator.

Scene selection is a stated rule, not a search:

  * ``plecta_win``   the scene where PLECTA beats the comparator by most;
  * ``typical``      the scene whose delta-F1 is closest to the median over
                     the whole comparison set;
  * ``comparator_win`` the scene where the comparator beats PLECTA by most.

Showing only the first would be a difference-maximising rule; all three are
recorded, together with the rule, in the manifest block of the output.

Writes ``results/plecta_examples_<comparator>.json``.

    python scripts/figures/extract_plecta_examples.py \
        --comparator greedy --comparator-label "Greedy continuation" \
        --predictions C:/Repos/comparisons/filaseg_stubmatch_strandface/predictions/testcmp10/baseline_skeleton \
        --scenes      C:/Repos/comparisons/filaseg_stubmatch_strandface/data/testcmp10
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plecta", default=r"C:/Repos/stubmatch")
    ap.add_argument("--comparator", default="greedy")
    ap.add_argument("--comparator-label", default="Greedy continuation")
    ap.add_argument("--record", default="plecta_greedy_baseline.json")
    ap.add_argument("--condition", default="degraded")
    ap.add_argument(
        "--predictions",
        default=r"C:/Repos/comparisons/filaseg_stubmatch_strandface/predictions/testcmp10/baseline_skeleton")
    ap.add_argument(
        "--scenes",
        default=r"C:/Repos/comparisons/filaseg_stubmatch_strandface/data/testcmp10")
    ap.add_argument("--mask-name", default="mask_w1.png")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.plecta)))
    from plecta.graph import read_mask                   # noqa: E402
    from plecta.predict import load_params, predict      # noqa: E402

    record = json.loads((REPO / "results" / args.record).read_text(encoding="utf-8"))
    rows = record["conditions"][args.condition]["per_scene"]
    deltas = np.array([r["plecta_f1"] - r["greedy_f1"] for r in rows])
    order = np.argsort(deltas)
    picks = [
        ("plecta_win", int(order[-1]), "largest PLECTA advantage"),
        ("typical", int(np.argmin(np.abs(deltas - np.median(deltas)))),
         "delta-F1 closest to the median over all %d scenes" % len(rows)),
        ("comparator_win", int(order[0]), "largest comparator advantage"),
    ]

    params = load_params()
    scenes_root, pred_root = Path(args.scenes), Path(args.predictions)
    panels = []
    for key, index, rule in picks:
        row = rows[index]
        scene = row["scene"]
        scene_dir = scenes_root / f"cov{int(row['density'])}" / scene
        mask = read_mask(scene_dir / args.mask_name)
        gt_layers, shape = load_layers(scene_dir / "gt_multilabel.npz")
        plecta_layers = list(predict(mask, params).values())
        comp_layers = load_label_tif(
            pred_root / f"cov{int(row['density'])}" / scene / "labels.tif",
            shape)

        entry = {"key": key, "rule": rule, "scene": scene,
                 "density": int(row["density"]),
                 "delta_f1": round(float(deltas[index]), 4),
                 "plecta_f1": round(float(row["plecta_f1"]), 4),
                 "comparator_f1": round(float(row["greedy_f1"]), 4),
                 "shape": list(shape),
                 "mask": pack(mask),
                 "reference": pack_labels(gt_layers),
                 "plecta": pack_labels(plecta_layers),
                 "comparator": pack_labels(comp_layers),
                 "n_reference": len(gt_layers),
                 "n_plecta": len(plecta_layers),
                 "n_comparator": len(comp_layers)}
        panels.append(entry)
        print(key, scene, "dF1", entry["delta_f1"], "instances",
              entry["n_reference"], entry["n_plecta"], entry["n_comparator"],
              flush=True)

    out = {
        "schema": 1,
        "comparator": args.comparator,
        "comparator_label": args.comparator_label,
        "condition": args.condition,
        "record": args.record,
        "n_scenes_in_comparison": len(rows),
        "selection_rule": "One scene per row, chosen by a stated rule and not "
                          "by inspection: the largest PLECTA advantage, the "
                          "scene closest to the median delta-F1, and the "
                          "largest comparator advantage.",
        "colour_rule": "Instance colours are arbitrary and carry identity "
                       "only within one panel; they are not matched between "
                       "columns.",
        "panels": panels,
    }
    path = REPO / "results" / f"plecta_examples_{args.comparator}.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("[wrote]", path, round(path.stat().st_size / 1e6, 2), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
