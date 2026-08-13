"""Extract the committed asset behind ``fig_real_sem``.

The qualitative real-SEM figure needs three instance labellings of one field --
the manual annotation, PLECTA on the manual-derived axis, and PLECTA on the
nnU-Net axis -- with instance colours that correspond across the three.  This
script runs PLECTA on the *whole* field, matches predicted instances to manual
ones there, and only then cuts out the crop that is drawn.  Running it on the
crop instead would change the reconstruction at the crop boundary and the
picture would no longer be of the run the numbers come from.

**Which field.**  B58_110.  It is one of the two fields held out of the nnU-Net
training run used here, and it is the weaker of those two on the manual-derived
axis, so it is not the flattering choice among the held-out fields.  B58_100 is
training-contaminated and is deliberately not drawn.

**Which crop.**  Chosen, not picked: over a grid of windows, the one whose local
pairwise F1 is jointly closest to the whole-field value on *both* axes, so the
crop is typical of the field rather than selected from it.  This is the rule
``fig_plecta_examples`` already uses at scene level ("the scene whose F1 is
closest to the median").  Local F1 is computed by restricting the common
fragments to those lying wholly inside the window and rescoring; nothing is
re-run.

The stored per-field numbers are reproduced before anything is written, as the
repository requires of any figure built from a fresh run.

    python scripts/figures/extract_real_sem_crop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY = Path(r"C:\Repos\comparisons\real_sem_study")

for path in (Path(r"C:\Repos\comparisons\graft_regime\scripts"),
             Path(r"C:\Repos\comparisons\metrics_study\scripts"),
             Path(r"C:\Repos\filaments_quantification\eval")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _local as L                                            # noqa: E402
import metrics_lib as ML                                      # noqa: E402
import common_metric as CM                                    # noqa: E402
import instance_io as IIO                                     # noqa: E402

FIELD = "b58_110"
AXES = (("manual", "skel"), ("unet", "nnunet"))
CROP = 512
STRIDE = 128
TOL = 2                       # the placement tolerance used throughout
MIN_REFERENCE = 25            # a window with almost nothing in it is not typical
OUT = REPO / "results" / "figure_assets" / "real_sem_crop.npz"


def load_reference(scene: Path, shape) -> dict:
    """The manual annotation, from the study's CSR-packed label file."""
    data = np.load(scene / "gt_multilabel.npz")
    ids, indptr, indices = data["ids"], data["indptr"], data["indices"]
    out = {}
    for k, ident in enumerate(ids):
        flat = np.zeros(int(np.prod(shape)), bool)
        flat[indices[indptr[k]:indptr[k + 1]]] = True
        out[int(ident)] = flat.reshape(shape)
    return out


def greedy_match(gt: dict, pred: dict, shape) -> dict:
    """Predicted id -> reference id, greedily by highest tolerant IoU.

    The same matching rule and the same tolerance detection F1 uses, so the
    colours in the figure correspond the way the table's object measure does.
    """
    gids, pids, iou = ML._tolerant_iou_matrix(gt, pred, TOL, shape)
    work = iou.copy()
    out = {}
    while work.size:
        k = int(np.argmax(work))
        i, j = divmod(k, work.shape[1])
        if work[i, j] <= 0.0:
            break
        out[pids[j]] = gids[i]
        work[i, :] = -1.0
        work[:, j] = -1.0
    return out


def label_image(masks: dict, shape, ident_of=None, unmatched=-1) -> np.ndarray:
    """Paint instances into one label image, later ids winning an overlap."""
    out = np.zeros(shape, np.int32)
    for ident, mask in masks.items():
        value = ident if ident_of is None else ident_of.get(ident, unmatched)
        out[np.asarray(mask, bool)] = value
    return out


def fragments_inside(frag: np.ndarray, counts_all: np.ndarray,
                     r0: int, c0: int) -> set:
    """Fragment ids lying wholly inside the window at (r0, c0).

    Whole containment rather than centroid membership, so that a fragment cut by
    the window edge is scored by neither the crop nor the deviation that chose
    it.
    """
    window = frag[r0:r0 + CROP, c0:c0 + CROP]
    counts_win = np.bincount(window.ravel(), minlength=counts_all.size)
    ids = np.flatnonzero(counts_win)
    return {int(f) for f in ids if f and counts_win[f] == counts_all[f]}


def main() -> int:
    stored = {row["field"] + "|" + row["axis"]: row for row in
              json.loads((STUDY / "results_v2.json").read_text())["rows"]}
    per_axis, sem = {}, None

    for key, variant in AXES:
        scene = STUDY / "scenes_v2" / FIELD / variant
        mask = L.read_mask(scene / "mask.png")
        prediction = L.plecta_predict(mask)
        centrelines = L.centerline_instances(prediction)
        reference = IIO.InstanceSet(shape=mask.shape,
                                    masks=load_reference(scene, mask.shape),
                                    source="gt")

        #  Parity first: this must reproduce the stored row exactly, or the
        #  picture is not of the run the manuscript reports.
        scores = L.score_prediction(scene, prediction, "mask.png")
        want = stored[FIELD + "|" + ("manual-skel" if key == "manual"
                                     else "nnU-Net")]
        if abs(scores["f1"] - want["pairwise_f1"]) > 1e-12:
            raise SystemExit(
                "parity failed on %s %s: got %.17g, stored %.17g"
                % (FIELD, key, scores["f1"], want["pairwise_f1"]))

        frag = CM.common_fragments(mask)
        ref_assign = CM.assign_fragments(frag, reference, 0.3, 6.0)
        pred_assign = CM.assign_fragments(frag, centrelines, 0.3, 6.0)
        match = greedy_match(reference.masks, centrelines.masks, mask.shape)

        per_axis[key] = {
            "mask": mask, "frag": frag,
            "counts": np.bincount(frag.ravel()),
            "ref_assign": ref_assign, "pred_assign": pred_assign,
            "field_f1": scores["f1"],
            "labels": label_image(centrelines.masks, mask.shape, match),
            "reference": reference,
        }
        if sem is None:
            from PIL import Image
            sem = np.array(Image.open(scene / "sem.png").convert("L"))
        print("[extract] %s %-6s field F1 %.4f, %d instances, %d matched"
              % (FIELD, key, scores["f1"], len(centrelines.masks), len(match)))

    #  The reference axis drawn in panel (b) is the skeleton of the manual
    #  annotation -- which is exactly the manual-derived input axis -- so the
    #  three instance panels are all one-pixel centrelines and comparable.
    from skimage.morphology import skeletonize
    reference = per_axis["manual"]["reference"]
    ref_labels = label_image(
        {i: skeletonize(np.asarray(m, bool)) for i, m in reference.masks.items()},
        per_axis["manual"]["mask"].shape)

    height, width = per_axis["manual"]["mask"].shape
    best = None
    for r0 in range(0, height - CROP + 1, STRIDE):
        for c0 in range(0, width - CROP + 1, STRIDE):
            deviation, local, n_ref = 0.0, {}, 0
            for key in per_axis:
                axis = per_axis[key]
                keep = fragments_inside(axis["frag"], axis["counts"], r0, c0)
                if not keep:
                    deviation = float("inf")
                    break
                ga = {f: g for f, g in axis["ref_assign"].items() if f in keep}
                pa = {f: p for f, p in axis["pred_assign"].items() if f in keep}
                sub = CM.pairwise_scores(ga, pa)
                local[key] = sub["f1"]
                n_ref = max(n_ref, len({g for g in ga.values() if g}))
                deviation += abs(sub["f1"] - axis["field_f1"])
            if n_ref < MIN_REFERENCE or not np.isfinite(deviation):
                continue
            if best is None or deviation < best[0]:
                best = (deviation, r0, c0, dict(local), n_ref)

    if best is None:
        raise SystemExit("no candidate window met the criteria")
    deviation, r0, c0, local, n_ref = best
    print("[extract] crop at row %d col %d (%dx%d): local F1 %s, "
          "total deviation %.4f over %d reference filaments"
          % (r0, c0, CROP, CROP,
             ", ".join("%s %.4f" % kv for kv in sorted(local.items())),
             deviation, n_ref))

    sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
    payload = {
        "sem": sem[sl].astype(np.uint8),
        "reference": ref_labels[sl].astype(np.int32),
        "manual": per_axis["manual"]["labels"][sl].astype(np.int32),
        "unet": per_axis["unet"]["labels"][sl].astype(np.int32),
        #  Stored as a plain string so the asset loads without allow_pickle.
        "meta": np.array(json.dumps({
            "field": FIELD,
            "crop_row": r0, "crop_col": c0, "crop_px": CROP,
            "stride_px": STRIDE, "tolerance_px": TOL,
            "field_f1": {k: per_axis[k]["field_f1"] for k in per_axis},
            "local_f1": local,
            "n_reference_in_crop": int(n_ref),
            "unmatched_label": -1,
            "selection": ("window whose local pairwise F1 is jointly closest "
                          "to the whole-field value on both axes"),
            "held_out": True,
        })),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    print("[extract] wrote %s (%.0f kB)" % (OUT, OUT.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
