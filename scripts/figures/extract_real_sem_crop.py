"""Extract the committed asset behind ``fig_real_sem``.

The qualitative real-SEM figure needs three instance labellings of one field --
the manual annotation, PLECTA on the manual-derived axis, and PLECTA on the
nnU-Net axis -- with instance colours that correspond across the three.  This
script runs PLECTA on the *whole* field, matches predicted instances to manual
ones there, and only then cuts out the crop that is drawn.  Running it on the
crop instead would change the reconstruction at the crop boundary and the
picture would no longer be of the run the numbers come from.

**Which fields.**  B58_110 and B58_300, which are both held out of the nnU-Net
training run used here.  B58_100 is training-contaminated under every available
run and is deliberately not drawn: a picture of a field the network memorised
would show a quality of mask transfer that does not exist.  Two fields also let
the figure show that the result is not one lucky field.

**Which crop.**  Chosen, not picked: over a grid of windows, the one whose local
pairwise F1 is jointly closest to that field's whole-field value on *both* axes,
so the crop is typical of the field rather than selected from it.  This is the
rule ``fig_plecta_examples`` already uses at scene level ("the scene whose F1 is
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
             Path(r"C:\Repos\comparisons\real_sem_study\scripts"),
             Path(r"C:\Repos\filaments_quantification\eval")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _local as L                                            # noqa: E402
import metrics_lib as ML                                      # noqa: E402
import common_metric as CM                                    # noqa: E402
import instance_io as IIO                                     # noqa: E402
#  PLECTA's own optional image layer, driven rather than reimplemented: it
#  measures a width per instance by FWHM across the centreline and stamps a
#  ribbon. This is the Section 2.5 rendering, and it runs after the grouping is
#  fixed, so it cannot move a pixel from one instance to another.
import width_render as WR                                     # noqa: E402

#: Both fields that the nnU-Net run used here held out of training. B58_100 is
#: contaminated under every available run and is deliberately not drawn: a
#: picture of a field the network memorised would show mask transfer that does
#: not exist.
FIELDS = ("b58_110", "b58_300")
AXES = (("manual", "skel"), ("unet", "nnunet"))
#: The window is wider than tall because the figure puts two panels side by side
#: across \linewidth: each is then ~3.1 in wide, and a square crop would make the
#: figure 3.3 in tall for content that is a texture and reads just as well at
#: 3:2. That is worth about a fifth of a page on a 22-page budget.
CROP_H, CROP_W = 344, 512
STRIDE = 88
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


def pack_instances(masks: dict, shape, ident_of=None, unmatched=-1) -> dict:
    """Keep every instance separately, as flat pixel indices over the field.

    Not a label image.  Flattening instances into one label array makes the last
    one painted win at every crossing, which is exactly where PLECTA's output is
    interesting: a crossing pixel legitimately belongs to two filaments, and the
    method is overlap-aware precisely so it can say so.  A label image throws
    that away and the picture then shows an arbitrary winner.  Instances are
    therefore carried through separately and blended at draw time.

    ``ident_of`` maps an instance id to the reference filament it was matched
    to, which is what the colour is keyed on; the instances stay distinct even
    when several are unmatched and share the ``unmatched`` colour.
    """
    order = sorted(masks)
    colours = np.array([(ident_of.get(k, unmatched) if ident_of is not None
                         else k) for k in order], dtype=np.int32)
    flat = [np.flatnonzero(np.asarray(masks[k], bool).ravel()) for k in order]
    return {"colour_ids": colours, "pixels": flat, "shape": shape}


def crop_pack(pack: dict, r0: int, c0: int) -> dict:
    """Restrict a packed instance set to a window, as CSR over crop-local ids.

    Instances left with no pixel inside the window are dropped.  Storing the
    survivors as ``(colour_ids, indptr, indices)`` is the same shape the study's
    own ``gt_multilabel.npz`` uses, so the asset needs no bespoke reader.
    """
    height, width = pack["shape"]
    keep_ids, runs = [], []
    for colour, flat in zip(pack["colour_ids"], pack["pixels"]):
        rows, cols = np.divmod(flat, width)
        inside = ((rows >= r0) & (rows < r0 + CROP_H)
                  & (cols >= c0) & (cols < c0 + CROP_W))
        if not inside.any():
            continue
        local = (rows[inside] - r0) * CROP_W + (cols[inside] - c0)
        keep_ids.append(int(colour))
        runs.append(local.astype(np.int32))
    indptr = np.zeros(len(runs) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in runs], out=indptr[1:])
    indices = (np.concatenate(runs) if runs
               else np.zeros(0, dtype=np.int32)).astype(np.int32)
    return {"colour_ids": np.array(keep_ids, dtype=np.int32),
            "indptr": indptr, "indices": indices}


def fragments_inside(frag: np.ndarray, counts_all: np.ndarray,
                     r0: int, c0: int) -> set:
    """Fragment ids lying wholly inside the window at (r0, c0).

    Whole containment rather than centroid membership, so that a fragment cut by
    the window edge is scored by neither the crop nor the deviation that chose
    it.
    """
    window = frag[r0:r0 + CROP_H, c0:c0 + CROP_W]
    counts_win = np.bincount(window.ravel(), minlength=counts_all.size)
    ids = np.flatnonzero(counts_win)
    return {int(f) for f in ids if f and counts_win[f] == counts_all[f]}


def extract_field(field: str, stored: dict) -> tuple[dict, dict]:
    """Run both axes of one field, choose its crop, return arrays and metadata."""
    per_axis, sem = {}, None

    for key, variant in AXES:
        scene = STUDY / "scenes_v2" / field / variant
        mask = L.read_mask(scene / "mask.png")
        prediction = L.plecta_predict(mask)
        centrelines = L.centerline_instances(prediction)
        reference = IIO.InstanceSet(shape=mask.shape,
                                    masks=load_reference(scene, mask.shape),
                                    source="gt")

        #  Parity first: this must reproduce the stored row exactly, or the
        #  picture is not of the run the manuscript reports.
        scores = L.score_prediction(scene, prediction, "mask.png")
        want = stored[field + "|" + ("manual-skel" if key == "manual"
                                     else "nnU-Net")]
        if abs(scores["f1"] - want["pairwise_f1"]) > 1e-12:
            raise SystemExit(
                "parity failed on %s %s: got %.17g, stored %.17g"
                % (field, key, scores["f1"], want["pairwise_f1"]))

        frag = CM.common_fragments(mask)
        ref_assign = CM.assign_fragments(frag, reference, 0.3, 6.0)
        pred_assign = CM.assign_fragments(frag, centrelines, 0.3, 6.0)
        match = greedy_match(reference.masks, centrelines.masks, mask.shape)

        #  The same instances re-drawn at the width the SEM says each filament
        #  is. Run on the whole field, like the grouping, and cropped after.
        widths, width_record = WR.render_all(scene, mask, prediction.masks)
        print("[extract]   width: %d/%d instances measured, scene median %s px"
              % (width_record["n_measured"], width_record["n_instances"],
                 width_record["scene_median_width_px"]))

        per_axis[key] = {
            "mask": mask, "frag": frag,
            "counts": np.bincount(frag.ravel()),
            "ref_assign": ref_assign, "pred_assign": pred_assign,
            "field_f1": scores["f1"],
            "centre_pack": pack_instances(centrelines.masks, mask.shape, match),
            "width_pack": pack_instances(widths, mask.shape, match),
            "width_record": width_record,
            "reference": reference,
        }
        if sem is None:
            from PIL import Image
            sem = np.array(Image.open(scene / "sem.png").convert("L"))
        print("[extract] %s %-6s field F1 %.4f, %d instances, %d matched"
              % (field, key, scores["f1"], len(centrelines.masks), len(match)))

    #  Two renderings of the annotation, because the two figures compare against
    #  different things. Its skeleton is exactly the manual-derived input axis,
    #  and is what the centreline panels are comparable with; its stored form is
    #  already painted at the width the annotator drew, which is what the
    #  width-rendered panels are comparable with. Neither is invented.
    from skimage.morphology import skeletonize
    reference = per_axis["manual"]["reference"]
    shape = per_axis["manual"]["mask"].shape
    ref_centre_pack = pack_instances(
        {i: skeletonize(np.asarray(m, bool)) for i, m in reference.masks.items()},
        shape)
    ref_width_pack = pack_instances(reference.masks, shape)

    height, width = per_axis["manual"]["mask"].shape
    best = None
    for r0 in range(0, height - CROP_H + 1, STRIDE):
        for c0 in range(0, width - CROP_W + 1, STRIDE):
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
        raise SystemExit("no candidate window met the criteria on " + field)
    deviation, r0, c0, local, n_ref = best
    print("[extract] %s crop at row %d col %d (%dx%d): local F1 %s, "
          "total deviation %.4f over %d reference filaments"
          % (field, r0, c0, CROP_W, CROP_H,
             ", ".join("%s %.4f" % kv for kv in sorted(local.items())),
             deviation, n_ref))

    sl = (slice(r0, r0 + CROP_H), slice(c0, c0 + CROP_W))
    arrays = {
        "sem": sem[sl].astype(np.uint8),
        #  The input axes themselves, as the binary masks PLECTA was handed.
        #  The axis figure shows these rather than any reconstruction: what
        #  separates the two conditions is the mask, and this is the mask.
        "mask_manual": per_axis["manual"]["mask"][sl].astype(bool),
        "mask_unet": per_axis["unet"]["mask"][sl].astype(bool),
    }
    for name, pack in (("reference", ref_centre_pack),
                       ("reference_width", ref_width_pack),
                       ("manual", per_axis["manual"]["centre_pack"]),
                       ("unet", per_axis["unet"]["centre_pack"]),
                       ("manual_width", per_axis["manual"]["width_pack"]),
                       ("unet_width", per_axis["unet"]["width_pack"])):
        cropped = crop_pack(pack, r0, c0)
        for part, array in cropped.items():
            arrays[f"{name}__{part}"] = array
    meta = {
        "crop_row": r0, "crop_col": c0,
        "crop_h": CROP_H, "crop_w": CROP_W,
        "field_f1": {k: per_axis[k]["field_f1"] for k in per_axis},
        "local_f1": local,
        "n_reference_in_crop": int(n_ref),
        "width_render": {k: per_axis[k]["width_record"] for k in per_axis},
    }
    return arrays, meta


def main() -> int:
    stored = {row["field"] + "|" + row["axis"]: row for row in
              json.loads((STUDY / "results_v2.json").read_text())["rows"]}

    payload: dict = {}
    meta: dict = {
        "fields": list(FIELDS),
        "stride_px": STRIDE, "tolerance_px": TOL,
        "unmatched_label": -1,
        "selection": ("per field, the window whose local pairwise F1 is jointly "
                      "closest to that field's whole-field value on both axes"),
        "held_out": {f: True for f in FIELDS},
        "per_field": {},
    }
    for field in FIELDS:
        arrays, field_meta = extract_field(field, stored)
        for name, array in arrays.items():
            payload[f"{field}__{name}"] = array
        meta["per_field"][field] = field_meta

    #  Stored as a plain string so the asset loads without allow_pickle.
    payload["meta"] = np.array(json.dumps(meta))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    print("[extract] wrote %s (%.0f kB)" % (OUT, OUT.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
