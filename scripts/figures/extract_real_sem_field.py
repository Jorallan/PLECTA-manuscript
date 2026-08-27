"""Extract the committed asset behind ``fig_real_sem``.

The qualitative real-SEM figure needs, for one field, the manual annotation and
two reconstructions -- PLECTA on the manual-derived axis and PLECTA on the
nnU-Net axis -- with instance colours that correspond across the three, plus the
two input axis masks themselves.  Everything is stored for the **whole field**:
PLECTA is run on the whole field, matched on the whole field, and drawn on the
whole field, so nothing in the picture is a window chosen after the fact.

**Which field.**  B58_110, which the nnU-Net run used here held out of training.
B58_100 is training-contaminated under every available real-trained run and is
deliberately not drawn: a picture of a field the network memorised would show a
quality of mask transfer that does not exist -- see ``NNUNET_PROVENANCE.md`` in
the study, which measures it at correlation 0.9999 against ds202's training
images.  B58_300 is also held out and would serve; B58_110 is the field the text
already names.  That the result is not one lucky field is a numerical claim, and
Table 5 makes it over all three fields, which a second picture states weakly.

**No crop.**  An earlier version drew a 512x344 window per field, chosen as the
one whose local pairwise F1 was jointly closest to the whole-field value on both
axes.  Choosing a window well is still choosing one; showing the frame the
number was computed on removes the question.

The stored per-field numbers are reproduced before anything is written, as the
repository requires of any figure built from a fresh run.

    python scripts/figures/extract_real_sem_field.py
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
import instance_io as IIO                                     # noqa: E402


def _alias_image_layer() -> None:
    """Let the study's ``width_render`` find PLECTA's image layer.

    The method repository renamed that package ``image_characterization`` ->
    ``plecta.image`` after this study was written, so ``width_render`` imports a
    name that no longer resolves.  It is aliased here rather than the comparator
    study edited: the study is a record of what ran, and this repository is not
    where it gets revised.

    One signature moved with the rename and is adapted, not guessed at.
    ``resample_run(pts, aids, tangent_sigma)`` lost its middle argument, which
    existed only to carry a per-pixel arm id through to an ``aid`` key in the
    returned dict.  ``width_render`` passes ``np.zeros(len(path))`` for it and
    reads ``u``, ``valid``, ``r``, ``c``, ``nr`` and ``nc`` back, never ``aid``;
    the rest of the function is unchanged line for line (``git log -S`` in the
    method repository).  So dropping the argument is a no-op for this caller,
    and the wrapper below drops it.  Everything else ``width_render`` calls --
    ``load_sem``, ``measure_cut``, ``render_ribbon``, ``smooth_polyline``,
    ``_order_pixels``, ``degree_map``, ``prune_spurs`` -- takes exactly the
    arguments it passes, and the parameter defaults its own docstring names
    (``cut_step`` 2, ``node_clear_px`` 6, ``half_len`` 22, ``min_width`` 3,
    ``max_width`` 40, ``smooth_window`` 9) are the current ones.
    """
    import importlib
    import types

    if str(L.PLECTA_REPO) not in sys.path:
        sys.path.insert(0, str(L.PLECTA_REPO))

    bundles = importlib.import_module("plecta.image.bundles")
    resample_run = bundles.resample_run

    shim = types.ModuleType("image_characterization.bundles")
    shim.__dict__.update(vars(bundles))
    shim.resample_run = (lambda pts, aids, tangent_sigma:
                         resample_run(pts, tangent_sigma))

    package = types.ModuleType("image_characterization")
    package.__path__ = []          # a package, so ``X.Y`` imports resolve
    sys.modules["image_characterization"] = package
    for name, module in (("measurement",
                          importlib.import_module("plecta.image.measurement")),
                         ("bundles", shim),
                         ("refine", importlib.import_module("plecta.image.refine"))):
        setattr(package, name, module)
        sys.modules["image_characterization." + name] = module


_alias_image_layer()
#  PLECTA's own optional image layer, driven rather than reimplemented: it
#  measures a width per instance by FWHM across the centreline and stamps a
#  ribbon. This is the Section 2.5 rendering, and it runs after the grouping is
#  fixed, so it cannot move a pixel from one instance to another.
import width_render as WR                                     # noqa: E402

#: The one field drawn, and the reason is in the docstring.
FIELD = "b58_110"
AXES = (("manual", "skel"), ("unet", "nnunet"))
TOL = 2                       # the placement tolerance used throughout
OUT = REPO / "results" / "figure_assets" / "real_sem_field.npz"


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
    """Keep every instance separately, as CSR over flat pixel indices.

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
    runs = [np.flatnonzero(np.asarray(masks[k], bool).ravel()).astype(np.int32)
            for k in order]
    indptr = np.zeros(len(runs) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in runs], out=indptr[1:])
    indices = (np.concatenate(runs) if runs
               else np.zeros(0, dtype=np.int32)).astype(np.int32)
    return {"colour_ids": colours, "indptr": indptr, "indices": indices}


def extract_field(field: str, stored: dict) -> tuple[dict, dict]:
    """Run both axes of one field and return its arrays and metadata."""
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

        match = greedy_match(reference.masks, centrelines.masks, mask.shape)

        #  The same instances re-drawn at the width the SEM says each filament
        #  is, on the whole field, like the grouping.
        widths, width_record = WR.render_all(scene, mask, prediction.masks)
        print("[extract]   width: %d/%d instances measured, scene median %s px"
              % (width_record["n_measured"], width_record["n_instances"],
                 width_record["scene_median_width_px"]))

        per_axis[key] = {
            "mask": mask,
            "field_f1": scores["f1"],
            "centre_pack": pack_instances(centrelines.masks, mask.shape, match),
            "width_pack": pack_instances(widths, mask.shape, match),
            "width_record": width_record,
            "n_instances": len(centrelines.masks),
            "n_matched": len(match),
            "reference": reference,
        }
        if sem is None:
            from PIL import Image
            sem = np.array(Image.open(scene / "sem.png").convert("L"))
        print("[extract] %s %-6s field F1 %.4f, %d instances, %d matched"
              % (field, key, scores["f1"], len(centrelines.masks), len(match)))

    #  Two renderings of the annotation, because two panels compare against
    #  different things. Its skeleton is exactly the manual-derived input axis,
    #  and is what a centreline panel is comparable with; its stored form is
    #  already painted at the width the annotator drew, which is what the
    #  width-rendered panels are comparable with. Neither is invented.
    from skimage.morphology import skeletonize
    reference = per_axis["manual"]["reference"]
    shape = per_axis["manual"]["mask"].shape
    ref_centre_pack = pack_instances(
        {i: skeletonize(np.asarray(m, bool)) for i, m in reference.masks.items()},
        shape)
    ref_width_pack = pack_instances(reference.masks, shape)

    arrays = {
        "sem": np.asarray(sem, np.uint8),
        #  The input axes themselves, as the binary masks PLECTA was handed.
        #  Panels (b) and (d) show these rather than any reconstruction: what
        #  separates the two conditions is the mask, and this is the mask.
        "mask_manual": per_axis["manual"]["mask"].astype(bool),
        "mask_unet": per_axis["unet"]["mask"].astype(bool),
    }
    for name, pack in (("reference", ref_centre_pack),
                       ("reference_width", ref_width_pack),
                       ("manual", per_axis["manual"]["centre_pack"]),
                       ("unet", per_axis["unet"]["centre_pack"]),
                       ("manual_width", per_axis["manual"]["width_pack"]),
                       ("unet_width", per_axis["unet"]["width_pack"])):
        for part, array in pack.items():
            arrays[f"{name}__{part}"] = array

    meta = {
        "shape": [int(shape[0]), int(shape[1])],
        "n_reference": len(reference.masks),
        "field_f1": {k: per_axis[k]["field_f1"] for k in per_axis},
        "n_instances": {k: per_axis[k]["n_instances"] for k in per_axis},
        "n_matched": {k: per_axis[k]["n_matched"] for k in per_axis},
        "width_render": {k: per_axis[k]["width_record"] for k in per_axis},
    }
    return arrays, meta


def main() -> int:
    stored = {row["field"] + "|" + row["axis"]: row for row in
              json.loads((STUDY / "results_v2.json").read_text())["rows"]}

    arrays, field_meta = extract_field(FIELD, stored)
    payload = {f"{FIELD}__{name}": array for name, array in arrays.items()}
    meta = {
        "fields": [FIELD],
        "tolerance_px": TOL,
        "unmatched_label": -1,
        "selection": "the whole field, uncropped",
        "held_out": {FIELD: True},
        "held_out_note": ("B58_110 is held out of the nnU-Net run used here "
                          "(ds202); B58_100 is contaminated under every "
                          "available real-trained run and is not drawn"),
        "per_field": {FIELD: field_meta},
    }
    #  Stored as a plain string so the asset loads without allow_pickle.
    payload["meta"] = np.array(json.dumps(meta))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    print("[extract] wrote %s (%.0f kB)" % (OUT, OUT.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
