"""Mask quality of the two real axes, as the thin-structure literature states it.

Section~3.7 already reports two numbers for each upstream axis: an axis recall
and an axis precision, both measured centreline-to-centreline with a symmetric
3-px tolerance, so that neither mask is rewarded for being thicker.  Those two
numbers are, under their usual names, the *completeness* and *correctness* of
the standard tolerant thin-structure triple.  What the manuscript never carried
is the third member, *quality*, the tolerant Jaccard -- which is also GraFT's
published Jaccard index in its intended reading, and therefore the one number
that lets a reader from that literature place these masks.

So this record adds the missing component rather than replacing anything, and
it does two things that make the addition safe to quote:

  * **parity.**  Completeness and correctness are recomputed here from the
    stored masks and checked against the values the manuscript already prints.
    A conversion that silently disagreed with the study would otherwise produce
    a plausible-looking third number attached to two wrong ones.
  * **a tolerance sweep.**  The triple is computed at 0, 1, 2 and 3 px, so the
    tolerance cannot be chosen after seeing the answer.  The sweep also
    contradicts what was expected of it, which is the reason it is kept in the
    record rather than collapsed to one row.  On the *synthetic* degraded mask
    the measure is flat above 1 px, so the tolerance looks like a formality.
    On the two *real* fields it is not flat at all -- quality climbs from 0.285
    to 0.622 on B58_100 between 1 and 3 px -- because there the two axes are
    two independent readings of the same filaments rather than one derived from
    the other, and they disagree in placement by a pixel or two nearly
    everywhere.  Both sweeps are stored, and the manuscript reports the real
    triple at 3 px, which is the tolerance its existing two components were
    measured at and also GraFT's own.

The comparison is method-independent by construction.  No reconstruction is
opened; this characterises the input PLECTA is handed, which is precisely the
distinction Section~3.7 has to keep.

Definitions follow ``metrics_study/scripts/metrics_lib.py::ccq``:

    completeness = |reference axis within tol of the test axis| / |reference|
    correctness  = |test axis within tol of the reference axis| / |test|
    quality      = TP / (TP + FP + FN)

Usage:
  python scripts/results/make_mask_quality_record.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
STUDY = Path("C:/Repos/comparisons/real_sem_study")
SCENES = STUDY / "scenes"

#: The same triple on the synthetic degraded mask, 50 scenes, already computed.
#: It is carried alongside because it is the calibration that makes the real
#: numbers readable: it is what the measure looks like when the test axis is
#: *derived from* the reference rather than being a second reading of the same
#: filaments.
SYNTH_STUDY = Path("C:/Repos/comparisons/metrics_study")
SYNTH_SUMMARY = SYNTH_STUDY / "results" / "summary.json"

#: manuscript name -> (field directory, manual-axis variant, U-Net variant)
FIELDS = {
    "B58_100": ("b58_100", "skel", "nnunet"),
    "B58_110": ("b58_110", "skel", "nnunet"),
}

TOLERANCES = (0, 1, 2, 3)

#: The tolerance the manuscript reports the real triple at: the one its
#: existing axis recall and precision were already measured at, which is also
#: GraFT's own.  Reporting the third component at a different tolerance from
#: the two beside it would be indefensible.
REPORTED_TOL = 3

#: The tolerance the two existing numbers were measured at, which is what the
#: parity check has to reproduce.
STUDY_TOL = 3
PARITY_TOL = 5e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis(path: Path) -> np.ndarray:
    """A mask as its one-pixel centreline."""
    return skeletonize(np.array(Image.open(path).convert("L")) > 127)


def near(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return mask
    return binary_dilation(mask, structure=np.ones((2 * tol + 1,) * 2, bool))


def ccq(ref: np.ndarray, test: np.ndarray, tol: int) -> dict:
    ref_near, test_near = near(ref, tol), near(test, tol)
    tp_ref = int((ref & test_near).sum())
    fn = int((ref & ~test_near).sum())
    tp_test = int((test & ref_near).sum())
    fp = int((test & ~ref_near).sum())
    n_ref, n_test = int(ref.sum()), int(test.sum())
    #  TP is ambiguous when the two directions disagree; the smaller is the
    #  conservative choice, and it is what keeps quality below both components.
    tp = min(tp_ref, tp_test)
    denom = tp + fp + fn
    return {
        "tol_px": tol,
        "completeness": tp_ref / n_ref if n_ref else float("nan"),
        "correctness": tp_test / n_test if n_test else float("nan"),
        "quality": tp / denom if denom else float("nan"),
        "n_reference_px": n_ref,
        "n_test_px": n_test,
    }


def main() -> None:
    stored = json.loads((RESULTS / "plecta_real_masks.json")
                        .read_text(encoding="utf-8"))
    stored_by_image = {row["image"]: row for row in stored["images"]}

    images, failures = [], []
    for name, (folder, manual, unet) in FIELDS.items():
        manual_png = SCENES / folder / manual / "mask.png"
        unet_png = SCENES / folder / unet / "mask.png"
        ref, test = axis(manual_png), axis(unet_png)
        sweep = [ccq(ref, test, tol) for tol in TOLERANCES]
        at_study = next(r for r in sweep if r["tol_px"] == STUDY_TOL)

        # Parity: the two components the manuscript already prints have to come
        # back out of this code path before the third is believed.
        before = stored_by_image[name]
        for key, computed in (("axis_recall", at_study["completeness"]),
                              ("axis_precision", at_study["correctness"])):
            delta = abs(float(before[key]) - computed)
            if delta > PARITY_TOL:
                failures.append("%s %s: stored %.4f, recomputed %.4f (%+.4f)"
                                % (name, key, before[key], computed, -delta))

        images.append({
            "image": name,
            "reference_axis": "manual-derived skeleton",
            "test_axis": "upstream U-Net",
            "masks": {
                "reference": {"file": str(manual_png.relative_to(STUDY))
                              .replace("\\", "/"), "sha256": sha256(manual_png)},
                "test": {"file": str(unet_png.relative_to(STUDY))
                         .replace("\\", "/"), "sha256": sha256(unet_png)},
            },
            "by_tolerance": sweep,
            "reported": next(r for r in sweep
                             if r["tol_px"] == REPORTED_TOL),
            "parity_against_manuscript": {
                "tol_px": STUDY_TOL,
                "axis_recall_stored": float(before["axis_recall"]),
                "axis_recall_recomputed": at_study["completeness"],
                "axis_precision_stored": float(before["axis_precision"]),
                "axis_precision_recomputed": at_study["correctness"],
            },
        })

    synth = json.loads(SYNTH_SUMMARY.read_text(encoding="utf-8"))
    block = synth["mask_quality"]["degraded"]["overall"]
    synthetic = {
        "role": ("the synthetic degraded axis against the generator's clean "
                 "axis; the test axis is derived from the reference"),
        "n_scenes": int(synth["mask_quality"]["degraded"]["n_scenes"]),
        "by_tolerance": [
            {"tol_px": tol,
             "completeness": block["ccq%d_completeness" % tol],
             "correctness": block["ccq%d_correctness" % tol],
             "quality": block["ccq%d_quality" % tol]}
            for tol in TOLERANCES],
        "graft_jaccard_as_published": block["graft_JI_vendor"],
        "graft_jaccard_note": ("GraFT's own JI, computed against the "
                               "unthinned prediction, so it charges the mask "
                               "for its thickness"),
        "source_record": {
            "study": str(SYNTH_STUDY).replace("\\", "/"),
            "file": str(SYNTH_SUMMARY.relative_to(SYNTH_STUDY))
                    .replace("\\", "/"),
            "sha256": sha256(SYNTH_SUMMARY),
            "path_in_file": "mask_quality/degraded/overall",
        },
    }

    payload = {
        "role": ("tolerant completeness / correctness / quality of the two "
                 "upstream real axes; characterises the input, not the "
                 "reconstruction"),
        "synthetic_reference": synthetic,
        "definition": ("both sides skeletonised, each tested against the other "
                       "dilated by tol; completeness is tolerant recall of the "
                       "manual axis, correctness tolerant precision of the "
                       "U-Net axis, quality the tolerant Jaccard"),
        "reported_tolerance_px": REPORTED_TOL,
        "source_record": {
            "study": str(STUDY).replace("\\", "/"),
            "metric_definition": ("C:/Repos/comparisons/metrics_study/scripts/"
                                  "metrics_lib.py::ccq"),
        },
        "images": images,
    }

    destination = RESULTS / "plecta_mask_quality.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print("wrote", destination)
    for image in images:
        rows = " | ".join(
            "tol %d: %.3f/%.3f/%.3f" % (r["tol_px"], r["completeness"],
                                        r["correctness"], r["quality"])
            for r in image["by_tolerance"])
        print(" ", image["image"], rows)
    if failures:
        raise SystemExit("PARITY FAILED:\n  " + "\n  ".join(failures))
    print("parity: completeness and correctness at %d px reproduce the stored "
          "axis recall and precision for both fields" % STUDY_TOL)


if __name__ == "__main__":
    main()
