"""Three annotated CNT SEM fields, two mask sources, five measures.

Supersedes ``make_real_masks_record.py``, which covered two fields under the
pairwise score alone.  Three things changed and all three are in this record:

  * **a third annotated field exists, B58_300, and it is genuinely held out.**
    Its image is absent from the ds202 training set at a tile correlation of
    0.15, verified by image content rather than by filename, so the study now
    has two clean U-Net conditions where it had one.
  * **B58_100's annotation was revised**, 280 to 307 reference filaments, so
    every number for that field moves slightly.  Its manual-derived pairwise F1
    goes 0.882 to 0.887, which is the useful part: the published value was not
    resting on annotation quirks.
  * **the measures reported are now four**, not one -- pairwise F1 with its
    precision and recall, join F1 with the number of decisions the mask posed,
    detection F1, and Crossing Fidelity -- because no one of them answers the
    question the section asks on its own.  See Section 2.6.

The chance-corrected resolution index the source study also computes is *not*
carried.  Over these rows it correlates with the exact rate at Pearson 0.997,
so on this material it is a linear restatement of a number already reported;
its one distinguishing property, that it can go negative and so separate
"declined to join" from "joined wrongly", never fired, the minimum observed
being 0.404.  A bespoke column that would need defending and carries almost no
information is not worth a page-limited manuscript's space.

Crossing Fidelity is quoted with a chance level, and it is **not** the 0.709
the manuscript used to print.  That number is ``null_pair_accuracy``, the
chance level of pair accuracy -- the statistic just dropped.  The do-nothing
prediction that leaves every arm unpaired resolves 0.011 to 0.102 of these
fields' crossings exactly, measured by
``real_sem_study/scripts/crossing_fidelity_chance.py``.

**B58_100 cannot be made a held-out condition.**  Its image is in the training
data of every nnU-Net run on disk, at a tile correlation of 0.9999 or above.
Its U-Net row is therefore an upper bound and is labelled as one wherever it is
printed.  What that contamination is worth is not left as a caveat: the same
three models were run on all three fields, and because ds204 trained on every
field while ds202 and ds203 trained on neither B58_110 nor B58_300, the gap can
be measured directly.  It is about +0.06 pairwise F1, and this record carries
it.

**The predictions inside the annotation folders are ds204 output** and trained
on every field.  Building a scene from them raises the score by that same +0.06
with nothing erroring.  ``NNUNET_PROVENANCE.md`` in the source study records
that trap and two others.

Sources, all in C:/Repos/comparisons/real_sem_study:
  results_v2.json             the three fields under the five measures
  results_axis_ccq_v2.json    the tolerant axis triple, parity-gated on B58_110
  nnunet_study/results.json   three models x three fields

    python scripts/results/make_real_fields_record.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
STUDY = Path("C:/Repos/comparisons/real_sem_study")
SCORES = STUDY / "results_v2.json"
AXIS = STUDY / "results_axis_ccq_v2.json"
MODELS = STUDY / "nnunet_study" / "results.json"
CHANCE = STUDY / "results_crossing_fidelity_chance.json"

#: One placement tolerance for the whole manuscript, and it is 2 px.  The axis
#: triple used to be reported at 3 px and the reconstruction measures at 2, for
#: no reason beyond the order the two studies were written in.  2 px is the
#: smallest value on the plateau of the synthetic sensitivity curve -- the
#: triple reads 0.912, 0.919 and 0.926 at 1, 2 and 3 px and collapses to 0.545
#: at 0 -- so it is the least generous defensible choice, and having one number
#: to explain rather than two is worth more than the third decimal.  The whole
#: sweep is carried below either way, so the sensitivity stays checkable.
AXIS_TOLERANCE_PX = 2

DISPLAY = {"b58_100": "B58_100", "b58_110": "B58_110", "b58_300": "B58_300"}
AXIS_KEY = {"manual-skel": "manual", "nnU-Net": "unet"}

#: The manual-derived condition must reproduce exactly: the model study reads
#: the same skeleton file, so anything at all there is a broken path.
MANUAL_PARITY_TOL = 1e-9

#: The U-Net condition may not, and the reason was measured rather than
#: assumed.  The two studies each register the same nnU-Net prediction into the
#: annotation grid independently, and the two registrations disagree on 51 to
#: 113 pixels of roughly 190,000 -- an intersection over union of 0.9994 or
#: better.  A handful of pixels can join or break a skeleton branch, so a
#: measure computed on that skeleton moves by a few thousandths.  This
#: tolerance is loose enough to pass that and far too tight to pass the wrong
#: prediction file, which is the failure the gate exists for: the ds204 file
#: shipped inside the annotation folders differs by about +0.06 pairwise F1.
UNET_PARITY_TOL = 1e-2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(row: dict, chance: float) -> dict:
    out = {key: row[key] for key in (
        "pairwise_f1", "pairwise_precision", "pairwise_recall",
        "join_f1", "join_precision", "join_recall", "n_decisions",
        "detection_f1", "n_crossings")}
    out["crossing_fidelity"] = row["crossings_exact"]
    out["crossing_fidelity_chance"] = chance
    return out


def contamination(models: dict) -> dict:
    """What training on the field is worth, where a clean comparison exists."""
    by_field = {}
    for row in models["rows"]:
        if row["model"] == "manual-skel":
            continue
        by_field.setdefault(row["field"], []).append(row)

    fields = []
    for field, rows in by_field.items():
        clean = [r for r in rows if r["trained_on_this_field"] is False]
        dirty = [r for r in rows if r["trained_on_this_field"] is True]
        entry = {
            "field": DISPLAY[field],
            "held_out_models": sorted(r["model"] for r in clean),
            "trained_on_models": sorted(r["model"] for r in dirty),
            "held_out_pairwise_f1": [r["pairwise_f1"] for r in clean],
            "trained_on_pairwise_f1": [r["pairwise_f1"] for r in dirty],
            "held_out_mask_foreground_px": [r["mask_foreground_px"]
                                            for r in clean],
            "trained_on_mask_foreground_px": [r["mask_foreground_px"]
                                              for r in dirty],
        }
        if clean and dirty:
            mean_clean = sum(entry["held_out_pairwise_f1"]) / len(clean)
            mean_dirty = sum(entry["trained_on_pairwise_f1"]) / len(dirty)
            entry["delta_pairwise_f1"] = mean_dirty - mean_clean
        fields.append(entry)

    gaps = [f["delta_pairwise_f1"] for f in fields if "delta_pairwise_f1" in f]
    return {
        "what": ("three nnU-Net models on three fields, PLECTA scored "
                 "identically throughout; the same field is scored under a "
                 "model that trained on it and under models that did not"),
        "control": ("on B58_100 every model trained on the field, the three "
                    "agree to within 0.008 pairwise F1, and there is no gap to "
                    "see; the effect appears only where a clean comparison "
                    "exists"),
        "mechanism": ("the trained-on model predicts far more foreground, "
                      "which is what raises the score"),
        "delta_pairwise_f1_min": min(gaps) if gaps else None,
        "delta_pairwise_f1_max": max(gaps) if gaps else None,
        "delta_pairwise_f1_mean": sum(gaps) / len(gaps) if gaps else None,
        "per_field": fields,
    }


def main() -> int:
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    axis = json.loads(AXIS.read_text(encoding="utf-8"))
    models = json.loads(MODELS.read_text(encoding="utf-8"))
    chance = {(r["field"], r["axis"]): r["null_exact_rate"]
              for r in json.loads(CHANCE.read_text(encoding="utf-8"))["rows"]}

    #: The two records must describe the same predictions.  The model study
    #: re-runs everything from scratch, so its manual-skel and ds202 rows are
    #: an independent path to the same numbers and this is a real check.
    failures, worst = [], {"manual": 0.0, "unet": 0.0}
    by_key = {(r["field"], r["model"]): r for r in models["rows"]}
    for row in scores["rows"]:
        manual = row["axis"] == "manual-skel"
        model = "manual-skel" if manual else "ds202"
        limit = MANUAL_PARITY_TOL if manual else UNET_PARITY_TOL
        other = by_key.get((row["field"], model))
        if other is None:
            failures.append("%s %s missing from the model study"
                            % (row["field"], model))
            continue
        for key in ("pairwise_f1", "join_f1", "detection_f1",
                    "crossings_exact"):
            deviation = abs(other[key] - row[key])
            side = "manual" if manual else "unet"
            worst[side] = max(worst[side], deviation)
            if deviation > limit:
                failures.append("%s %s %s: %.6f against %.6f (limit %g)"
                                % (row["field"], model, key, row[key],
                                   other[key], limit))
    if failures:
        raise SystemExit("PARITY FAILED, refusing to write:\n  "
                         + "\n  ".join(failures))

    axis_by_field = {im["field"]: im for im in axis["images"]}
    fields = []
    for name in ("b58_100", "b58_110", "b58_300"):
        rows = [r for r in scores["rows"] if r["field"] == name]
        triple = axis_by_field[name]
        reported = next(t for t in triple["by_tolerance"]
                        if t["tol_px"] == AXIS_TOLERANCE_PX)
        held_out = next(r["held_out"] for r in rows if r["axis"] == "nnU-Net")
        fields.append({
            "image": DISPLAY[name],
            "n_reference": rows[0]["n_reference"],
            "unet_held_out": bool(held_out),
            "axis_triple": {
                "reported_tolerance_px": AXIS_TOLERANCE_PX,
                "completeness": reported["completeness"],
                "correctness": reported["correctness"],
                "quality": reported["quality"],
                "by_tolerance": triple["by_tolerance"],
            },
            "conditions": {
                AXIS_KEY[r["axis"]]: measures(
                    r, chance[(name, "skel" if r["axis"] == "manual-skel"
                               else "nnunet")])
                for r in rows},
        })

    payload = {
        "role": ("PLECTA on three manually annotated CNT SEM fields, from the "
                 "manual-derived axis and from the upstream nnU-Net axis, "
                 "under the four measures of Section 2.6"),
        "crossing_fidelity": ("the fraction of reference crossings whose "
                              "incident arms the prediction partitions exactly "
                              "as the reference does. Its chance level is the "
                              "do-nothing prediction that leaves every arm "
                              "unpaired, measured per field per axis; it is "
                              "not the 0.709 the manuscript used to print, "
                              "which is the chance level of pair accuracy"),
        "discipline": scores["discipline"],
        "tolerance_px": scores["tolerance_px"],
        "held_out": ("B58_110 and B58_300 are absent from the ds202 training "
                     "set, verified by image content at tile correlations of "
                     "0.10 and 0.15; B58_100's image is present in every "
                     "available run at 0.9999 or above and cannot be made a "
                     "held-out condition, so its U-Net row is an upper bound"),
        "annotation_revision": ("B58_100's annotation was revised from 280 to "
                                "307 reference filaments after the two-field "
                                "study; its manual-derived pairwise F1 moves "
                                "from 0.882 to 0.887"),
        "axis_triple": ("tolerant completeness / correctness / quality of the "
                        "nnU-Net axis against the manual-derived skeleton; it "
                        "characterises the mask PLECTA is handed and opens no "
                        "reconstruction"),
        "parity": {
            "what": ("every reconstruction measure is reproduced "
                     "independently by the three-model study, which rebuilds "
                     "and rescores from scratch; and the axis triple "
                     "reproduces the published two-field values for B58_110, "
                     "whose masks are byte-identical to the published build"),
            "manual_axis_tolerance": MANUAL_PARITY_TOL,
            "manual_axis_max_deviation": worst["manual"],
            "unet_axis_tolerance": UNET_PARITY_TOL,
            "unet_axis_max_deviation": worst["unet"],
            "why_the_unet_axis_is_not_exact": (
                "the two studies register the same nnU-Net prediction into "
                "the annotation grid independently, and the two registrations "
                "disagree on 51 to 113 pixels of about 190,000, an "
                "intersection over union of 0.9994 or better; a handful of "
                "pixels can join or break a skeleton branch, so a measure "
                "computed on that skeleton moves by a few thousandths"),
        },
        "source_record": {
            "study": str(STUDY).replace("\\", "/"),
            "scores": {"file": "results_v2.json", "sha256": sha256(SCORES)},
            "axis": {"file": "results_axis_ccq_v2.json",
                     "sha256": sha256(AXIS)},
            "models": {"file": "nnunet_study/results.json",
                       "sha256": sha256(MODELS)},
            "chance": {"file": "results_crossing_fidelity_chance.json",
                       "sha256": sha256(CHANCE)},
            "provenance": "NNUNET_PROVENANCE.md",
        },
        "fields": fields,
        "contamination": contamination(models),
    }

    destination = RESULTS / "plecta_real_fields.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print("wrote", destination)
    print("\n%-9s %-8s %5s %8s %8s %9s %8s %8s %8s"
          % ("field", "axis", "held", "pairwise", "join F1", "decisions",
             "det F1", "CF", "CF chance"))
    for field in fields:
        for key, condition in field["conditions"].items():
            print("%-9s %-8s %5s %8.3f %8.3f %9d %8.3f %8.3f %8.3f"
                  % (field["image"], key,
                     "--" if key == "manual" else field["unet_held_out"],
                     condition["pairwise_f1"], condition["join_f1"],
                     condition["n_decisions"], condition["detection_f1"],
                     condition["crossing_fidelity"],
                     condition["crossing_fidelity_chance"]))
    c = payload["contamination"]
    print("\ntraining contamination is worth +%.3f to +%.3f pairwise F1"
          % (c["delta_pairwise_f1_min"], c["delta_pairwise_f1_max"]))
    for entry in c["per_field"]:
        if "delta_pairwise_f1" in entry:
            print("   %-9s held out %s -> trained on %s : +%.3f"
                  % (entry["field"],
                     ["%.3f" % v for v in entry["held_out_pairwise_f1"]],
                     ["%.3f" % v for v in entry["trained_on_pairwise_f1"]],
                     entry["delta_pairwise_f1"]))
    print("\nparity: the three-model study reproduces every measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
