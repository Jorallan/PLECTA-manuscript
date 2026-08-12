"""Two gentler readings of the two real SEM fields, and what they are worth.

Section 3.7 reports the real fields under the common-fragment pairwise score,
and the distance between the manual-derived and the U-Net condition is large:
0.882 against 0.438 on B58_100. That distance is the paper's statement about
upstream mask quality, and two objections can be raised against measuring it
that way. This record answers both with measurements rather than argument, and
neither answer flatters the method.

**Objection 1: a strict object IoU is unusable across two mask sources.**
DNAi's detection F1 matches predicted objects to reference objects by pixel
overlap. On the manual-derived axis the input mask is a skeleton *of the
annotation*, so a predicted centreline lies on the reference centreline pixel
for pixel and the measure reads 0.94. On the U-Net axis the mask is an
independent reading of the same field, which Section 3.7 already measures as
disagreeing in placement by a pixel or two nearly everywhere, and two one-pixel
centrelines two pixels apart intersect in nothing. The strict measure reads
0.14, and at an IoU threshold of 0.5 it reads exactly zero. Recomputed with the
3-px placement tolerance the section already uses for the axis triple, the same
predictions read 0.67 and 0.62. The control that makes this a correction rather
than an inflation is that the manual-derived figures do not move at all: there
was no misplacement there to forgive.

**Objection 2: pairwise F1 charges the method for the mask's fragmentation.**
It counts co-clustered *pairs*, so a filament the mask breaks into n pieces
carries C(n,2) of them and one missed join is charged many times over. The join
score counts each decision once instead -- for every pair of fragment ends the
mask brings within PLECTA's own gap_max_len, did the method join them, and
should it have. It is conditioned on the mask, so the two conditions are asked
comparably hard questions, and it reports how many decisions each mask posed.
It closes about a third of the distance and no more: 0.882 -> 0.438 is a factor
of 2.0, 0.925 -> 0.536 a factor of 1.7. The rest of the collapse is real.

The join decomposition is the part worth reporting on its own. On the U-Net
axis PLECTA runs at a join precision near 0.68 against a join recall near 0.44
to 0.53 -- it is not fusing filaments that the mask separated, it is declining
joins the mask made available. The failure under a worse mask is conservatism.

Source study: C:/Repos/comparisons/graft_regime/scripts/real_detection_f1.py,
which re-runs PLECTA from the frozen package rather than reading a stored file,
so its pairwise score has to reproduce the four values the manuscript already
prints before any other column it emits means anything. It also checks its join
score against the study that defined that metric
(C:/Repos/comparisons/plecta_vs_strandtrace), and carries a self-test proving
the tolerance forgives a two-pixel displacement and still refuses a ten-pixel
one.

Usage:
  python scripts/results/make_real_field_metrics_record.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
STUDY = Path("C:/Repos/comparisons/graft_regime")
SOURCE = STUDY / "results" / "real_detection_f1.json"
REAL_MASKS = RESULTS / "plecta_real_masks.json"

#: The manuscript's own axis tolerance. Reporting a detection score at a
#: tolerance the paper does not already use elsewhere would be choosing a
#: number after seeing the answer.
REPORTED_TOL = "3"
STRICT_TOL = "0"
IOU = "0.1"
PARITY_TOL = 5e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    study = json.loads(SOURCE.read_text(encoding="utf-8"))
    printed = {r["image"]: r for r in
               json.loads(REAL_MASKS.read_text(encoding="utf-8"))["images"]}

    key_of = {"manual-derived": "plecta_manual_axis", "U-Net": "plecta_unet_axis"}
    conditions, failures = [], []
    for row in study["rows"]:
        was = float(printed[row["field"]][key_of[row["input_axis"]]]["f1"])
        if abs(was - float(row["pairwise_f1"])) > PARITY_TOL:
            failures.append("%s %s: manuscript %.4f, study %.4f"
                            % (row["field"], row["input_axis"], was,
                               row["pairwise_f1"]))
        det = row["detection_by_tolerance"]
        join = row["join"]
        conditions.append({
            "image": row["field"],
            "input_axis": row["input_axis"],
            "pairwise_f1": row["pairwise_f1"],
            "detection_f1_strict": det[STRICT_TOL][IOU]["detection_f1"],
            "detection_f1_tolerant": det[REPORTED_TOL][IOU]["detection_f1"],
            "detection_f1_strict_iou_half": det[STRICT_TOL]["0.5"]["detection_f1"],
            "detection_f1_tolerant_iou_half": det[REPORTED_TOL]["0.5"]["detection_f1"],
            "detection_by_tolerance": {
                tol: det[tol][IOU]["detection_f1"] for tol in det},
            "join_f1": join["join_f1"],
            "join_precision": join["join_precision"],
            "join_recall": join["join_recall"],
            "n_decisions": join["n_decisions"],
            "n_true_joins": join["n_true_joins"],
            "unassigned_fraction_at_default":
                row["pairwise_by_assignment_distance"]["6"]["unassigned_fraction"],
            "unassigned_fraction_at_zero":
                row["pairwise_by_assignment_distance"]["0"]["unassigned_fraction"],
        })

    if failures:
        raise SystemExit("PARITY FAILED, refusing to write:\n  "
                         + "\n  ".join(failures))

    payload = {
        "role": ("two gentler readings of the two real fields of Section 3.7, "
                 "on the identical predictions the pairwise score is computed "
                 "from; neither replaces it"),
        "parity": ("the study re-runs PLECTA from the frozen package and its "
                   "pairwise score reproduces all four values "
                   "results/plecta_real_masks.json already prints, within "
                   "%g" % PARITY_TOL),
        "detection_metric": ("DNAi's greedy-IoU object detection F1, at its "
                             "shipped IoU threshold of %s. The tolerance is "
                             "the symmetric placement tolerance of the "
                             "completeness/correctness/quality triple: a "
                             "reference pixel counts as found when the "
                             "prediction passes within it. At 0 px it is the "
                             "vendor's strict overlap exactly." % IOU),
        "join_metric": ("one count per decision the mask poses -- a pair of "
                        "fragment ends within PLECTA's own gap_max_len of "
                        "85 px -- instead of one per pair of fragments, so a "
                        "filament the mask breaks into n pieces poses n-1 "
                        "decisions rather than C(n,2) pairs"),
        "why_the_tolerance_is_not_a_free_gain": (
            "on the manual-derived axis, where the input mask is a skeleton of "
            "the annotation itself and nothing is displaced, the tolerance "
            "moves the detection score by 0.000; the whole of its effect is on "
            "the U-Net axis, where two independent readings of the same field "
            "are being compared"),
        "what_it_does_not_show": (
            "neither reading closes the gap between the two mask sources. The "
            "join score narrows it from a factor of 2.0 to a factor of 1.7 and "
            "the tolerant detection score leaves 0.62 to 0.67 against 0.93 to "
            "0.94. Both conditions remain exploratory development cases, n=2 "
            "images, no inference"),
        "reported_tolerance_px": int(REPORTED_TOL),
        "iou_threshold": float(IOU),
        "join_reach_px": 85.0,
        "source_record": {
            "study": str(STUDY).replace("\\", "/"),
            "script": "scripts/real_detection_f1.py",
            "json": {"file": "results/real_detection_f1.json",
                     "sha256": sha256(SOURCE)},
        },
        "conditions": conditions,
    }

    destination = RESULTS / "plecta_real_field_metrics.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print("wrote", destination)
    print("\n%-9s %-15s %9s %9s %9s %9s %9s %10s"
          % ("field", "input axis", "pairwise", "det(0px)", "det(3px)",
             "join F1", "join P", "decisions"))
    for c in conditions:
        print("%-9s %-15s %9.3f %9.3f %9.3f %9.3f %9.3f %10d"
              % (c["image"], c["input_axis"], c["pairwise_f1"],
                 c["detection_f1_strict"], c["detection_f1_tolerant"],
                 c["join_f1"], c["join_precision"], c["n_decisions"]))
    print("\nparity: all four pairwise values reproduce "
          "results/plecta_real_masks.json.")


if __name__ == "__main__":
    main()
