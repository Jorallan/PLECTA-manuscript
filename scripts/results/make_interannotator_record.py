"""Collect the ALLAN/ZAGH two-reader study into one machine-readable record.

Source: ``comparisons/real_sem_study/interannotator/allan_vs_zagh_2026-08-14``,
run under its own frozen PROTOCOL.md. Three of the manuscript's annotated
fields were independently re-annotated by a second reader; a fourth field
(S3_010) carries only a ZAGH annotation and contributes no agreement
observation, so it is excluded here and the exclusion is recorded.

What the record is for: the manuscript's evidence boundary previously said no
inter-reader estimate existed. One does now, and it does not flatter the
method -- cross-reader PLECTA scores fall to roughly the level at which the
two readers agree with each other. That is a ceiling, not a regression, and
the record carries both so the prose cannot quote one without the other.

Nothing here is inferential: three paired fields, plain ranges, no interval.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

STUDY = Path(r"C:\Repos\comparisons\real_sem_study\interannotator"
             r"\allan_vs_zagh_2026-08-14")
TABLES = STUDY / "tables"
OUT = Path(__file__).resolve().parents[2] / "results" / "plecta_interannotator.json"

# S3_010 has no ALLAN export; it is a ZAGH-vs-network field only.
PAIRED = ("b58_s2_100", "b58_s2_110", "b58_s3_100")


def _rows(name: str) -> list[dict]:
    with (TABLES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    scored = _rows("plecta_vs_references.csv")
    partition = _rows("direct_annotation_partition.csv")
    detection = _rows("direct_instance_detection.csv")
    sources = _rows("source_summary.csv")

    by_key = {(r["field"], r["input"], r["reference"]): r for r in scored}
    counts = {(r["field"], r["label"]): r for r in sources
              if r["label"] in ("ALLAN", "ZAGH")}

    fields = []
    for f in PAIRED:
        allan, zagh = counts[(f, "ALLAN")], counts[(f, "ZAGH")]
        part = {r["reference"]: r for r in partition if r["field"] == f}
        det = next(r for r in detection if r["field"] == f)
        fields.append({
            "field": f,
            "display": allan["field_display"],
            "n_instances": {"ALLAN": int(allan["n_instances"]),
                            "ZAGH": int(zagh["n_instances"])},
            "object_ratio_zagh_over_allan":
                int(zagh["n_instances"]) / int(allan["n_instances"]),
            "axis_components_8": {
                "ALLAN": int(allan["axis_union_components_8"]),
                "ZAGH": int(zagh["axis_union_components_8"])},
            "component_ratio_zagh_over_allan":
                int(zagh["axis_union_components_8"])
                / int(allan["axis_union_components_8"]),
            # PLECTA, per input skeleton, against each reader.
            "plecta": {
                "allan_skel_vs_allan": float(by_key[(f, "ALLAN_SKEL", "ALLAN")]["f1"]),
                "allan_skel_vs_zagh": float(by_key[(f, "ALLAN_SKEL", "ZAGH")]["f1"]),
                "zagh_skel_vs_zagh": float(by_key[(f, "ZAGH_SKEL", "ZAGH")]["f1"]),
                "zagh_skel_vs_allan": float(by_key[(f, "ZAGH_SKEL", "ALLAN")]["f1"]),
                "nnunet_vs_allan": float(by_key[(f, "NNUNET_MASK", "ALLAN")]["f1"]),
                "nnunet_vs_zagh": float(by_key[(f, "NNUNET_MASK", "ZAGH")]["f1"]),
            },
            # Reader against reader, with no method in the loop at all.
            "reader_agreement": {
                "partition_f1_allan_exam": float(part["ALLAN"]["f1"]),
                "partition_f1_zagh_exam": float(part["ZAGH"]["f1"]),
                "detection_f1": float(det["detection_f1"]),
                "detection_recall_of_allan": float(det["detection_recall"]),
                "detection_precision_wrt_zagh": float(det["detection_precision"]),
            },
        })

    def span(pick):
        vals = [pick(f) for f in fields]
        return {"min": min(vals), "max": max(vals)}

    payload = {
        "role": "PLECTA and two independent human readers on three annotated "
                "CNT SEM fields, scored through eval/core/common_metric.py",
        "discipline": "exploratory, 3 paired fields, plain ranges, no inference",
        "study": str(STUDY),
        "excluded": {
            "b58_s3_010": "ZAGH-only; no ALLAN export exists, so it yields no "
                          "inter-reader observation and is not counted here",
        },
        "ceiling": "cross-reader PLECTA F1 must be read against the direct "
                   "reader-versus-reader agreement in reader_agreement, not "
                   "against the same-reader column; the two readers do not "
                   "implement one instance definition",
        "n_paired_fields": len(fields),
        "spans": {
            "same_reader_f1": span(lambda f: min(f["plecta"]["allan_skel_vs_allan"],
                                                 f["plecta"]["zagh_skel_vs_zagh"])),
            "same_reader_f1_high": span(lambda f: max(f["plecta"]["allan_skel_vs_allan"],
                                                      f["plecta"]["zagh_skel_vs_zagh"])),
            "cross_reader_f1": span(lambda f: min(f["plecta"]["allan_skel_vs_zagh"],
                                                  f["plecta"]["zagh_skel_vs_allan"])),
            "cross_reader_f1_high": span(lambda f: max(f["plecta"]["allan_skel_vs_zagh"],
                                                       f["plecta"]["zagh_skel_vs_allan"])),
            "reader_partition_f1": span(
                lambda f: min(f["reader_agreement"]["partition_f1_allan_exam"],
                              f["reader_agreement"]["partition_f1_zagh_exam"])),
            "reader_partition_f1_high": span(
                lambda f: max(f["reader_agreement"]["partition_f1_allan_exam"],
                              f["reader_agreement"]["partition_f1_zagh_exam"])),
            "object_ratio": span(lambda f: f["object_ratio_zagh_over_allan"]),
            "component_ratio": span(lambda f: f["component_ratio_zagh_over_allan"]),
            "nnunet_f1": span(lambda f: min(f["plecta"]["nnunet_vs_allan"],
                                            f["plecta"]["nnunet_vs_zagh"])),
            "nnunet_f1_high": span(lambda f: max(f["plecta"]["nnunet_vs_allan"],
                                                 f["plecta"]["nnunet_vs_zagh"])),
        },
        "fields": fields,
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
