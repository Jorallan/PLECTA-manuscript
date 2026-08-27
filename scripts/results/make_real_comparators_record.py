"""SIFNE and the Basu Stage-B reimplementation on the three real CNT SEM fields.

Source study: ``C:/Repos/comparisons/real_sem_comparators`` (PROTOCOL.md frozen
2026-08-27 before any field of that study was run or scored).  It runs the two
comparators that read a binary axis on the *identical* manual-derived masks and
the *identical* overlap-aware reference PLECTA's published real-field numbers
were computed on, through the same scoring path, so the three can be read side
by side.  The ``nnunet`` condition was not run and is out of scope there.

Three things this record carries that the study's own file does not:

  * **the parity assertion is re-run here**, not merely quoted.  The study's
    PLECTA rows must reproduce ``plecta_real_fields.json``
    ``conditions.manual`` exactly, or nothing is written.  A comparator number
    is only readable against PLECTA's if the same scorer produced both.
  * **areal coverage of each field**, measured here from the overlap-aware
    reference itself.  It is the union of the full-width annotated strands
    over the frame -- the quantity Section 2.2 defines and the synthetic
    coverage strata label -- and it is *not* the 0.032--0.040 foreground
    fraction of the thin axis mask that the study's own protocol quotes.  Both
    are written, under names that cannot be confused, because a reader who
    took the axis figure for coverage would conclude these fields are ten
    times sparser than they are.
  * **the caveats, verbatim from the study**, so that anything quoting this
    record quotes them too: both comparators run at parameters selected on
    512x512 *synthetic* development scenes and applied unchanged to
    1024x1536 real fields, and Basu Stage B is our reimplementation and a
    floor, never an external comparator result.

    python scripts/results/make_real_comparators_record.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DEFAULT_COMPARISONS_ROOT = ROOT.parent / "comparisons"

#: The study reports the manual-derived (``skel``) axis only.
AXIS = "manual-skel"
FIELDS = {"b58_100": "B58_100", "b58_110": "B58_110", "b58_300": "B58_300"}
METHODS = {
    "PLECTA": "plecta",
    "SIFNE": "sifne",
    "Basu Stage B (reimplementation)": "basu",
}
#: Every quantity the table or a macro may read, so a missing key fails here
#: rather than as a KeyError three files away.
MEASURES = ("pairwise_f1", "pairwise_precision", "pairwise_recall",
            "join_f1", "join_precision", "join_recall", "n_decisions",
            "detection_f1", "crossing_fidelity", "n_crossings")
#: PLECTA is quoted from the published record, not re-run, so it carries none
#: of these; the two comparators carry all of them.
COMPARATOR_ONLY = ("adjusted_rand_index", "vi_split_bits", "vi_merge_bits",
                   "vi_total_bits", "fragment_recovery_recovery_rate",
                   "n_pred_instances", "seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def areal_coverage(scene_dir: Path) -> dict:
    """The fraction of the frame the full-width reference strands cover.

    The per-instance reference is overlap-aware, so a pixel shared by two
    strands appears once per strand in the index list; the covered *area* is
    the union of those indices and the sum of the instance areas is not it.
    Both are written, with their ratio, because the difference between them is
    how much of the frame is contested and that is worth a number.
    """
    ref = scene_dir / "gt_multilabel.npz"
    meta = json.loads((scene_dir / "scene_meta.json").read_text(encoding="utf-8"))
    with np.load(ref, allow_pickle=False) as npz:
        height, width = (int(v) for v in npz["shape"])
        indices = npz["indices"]
        n_instances = int(npz["ids"].size)
        union = int(np.unique(indices).size)
        assigned = int(indices.size)
    frame = height * width
    stats = meta["mask_stats"]
    return {
        "frame_px": frame,
        "shape": [height, width],
        "n_reference_instances": n_instances,
        "reference_union_px": union,
        "areal_coverage": union / frame,
        "reference_assigned_px": assigned,
        "mean_instances_per_covered_px": assigned / union,
        "axis_mask_foreground_px": int(stats["foreground_px"]),
        "axis_mask_coverage": float(stats["coverage"]),
        "reference_sha256": sha256(ref),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons-root", type=Path, default=DEFAULT_COMPARISONS_ROOT,
        help="root of the comparisons checkout")
    args = parser.parse_args()
    study = args.comparisons_root / "real_sem_comparators"
    scenes = args.comparisons_root / "real_sem_study" / "scenes_v2"
    src = study / "results" / "real_fields_comparators.json"
    gate = study / "results" / "parity_gate.json"
    conversion = study / "results" / "parity_conversion.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    conv_payload = json.loads(conversion.read_text(encoding="utf-8"))

    if not gate_payload.get("passed"):
        raise SystemExit("the study's PLECTA parity gate did not pass; refusing")

    fields = json.loads((RESULTS / "plecta_real_fields.json")
                        .read_text(encoding="utf-8"))
    published = {f["image"]: f["conditions"]["manual"] for f in fields["fields"]}
    n_reference = {f["image"]: f["n_reference"] for f in fields["fields"]}

    rows: list[dict] = []
    for key, display in FIELDS.items():
        for label, tag in METHODS.items():
            found = [r for r in payload["rows"]
                     if r["field"] == key and r["method"] == label]
            if len(found) != 1:
                raise SystemExit(
                    f"expected one {label} row for {key}, got {len(found)}")
            row = found[0]
            if row["axis"] != AXIS:
                raise SystemExit(f"{key}/{label}: unexpected axis {row['axis']!r}")
            if row.get("status") != "ok":
                raise SystemExit(f"{key}/{label}: status {row.get('status')!r}")
            out = {"field": display, "method": tag, "label": label,
                   **{k: row[k] for k in MEASURES}}
            if tag != "plecta":
                out.update({k: row.get(k) for k in COMPARATOR_ONLY})
                out["params"] = row["params"]
            else:
                #  PLECTA is quoted rather than re-run, and
                #  plecta_real_fields.json stores no ARI or VI per field, so
                #  the study's own record carries none for it either. The
                #  parity gate does: it re-ran PLECTA through this scoring
                #  path and recorded them under `not_in_record` precisely
                #  because the published record had nowhere to put them. They
                #  are the same run the gate matched at 0.0 on everything the
                #  published record does store, so they belong to these rows.
                gate_row = [r for r in gate_payload["rows"]
                            if r["field"] == key]
                if len(gate_row) != 1:
                    raise SystemExit(f"no parity-gate row for {key}")
                extra = gate_row[0]["not_in_record"]
                out.update({k: extra[k] for k in
                            ("adjusted_rand_index", "vi_split_bits",
                             "vi_merge_bits", "vi_total_bits")})
            rows.append(out)

    #  The study's parity gate is re-asserted here rather than trusted: its
    #  PLECTA rows must be the published ones, to the last bit, or the
    #  comparator rows beside them are not on the same exam.
    worst, worst_at = 0.0, ""
    for row in rows:
        if row["method"] != "plecta":
            continue
        want = published[row["field"]]
        for key in MEASURES:
            deviation = abs(float(row[key]) - float(want[key]))
            if deviation > worst:
                worst, worst_at = deviation, f"{row['field']} {key}"
    if worst > 1e-9:
        raise SystemExit(
            "the study's PLECTA rows do not reproduce plecta_real_fields.json: "
            f"max |difference| {worst:.3e} at {worst_at}")

    coverage = {}
    for key, display in FIELDS.items():
        cov = areal_coverage(scenes / key / "skel")
        if cov["n_reference_instances"] != n_reference[display]:
            raise SystemExit(
                f"{display}: the reference carries "
                f"{cov['n_reference_instances']} instances, "
                f"plecta_real_fields.json says {n_reference[display]}")
        coverage[display] = cov

    record = {
        "role": ("SIFNE and our Basu Stage-B reimplementation on the three "
                 "manual-derived axis masks of the annotated CNT SEM fields, "
                 "beside PLECTA's published numbers on identical inputs and "
                 "through an identical scoring path"),
        "discipline": payload["discipline"],
        "axis": payload["axis"],
        "parameter_framing": payload["parameter_framing"],
        "basu_caveat": payload["basu_caveat"],
        "sifne_caveat": payload["sifne_caveat"],
        "coverage_definition": (
            "areal_coverage is the union of the full-width overlap-aware "
            "reference strands over the frame -- Section 2.2's definition, and "
            "the quantity the synthetic 20-60 % strata label. It is NOT "
            "axis_mask_coverage, the 0.032-0.040 foreground fraction of the "
            "thin input mask, which is a centreline density"),
        "source_record": {
            "study": str(study),
            "protocol": "PROTOCOL.md, frozen 2026-08-27 before any field ran",
            "file": "results/real_fields_comparators.json",
            "sha256": sha256(src),
            "parity_gate": {"file": "results/parity_gate.json",
                            "sha256": sha256(gate)},
            "parity_conversion": {"file": "results/parity_conversion.json",
                                  "sha256": sha256(conversion)},
            "scenes": str(scenes),
        },
        "parity": {
            "plecta_rows_reproduce_published": {
                "checked": list(MEASURES),
                "max_abs_difference": worst,
                "max_abs_difference_at": worst_at,
                "tolerance": 1e-9,
            },
            "study_gate_max_abs_difference": gate_payload["max_abs_difference"],
            "adapter_conversion_gate": {
                "max_abs_difference_vs_stored":
                    conv_payload["max_abs_difference_vs_stored"],
                "max_abs_difference_vs_stored_at":
                    conv_payload["max_abs_difference_vs_stored_at"],
                "rounding_note": conv_payload["rounding_note"],
                "basu_vs_live_parent_runner":
                    conv_payload["max_abs_difference_basu_vs_live_parent_runner"],
                "soft_spot":
                    conv_payload["stored_basu_record_is_not_reproducible"],
            },
        },
        "coverage": coverage,
        "rows": rows,
    }
    out = RESULTS / "plecta_real_comparators.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  PLECTA parity: max |difference| {worst:.3e}"
          + (f" at {worst_at}" if worst_at else ""))
    for display, cov in coverage.items():
        print(f"  {display}: areal coverage {100 * cov['areal_coverage']:.1f} %"
              f"  (thin axis mask {100 * cov['axis_mask_coverage']:.1f} %)")
    for row in rows:
        print(f"  {row['field']:>8} {row['method']:>7}: "
              f"F1 {row['pairwise_f1']:.3f}  join {row['join_f1']:.3f}  "
              f"det {row['detection_f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
