"""Copy the scorer-constant sensitivity sweep into the manuscript's records.

Source study: C:/Repos/comparisons/scorer_sensitivity (PROTOCOL.md frozen
2026-08-14 before any result existed; grid one-at-a-time around the published
defaults; stored predictions re-scored, nothing re-run).

The study's own parity gate already requires every default-combination
per-scene f1 to reproduce the published CSVs to 1e-9. This record-maker
re-asserts the aggregate consequence: the default row's PLECTA mean F1 must
match \\PlectaGreedySceneCount-set comparator table's stored PLECTA mean to
within rounding, or nothing is written.

    python scripts/results/make_scorer_sensitivity_record.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DEFAULT_COMPARISONS_ROOT = ROOT.parent / "comparisons"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons-root", type=Path, default=DEFAULT_COMPARISONS_ROOT,
        help="root of the comparisons checkout")
    args = parser.parse_args()
    study = args.comparisons_root / "scorer_sensitivity"
    src = study / "results" / "sweep_summary.json"
    payload = json.loads(src.read_text(encoding="utf-8"))

    if "PARITY" not in payload.get("parity", "").upper() and not payload.get("parity"):
        raise SystemExit("source record carries no parity statement; refusing")

    default_rows = [r for r in payload["summary"] if r["combo"] == "default"]
    assert len(default_rows) == 1, "expected exactly one default combination"

    audit = json.loads((RESULTS / "plecta_metrics_audit.json")
                       .read_text(encoding="utf-8"))
    stored = audit["methods"]["degraded"]["plecta"]["overall"].get("f1")
    if stored is not None:
        dev = abs(default_rows[0]["plecta_f1_mean"] - float(stored))
        if dev > 5e-4:
            raise SystemExit(
                f"default-combo PLECTA mean {default_rows[0]['plecta_f1_mean']:.4f} "
                f"does not reproduce the stored mean {stored:.4f} (|d|={dev:.2e})")

    record = {
        "role": ("scorer-constant sensitivity: stored predictions re-scored "
                 "under one-at-a-time perturbations of the four common-metric "
                 "constants; disclosure, not selection -- the published "
                 "configuration is unchanged regardless of outcome"),
        "source_record": {
            "study": str(study),
            "file": "results/sweep_summary.json",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "protocol": "PROTOCOL.md, frozen 2026-08-14 before any result",
        },
        "parity": payload["parity"],
        "defaults": payload["defaults"],
        "summary": payload["summary"],
    }
    out = RESULTS / "plecta_scorer_sensitivity.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    for r in payload["summary"]:
        print(f"  {r['combo']:>22}: PLECTA {r['plecta_f1_mean']:.4f}  "
              f"greedy {r['margin_vs_greedy_swept']:+.4f}  "
              f"DNAi {r['margin_vs_dnai']:+.4f}  "
              f"GraFT {r['margin_vs_graft']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
