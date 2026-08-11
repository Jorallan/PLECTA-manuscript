"""Copy the comparator fairness analysis into the manuscript.

Two questions were put to the comparison after the first results, and both
changed what the manuscript can say.

1. Was the greedy baseline fairly tuned? Its two parameters had been set by
   hand with no scored selection record, while PLECTA's configuration was
   selected over 84 development scenes. A 10x10 grid was swept over the same
   84 development scenes. The hand-set values turned out to rank 5th of 100,
   so the baseline was not materially under-tuned, but the swept optimum is
   better and is what the manuscript now reports.

2. Is PLECTA better because it decides better, or only because it attempts
   cases the comparator cannot? Restricting the score to the evidence a
   degree-limited, gap-blind method can reach separates the two. About a third
   of the margin is capability coverage; the rest is decision quality.

Copies only; runs neither method.

Source: C:/Repos/comparisons/dnai_comparison/results

Usage:
  python scripts/results/make_comparator_analysis_record.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SOURCE = Path("C:/Repos/comparisons/dnai_comparison/results")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sweep_path = SOURCE / "greedy_sweep" / "greedy_swept_summary.json"
    evidence_path = SOURCE / "junction_evidence.json"
    paired_path = SOURCE / "junction_evidence_paired_ci.json"

    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))

    payload = {
        "description": (
            "Fairness analysis of the comparator study: a development-set "
            "sweep of the greedy baseline, and a decomposition of the margin "
            "over DNAi into capability coverage and decision quality."
        ),
        "greedy_sweep": sweep,
        "junction_evidence": evidence,
        "paired_full_vs_restricted": paired,
        "provenance": {
            "study": "C:/Repos/comparisons/dnai_comparison",
            "greedy_sweep_sha256": sha256(sweep_path),
            "junction_evidence_sha256": sha256(evidence_path),
            "paired_sha256": sha256(paired_path),
            "selection_scenes": (
                "84 development scenes; the 50-scene comparison set was never "
                "consulted for selection and the locked set was never read"),
        },
        "statistics": {
            "aggregate": "density-stratified paired bootstrap over 50 scenes",
            "per_density": "plain means, n = 10, no interval, no inference",
        },
    }

    destination = RESULTS / "plecta_comparator_analysis.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {destination}")

    sel = sweep["selected_configuration"]
    print(f"  greedy baseline selected on development scenes: "
          f"max_gap_px={sel['max_gap_px']}, max_turn_deg={sel['max_turn_deg']}")
    for cond in ("degraded", "clean"):
        key = f"{cond}|restricted|plecta_minus_dnai_tuned"
        full = paired[f"{cond}|full|plecta_minus_dnai_tuned"]
        res = paired[key]
        share = 1.0 - res["mean_diff"] / full["mean_diff"]
        print(f"  {cond}: DNAi margin {full['mean_diff']:+.3f} full -> "
              f"{res['mean_diff']:+.3f} restricted "
              f"({share*100:.0f}% of it was capability coverage)")


if __name__ == "__main__":
    main()
