"""Copy the DNAi comparison into the manuscript as a self-contained record.

The study lives at C:/Repos/comparisons/dnai_comparison. It runs DNAi's own
junction-disentanglement entry point on the same 50-scene set, the same masks
and the same scorer used for PLECTA and the greedy baseline.

Two configurations are carried:

  shipped   DNAi at its released defaults.
  tuned     DNAi at the best configuration found on development scenes. The
            one parameter that matters is the distance at which junction
            pixels are clustered by single linkage. At its shipped 10 px this
            chains across our crossings: the largest cluster spans 89 px at
            60 % coverage and is replaced by a single small erasure disk, so
            the crossings along it are never separated and the tangle survives
            as one instance. Reducing it to 2 px moves development F1 from
            0.211 to 0.378, more than the whole fibre-width sweep achieved.

The tuned configuration is the one reported. Dilating the input to a thicker
mask was tested across 41 development configurations and made DNAi worse at
every setting, so it is not used.

This script only copies and re-summarises; it runs neither method.

Usage:
  python scripts/results/make_dnai_record.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SOURCE = Path("C:/Repos/comparisons/dnai_comparison/results")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary_path = SOURCE / "summary.json"
    per_scene_path = SOURCE / "per_scene.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    with per_scene_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle)]

    payload = {
        "description": (
            "PLECTA against DNAi's junction-disentanglement stage on 50 scenes, "
            "identical masks, one scorer. Reported at DNAi's best configuration "
            "found on development scenes."
        ),
        "comparator": {
            "name": "DNAi",
            "citation": "Playout et al., Nucleic Acids Research 54(7):gkag335 (2026)",
            "doi": "10.1093/nar/gkag335",
            "repository": summary["provenance"].get("dnai_repo"),
            "commit": summary["provenance"].get("dnai_commit"),
            "licence": summary["provenance"].get("dnai_licence"),
            "entry_point": summary["provenance"].get("dnai_entry_point"),
            "segmentation_network_used": False,
            "note": (
                "Only the mask post-processing stage is used. Its segmentation "
                "network is deliberately bypassed so that grouping quality is "
                "not confounded with segmentation quality."
            ),
        },
        "provenance": {
            "study": "C:/Repos/comparisons/dnai_comparison",
            "summary_sha256": sha256(summary_path),
            "per_scene_sha256": sha256(per_scene_path),
            "scorer": summary["provenance"].get("scorer"),
            "parity_gate": summary["provenance"].get("parity_gate"),
        },
        "statistics": {
            "aggregate_intervals": (
                "density-stratified paired bootstrap over all 50 scenes"),
            "per_density": (
                "plain means, n = 10 per stratum, no interval and no inference"),
        },
        "per_method": summary["per_method"],
        "per_density": summary["per_density"],
        "paired": summary["paired"],
        "per_scene": rows,
    }

    destination = RESULTS / "plecta_dnai_comparison.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination} ({len(rows)} per-scene rows)")


if __name__ == "__main__":
    main()
