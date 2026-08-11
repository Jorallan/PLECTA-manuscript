"""Copy the GraFT comparison, and the overlap measurement it produced.

GraFT (Oesterlund et al., Sci Adv 12:eadz4132, 2026) traces filaments from a
graph built on a skeleton. Its published entry point begins with a Frangi and
Otsu front end that enhances and binarises grayscale ridge images, which is
degenerate on a binary one-pixel axis, so the mask is injected at its
skeleton-to-graph stage and everything downstream is its own code unchanged.

That decision was verified rather than assumed, and the injection turns out to
help GraFT rather than hobble it: run end to end on the grayscale images it
scores near the connected-component floor, because its segmentation stage
recovers only a third to a half of the reference skeleton.

The same study measured something this manuscript needs for its own sake. The
primary metric splits the skeleton at junctions, discards the junction pixels,
and assigns each surviving fragment to exactly one instance on both sides, so it
cannot see whether an instance shares a crossing pixel. Flattening either
method's overlap-aware output to a single label image is therefore almost free.
That is a fairness point for GraFT, whose near-hard partition costs it nothing
here, and a limitation of this paper, whose titular property the primary
endpoint does not measure.

Copies only; runs neither method.

Source: C:/Repos/comparisons/graft_comparison

Usage:
  python scripts/results/make_graft_record.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
STUDY = Path("C:/Repos/comparisons/graft_comparison")
SUMMARY = STUDY / "results" / "summary.json"
OVERLAP = STUDY / "results" / "overlap_cost_degraded.csv"

# The configuration reported: mask injected at the skeleton stage, with the one
# parameter that was translated on development scenes.
ARM = "matched_inject_a160"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap_by_density() -> dict:
    with OVERLAP.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["density"])].append(row)

    def mean(subset, key):
        values = [float(r[key]) for r in subset
                  if r.get(key) not in ("", "nan", None)]
        return statistics.fmean(values) if values else None

    out = {}
    for density, subset in sorted(grouped.items()):
        out[str(density)] = {
            "n": len(subset),
            "reference_shared_fraction": mean(subset, "gt_shared_frac"),
            "plecta_shared_fraction": mean(subset, "plecta_shared_frac"),
            "graft_shared_fraction": mean(subset, "graft_shared_frac"),
            "plecta_f1_cost_of_flattening":
                mean(subset, "plecta_f1_cost_of_flattening"),
            "graft_f1_cost_of_flattening":
                mean(subset, "graft_f1_cost_of_flattening"),
        }
    pooled = {
        "plecta_f1_cost_of_flattening_max": max(
            abs(v["plecta_f1_cost_of_flattening"]) for v in out.values()),
        "graft_f1_cost_of_flattening_max": max(
            abs(v["graft_f1_cost_of_flattening"]) for v in out.values()),
    }
    return {"by_density": out, "pooled": pooled}


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    arms = {}
    for variant in ("degraded", "clean"):
        for method, key in (("plecta", f"stubmatch|native|{variant}"),
                            ("graft", f"graft|{ARM}|{variant}"),
                            ("graft_native", f"graft|native|{variant}"),
                            ("graft_shipped", f"graft|matched_inject|{variant}")):
            block = summary["arms"].get(key)
            if block is None:
                continue
            arms[f"{method}|{variant}"] = {
                "n_scenes": block["n_scenes"],
                "n_ok": block["n_ok"],
                "n_failed": block["n_failed"],
                "failures": block.get("failures", []),
                "overall": block["overall"],
                "by_density": block.get("by_density", {}),
            }

    paired = {}
    for variant in ("degraded", "clean"):
        key = f"PLECTA_minus_GraFT|{ARM}|{variant}"
        block = summary["paired"].get(key)
        if block:
            paired[variant] = block

    payload = {
        "description": (
            "PLECTA against GraFT on 50 scenes, identical masks, one scorer. "
            "GraFT receives the mask at its skeleton-to-graph stage; its own "
            "front end is degenerate on a binary axis and scores near the "
            "connected-component floor when run end to end."
        ),
        "comparator": {
            "name": "GraFT",
            "citation": ("Oesterlund et al., Science Advances 12(13):eadz4132 "
                         "(2026)"),
            "doi": "10.1126/sciadv.adz4132",
            "repository": summary["provenance"].get("graft_repo"),
            "commit": summary["provenance"].get("graft_commit"),
            "licence": summary["provenance"].get("graft_licence"),
            "reported_configuration": ARM,
            "parameter_translation": (
                "minimum continuation angle 140 to 160 degrees, selected on 10 "
                "development scenes disjoint from the comparison set; all other "
                "shipped values kept, none improved on development data"),
        },
        "provenance": {
            "study": str(STUDY).replace("\\", "/"),
            "summary_sha256": sha256(SUMMARY),
            "overlap_sha256": sha256(OVERLAP),
            "parity_gate": summary["provenance"].get("parity_gate"),
        },
        "statistics": {
            "aggregate": "density-stratified paired bootstrap over the scenes "
                         "both methods completed",
            "per_density": "plain means, n stated, no interval",
            "timeouts": ("scenes GraFT did not finish are excluded from its "
                         "means; they are the hardest of the densest tier, so "
                         "the reported GraFT means are optimistic"),
        },
        "arms": arms,
        "paired": paired,
        "overlap_cost": overlap_by_density(),
    }

    destination = RESULTS / "plecta_graft_comparison.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {destination}")
    for variant, block in paired.items():
        f1 = block["metrics"]["f1"]
        print(f"  {variant}: {f1['mean_diff']:+.3f} "
              f"[{f1['ci95_lo']:+.3f}, {f1['ci95_hi']:+.3f}], "
              f"PLECTA wins {f1['plecta_wins']}/{f1['n']}")
    cost = payload["overlap_cost"]["pooled"]
    print(f"  flattening costs PLECTA at most "
          f"{cost['plecta_f1_cost_of_flattening_max']:.4f} F1, GraFT at most "
          f"{cost['graft_f1_cost_of_flattening_max']:.4f}")


if __name__ == "__main__":
    main()
