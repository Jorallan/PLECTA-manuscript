"""Build the greedy-continuation baseline comparison record.

PLECTA's narrowed contribution is that a junction is solved as one exact
maximum-weight partial matching, with an explicit priced option to leave an arm
unmatched, rather than by pairing endpoints greedily in priority order. That is
the rule used by SIFNE, GraFT and the continuity baseline of Wegmayr et al., so
the claim needs a greedy comparator on identical input and an identical scorer.

`eval/baseline/baselines.py::baseline_skeleton_minturn` is exactly that rule:
skeletonise, prune spurs, split into branches, then greedily accept the
lowest-turning-angle pairing between branch ends inside a distance gate and a
turning-angle gate. It was run alongside PLECTA on a purpose-built 50-scene set
generated after both configurations were fixed, with both methods receiving the
same mask and both scored through `eval/core/common_metric.py`.

This script copies those per-scene rows into the manuscript repository so the
reported numbers have a self-contained source, and computes the paired
statistics. It does not run either method.

Source (read-only):
  C:/Repos/comparisons/filaseg_stubmatch_strandface/results/testcmp10_per_scene.csv
  C:/Repos/comparisons/filaseg_stubmatch_strandface/results/testcmp10_per_scene_clean.csv

Usage:
  python scripts/results/make_greedy_baseline_record.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SOURCE = Path("C:/Repos/comparisons/filaseg_stubmatch_strandface/results")

# The two methods we report. `filaseg` and `strandtrace` rows also exist in the
# source and are deliberately not carried over: the first is the superseded
# predecessor, the second an unpublished in-house method. Neither belongs in the
# publication-facing comparison.
PLECTA = "stubmatch"
GREEDY = "baseline_skeleton"

METRICS = ("f1", "precision", "recall", "fragment_recovery_recovery_rate",
           "adjusted_rand_index", "vi_split_bits", "vi_merge_bits",
           "vi_total_bits")

N_BOOT = 20_000
SEED = 20_260_810


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def paired(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Return per-scene paired records, and any scene missing a method."""
    by_scene: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["status"] != "ok":
            continue
        by_scene[(int(row["density"]), row["scene"])][row["method"]] = row

    out, dropped = [], []
    for (density, scene), methods in sorted(by_scene.items()):
        if PLECTA not in methods or GREEDY not in methods:
            dropped.append(f"cov{density}/{scene}")
            continue
        record = {"density": density, "scene": scene}
        for metric in METRICS:
            record[f"plecta_{metric}"] = float(methods[PLECTA][metric])
            record[f"greedy_{metric}"] = float(methods[GREEDY][metric])
        out.append(record)
    return out, dropped


def stratified_paired_ci(records: list[dict], metric: str) -> dict:
    """Density-stratified nonparametric bootstrap of the paired difference.

    Matches the convention used for every other interval in this manuscript:
    resample scenes with replacement inside each density stratum, recompute the
    mean paired difference, take the 2.5th and 97.5th percentiles.
    """
    diffs = np.array([r[f"plecta_{metric}"] - r[f"greedy_{metric}"]
                      for r in records])
    strata = np.array([r["density"] for r in records])
    rng = np.random.default_rng(SEED)

    replicates = np.empty(N_BOOT)
    index_by_stratum = [np.flatnonzero(strata == d) for d in np.unique(strata)]
    for b in range(N_BOOT):
        picked = np.concatenate([
            rng.choice(idx, size=idx.size, replace=True)
            for idx in index_by_stratum
        ])
        replicates[b] = diffs[picked].mean()

    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return {
        "mean_difference": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "plecta_better": int((diffs > 0).sum()),
        "greedy_better": int((diffs < 0).sum()),
        "ties": int((diffs == 0).sum()),
    }


def summarise(records: list[dict]) -> dict:
    out: dict = {"n_scenes": len(records), "metrics": {}}
    for metric in METRICS:
        out["metrics"][metric] = {
            "plecta_mean": float(np.mean([r[f"plecta_{metric}"] for r in records])),
            "greedy_mean": float(np.mean([r[f"greedy_{metric}"] for r in records])),
            "paired": stratified_paired_ci(records, metric),
        }
    by_density: dict[str, dict] = {}
    for density in sorted({r["density"] for r in records}):
        subset = [r for r in records if r["density"] == density]
        by_density[str(density)] = {
            "n": len(subset),
            "plecta_f1": float(np.mean([r["plecta_f1"] for r in subset])),
            "greedy_f1": float(np.mean([r["greedy_f1"] for r in subset])),
        }
    out["by_density"] = by_density
    return out


def main() -> None:
    payload: dict = {
        "description": (
            "PLECTA against a greedy minimum-turning-angle continuation "
            "baseline on a 50-scene set generated after both configurations "
            "were fixed. Identical input masks; both scored through "
            "eval/core/common_metric.py."
        ),
        "baseline": {
            "implementation": "eval/baseline/baselines.py::baseline_skeleton_minturn",
            "rule": ("skeletonise, prune spurs, split into branches, then "
                     "greedily accept the lowest-turning-angle pairing between "
                     "branch ends within a distance and turning-angle gate"),
            "max_gap_px": 28.0,
            "max_turn_deg": 20.0,
        },
        "bootstrap": {"replicates": N_BOOT, "seed": SEED,
                      "resampling": "within_density_paired_scene"},
        "conditions": {},
    }

    for variant, filename in (("degraded", "testcmp10_per_scene.csv"),
                              ("clean", "testcmp10_per_scene_clean.csv")):
        path = SOURCE / filename
        records, dropped = paired(load(path))
        payload["conditions"][variant] = {
            "source_file": str(path).replace("\\", "/"),
            "source_sha256": sha256(path),
            "scenes_without_both_methods": dropped,
            "summary": summarise(records),
            "per_scene": records,
        }

    destination = RESULTS / "plecta_greedy_baseline.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {destination}")
    for variant, block in payload["conditions"].items():
        stats = block["summary"]["metrics"]["f1"]["paired"]
        print(f"  {variant}: n={block['summary']['n_scenes']} "
              f"dF1={stats['mean_difference']:+.4f} "
              f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] "
              f"PLECTA better in {stats['plecta_better']}")


if __name__ == "__main__":
    main()
