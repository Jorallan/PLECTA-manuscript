"""Collect the bending-stiffness sensitivity sweep into one record.

Source: ``exploration/curviness_regime/v2_per_scene.csv``, produced by that
study's ``run_sweep_v2.py`` under its frozen PROTOCOL.md. The study is gated on
two parity checks that run before any scene exists: regenerating a stored
testcmp10 scene reproduces its ``mask_w1.png`` byte for byte, and frozen PLECTA
scored through the comparison harness reproduces all ten stored per-scene F1
values at a delta of exactly zero.

What the record is for: ``curviness`` is the generator's bending-stiffness
knob, and a reader is entitled to ask whether the headline numbers are an
artefact of its setting. They are not, and the record carries the comparison
that shows it -- the spread of the level means against the scene-to-scene
spread within a level. Reporting the first without the second would invite the
reader to read noise as a trend.

Exploratory: 10 scenes per cell, two coverage targets, plain means, no
interval and none claimed.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "exploration" / "curviness_regime" / "v2_per_scene.csv"
OUT = REPO / "results" / "plecta_curviness_sensitivity.json"

DEFAULT = 0.035
# Measured off the generated centrelines by the study's measure_geometry.py.
GEOMETRY = {
    0.01750: {"median_radius_px": 975, "turn_per_100px_deg": 5.7,
              "p95_local_bend_20px_deg": 3.0},
    0.02625: {"median_radius_px": 632, "turn_per_100px_deg": 9.1,
              "p95_local_bend_20px_deg": 4.6},
    0.03500: {"median_radius_px": 496, "turn_per_100px_deg": 11.4,
              "p95_local_bend_20px_deg": 5.9},
    0.04375: {"median_radius_px": 383, "turn_per_100px_deg": 15.0,
              "p95_local_bend_20px_deg": 7.9},
    0.05250: {"median_radius_px": 308, "turn_per_100px_deg": 16.9,
              "p95_local_bend_20px_deg": 9.2},
}


def main() -> int:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["cov"], r["variant"], float(r["curviness"]))].append(float(r["f1"]))

    covs = sorted({r["cov"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    levels = sorted({float(r["curviness"]) for r in rows})

    series = []
    for cov in covs:
        for var in variants:
            means = [mean(cells[(cov, var, lv)]) for lv in levels]
            sds = [stdev(cells[(cov, var, lv)]) for lv in levels]
            series.append({
                "coverage": cov,
                "mask": var,
                "per_level": [
                    {"curviness": lv, "f1_mean": m, "f1_sd": s,
                     "n": len(cells[(cov, var, lv)])}
                    for lv, m, s in zip(levels, means, sds)],
                "spread_of_level_means": max(means) - min(means),
                "mean_within_level_sd": mean(sds),
                # The whole point: is the effect bigger than the noise?
                "spread_exceeds_noise": (max(means) - min(means)) > mean(sds),
            })

    payload = {
        "role": "sensitivity of frozen PLECTA to the generator's bending "
                "stiffness (`curviness`), on synth_thick scenes at the "
                "manuscript's own coverage targets",
        "discipline": "exploratory, 10 scenes per cell, plain means, no inference",
        "not_paired": "coverage targeting changes the filament count per "
                      "level, so a shared seed does not give the same scene; "
                      "the levels are independent samples, not matched "
                      "counterfactuals",
        "gates": {
            "generator": "regenerating testcmp10/cov20/synth_0000 from its "
                         "stored seed reproduces mask_w1.png byte for byte",
            "scorer": "frozen PLECTA over testcmp10/cov20 reproduces all ten "
                      "stored per-scene F1 values at max|delta| = 0.0",
        },
        "default_curviness": DEFAULT,
        "swept_fraction_of_default": [min(levels) / DEFAULT, max(levels) / DEFAULT],
        "geometry": [dict(curviness=k, **v) for k, v in sorted(GEOMETRY.items())],
        "series": series,
        "headline": {
            "max_spread_of_level_means": max(s["spread_of_level_means"] for s in series),
            "min_within_level_sd": min(s["mean_within_level_sd"] for s in series),
            "any_series_where_spread_exceeds_noise":
                any(s["spread_exceeds_noise"] for s in series),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")
    h = payload["headline"]
    print(f"  max spread of level means : {h['max_spread_of_level_means']:.3f}")
    print(f"  min within-level scene sd : {h['min_within_level_sd']:.3f}")
    print(f"  any series with spread > noise: {h['any_series_where_spread_exceeds_noise']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
