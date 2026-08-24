"""Collect the 2.5-D depth-decision study into one machine-readable record.

Source: ``exploration/depth_order_eval/depth_scores.json``, written by that
study's ``score_depth.py``.

Two input conditions over the same 12 scenes, and the pairing is the point:

* ``oracle`` takes the ground-truth 2-D centrelines, so the 2-D grouping never
  runs and the depth decision is measured on its own;
* ``clean`` runs PLECTA's own 2-D reconstruction from the non-degraded binary
  axis mask, so it carries upstream error.

Both then place z from the same image evidence: ONE channel, the flank-vs-core
median intensity of each rod, with the denominator floored at twice the local
noise estimate (``2 * sigma_hat``). The signed score ``s`` decides the
direction and weights it, and a crossing abstains when ``|s| < 0.40``. See
sections/07_technical.tex for the normative account of the change.

Re-measured under that shipped rule on 2026-08-23 by
``scripts/results/run_depth_order_eval.py``, which regenerates every
``pred_depth.json`` and then calls ``score_depth.py``. Everything before that
date was measured under the previous two-channel logistic rule and is
superseded. ``score_depth.py`` only scores what is already on disk, so re-running
it alone does NOT re-measure the rule -- go through the driver.
Reporting either alone would misattribute: ``coa_all`` falls sharply between
them, but ``coa_decided`` does not, and the gap is the crossing match rate.
The record therefore always carries all three together.

Exploratory: 6 scenes per coverage, plain means, no interval and none claimed.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "exploration" / "depth_order_eval" / "depth_scores.json"
OUT = REPO / "results" / "plecta_depth_order.json"
PARAMS = Path(os.environ.get(
    "PLECTA_PARAMS_YAML", r"C:\Repos\PLECTA\plecta\parameters.yaml"))


def _shipped_rule() -> str:
    """The evidence rule these scores were produced under, from the released file.

    Three rules are implemented and the shipped one has changed once already,
    moving every number in this record. Naming it here, from
    ``parameters.yaml`` rather than from a description, is what lets a reader
    check the record against the code: if the two disagree, they disagree
    visibly instead of silently.
    """
    try:
        import yaml
    except ModuleNotFoundError:                       # pragma: no cover
        return "unknown (PyYAML not installed)"
    if not PARAMS.is_file():
        return f"unknown (no {PARAMS})"
    tree = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    return str(tree["depth_3d"]["evidence_weights"]["scoring"])

# Reported per cell. Chance for every accuracy here is 0.5 -- the over/under
# decision is binary and symmetric -- so the margin over 0.5 is the evidence.
FIELDS = ("coa_all", "coa_decided", "abstain_rate", "match_rate",
          "coa_shallow", "coa_steep", "coa_near_sep", "coa_far_sep",
          "order_acc_crossing_pairs", "order_acc_all_pairs",
          "frac_pairs_within_component", "n_components",
          "layer_exact_agreement", "n_layers_pred", "n_layers_gt",
          "n_gt_crossings")

# Kendall tau over the same instance pairs is exactly 2*accuracy - 1, so it is
# not reported: it would be the same measurement printed twice under a second
# name.


def main() -> int:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["condition"], r["coverage"])].append(r)

    conditions = sorted({r["condition"] for r in rows})
    coverages = sorted({r["coverage"] for r in rows})

    def agg(sub, key):
        vals = [r[key] for r in sub if r.get(key) is not None]
        vals = [v for v in vals if v == v]          # drop NaN
        return mean(vals) if vals else None

    grid = []
    for cond in conditions:
        for cov in coverages:
            sub = cells[(cond, cov)]
            grid.append({
                "condition": cond,
                "coverage": cov,
                "n_scenes": len(sub),
                **{k: agg(sub, k) for k in FIELDS},
            })

    # What the global order *could* be, given that only pairs inside one
    # component carry evidence and the rest are a tie-break. If the measured
    # all-pairs accuracy sits at this value, the weak global figure is the
    # crossing graph being disconnected and not the depth decision failing.
    for c in grid:
        f = c["frac_pairs_within_component"]
        c["order_acc_all_pairs_predicted"] = (
            f * c["order_acc_crossing_pairs"] + (1.0 - f) * 0.5)
        c["order_acc_all_pairs_residual"] = (
            c["order_acc_all_pairs"] - c["order_acc_all_pairs_predicted"])

    decided = [c["coa_decided"] for c in grid]
    allc = [c["coa_all"] for c in grid]
    match = [c["match_rate"] for c in grid]

    payload = {
        "role": "PLECTA's depth stage: is the instance chosen as `over` at "
                "each projected crossing the one physically in front?",
        "discipline": "exploratory, 6 scenes per coverage, plain means, "
                      "no inference",
        "scenes": "eval/generators/synth_depth.py --domain semlike, "
                  "seeds 20530101+, 512 px",
        "input_mask": "mask_clean.png, the non-degraded axis; measured median "
                      "thickness 2.00 px. All 2.5-D results use this one mask "
                      "so conditions differ only in what is being tested",
        "configuration": "image evidence on, grazing overlaps not cleared, "
                         "undecided pairs stacked compactly -- the shipped "
                         "defaults",
        #  Which rule combined the channels, read live rather than described.
        #  The shipped rule changed once during this work and the numbers moved
        #  with it, so a record that does not name its rule cannot be checked
        #  against the code that produced it.
        "evidence_rule": _shipped_rule(),
        "chance_level": 0.5,
        "conditions": {
            "oracle": "ground-truth 2-D centrelines; the 2-D grouping does not "
                      "run, so this isolates the depth decision",
            "clean": "PLECTA's own 2-D reconstruction from the binary axis "
                     "mask; carries upstream error",
        },
        "read_together": "coa_decided is correct/decided and a method that "
                         "abstained on everything hard would score 1.000, so "
                         "it is never quoted without coa_all and abstain_rate",
        "headline": {
            "coa_decided_min": min(decided),
            "coa_decided_max": max(decided),
            "coa_all_min": min(allc),
            "coa_all_max": max(allc),
            "match_rate_min": min(match),
            "match_rate_max": max(match),
            # The decomposition: the decision holds while recovery does not.
            "coa_decided_span": max(decided) - min(decided),
            "coa_all_span": max(allc) - min(allc),
            "frac_within_min": min(c["frac_pairs_within_component"] for c in grid),
            "frac_within_max": max(c["frac_pairs_within_component"] for c in grid),
            "order_residual_absmax": max(abs(c["order_acc_all_pairs_residual"])
                                         for c in grid),
        },
        "grid": grid,
        "per_scene": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")
    h = payload["headline"]
    print(f"  coa_decided {h['coa_decided_min']:.3f}-{h['coa_decided_max']:.3f}"
          f"  (span {h['coa_decided_span']:.3f})")
    print(f"  coa_all     {h['coa_all_min']:.3f}-{h['coa_all_max']:.3f}"
          f"  (span {h['coa_all_span']:.3f})")
    print(f"  match_rate  {h['match_rate_min']:.3f}-{h['match_rate_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
