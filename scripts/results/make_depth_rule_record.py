"""Emit the measurement that replaced the depth stage's evidence rule.

Reads ``results/plecta_depth_rule_measured.json``, written by
``scripts/results/run_depth_rule_comparison.py``, and emits the
``\\PlectaDepthRule*`` macros the supplementary quotes.

These numbers used to be transcribed out of prose. The 2026-08-22 study that
justified the replacement wrote no machine-readable output, so every published
quantity sat outside the repository's own gate and could drift from the code
silently. The twenty-one macros below are now recomputed from the held-out
scenes on every run.

Recomputing changed them, and the change is instructive rather than alarming.
The conclusion is unmoved: the same accuracy at matched coverage, with
markedly better-ordered confidence. The magnitudes moved because the earlier
figures were measured before several unrelated changes to ``plecta/depth.py``,
and the AUC in particular is not comparable across that drift. What did not
survive is the earlier accuracy delta's sign at *full* coverage; see the note
on ``accuracy_delta_at_full_coverage`` below, which is reported here precisely
because it is the reading that goes against the shipped rule.

Every quantity this module publishes is computed. None is transcribed. That
was not true before: thirteen values describing what the rule change *deleted*
-- the removed fixed denominator floor and the saturation behaviour it caused,
the removed gradient/sharpness channel's standalone behaviour, and an axial
channel measured inert before removal -- were typed out of prose and emitted
alongside the computed ones. No configuration of the shipped code can produce
them, because the code that produced them is gone, so they were diagnoses of
why the old rule went rather than properties of the shipped one. They no longer
appear in the manuscript or in this module's output; they are kept for
reference in the untracked ``notes/depth_rule_transcribed_measurements.md``.

Scope. Only measurements live here. The rule's tunable constants (core arc,
sampling window, minimum flank count, abstention threshold, exact-solver
limit) are configuration, read live from ``plecta/parameters.yaml`` by
``make_parameter_table.py``, and reach the prose as ``\\PlectaParam*``.
Nothing is defined twice.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "plecta_depth_rule_measured.json"
OUT_JSON = REPO / "results" / "plecta_depth_rule.json"
OUT_TEX = REPO / "results" / "plecta_depth_rule.tex"


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> int:
    if not SRC.is_file():
        raise SystemExit(
            f"{SRC} not found. Run scripts/results/run_depth_rule_comparison.py "
            "first; these numbers are computed, not stored.")
    m = json.loads(SRC.read_text(encoding="utf-8"))

    op = m["operating_point"]
    cov_a, cov_b = sorted(m["accuracy_at_coverage"], reverse=True)

    computed = [
        ("PlectaDepthRuleHeldoutScenes", str(m["n_scenes"])),
        ("PlectaDepthRuleHeldoutCrossings", str(m["n_gt_crossings"])),
        #  Each rule where it actually runs.
        ("PlectaDepthRuleCovShipped", f"{op['shipped']['coverage'] * 100:.1f}"),
        ("PlectaDepthRuleCovLegacy", f"{op['legacy']['coverage'] * 100:.1f}"),
        ("PlectaDepthRuleAccShipped", f"{op['shipped']['accuracy']:.4f}"),
        ("PlectaDepthRuleAccLegacy", f"{op['legacy']['accuracy']:.4f}"),
        #  Matched-coverage comparison, which is the honest accuracy reading.
        ("PlectaDepthRuleCovAPct", f"{float(cov_a) * 100:g}"),
        ("PlectaDepthRuleAccAtCovA",
         _signed(m["accuracy_at_coverage"][cov_a]["delta"])),
        ("PlectaDepthRuleCovBPct", f"{float(cov_b) * 100:g}"),
        ("PlectaDepthRuleAccAtCovB",
         _signed(m["accuracy_at_coverage"][cov_b]["delta"])),
        #  Headline accuracy: both rules answering as often as the shipped one
        #  does. This is the quantity the prose calls "at matched coverage".
        ("PlectaDepthRuleAccDelta",
         _signed(m["accuracy_delta_at_matched_coverage"])),
        ("PlectaDepthRuleAccCILow",
         _signed(m["accuracy_delta_matched_ci"][0])),
        ("PlectaDepthRuleAccCIHigh",
         _signed(m["accuracy_delta_matched_ci"][1])),
        #  The adversarial reading, kept and reported: both rules forced to
        #  answer every crossing, including the ones they would decline. It
        #  goes against the shipped rule, because abstention is exactly where
        #  better-ordered confidence pays.
        ("PlectaDepthRuleAccDeltaFull",
         _signed(m["accuracy_delta_at_full_coverage"])),
        ("PlectaDepthRuleAccFullCILow", _signed(m["accuracy_delta_ci"][0])),
        ("PlectaDepthRuleAccFullCIHigh", _signed(m["accuracy_delta_ci"][1])),
        #  Confidence ordering, the quantity the replacement was made for.
        ("PlectaDepthRuleAucOld", f"{m['auc']['legacy']:.4f}"),
        ("PlectaDepthRuleAucNew", f"{m['auc']['shipped']:.4f}"),
        ("PlectaDepthRuleAucDelta", _signed(m["auc_delta"])),
        ("PlectaDepthRuleAucCILow", _signed(m["auc_delta_ci"][0])),
        ("PlectaDepthRuleAucCIHigh", _signed(m["auc_delta_ci"][1])),
    ]

    lines = [
        "% Generated by scripts/results/make_depth_rule_record.py; do not edit.",
        "% Computed from results/plecta_depth_rule_measured.json, which",
        "% run_depth_rule_comparison.py writes from the held-out scenes.",
        "% The rule's tunable constants are NOT here: they are \\PlectaParam*,",
        "% read live from plecta/parameters.yaml.",
        "% Every macro below is computed; none is transcribed.",
    ]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in computed]
    #  Convenience pairs the prose sets as one token.
    lines += [
        r"\newcommand{\PlectaDepthRuleAccCI}"
        r"{[\PlectaDepthRuleAccCILow,\PlectaDepthRuleAccCIHigh]}",
        r"\newcommand{\PlectaDepthRuleAucCI}"
        r"{[\PlectaDepthRuleAucCILow,\PlectaDepthRuleAucCIHigh]}",
        r"\newcommand{\PlectaDepthRuleAccFullCI}"
        r"{[\PlectaDepthRuleAccFullCILow,\PlectaDepthRuleAccFullCIHigh]}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    OUT_JSON.write_text(json.dumps({
        "role": "the evidence-rule replacement, as published",
        "computed_from": str(SRC.relative_to(REPO)),
        "computed": dict(computed),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT_TEX}  ({len(computed)} computed, 0 transcribed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
