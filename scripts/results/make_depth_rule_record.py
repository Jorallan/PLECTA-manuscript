"""Emit the measurement that replaced the depth stage's evidence rule.

Reads ``results/plecta_depth_rule_measured.json``, written by
``scripts/results/run_depth_rule_comparison.py``, and emits the
``\\PlectaDepthRule*`` macros the supplementary quotes.

These numbers used to be transcribed out of prose. The 2026-08-22 study that
justified the replacement wrote no machine-readable output, so roughly
twenty-five published quantities sat outside the repository's own gate and
could drift from the code silently. They are now recomputed from the held-out
scenes on every run.

Recomputing changed them, and the change is instructive rather than alarming.
The conclusion is unmoved: the same accuracy at matched coverage, with
markedly better-ordered confidence. The magnitudes moved because the earlier
figures were measured before several unrelated changes to ``plecta/depth.py``,
and the AUC in particular is not comparable across that drift. What did not
survive is the earlier accuracy delta's sign at *full* coverage; see the note
on ``accuracy_delta_at_full_coverage`` below, which is reported here precisely
because it is the reading that goes against the shipped rule.

Two quantities cannot be recomputed and stay transcribed, flagged as such in
the JSON. Both describe channels the change deleted -- the sharpness channel's
standalone behaviour, and the axial channel measured inert before removal --
so no configuration of the current code can produce them. They are diagnoses
of why the old rule went, not properties of the new one.

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

#  Constants of the REMOVED rule, and diagnoses of channels the change
#  deleted. No live configuration holds these and no run can reproduce them;
#  they exist only as the record of why the rule was replaced.
TRANSCRIBED = (
    ("PlectaDepthRuleLegacyFloor", "0.02",
     "the removed rule's fixed denominator floor"),
    ("PlectaDepthRuleSaturationCut", "1.98",
     "score at which the removed rule became unable to abstain"),
    ("PlectaDepthRuleSatSynthPct", "32.3",
     "per cent of synthetic crossings at that saturation"),
    ("PlectaDepthRuleSatRealPct", "61.5",
     "per cent of real crossings at that saturation"),
    ("PlectaDepthRuleSatBelowNoiseAcc", "0.70",
     "accuracy of saturated crossings below the noise floor"),
    ("PlectaDepthRuleSatAboveNoiseAcc", "0.93",
     "accuracy of saturated crossings above it"),
    ("PlectaDepthRuleSharpDecidedPct", "0.2",
     "per cent of crossings the removed sharpness channel decided alone"),
    ("PlectaDepthRuleSharpKappa", "+0.09",
     "its agreement with the intensity channel"),
    ("PlectaDepthRuleSharpAccLow", "0.54",
     "its standalone accuracy, low end"),
    ("PlectaDepthRuleSharpAccHigh", "0.59",
     "its standalone accuracy, high end"),
    ("PlectaDepthRuleAxialDelta", "-0.0008",
     "the axial channel, measured inert and not included"),
    ("PlectaDepthRuleAxialCILow", "-0.0059", "its CI, low"),
    ("PlectaDepthRuleAxialCIHigh", "+0.0048", "its CI, high"),
)


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
    ]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in computed]
    lines += [
        "% Transcribed, and not recomputable: these describe channels the",
        "% change deleted, so no configuration of the current code produces",
        "% them. They are the diagnosis of why the old rule went.",
    ]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v, _why in TRANSCRIBED]
    #  Convenience pairs the prose sets as one token.
    lines += [
        r"\newcommand{\PlectaDepthRuleAccCI}"
        r"{[\PlectaDepthRuleAccCILow,\PlectaDepthRuleAccCIHigh]}",
        r"\newcommand{\PlectaDepthRuleAucCI}"
        r"{[\PlectaDepthRuleAucCILow,\PlectaDepthRuleAucCIHigh]}",
        r"\newcommand{\PlectaDepthRuleAxialCI}"
        r"{[\PlectaDepthRuleAxialCILow,\PlectaDepthRuleAxialCIHigh]}",
        r"\newcommand{\PlectaDepthRuleAccFullCI}"
        r"{[\PlectaDepthRuleAccFullCILow,\PlectaDepthRuleAccFullCIHigh]}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    OUT_JSON.write_text(json.dumps({
        "role": "the evidence-rule replacement, as published",
        "computed_from": str(SRC.relative_to(REPO)),
        "computed": dict(computed),
        "transcribed": {n: {"value": v, "what": w} for n, v, w in TRANSCRIBED},
        "transcribed_note": "channels deleted by the change; no run of the "
                            "current code can reproduce these",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT_TEX}  ({len(computed)} computed, "
          f"{len(TRANSCRIBED)} transcribed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
