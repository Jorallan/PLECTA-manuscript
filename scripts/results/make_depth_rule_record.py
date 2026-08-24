"""Emit the measurement that replaced the depth stage's evidence rule.

Why this exists. The supplementary section on the depth stage
(``sections/07_technical.tex``) quotes roughly twenty-five numbers: the two
measured defects of the rule that was removed, and the held-out comparison of
the two rules. Typed as literals they sit outside the repository's own gate
(``CLAUDE.md``: never write a measured number into the prose as a literal;
``docs/REPRODUCIBILITY.md``: every number in the article and the supplementary
is generated) and can drift away from the code silently. Two products, so they
cannot:

* ``results/plecta_depth_rule.json`` -- the machine-readable record;
* ``results/plecta_depth_rule.tex`` -- ``\\PlectaDepthRule*`` macros, ``\\input``
  by ``supplementary.tex``.

Scope. Only *measurements* live here. The rule's own tunable constants -- the
core arc, the sampling window, the minimum flank count, the abstention
threshold and the exact-solver limit -- are configuration, are read live from
``plecta/parameters.yaml`` by ``make_parameter_table.py``, and reach the prose
as ``\\PlectaParam*``. Retune one there and the supplementary follows on the next
run. Nothing is defined twice.

Provenance, stated plainly. The 2026-08-22 comparison of the two evidence rules
-- 20 optically rendered held-out scenes, 1355 crossings, seeds disjoint from
every set used to choose either rule or its constants -- wrote no
machine-readable record. Its results survive only as prose, in
``exploration/depth_order_eval/MEMO.md`` (note of 2026-08-22), in
``plecta/parameters.yaml``'s ``evidence_weights`` commentary, and in
``docs/internal/DEPTH_HANDOFF.md``. They are transcribed once, here, in
``MEASURED`` below, with that provenance carried into the JSON.

  TODO(depth-rule-record): re-derive ``MEASURED`` from the study's own output
  when the comparison is re-run and made to write a JSON, and delete the
  transcription. Until then this table is the single place any of these numbers
  is written down for the manuscript, which is the point: one place to correct
  instead of four.

Two entries are constants of the *removed* rule rather than measurements --
its fixed ``0.02`` denominator floor and the ``1.98`` score at which it became
structurally unable to abstain. They are kept here because no live
configuration holds them any more; they exist only as part of this record.

Nothing here is affected by the pending depth ablation rerun. That rerun
regenerates ``results/plecta_depth_order.json`` (Table 3), which was measured
under the *previous* rule; the numbers below already describe the shipped one.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "results" / "plecta_depth_rule.json"
OUT_TEX = REPO / "results" / "plecta_depth_rule.tex"

PROVENANCE = {
    "study": "depth evidence-rule replacement, measured 2026-08-22",
    "held_out_set": ("20 optically rendered synthetic scenes, 1355 crossings; "
                     "seeds disjoint from every set used to choose either rule "
                     "or its constants"),
    "prose_sources": [
        "exploration/depth_order_eval/MEMO.md, note of 2026-08-22",
        "C:/Repos/stubmatch/plecta/parameters.yaml, depth_3d.evidence_weights",
        "docs/internal/DEPTH_HANDOFF.md, note of 2026-08-22",
    ],
    "machine_readable_source": None,
    "note": ("transcribed, not computed here; see TODO(depth-rule-record) in "
             "this generator"),
}

MEASURED = (
    # (macro, value, printf format, what it is)
    ("PlectaDepthRuleHeldoutScenes", 20, "%d",
     "held-out optically rendered scenes"),
    ("PlectaDepthRuleHeldoutCrossings", 1355, "%d",
     "crossings in that held-out set"),

    # -- the previous rule's fixed denominator floor, and what it cost --
    ("PlectaDepthRuleLegacyFloor", 0.02, "%.2f",
     "grey levels; the previous rule's fixed denominator floor"),
    ("PlectaDepthRuleSaturationCut", 1.98, "%.2f",
     "|s| at or above which the previous rule was structurally unable to "
     "abstain"),
    ("PlectaDepthRuleSatSynthPct", 32.3, "%.1f",
     "per cent of synthetic crossings saturated by that floor"),
    ("PlectaDepthRuleSatRealPct", 61.5, "%.1f",
     "per cent of real crossings saturated by that floor"),
    ("PlectaDepthRuleSatBelowNoiseAcc", 0.70, "%.2f",
     "accuracy of saturated crossings whose separation fell below the noise"),
    ("PlectaDepthRuleSatAboveNoiseAcc", 0.93, "%.2f",
     "accuracy of saturated crossings whose separation cleared the noise"),

    # -- the removed gradient (sharpness) channel --
    ("PlectaDepthRuleSharpDecidedPct", 0.2, "%.1f",
     "per cent of real crossings the sharpness channel decided"),
    ("PlectaDepthRuleSharpKappa", 0.09, "+%.2f",
     "Cohen's kappa between the sharpness and the intensity channel"),
    ("PlectaDepthRuleSharpAccLow", 0.54, "%.2f",
     "low end of the sharpness channel's standalone accuracy"),
    ("PlectaDepthRuleSharpAccHigh", 0.59, "%.2f",
     "high end of the sharpness channel's standalone accuracy"),

    # -- what the replacement is worth, at matched coverage --
    ("PlectaDepthRuleAccDelta", 0.0025, "+%.4f",
     "accuracy difference, one-channel minus two-channel, matched coverage"),
    ("PlectaDepthRuleAccCILow", -0.0042, "%+.4f",
     "lower 95 per cent confidence bound on that difference"),
    ("PlectaDepthRuleAccCIHigh", 0.0079, "%+.4f",
     "upper 95 per cent confidence bound on that difference"),

    # -- the ordering of the stage's own confidence --
    ("PlectaDepthRuleAucOld", 0.7772, "%.4f",
     "AUC of correctness against confidence, previous rule"),
    ("PlectaDepthRuleAucNew", 0.8698, "%.4f",
     "AUC of correctness against |s|, shipped rule"),
    ("PlectaDepthRuleAucDelta", 0.0926, "+%.4f", "the gain in that AUC"),
    ("PlectaDepthRuleAucCILow", 0.0563, "%+.4f",
     "lower 95 per cent confidence bound on the AUC gain"),
    ("PlectaDepthRuleAucCIHigh", 0.1321, "%+.4f",
     "upper 95 per cent confidence bound on the AUC gain"),

    # -- what the better ordering converts into, given more abstention --
    ("PlectaDepthRuleCovAPct", 82.5, "%.1f", "the first coverage reported"),
    ("PlectaDepthRuleAccAtCovA", 0.0188, "+%.4f",
     "accuracy gain at that coverage"),
    ("PlectaDepthRuleCovBPct", 80, "%d", "the second coverage reported"),
    ("PlectaDepthRuleAccAtCovB", 0.0240, "+%.4f",
     "accuracy gain at that coverage"),

    # -- the axial channel that was built and measured inert --
    ("PlectaDepthRuleAxialDelta", -0.0008, "%+.4f",
     "accuracy difference from adding an axial continuity channel"),
    ("PlectaDepthRuleAxialCILow", -0.0059, "%+.4f",
     "lower 95 per cent confidence bound on that difference"),
    ("PlectaDepthRuleAxialCIHigh", 0.0048, "%+.4f",
     "upper 95 per cent confidence bound on that difference"),
)

#  Intervals the prose sets in one pair of brackets.
INTERVALS = (
    ("PlectaDepthRuleAccCI",
     "PlectaDepthRuleAccCILow", "PlectaDepthRuleAccCIHigh"),
    ("PlectaDepthRuleAucCI",
     "PlectaDepthRuleAucCILow", "PlectaDepthRuleAucCIHigh"),
    ("PlectaDepthRuleAxialCI",
     "PlectaDepthRuleAxialCILow", "PlectaDepthRuleAxialCIHigh"),
)

#  The honest headline, written once so every place that states it states the
#  same thing. Never "more accurate".
HEADLINE = ("the same accuracy with better-ordered confidence, from a simpler "
            "evidence path")


def main() -> int:
    record = {"measured": {}, "provenance": PROVENANCE, "headline": HEADLINE}
    lines = ["% Generated by scripts/results/make_depth_rule_record.py;"
             " do not edit.",
             "% The 2026-08-22 evidence-rule replacement study. Transcribed,"
             " not computed:",
             "% see the generator's TODO(depth-rule-record). The rule's tunable"
             " constants are",
             "% NOT here -- they are \\PlectaParam*, read live from"
             " plecta/parameters.yaml."]

    for macro, value, fmt, what in MEASURED:
        record["measured"][macro] = {"value": value, "what": what}
        lines.append(f"\\newcommand{{\\{macro}}}{{{fmt % value}}}")
    for macro, low, high in INTERVALS:
        lines.append(f"\\newcommand{{\\{macro}}}{{[\\{low},\\{high}]}}")

    OUT_JSON.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX} ({len(MEASURED)} transcribed values, "
          f"{len(INTERVALS)} interval macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
