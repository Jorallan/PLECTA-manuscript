"""Generate PLECTA LaTeX macros and tables from machine-readable results.

The 128-scene held-out JSON and 100-row robustness CSV are publication-facing
numeric sources. The ablation JSON is development-only and is labelled as such
in every generated table.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HELDOUT = RESULTS / "plecta_heldout.json"
ABLATION = RESULTS / "plecta_ablation_dev.json"
ROBUSTNESS = RESULTS / "plecta_robustness.csv"
STAGE4 = RESULTS / "plecta_stage4_dev.json"
# plecta_real_masks.json is a superseded two-field record kept only as the
# parity anchor other record-makers cite; nothing here reads it any more.
CC_BASELINE = RESULTS / "plecta_cc_baseline.json"
WIDTH_VALIDATION = RESULTS / "plecta_width_validation.json"
GREEDY_BASELINE = RESULTS / "plecta_greedy_baseline.json"
DENSITY_FACTORIAL = RESULTS / "plecta_density_factorial.json"
DNAI = RESULTS / "plecta_dnai_comparison.json"
RUNTIME = RESULTS / "runtime_comparison.json"
#  OUR REIMPLEMENTATION of Basu, Liu & Rohde (TCBB 2015) Stage B. Optional:
#  if the record is absent the comparator table simply omits the column.
#  Written by comparisons/basu_comparison/scripts/make_manuscript_record.py.
BASU = RESULTS / "plecta_basu_reimplementation.json"
SIFNE = RESULTS / "plecta_sifne_comparison.json"
COMPARATOR_ANALYSIS = RESULTS / "plecta_comparator_analysis.json"
GREEDY_SWEPT = RESULTS / "plecta_greedy_swept_paired.json"
GRAFT = RESULTS / "plecta_graft_comparison.json"
METRICS_AUDIT = RESULTS / "plecta_metrics_audit.json"
METRICS_SUPPLEMENT = RESULTS / "plecta_metrics_supplement.json"
MASK_QUALITY = RESULTS / "plecta_mask_quality.json"
GRAFT_REGIME = RESULTS / "plecta_graft_regime.json"
REAL_FIELD_METRICS = RESULTS / "plecta_real_field_metrics.json"
REAL_FIELDS = RESULTS / "plecta_real_fields.json"
INTERANNOTATOR = RESULTS / "plecta_interannotator.json"
CURVINESS = RESULTS / "plecta_curviness_sensitivity.json"
DEPTH_ORDER = RESULTS / "plecta_depth_order.json"
METRIC_PANELS = RESULTS / "plecta_metric_panels.json"
SENSITIVITY = RESULTS / "development_sensitivity.json"
SCORER_SENSITIVITY = RESULTS / "plecta_scorer_sensitivity.json"
# The comparator studies live in a sibling repository; the default assumes
# the standard side-by-side checkout layout and --comparisons-root overrides it.
DEFAULT_COMPARISONS_ROOT = ROOT.parent / "comparisons"
CROSSING_CHANCE_RELATIVE = Path("metrics_study/results/crossing_fidelity_chance.json")
N_BOOT = 20_000
SEED = 20_260_810

METRICS = {
    "f1": "FOne",
    "precision": "Precision",
    "recall": "Recall",
    "fragment_recovery_recovery_rate": "Recovery",
    "adjusted_rand_index": "ARI",
    "vi_split_bits": "VISplit",
    "vi_merge_bits": "VIMerge",
}

DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def boot_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def stratified_boot_mean(rows: list[dict], key: str, rng: np.random.Generator) -> tuple[float, float]:
    groups: dict[int, np.ndarray] = {}
    for density in sorted({int(r["density"]) for r in rows}):
        groups[density] = np.asarray(
            [float(r[key]) for r in rows if int(r["density"]) == density],
            dtype=float,
        )
    means = np.zeros(N_BOOT, dtype=float)
    total = sum(len(v) for v in groups.values())
    for values in groups.values():
        idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
        means += values[idx].sum(axis=1) / total
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def macro(name: str, value: str) -> str:
    # TeX control-word names may contain letters only.
    command = "".join(DIGIT_WORDS.get(char, char)
                      for char in name if char.isalnum())
    if not command or not command.isalpha():
        raise ValueError(f"invalid generated macro name: {name!r} -> {command!r}")
    return f"\\newcommand{{\\{command}}}{{{value}}}"


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def greedy_macros(greedy: dict) -> list[str]:
    """Macros for the greedy-continuation comparison.

    Reported on the degraded condition, which is the primary input everywhere
    else in the paper; the clean condition is carried as a secondary check.
    """
    lines = []
    for variant, tag in (("degraded", ""), ("clean", "Clean")):
        block = greedy["conditions"][variant]["summary"]
        metrics = block["metrics"]
        f1 = metrics["f1"]
        lines.extend([
            macro(f"PlectaGreedy{tag}SceneCount", str(block["n_scenes"])),
            macro(f"PlectaGreedy{tag}PlectaFOne", fmt(f1["plecta_mean"])),
            macro(f"PlectaGreedy{tag}BaselineFOne", fmt(f1["greedy_mean"])),
            macro(f"PlectaGreedy{tag}DeltaFOne",
                  f"{f1['paired']['mean_difference']:+.3f}"),
            macro(f"PlectaGreedy{tag}DeltaCILow",
                  f"{f1['paired']['ci_low']:+.3f}"),
            macro(f"PlectaGreedy{tag}DeltaCIHigh",
                  f"{f1['paired']['ci_high']:+.3f}"),
            macro(f"PlectaGreedy{tag}WinCount",
                  str(f1["paired"]["plecta_better"])),
        ])
        for key, stem in (("fragment_recovery_recovery_rate", "Recovery"),
                          ("adjusted_rand_index", "ARI"),
                          ("vi_total_bits", "VITotal")):
            lines.extend([
                macro(f"PlectaGreedy{tag}Plecta{stem}",
                      fmt(metrics[key]["plecta_mean"])),
                macro(f"PlectaGreedy{tag}Baseline{stem}",
                      fmt(metrics[key]["greedy_mean"])),
            ])
    return lines



def dnai_macros(dnai: dict) -> list[str]:
    """Macros for the DNAi comparison.

    Aggregate paired intervals only. Per-density values are plain means with
    n = 10 and carry no interval, so none is emitted.
    """
    lines = []
    for variant, tag in (("degraded", ""), ("clean", "Clean")):
        block = dnai["paired"][variant]["plecta_minus_dnai_tuned__f1"]
        lines.extend([
            macro(f"PlectaDNAi{tag}SceneCount", str(block["n_paired_scenes"])),
            macro(f"PlectaDNAi{tag}PlectaFOne", fmt(block["mean_a"])),
            macro(f"PlectaDNAi{tag}FOne", fmt(block["mean_b"])),
            macro(f"PlectaDNAi{tag}DeltaFOne", f"{block['mean_diff']:+.3f}"),
            macro(f"PlectaDNAi{tag}DeltaCILow",
                  f"{block['ci95_lo_stratified']:+.3f}"),
            macro(f"PlectaDNAi{tag}DeltaCIHigh",
                  f"{block['ci95_hi_stratified']:+.3f}"),
            macro(f"PlectaDNAi{tag}WinCount", str(block["wins_a"])),
        ])
    shipped = dnai["per_method"]["degraded"]["dnai_default"]["f1"]["mean"]
    lines.append(macro("PlectaDNAiShippedFOne", fmt(shipped)))
    best = dnai["per_density"]["clean"]["dnai_tuned"]["20"]["f1"]["mean"]
    lines.append(macro("PlectaDNAiCleanSparseFOne", fmt(best)))
    return lines



def comparator_analysis_macros(analysis: dict, swept: dict) -> list[str]:
    """Macros for the comparator fairness analysis.

    The greedy baseline is reported at the configuration selected on the 84
    development scenes, not at the hand-set values it was first run with. The
    restricted comparison removes every pair a degree-limited, gap-blind method
    is structurally barred from attempting, which separates capability coverage
    from decision quality.
    """
    sweep = analysis["greedy_sweep"]
    paired = analysis["paired_full_vs_restricted"]
    evidence = analysis["junction_evidence"]

    selected = sweep["selected_configuration"]
    handset = sweep["hand_set_configuration"]
    lines = [
        macro("PlectaGreedyGapPx", f"{selected['max_gap_px']:.0f}"),
        macro("PlectaGreedyTurnDeg", f"{selected['max_turn_deg']:.0f}"),
        macro("PlectaGreedyHandsetGapPx", f"{handset['max_gap_px']:.0f}"),
        macro("PlectaGreedyHandsetTurnDeg", f"{handset['max_turn_deg']:.0f}"),
    ]

    for cond, tag in (("degraded", ""), ("clean", "Clean")):
        # The swept baseline's own per-scene rows, not the hand-set ones the
        # junction-evidence file happens to carry.
        sw = swept["conditions"][cond]
        lines.extend([
            macro(f"PlectaGreedySwept{tag}BaselineFOne",
                  fmt(sw["greedy_swept_mean"])),
            macro(f"PlectaGreedySwept{tag}DeltaFOne",
                  f"{sw['mean_diff']:+.3f}"),
            macro(f"PlectaGreedySwept{tag}DeltaCILow",
                  f"{sw['ci95_lo']:+.3f}"),
            macro(f"PlectaGreedySwept{tag}DeltaCIHigh",
                  f"{sw['ci95_hi']:+.3f}"),
            macro(f"PlectaGreedySwept{tag}WinCount",
                  str(sw["wins_plecta"])),
        ])
        res = paired[f"{cond}|restricted|plecta_minus_dnai_tuned"]
        full_ref = paired[f"{cond}|full|plecta_minus_dnai_tuned"]
        coverage = 1.0 - res["mean_diff"] / full_ref["mean_diff"]
        lines.extend([
            macro(f"PlectaDNAiRestricted{tag}DeltaFOne",
                  f"{res['mean_diff']:+.3f}"),
            macro(f"PlectaDNAiRestricted{tag}DeltaCILow",
                  f"{res['ci95_lo_stratified']:+.3f}"),
            macro(f"PlectaDNAiRestricted{tag}DeltaCIHigh",
                  f"{res['ci95_hi_stratified']:+.3f}"),
            macro(f"PlectaDNAiRestricted{tag}WinCount",
                  str(res["wins_plecta"])),
            macro(f"PlectaDNAiCoverageShare{tag}",
                  f"{coverage*100:.0f}"),
            macro(f"PlectaDNAiDecisionShare{tag}",
                  f"{(1-coverage)*100:.0f}"),
        ])
        census = evidence["per_variant"][cond]["census_pooled"]
        for key, stem in (("frac_high", "HighDegree"),
                          ("frac_gap_disconnected", "Gap")):
            if key in census:
                lines.append(macro(f"PlectaEvidence{tag}{stem}",
                                   f"{census[key]*100:.1f}"))
    return lines



def graft_macros(graft: dict) -> list[str]:
    """Macros for the GraFT comparison and the overlap measurement.

    The overlap block is the one place the paper reports what its own primary
    metric cannot see, so it is emitted even though no comparator number
    depends on it.
    """
    lines = []
    for variant, tag in (("degraded", ""), ("clean", "Clean")):
        f1 = graft["paired"][variant]["metrics"]["f1"]
        arm = graft["arms"][f"graft|{variant}"]
        lines.extend([
            macro(f"PlectaGraFT{tag}FOne", fmt(arm["overall"]["f1"])),
            macro(f"PlectaGraFT{tag}SceneCount", str(f1["n"])),
            macro(f"PlectaGraFT{tag}DeltaFOne", f"{f1['mean_diff']:+.3f}"),
            macro(f"PlectaGraFT{tag}DeltaCILow", f"{f1['ci95_lo']:+.3f}"),
            macro(f"PlectaGraFT{tag}DeltaCIHigh", f"{f1['ci95_hi']:+.3f}"),
            macro(f"PlectaGraFT{tag}WinCount", str(f1["plecta_wins"])),
            macro(f"PlectaGraFT{tag}Failed", str(arm["n_failed"])),
            macro(f"PlectaGraFT{tag}VISplit",
                  fmt(arm["overall"]["vi_split_bits"])),
            macro(f"PlectaGraFT{tag}VIMerge",
                  fmt(arm["overall"]["vi_merge_bits"])),
        ])
    native = graft["arms"].get("graft_native|degraded")
    if native:
        lines.append(macro("PlectaGraFTNativeFOne",
                           fmt(native["overall"]["f1"])))

    overlap = graft["overlap_cost"]
    low = overlap["by_density"]["20"]
    high = overlap["by_density"]["60"]
    lines.extend([
        macro("PlectaOverlapReferenceLow",
              f"{low['reference_shared_fraction']*100:.0f}"),
        macro("PlectaOverlapReferenceHigh",
              f"{high['reference_shared_fraction']*100:.0f}"),
        macro("PlectaOverlapExpressedLow",
              f"{low['plecta_shared_fraction']*100:.0f}"),
        macro("PlectaOverlapExpressedHigh",
              f"{high['plecta_shared_fraction']*100:.0f}"),
        macro("PlectaOverlapGraFTHigh",
              f"{high['graft_shared_fraction']*100:.0f}"),
        macro("PlectaOverlapFlattenCost",
              f"{overlap['pooled']['plecta_f1_cost_of_flattening_max']:.4f}"),
    ])
    return lines


def matcher_ablation_macros(audit: dict) -> list[str]:
    """Macros for the matcher-only ablation.

    PLECTA's own pipeline with the per-junction maximum-weight matching replaced
    by greedy cheapest-first acceptance and nothing else altered: same graph,
    same cost, same admissibility gate, same round schedule. Development scenes
    only, and the greedy arm runs at parameters selected with the exact matcher
    in the loop, so the comparison is biased in the exact matcher's favour.
    """
    parity = audit["provenance"]["ablation_control_parity"]["degraded"]
    lines = [
        macro("PlectaMatcherParitySceneCount", str(parity["n_scenes"])),
        macro("PlectaMatcherParityIdentical", str(parity["n_identical"])),
    ]
    for variant, tag in (("degraded", ""), ("clean", "Clean")):
        block = audit["matcher_ablation"][variant]
        exact = block["arms"]["exact"]
        greedy = block["arms"]["greedy_junction"]
        both = block["arms"]["greedy_both"]
        paired = block["paired_vs_exact"]["exact_minus_greedy_junction"]["f1"]
        lines.extend([
            macro(f"PlectaMatcher{tag}SceneCount", str(exact["n_scenes"])),
            macro(f"PlectaMatcher{tag}ExactFOne", fmt(exact["f1"])),
            macro(f"PlectaMatcher{tag}GreedyFOne", fmt(greedy["f1"])),
            macro(f"PlectaMatcher{tag}DeltaFOne", f"{paired['mean']:+.3f}"),
            macro(f"PlectaMatcher{tag}DeltaCILow", f"{paired['ci_lo']:+.3f}"),
            macro(f"PlectaMatcher{tag}DeltaCIHigh", f"{paired['ci_hi']:+.3f}"),
            macro(f"PlectaMatcher{tag}WinCount", str(paired["n_wins_a"])),
            macro(f"PlectaMatcher{tag}ExactVISplit", fmt(exact["vi_split_bits"])),
            macro(f"PlectaMatcher{tag}GreedyVISplit", fmt(greedy["vi_split_bits"])),
            macro(f"PlectaMatcher{tag}ExactVIMerge", fmt(exact["vi_merge_bits"])),
            macro(f"PlectaMatcher{tag}GreedyVIMerge", fmt(greedy["vi_merge_bits"])),
            macro(f"PlectaMatcher{tag}ExactJunctionRate", fmt(exact["jr_exact_rate"])),
            macro(f"PlectaMatcher{tag}GreedyJunctionRate", fmt(greedy["jr_exact_rate"])),
            macro(f"PlectaMatcher{tag}InstanceExcess",
                  f"{(greedy['n_pred_instances'] / exact['n_pred_instances'] - 1) * 100:.0f}"),
            # Making the global gap matching greedy as well, on top of the
            # junction matcher: the further cost is what locates the effect at
            # the junction rather than in matching generally.
            macro(f"PlectaMatcher{tag}GapAlsoGreedyCost",
                  f"{both['f1'] - greedy['f1']:+.3f}"),
        ])

    by_degree = audit["matcher_ablation"]["degraded"]["by_degree"]
    for degree in ("3", "4", "5", "6"):
        exact_rate = by_degree["exact"][degree]["exact_rate"]
        greedy_rate = by_degree["greedy_junction"][degree]["exact_rate"]
        stem = DIGIT_WORDS[degree]
        lines.extend([
            macro(f"PlectaMatcherDegree{stem}Count",
                  str(by_degree["exact"][degree]["n_junctions"])),
            macro(f"PlectaMatcherDegree{stem}Exact", fmt(exact_rate)),
            macro(f"PlectaMatcherDegree{stem}Greedy", fmt(greedy_rate)),
            macro(f"PlectaMatcherDegree{stem}Absolute",
                  f"{greedy_rate - exact_rate:+.3f}"),
            macro(f"PlectaMatcherDegree{stem}Relative",
                  f"{(greedy_rate - exact_rate) / exact_rate * 100:+.0f}"),
        ])
    return lines


def junction_resolution_macros(audit: dict, chance: dict) -> list[str]:
    """Macros for Crossing Fidelity, the measure that states the claim.

    At every reference crossing the partition of incident arms induced by the
    prediction is compared with the reference's; Crossing Fidelity is the
    fraction resolved identically. Pooled over crossings, never averaged over
    per-scene ratios, because a dense scene carries several times as many
    crossing decisions as a sparse one.

    **The chance level is measured, not inherited.** An earlier draft printed
    ``null_pair_accuracy`` here -- 0.709 -- beside the sentence "leaving every
    arm unpaired already resolves this fraction of crossings". That is the
    chance level of *pair accuracy*, which is a different statistic and is no
    longer reported. The do-nothing prediction's own exact rate was measured by
    running the stored scorer with an empty prediction, and it is 0.088 on the
    degraded set and 0.080 on the clean one -- an order of magnitude lower, so
    the earlier number understated the result rather than flattering it.

    The chance-corrected resolution index is not emitted. Over the real-field
    rows it correlates with the exact rate at Pearson 0.997 and never went
    negative, so it restated a reported number and its one distinguishing
    property never fired.
    """
    names = {"plecta": "Plecta", "greedy_swept": "Greedy",
             "dnai": "DNAi", "graft": "GraFT"}
    lines = []
    for variant, tag in (("degraded", ""), ("clean", "Clean")):
        block = audit["junction_resolution"][variant]
        degrees = block["reference_degree_distribution"]
        total = sum(int(v) for v in degrees.values())
        lines.extend([
            macro(f"PlectaJunction{tag}Count", str(total)),
            macro(f"PlectaJunction{tag}Chance",
                  fmt(chance["summary"][variant]["null_exact_rate_pooled"])),
            macro(f"PlectaJunction{tag}DegreeFourShare",
                  f"{int(degrees['4']) / total * 100:.0f}"),
        ])
        for key, stem in names.items():
            overall = block[key]["overall"]
            lines.extend([
                macro(f"PlectaJunction{tag}{stem}Rate", fmt(overall["exact_rate"])),
                macro(f"PlectaJunction{tag}{stem}DegreeFour",
                      fmt(block[key]["by_degree"]["4"]["exact_rate"])),
                macro(f"PlectaJunction{tag}{stem}DegreeFivePlus",
                      fmt(block[key]["by_degree"]["5plus"]["exact_rate"])),
            ])
        for key, stem in (("greedy_swept", "Greedy"), ("dnai", "DNAi"),
                          ("graft", "GraFT")):
            paired = audit["paired_differences"][variant][
                f"plecta_minus_{key}"]["jr_exact_rate"]
            lines.extend([
                macro(f"PlectaJunction{tag}{stem}Delta", f"{paired['mean']:+.3f}"),
                macro(f"PlectaJunction{tag}{stem}DeltaCILow",
                      f"{paired['ci_lo']:+.3f}"),
                macro(f"PlectaJunction{tag}{stem}DeltaCIHigh",
                      f"{paired['ci_hi']:+.3f}"),
            ])
    return lines


def graft_frame_macros(audit: dict, supplement: dict) -> list[str]:
    """Macros for GraFT's own bundled synthetic frame.

    One 500x500 frame with per-filament ground truth, MIT licensed, scored
    through the identical pipeline. n = 1: no interval exists and none is
    emitted. Its only claim is that the comparison does not depend on our
    generator.
    """
    frame = audit["graft_own_data_feasibility"]
    methods = frame["methods"]
    provenance = supplement["graft_native_frame"]
    crossings = methods["plecta"]["jr_n_junctions"]
    lines = [
        macro("PlectaGraFTFrameSide", str(provenance["side_px"])),
        macro("PlectaGraFTFrameInstanceCount", str(frame["n_gt_instances"])),
        macro("PlectaGraFTFrameCrossingCount", str(crossings)),
        macro("PlectaGraFTFrameReferenceAgreement",
              fmt(provenance["reference_matches_vendor_nonoise_image"])),
    ]
    for key, stem in (("plecta", "Plecta"), ("greedy_swept", "Greedy"),
                      ("graft_frontend", "GraFTNative"),
                      ("graft_a160", "GraFTInjected"), ("cc_floor", "Floor")):
        block = methods[key]
        lines.extend([
            macro(f"PlectaGraFTFrame{stem}FOne", fmt(block["f1"])),
            macro(f"PlectaGraFTFrame{stem}Solved",
                  f"{block['jr_exact_rate'] * crossings:.0f}"),
        ])
    return lines


def shared_ownership_macros(audit: dict, supplement: dict) -> list[str]:
    """Macros for how the shared-pixel representation is drawn.

    Reported because the paper presents overlap-aware output as a property of
    the method: PLECTA recovers just under half of the reference's
    jointly-owned pixels -- the largest such recall among the compared methods,
    not a majority -- at low precision, marking several times more of them than
    the reference contains, the latter being a rendering choice rather than a
    grouping error. The comparator recalls are emitted so the prose can rank
    without hand-writing a number.
    """
    lines = []
    for variant, tag in (("degraded", ""), ("clean", "Clean")):
        overall = audit["methods"][variant]["plecta"]["overall"]
        pixels = supplement["shared_ownership_pixels"][variant]["plecta"]
        lines.extend([
            macro(f"PlectaShared{tag}Precision", fmt(overall["shared_precision"])),
            macro(f"PlectaShared{tag}Recall", fmt(overall["shared_recall"])),
            macro(f"PlectaShared{tag}Excess",
                  f"{pixels['predicted_over_reference']:.1f}"),
        ])
    for method, stem in (("graft", "GraFT"), ("dnai", "DNAi")):
        overall = audit["methods"]["degraded"][method]["overall"]
        lines.append(
            macro(f"PlectaShared{stem}Recall", fmt(overall["shared_recall"])))
    return lines


TOL_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three"}


def mask_quality_macros(quality: dict) -> list[str]:
    """Macros for the tolerant completeness / correctness / quality triple.

    Only the synthetic reference arm is emitted here.  The per-field triple
    moved to ``real_fields_macros`` when the third annotated field arrived,
    because the three fields have to be built by one path or the two-field and
    three-field numbers drift apart; this record now covers the synthetic
    comparison alone, which is what the section uses it for.
    """
    lines = []
    synthetic = quality["synthetic_reference"]
    for row in synthetic["by_tolerance"]:
        word = TOL_WORDS[row["tol_px"]]
        lines.append(macro(f"PlectaMaskQualitySynthetic{word}",
                           fmt(row["quality"])))
    lines.extend([
        macro("PlectaMaskQualitySyntheticSceneCount",
              str(synthetic["n_scenes"])),
        macro("PlectaMaskQualityGraFTAsPublished",
              fmt(synthetic["graft_jaccard_as_published"])),
    ])
    return lines


def real_field_metrics_macros(metrics: dict, tolerance: int) -> list[str]:
    """The tolerance control, and the centreline-Dice arm that was rejected.

    What survives here after the three-field rebuild is the evidence that the
    placement tolerance is a correction and not an inflation: the detection
    score is emitted at 0 px and at the manuscript's own tolerance for *both*
    axes, and the manual-derived pair is identical, because on that axis the
    input mask is a skeleton of the annotation itself and there is nothing
    displaced to forgive. A tolerance that raised every number would prove
    nothing. The reconstruction scores themselves come from
    ``plecta_real_fields.json``, which covers all three fields; this record
    covers the two it was written for, which is all the control needs.
    """
    lines = [macro("PlectaRealJoinReach", "%.0f" % metrics["join_reach_px"])]
    for c in metrics["conditions"]:
        stem = c["image"].replace("_", "")
        axis = "Manual" if c["input_axis"] == "manual-derived" else "UNet"
        lines.extend([
            macro(f"PlectaDetection{stem}{axis}Strict",
                  fmt(c["detection_f1_strict"])),
            macro(f"PlectaDetection{stem}{axis}Tolerant",
                  fmt(c["detection_by_tolerance"][str(tolerance)])),
            macro(f"PlectaClDice{stem}{axis}Strict", fmt(c["cldice_f1_strict"])),
            macro(f"PlectaClDice{stem}{axis}Widened",
                  fmt(c["cldice_f1_widened"])),
        ])
    return lines


def interannotator_macros(payload: dict) -> list[str]:
    """The two-annotator study, as ranges over the three paired fields.

    Every cross-annotator number is emitted beside the annotator-versus-annotator
    agreement it must be read against. Quoting the collapse without the
    ceiling would misattribute a disagreement between two humans about what
    an instance *is* to a failure of the method, and the pairing here is what
    stops the prose doing that.
    """
    spans = payload["spans"]

    def pair(name: str, low_key: str, high_key: str) -> list[str]:
        return [macro(f"PlectaInterAnnotator{name}Low", fmt(spans[low_key]["min"])),
                macro(f"PlectaInterAnnotator{name}High", fmt(spans[high_key]["max"]))]

    lines = [macro("PlectaInterAnnotatorFieldCount", str(payload["n_paired_fields"]))]
    lines += pair("SameF", "same_reader_f1", "same_reader_f1_high")
    lines += pair("CrossF", "cross_reader_f1", "cross_reader_f1_high")
    lines += pair("AgreeF", "reader_partition_f1", "reader_partition_f1_high")
    lines += pair("NetF", "nnunet_f1", "nnunet_f1_high")
    lines += [
        macro("PlectaInterAnnotatorObjectRatioLow",
              fmt(spans["object_ratio"]["min"], 2)),
        macro("PlectaInterAnnotatorObjectRatioHigh",
              fmt(spans["object_ratio"]["max"], 2)),
        macro("PlectaInterAnnotatorComponentRatioLow",
              "%.0f" % spans["component_ratio"]["min"]),
        macro("PlectaInterAnnotatorComponentRatioHigh",
              "%.0f" % spans["component_ratio"]["max"]),
    ]
    return lines


def basu_macros(basu: dict | None) -> list[str]:
    """Our reimplementation of Basu et al.'s Stage B, reported as a floor.

    Emitted only if the record exists, so the manuscript degrades gracefully to
    no Basu column and no Basu prose if the study is ever withdrawn. Every one
    of these is OUR code, not the authors', and Section 3.3 and the Table 3
    caption both say so; the macro names carry no such marker, so they must
    never be quoted without the surrounding qualification.
    """
    if not basu:
        return []
    o = basu["overall"]
    ratio = (float(np.mean([r["n_filaments"] for r in basu["per_scene"]]))
             / float(np.mean([r.get("ref_instances", float("nan"))
                              for r in basu["per_scene"]]))
             if basu["per_scene"] and basu["per_scene"][0].get("ref_instances")
             else None)
    lines = [
        macro("PlectaBasuFOne", fmt(o["f1"])),
        macro("PlectaBasuARI", fmt(o["adjusted_rand_index"])),
        macro("PlectaBasuVITotal", fmt(o["vi_total_bits"])),
        macro("PlectaBasuRecovery", fmt(o["fragment_recovery_recovery_rate"])),
        macro("PlectaBasuCrossingFidelity",
              fmt(basu["junction_resolution"]["exact_rate"])),
        macro("PlectaBasuDelta", "%g" % basu["params"]["delta"]),
        macro("PlectaBasuTheta", "%g" % basu["params"]["theta_deg"]),
    ]
    if ratio:
        lines.append(macro("PlectaBasuInstanceRatio", "%.1f" % ratio))
    return lines


def sifne_macros(sifne: dict | None) -> list[str]:
    r"""SIFNE, run from its own released source, so a genuine external comparator.

    Emitted only if the record exists, so the manuscript degrades gracefully to
    no SIFNE column and no SIFNE prose if the study is ever withdrawn.

    Unlike the Basu macros these ARE an externally run result and may be quoted
    as one. The single qualification they carry is tuning: SIFNE's shipped
    parameters excise a square at every crossing pixel and discard between a
    third and two thirds of the input mask at these densities, so three of them
    were re-selected on development scenes exactly as DNAi's linkage distance
    was. \PlectaSifneShippedFOne is the shipped-default score and exists so the
    tuned figure is never quoted without it available beside it.
    """
    if not sifne:
        return []
    o = sifne["overall"]
    p = sifne["tuned_params"]
    lines = [
        macro("PlectaSifneFOne", fmt(o["f1"])),
        macro("PlectaSifneARI", fmt(o["adjusted_rand_index"])),
        macro("PlectaSifneVITotal", fmt(o["vi_total_bits"])),
        macro("PlectaSifneRecovery", fmt(o["fragment_recovery_recovery_rate"])),
        macro("PlectaSifneCrossingFidelity",
              fmt(sifne["junction_resolution"]["exact_rate"])),
        macro("PlectaSifneSceneCount", str(sifne["n_scenes"])),
        macro("PlectaSifneJuncSize", "%g" % p["Size_Junc"]),
        macro("PlectaSifneFanR", "%g" % p["FanR"]),
        macro("PlectaSifneShippedJuncSize",
              "%g" % sifne["shipped_params"]["Size_Junc"]),
    ]
    if sifne.get("shipped_default_f1") == sifne.get("shipped_default_f1"):
        lines.append(macro("PlectaSifneShippedFOne",
                           fmt(sifne["shipped_default_f1"])))

    #  Arm 2, SIFNE's own published topology REBUILT BY US. Weaker standing
    #  than the GraFT regime arm, where that generator was vendored and run
    #  unmodified: SIFNE released none, so this is our reconstruction of a
    #  topology described in words. The macro names carry no marker, so the
    #  prose that quotes them must say so.
    own = sifne.get("own_topology", {}).get("groups", {})
    for tag, key in (("Sparse", "web_sparse"), ("Dense", "web_dense")):
        g = own.get(key)
        if not g:
            continue
        lines += [
            macro(f"PlectaSifneOwn{tag}Density", "%.3f" % g["density"]),
            macro(f"PlectaSifneOwn{tag}PlectaFOne", fmt(g["plecta_f1"])),
            macro(f"PlectaSifneOwn{tag}SifneFOne", fmt(g["sifne_f1"])),
        ]
    return lines


def depth_order_macros(payload: dict) -> list[str]:
    """The depth stage, emitted so the decomposition cannot be split up.

    The two spans are the result: the over/under decision barely moves between
    the oracle and the reconstructed input, while the all-crossings score moves
    a great deal, and it moves with the crossing match rate. Quoting the second
    without the first would read as a depth failure when it is a 2-D recovery
    failure, so both spans and the match rate are emitted together.
    """
    h = payload["headline"]
    cell = {(g["condition"], g["coverage"]): g for g in payload["grid"]}
    lines = [
        macro("PlectaDepthChance", "%.1f" % payload["chance_level"]),
        macro("PlectaDepthSceneCount",
              str(sum(g["n_scenes"] for g in payload["grid"])
                  // len({g["condition"] for g in payload["grid"]}))),
        macro("PlectaDepthDecidedLow", fmt(h["coa_decided_min"])),
        macro("PlectaDepthDecidedHigh", fmt(h["coa_decided_max"])),
        macro("PlectaDepthDecidedSpan", fmt(h["coa_decided_span"])),
        macro("PlectaDepthAllLow", fmt(h["coa_all_min"])),
        macro("PlectaDepthAllHigh", fmt(h["coa_all_max"])),
        macro("PlectaDepthAllSpan", fmt(h["coa_all_span"])),
        macro("PlectaDepthMatchLow", fmt(h["match_rate_min"])),
        macro("PlectaDepthMatchHigh", fmt(h["match_rate_max"])),
        macro("PlectaDepthAbstainLow", "%.0f" % (h["abstain_rate_min"] * 100)),
        macro("PlectaDepthAbstainHigh", "%.0f" % (h["abstain_rate_max"] * 100)),
        macro("PlectaDepthWithinLow", fmt(h["frac_within_min"])),
        macro("PlectaDepthWithinHigh", fmt(h["frac_within_max"])),
        macro("PlectaDepthOrderResidual", fmt(h["order_residual_absmax"])),
    ]
    for (cond, cov), g in cell.items():
        stem = f"{cond.capitalize()}{cov.replace('cov', 'Cov')}"
        lines.extend([
            macro(f"PlectaDepth{stem}All", fmt(g["coa_all"])),
            macro(f"PlectaDepth{stem}Decided", fmt(g["coa_decided"])),
            macro(f"PlectaDepth{stem}Match", fmt(g["match_rate"])),
            macro(f"PlectaDepth{stem}CrossOrder",
                  fmt(g["order_acc_crossing_pairs"])),
            macro(f"PlectaDepth{stem}AllOrder", fmt(g["order_acc_all_pairs"])),
            macro(f"PlectaDepth{stem}Within",
                  fmt(g["frac_pairs_within_component"])),
            macro(f"PlectaDepth{stem}OrderPred",
                  fmt(g["order_acc_all_pairs_predicted"])),
            macro(f"PlectaDepth{stem}Layers", fmt(g["layer_exact_agreement"])),
        ])
    return lines


def depth_order_table(payload: dict) -> str:
    """The 2x2 grid, generated so no number is retyped into the source."""
    cell = {(g["condition"], g["coverage"]): g for g in payload["grid"]}
    covs = sorted({g["coverage"] for g in payload["grid"]})
    rows = [
        ("Projected crossings per scene", "n_gt_crossings", "%.0f"),
        ("Crossings recovered", "match_rate", "%.3f"),
        ("Order accuracy, all crossings", "coa_all", "%.3f"),
        ("Order accuracy, decided only", "coa_decided", "%.3f"),
        ("Abstained", "abstain_rate", "%.3f"),
        ("Depth order, crossing pairs", "order_acc_crossing_pairs", "%.3f"),
        ("Depth order, all pairs", "order_acc_all_pairs", "%.3f"),
        ("Instance pairs sharing a component", "frac_pairs_within_component",
         "%.3f"),
        ("Layer agreement", "layer_exact_agreement", "%.3f"),
    ]
    head = " & ".join(c.replace("cov", "") + r"\,\%" for c in covs)
    out = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Given 2-D geometry} "
        r"& \multicolumn{2}{c}{Reconstructed 2-D} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        "Areal coverage & " + head + " & " + head + r" \\",
        r"\midrule",
    ]
    for label, key, fmt_s in rows:
        vals = [fmt_s % cell[(c, cov)][key]
                for c in ("oracle", "clean") for cov in covs]
        out.append(label + " & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def curviness_macros(payload: dict) -> list[str]:
    """Bending-stiffness sensitivity, as an effect size beside its own noise.

    The two numbers that matter are emitted together and are meant to be
    quoted together: the largest spread of level means across the sweep, and
    the smallest scene-to-scene standard deviation within any single level.
    When the first is below the second there is no trend to report, and
    printing the spread on its own would invite exactly the reading the sweep
    was run to rule out.
    """
    head = payload["headline"]
    w = payload["wavelength"]
    wh = w["headline"]
    low, high = payload["swept_fraction_of_default"]
    geo = {g["curviness"]: g for g in payload["geometry"]}
    lo_g, hi_g = geo[min(geo)], geo[max(geo)]
    return [
        macro("PlectaCurvinessDefault", "%.3f" % payload["default_curviness"]),
        macro("PlectaCurvinessSweepLow", "%.0f" % (low * 100)),
        macro("PlectaCurvinessSweepHigh", "%.0f" % (high * 100)),
        macro("PlectaCurvinessRadiusLow", "%.0f" % hi_g["median_radius_px"]),
        macro("PlectaCurvinessRadiusHigh", "%.0f" % lo_g["median_radius_px"]),
        macro("PlectaCurvinessBendHigh",
              "%.1f" % hi_g["p95_local_bend_20px_deg"]),
        macro("PlectaCurvinessMaxSpread",
              fmt(head["max_spread_of_level_means"])),
        macro("PlectaCurvinessMinSceneSD", fmt(head["min_within_level_sd"])),
        macro("PlectaCurvinessSceneCount",
              str(payload["series"][0]["per_level"][0]["n"])),
        macro("PlectaCurvinessLevelCount", str(len(payload["geometry"]))),
        # The wavelength axis, reported beside the amplitude one.
        macro("PlectaSmoothDefault", "%.0f" % w["default_px"]),
        macro("PlectaSmoothLow", "%.0f" % min(w["levels_px"])),
        macro("PlectaSmoothHigh", "%.0f" % max(w["levels_px"])),
        macro("PlectaSmoothMaxSpread", fmt(wh["max_spread_of_level_means"])),
        macro("PlectaSmoothMinSceneSD", fmt(wh["min_within_level_sd"])),
        macro("PlectaSmoothFlatSeries",
              str(wh["n_series"] - wh["n_series_where_spread_exceeds_noise"])),
        macro("PlectaSmoothSeriesCount", str(wh["n_series"])),
    ]


def real_fields_macros(payload: dict) -> list[str]:
    """The three annotated fields under the five reported measures.

    One placement tolerance is emitted, ``PlectaPlacementTolerance``, and every
    tolerance-bearing number in the manuscript reads it. There were two before
    the third field arrived -- 3 px for the axis triple and 2 px for the
    detection score -- for no reason beyond the order the two studies were
    written in, and a reader has no way to tell that from the page.
    """
    tolerance = payload["fields"][0]["axis_triple"]["reported_tolerance_px"]
    n_fields = len(payload["fields"])
    #  Spelled out as well as in digits, because two sentences in the
    #  manuscript open on this count and a sentence does not begin with a
    #  numeral.  DIGIT_WORDS covers one digit; past nine, decide how a
    #  two-digit count should read before extending this.
    if n_fields > 9:
        raise ValueError(
            f"{n_fields} annotated fields: no spelled-out form is defined "
            "past nine, so PlectaRealFieldCountWord would be wrong")
    word = DIGIT_WORDS[str(n_fields)]
    lines = [macro("PlectaPlacementTolerance", str(tolerance)),
             macro("PlectaRealFieldCount", str(n_fields)),
             macro("PlectaRealFieldCountWord", word),
             macro("PlectaRealFieldCountLower", word.lower())]
    #  PlectaRealHeldOutCount retired 2026-08-25: the manuscript no longer
    #  states which fields the upstream segmenter had seen. `unet_held_out`
    #  survives in plecta_real_fields.json, so restoring the macro is one line.

    total_crossings = 0
    for field in payload["fields"]:
        stem = field["image"].replace("_", "")
        triple = field["axis_triple"]
        lines.extend([
            macro(f"Plecta{stem}ReferenceCount", str(field["n_reference"])),
            macro(f"Plecta{stem}AxisRecall", fmt(triple["completeness"])),
            macro(f"Plecta{stem}AxisPrecision", fmt(triple["correctness"])),
        ])
        for row in triple["by_tolerance"]:
            word = TOL_WORDS[row["tol_px"]]
            lines.append(macro(f"PlectaMaskQuality{stem}Quality{word}",
                               fmt(row["quality"])))
        for key, axis in (("manual", "Manual"), ("unet", "UNet")):
            c = field["conditions"][key]
            total_crossings += c["n_crossings"]
            lines.extend([
                macro(f"Plecta{stem}{axis}FOne", fmt(c["pairwise_f1"])),
                macro(f"Plecta{stem}{axis}Precision",
                      fmt(c["pairwise_precision"])),
                macro(f"Plecta{stem}{axis}Recall", fmt(c["pairwise_recall"])),
                macro(f"PlectaJoin{stem}{axis}FOne", fmt(c["join_f1"])),
                macro(f"PlectaJoin{stem}{axis}Precision",
                      fmt(c["join_precision"])),
                macro(f"PlectaJoin{stem}{axis}Recall", fmt(c["join_recall"])),
                macro(f"PlectaJoin{stem}{axis}Decisions",
                      "%d" % c["n_decisions"]),
                macro(f"PlectaDetectionF{stem}{axis}", fmt(c["detection_f1"])),
                macro(f"PlectaFidelity{stem}{axis}",
                      fmt(c["crossing_fidelity"])),
                macro(f"PlectaFidelity{stem}{axis}Chance",
                      fmt(c["crossing_fidelity_chance"])),
                macro(f"PlectaCrossingCount{stem}{axis}",
                      "%d" % c["n_crossings"]),
            ])
    lines.append(macro("PlectaRealCrossingTotal", "%d" % total_crossings))

    #  The PlectaContamination* macros are RETIRED, 2026-08-25, by author
    #  decision, together with the results subsection that used them.
    #
    #  They reported what training on a field is worth, measured by running
    #  several nnU-Net models over all three fields and comparing a field
    #  scored under models that excluded it against one that did not. That
    #  experiment is not deleted: `payload["contamination"]` is still written
    #  by scripts/results/make_real_fields_record.py and still carries every
    #  number, so reinstating the block below restores the macros without
    #  re-running anything.
    #
    #  Emitting them while nothing cites them would leave ten dead macros in
    #  plecta_results.tex, so they are not emitted.
    return lines


def metric_panels_macros(panels: dict) -> list[str]:
    """The two ends of each comparison curve, so the prose can cite the figure.

    Only the sparsest and densest strata are emitted. The intermediate points
    are on the page already, in the figures, and a macro for every cell would
    be thirty numbers nobody cites.
    """
    lines = [macro("PlectaDetectionIoU", "%.1f" % panels["iou_threshold"]),
             macro("PlectaPanelCrossingChance",
                   fmt(panels["chance"]["degraded"]["pooled"])),
             macro("PlectaPanelCleanCrossingChance",
                   fmt(panels["chance"]["clean"]["pooled"]))]
    words = {"f1": "FOne", "detection_f1": "Detection",
             "crossing_fidelity": "Fidelity", "vi_total_bits": "VITotal",
             "ari": "ARI"}
    ends = (("Low", str(panels["densities"][0])),
            ("High", str(panels["densities"][-1])))
    for condition in ("degraded", "clean"):
        stem = condition.title()
        for method, name in (("plecta", "Plecta"), ("graft", "GraFT"),
                             ("dnai", "DNAi")):
            for end, density in ends:
                cell = panels["panels"][condition][method][density]
                for key, word in words.items():
                    lines.append(macro(f"PlectaPanel{stem}{name}{end}{word}",
                                       fmt(cell[key])))
    #: The empirical case for reporting ARI in one sentence rather than in
    #: every table: it never leaves pairwise F1 by more than this.
    gap = max(abs(cell["ari"] - cell["f1"])
              for condition in ("degraded", "clean")
              for method in ("plecta", "graft", "dnai")
              for cell in panels["panels"][condition][method].values())
    lines.append(macro("PlectaPanelARIMaxGap", fmt(gap)))

    #: How far apart the three methods lie at the densest stratum, under the
    #: measure that ranks them and under the one that states the claim.  The
    #: second is the argument for not ranking by the second.
    dense = str(panels["densities"][-1])
    for key, name in (("crossing_fidelity", "PlectaJunctionSpreadHigh"),
                      ("f1", "PlectaPairwiseSpreadHigh")):
        values = [panels["panels"]["degraded"][m][dense][key]
                  for m in ("plecta", "graft", "dnai")]
        lines.append(macro(name, fmt(max(values) - min(values))))
    return lines


def sensitivity_macros(sensitivity: dict) -> list[str]:
    """What the thickness of the input axis mask costs, on development scenes.

    This replaces a retired figure. It is the one density axis that moves the
    score: the rendered silhouette does not (Section 3.2, causally, on
    byte-identical masks), and this does.
    """
    paired = sensitivity["paired"]["w3_minus_w1"]
    return [
        macro("PlectaSensitivityWidthSceneCount", str(paired["n"])),
        macro("PlectaSensitivityWidthLow", "1"),
        macro("PlectaSensitivityWidthHigh", "3"),
        macro("PlectaSensitivityWidthDeltaFOne", "%+.3f" % paired["mean"]),
        macro("PlectaSensitivityWidthDeltaCILow", fmt(paired["ci_lo"])),
        macro("PlectaSensitivityWidthDeltaCIHigh", fmt(paired["ci_hi"])),
    ]


def graft_regime_macros(regime: dict) -> list[str]:
    """Macros for the run on GraFT's own generator.

    Two metric families are emitted for the two ends of the density ladder,
    because the comparison's point is that they disagree: the manuscript's
    pairwise endpoint ranks the methods one way and GraFT's own filament
    matched coverage ranks them the other, on the identical predictions.
    """
    strata = regime["by_stratum"]
    lo, hi = strata[0], strata[-1]
    lines = [
        macro("PlectaRegimeSceneCount",
              str(sum(s["n_scenes"] for s in strata))),
        macro("PlectaRegimeStratumCount", str(len(strata))),
        macro("PlectaRegimeDensityLow", "%.4f" % lo["density_mean"]),
        macro("PlectaRegimeDensityHigh", "%.4f" % hi["density_mean"]),
        macro("PlectaRegimeDensityPublished", "0.020"),
        #  GraFT's own front end, run end to end on the noisy image, is
        #  reported as a sentence rather than a curve: it is a fourth arm of
        #  one comparator and it changes no ordering.
        macro("PlectaRegimeLowGraFTNativeFOne",
              fmt(lo["methods"]["graft_native"]["f1"])),
        macro("PlectaRegimeHighGraFTNativeFOne",
              fmt(hi["methods"]["graft_native"]["f1"])),
        #  The counts behind the emission ratio, because "104 objects where 130
        #  filaments exist" is the sentence a reader can check.
        macro("PlectaRegimeHighTrueCount",
              "%.0f" % hi["methods"]["plecta"]["n_instances_true"]),
        macro("PlectaRegimeHighPlectaCount",
              "%.0f" % hi["methods"]["plecta"]["n_instances_emitted"]),
        macro("PlectaRegimeHighGraFTCount",
              "%.0f" % hi["methods"]["graft"]["n_instances_emitted"]),
        macro("PlectaRegimeHighDNAiCount",
              "%.0f" % hi["methods"]["dnai"]["n_instances_emitted"]),
        #  How nearly the permissive measure is that count and nothing else.
        #  PLECTA's residual is exactly zero on all forty scenes, so this is
        #  reported as the largest residual over the three methods rather than
        #  as a per-method figure that would read as three separate claims.
        macro("PlectaRegimeCoverageCountResidual",
              "%.2f" % max(regime["overall"][m]
                           ["max_abs_deviation_from_emission_ratio"]
                           for m in ("plecta", "graft", "dnai"))),
    ]
    #  The field-standard pair, pooled over the whole ladder, so the paragraph
    #  can say the in-house endpoint is not carrying the ordering alone. VI is
    #  emitted as split, merge and total together: the repository rule is that
    #  split never appears without merge, because a method that fuses
    #  everything never splits anything and scores near zero on it.
    for key, stem in (("plecta", "Plecta"), ("graft", "GraFT"),
                      ("dnai", "DNAi")):
        b = regime["overall"][key]
        lines.extend([
            #  Pooled over the whole ladder, so the Conclusions can state what
            #  the method scores on a generator that is not ours in one number
            #  rather than as the two ends of a range.
            macro(f"PlectaRegimeOverall{stem}FOne", fmt(b["f1"])),
            macro(f"PlectaRegimeOverall{stem}ARI", fmt(b["adjusted_rand_index"])),
            macro(f"PlectaRegimeOverall{stem}VISplit", fmt(b["vi_split_bits"])),
            macro(f"PlectaRegimeOverall{stem}VIMerge", fmt(b["vi_merge_bits"])),
            macro(f"PlectaRegimeOverall{stem}VITotal",
                  fmt(b["vi_split_bits"] + b["vi_merge_bits"])),
        ])
    for tag, block in (("Low", lo), ("High", hi)):
        for key, stem in (("plecta", "Plecta"), ("graft", "GraFT"),
                          ("dnai", "DNAi")):
            b = block["methods"][key]
            lines.extend([
                macro(f"PlectaRegime{tag}{stem}FOne", fmt(b["f1"])),
                macro(f"PlectaRegime{tag}{stem}Matched",
                      fmt(b["filament_matched_coverage"])),
                #  The same matching once a detection has to cover half the
                #  filament it claims. The gap between this and the line above
                #  is what the permissive form is not asking.
                macro(f"PlectaRegime{tag}{stem}MatchedHalf",
                      fmt(b["filament_matched_coverage_half"])),
                macro(f"PlectaRegime{tag}{stem}Emitted",
                      "%.2f" % (b["n_instances_emitted"]
                                / b["n_instances_true"])),
            ])
    return lines


def factorial_macros(factorial: dict) -> list[str]:
    """Macros for the bundle-width negative control.

    Within one geometry the axis mask and the centreline reference are
    byte-identical across the three bundle widths while the thick support, and
    therefore areal coverage, varies. Any score movement across widths would
    mean something reads the silhouette.
    """
    control = factorial["width_negative_control"]
    #  A null needs a scale, or "flat" says nothing.  The spread the same F1
    #  axis shows between geometries is that scale, so it is reported next to
    #  the within-geometry range it should be read against.
    scene_f1 = [float(row["f1"]) for row in factorial["per_scene"]]
    lines = [
        macro("PlectaFactorialSceneCount", str(factorial["n_scenes"])),
        macro("PlectaFactorialGeometryCount", str(control["n_geometries"])),
        macro("PlectaFactorialZeroRangeCount",
              str(control["n_geometries_with_zero_range"])),
        macro("PlectaFactorialMaxFOneRange",
              fmt(control["max_within_geometry_f1_range"], 4)),
        macro("PlectaFactorialFOneSpread",
              fmt(max(scene_f1) - min(scene_f1))),
        macro("PlectaFactorialMaxCoverageRatio",
              f"{control['max_areal_coverage_ratio_within_geometry']:.2f}"),
    ]
    for key, stem in (("ld020", "Low"), ("ld040", "Mid"), ("ld060", "High")):
        block = factorial["by_length_density"][key]
        lines.extend([
            macro(f"PlectaFactorial{stem}LengthDensity",
                  fmt(block["centreline_coverage_mean"], 4)),
            macro(f"PlectaFactorial{stem}FOne", fmt(block["f1_mean"])),
            macro(f"PlectaFactorial{stem}CoverageMin",
                  fmt(block["areal_coverage_min"], 3)),
            macro(f"PlectaFactorial{stem}CoverageMax",
                  fmt(block["areal_coverage_max"], 3)),
        ])
    return lines


def factorial_table(factorial: dict) -> str:
    lines = ["% Generated by scripts/results/make_plecta_results.py; do not edit.",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "Centreline length density & $n$ & Areal coverage & "
             "Bundle width & $F_1$ \\\\",
             "\\midrule"]
    for key in ("ld020", "ld040", "ld060"):
        block = factorial["by_length_density"][key]
        lines.append(
            f"{fmt(block['centreline_coverage_mean'], 4)} & {block['n']} & "
            f"{fmt(block['areal_coverage_min'], 3)}--"
            f"{fmt(block['areal_coverage_max'], 3)} & 6, 11, 16 px & "
            f"{fmt(block['f1_mean'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def greedy_table(greedy: dict) -> str:
    block = greedy["conditions"]["degraded"]["summary"]["metrics"]
    rows = [
        ("Common-fragment $F_1$", "f1", False),
        ("Precision", "precision", False),
        ("Recall", "recall", False),
        ("Reference-instance recovery", "fragment_recovery_recovery_rate", False),
        ("Adjusted Rand index", "adjusted_rand_index", False),
        ("Variation of information (total)", "vi_total_bits", True),
    ]
    lines = ["% Generated by scripts/results/make_plecta_results.py; do not edit.",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Measure & PLECTA & Greedy continuation & Difference \\\\",
             "\\midrule"]
    for label, key, lower_better in rows:
        stats = block[key]
        paired = stats["paired"]
        arrow = "$\\downarrow$" if lower_better else ""
        lines.append(
            f"{label}~{arrow} & {fmt(stats['plecta_mean'])} & "
            f"{fmt(stats['greedy_mean'])} & "
            f"{paired['mean_difference']:+.3f} "
            f"[{paired['ci_low']:+.3f}, {paired['ci_high']:+.3f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def comparator_table(audit: dict, runtime: dict | None = None,
                     basu: dict | None = None,
                     sifne: dict | None = None) -> str:
    """One table for all four methods on the degraded masks.

    Supersedes greedy_table(), which compared two methods on six measures and
    restated most of them in the prose above it. Methods are columns because
    there are four of them against eight measures, and because a reader compares
    methods down a column of like-scaled numbers more easily than across a row.

    There is no paired-difference column. It carried PLECTA minus the greedy
    continuation on every row, and the greedy continuation is reported as a
    floor rather than as a claim under test -- a reader takes that difference
    off the two columns beside each other. The one interval worth stating is the
    headline pairwise difference, which Section 3.3 gives in prose from
    \\PlectaGreedySwept* macros, and the claim that does need intervals -- the
    matcher-only arm -- keeps all of its own in the text.

    Variation of information is reported as the total, never as the split
    component alone: split alone is one-sided, since a method that merges
    everything never splits anything and would score near zero on it. Where the
    balance between the components is the point -- GraFT's failure mode, and the
    matcher-only ablation -- the prose gives both components for that method.

    Junction resolution is pooled over crossings (junction_resolution), not
    averaged over per-scene rates (methods[...]["jr_exact_rate"]); the two
    differ, and the pooled form is the one the prose reports. GraFT's column is
    over the scenes it completed, so its crossing count is smaller; the caption
    says so.
    """
    means = audit["methods"]["degraded"]
    junction = audit["junction_resolution"]["degraded"]

    #  COLUMNS ARE ORDERED BY COMMON-FRAGMENT F1, BEST FIRST, and the order is
    #  computed from the numbers rather than written down, so it stays correct
    #  if any of them change. A reader scanning left to right then meets the
    #  methods in the order the endpoint ranks them.
    #
    #  The greedy continuation rule is excluded: it is our own invention rather
    #  than anyone's published method, and belongs with the controls in the
    #  ablation table, not in a table whose purpose is to compare against other
    #  people's published work.
    #
    #  Two columns carry a marker rather than a position. Basu* is OUR
    #  reimplementation, because no implementation of that paper was ever
    #  released; SIFNE-dagger is externally run but at parameters we selected on
    #  development scenes, its own being degenerate at these densities. Sorting
    #  by score puts the reimplementation above two externally run methods, so
    #  the markers and the caption carry that qualification and the position
    #  does not.
    #  Source records: results/plecta_basu_reimplementation.json and
    #  results/plecta_sifne_comparison.json.
    entries: list[dict] = [
        {"key": "plecta", "label": "PLECTA", "src": "audit"},
        {"key": "dnai", "label": "DNAi", "src": "audit"},
        {"key": "graft", "label": "GraFT", "src": "audit"},
    ]
    if sifne:
        entries.append({"key": "sifne", "label": r"SIFNE$^{\dagger}$",
                        "src": "record", "rec": sifne})
    if basu:
        entries.append({"key": "basu", "label": r"Basu$^{*}$",
                        "src": "record", "rec": basu})

    def value(e: dict, key: str) -> float:
        if e["src"] == "audit":
            return means[e["key"]]["overall"][key]
        return e["rec"]["overall"][key]

    def crossing(e: dict) -> float:
        if e["src"] == "audit":
            return junction[e["key"]]["overall"]["exact_rate"]
        return e["rec"]["junction_resolution"]["exact_rate"]

    entries.sort(key=lambda e: -value(e, "f1"))

    body: list[tuple[str, list[str]]] = []
    for label, key in (("Common-fragment $F_1$", "f1"),
                       ("Precision", "precision"),
                       ("Recall", "recall"),
                       ("Reference-instance recovery",
                        "fragment_recovery_recovery_rate"),
                       ("Adjusted Rand index", "adjusted_rand_index"),
                       (r"Variation of information~$\downarrow$", "vi_total_bits")):
        body.append((label, [fmt(value(e, key)) for e in entries]))

    #  One crossing row, not two. The chance-corrected resolution index used to
    #  sit under this one; over the real-field rows it correlates with the exact
    #  rate at Pearson 0.997 and never went negative, so it restated the row
    #  above it. The chance level belongs in the caption, being a property of
    #  the reference rather than of any method.
    resolution = [("Crossing Fidelity", [fmt(crossing(e)) for e in entries])]

    #  Median seconds per scene at the two ends of the density range. Basu* is
    #  our Python code for one stage, and SIFNE-dagger is MATLAB including
    #  interpreter start-up; neither is comparable with the other columns and
    #  the caption says so.
    timing = []
    if runtime:
        medians = runtime["per_density_median_seconds"]

        def span(e: dict) -> str:
            if e["src"] == "audit":
                block = medians.get(e["key"])
            else:
                block = e["rec"].get("runtime", {}).get("median_seconds")
            if not block or block.get("cov20") is None or block.get("cov60") is None:
                return "---"
            return f"{block['cov20']:.2f}--{block['cov60']:.1f}"

        timing = [(r"Median s per scene, 20/60\%", [span(e) for e in entries])]

    ncol = "r" * len(entries)
    header = "Measure & " + " & ".join(e["label"] for e in entries)
    lines = ["% Generated by scripts/results/make_plecta_results.py; do not edit.",
             f"\\begin{{tabular}}{{l{ncol}}}", "\\toprule",
             header + " \\\\",
             "\\midrule"]
    for group in (body, resolution, timing):
        if not group:
            continue
        if group is not body:
            lines.append("\\addlinespace")
        for label, values in group:
            lines.append(f"{label} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def make_macros(held: dict, robustness: list[dict], ablation: dict,
                stage4: dict, cc_baseline: dict,
                width_validation: dict, greedy: dict,
                factorial: dict, dnai: dict,
                analysis: dict, swept: dict,
                graft: dict, audit: dict, supplement: dict,
                mask_quality: dict, graft_regime: dict,
                real_field_metrics: dict, real_fields: dict,
                interannotator: dict, curviness: dict,
                depth_order: dict,
                metric_panels: dict, sensitivity: dict,
                scorer_sensitivity: dict,
                crossing_chance: dict,
                basu: dict | None = None,
                sifne: dict | None = None) -> str:
    rows = [r for r in held["per_scene"] if r.get("status") == "ok"]
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        macro("PlectaHeldoutSceneCount", str(len(rows))),
        macro("PlectaHeldoutFailureCount", str(held["summary"]["all"]["n_failed"])),
    ]
    for i, (key, stem) in enumerate(METRICS.items()):
        values = np.asarray([float(r[key]) for r in rows])
        lo, hi = stratified_boot_mean(rows, key, np.random.default_rng(SEED + i))
        lines.extend(
            [
                macro(f"PlectaHeldout{stem}Mean", fmt(values.mean())),
                macro(f"PlectaHeldout{stem}CILow", fmt(lo)),
                macro(f"PlectaHeldout{stem}CIHigh", fmt(hi)),
            ]
        )

    for density in (20, 30, 40, 60):
        subset = [r for r in rows if int(r["density"]) == density]
        values = np.asarray([float(r["f1"]) for r in subset])
        lo, hi = boot_mean(values, np.random.default_rng(SEED + density))
        lines.extend(
            [
                macro(f"PlectaDensity{density}SceneCount", str(len(subset))),
                macro(f"PlectaDensity{density}FOneMean", fmt(values.mean())),
                macro(f"PlectaDensity{density}FOneCILow", fmt(lo)),
                macro(f"PlectaDensity{density}FOneCIHigh", fmt(hi)),
            ]
        )

    seconds = np.asarray([float(r["seconds"]) for r in rows])
    lines.extend(
        [
            macro("PlectaHeldoutRuntimeMean", fmt(seconds.mean(), 2)),
            macro("PlectaHeldoutRuntimeMedian", fmt(np.median(seconds), 2)),
            macro("PlectaHeldoutRuntimePNinety", fmt(np.quantile(seconds, 0.9), 2)),
            macro("PlectaHeldoutRuntimeMaximum", fmt(seconds.max(), 2)),
        ]
    )

    cc = cc_baseline["summary"]
    lines.extend(
        [
            macro("PlectaCCFloorFOne", fmt(cc["f1"])),
            macro("PlectaCCFloorPrecision", fmt(cc["precision"])),
            macro("PlectaCCFloorRecall", fmt(cc["recall"])),
            macro("PlectaCCFloorARI", fmt(cc["adjusted_rand_index"])),
            macro("PlectaCCFloorVIMerge", fmt(cc["vi_merge_bits"])),
        ]
    )

    width = width_validation["width_vs_renderer_truth"]
    interval = width_validation["quality_stratified_interval"]
    lines.extend(
        [
            macro("PlectaWidthValidationSceneCount", str(width_validation["n_scenes"])),
            macro("PlectaWidthValidationInstanceCount", str(width_validation["n_instances"])),
            macro("PlectaWidthValidationPearson", fmt(width["pearson_r"], 4)),
            macro("PlectaWidthValidationMAE", fmt(width["mae_px"], 3)),
            macro("PlectaWidthValidationMedianAE", fmt(width["median_absolute_error_px"], 3)),
            macro("PlectaWidthValidationCoverage", fmt(interval["scene_cross_validated_coverage"], 3)),
        ]
    )

    by_key: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in robustness:
        by_key[(int(row["density"]), row["scene"])][row["mask_variant"]] = row
    pairs = [v for v in by_key.values() if set(v) == {"clean", "degraded"}]
    for variant in ("degraded", "clean"):
        vals = np.asarray([float(v[variant]["f1"]) for v in pairs])
        lines.append(macro(f"PlectaRobustness{variant.title()}FOneMean", fmt(vals.mean())))
    deltas_by_density: dict[int, list[float]] = defaultdict(list)
    for variants in pairs:
        density = int(variants["clean"]["density"])
        deltas_by_density[density].append(
            float(variants["clean"]["f1"]) - float(variants["degraded"]["f1"])
        )
    bootstrap = np.zeros(N_BOOT)
    total = sum(len(v) for v in deltas_by_density.values())
    rng = np.random.default_rng(SEED + 100)
    for values in deltas_by_density.values():
        arr = np.asarray(values)
        idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
        bootstrap += arr[idx].sum(axis=1) / total
    all_deltas = np.concatenate([np.asarray(v) for v in deltas_by_density.values()])
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    lines.extend(
        [
            macro("PlectaRobustnessSceneCount", str(len(pairs))),
            macro("PlectaCleanMinusDegradedFOneMean", fmt(all_deltas.mean(), 4)),
            macro("PlectaCleanMinusDegradedFOneCILow", fmt(lo, 4)),
            macro("PlectaCleanMinusDegradedFOneCIHigh", fmt(hi, 4)),
        ]
    )

    base = next(r for r in ablation["rows"] if r["ablation"] == "frozen")
    lines.append(macro("PlectaDevelopmentSceneCount", str(base["n_scenes"])))
    for row in ablation["rows"]:
        if row["ablation"] == "frozen":
            continue
        label = "".join(word.title() for word in row["ablation"].replace("-", " ").split())
        lines.append(macro(f"PlectaAblation{label}DeltaFOne", fmt(float(row["f1"]) - float(base["f1"]), 4)))

    frozen = stage4["variants"]["frozen"]
    ribbon = stage4["variants"]["ribbon_default"]
    absorb = stage4["variants"]["ribbon_absorb_0p6"]
    lines.extend(
        [
            macro("PlectaStageFourFOne", fmt(ribbon["f1"], 3)),
            macro("PlectaStageFourDeltaFOne", fmt(ribbon["f1"] - frozen["f1"], 4)),
            macro("PlectaStageFourDisconnected", fmt(ribbon["fraction_disconnected"], 3)),
            macro("PlectaStageFourBranched", fmt(ribbon["fraction_branched"], 3)),
            macro("PlectaStageFourAbsorbFOne", fmt(absorb["f1"], 3)),
            # The core centreline output before any rendering. Reported so the
            # rendered 0.000 is not read as a property of the grouping itself:
            # accepted gap links carry identity but are not painted, so a core
            # instance is often delivered as several disjoint pixel runs.
            macro("PlectaCoreDisconnected", fmt(frozen["fraction_disconnected"], 3)),
            macro("PlectaCoreBranched", fmt(frozen["fraction_branched"], 3)),
            *greedy_macros(greedy),
            *factorial_macros(factorial),
            *dnai_macros(dnai),
            *comparator_analysis_macros(analysis, swept),
            *graft_macros(graft),
            *matcher_ablation_macros(audit),
            *junction_resolution_macros(audit, crossing_chance),
            *graft_frame_macros(audit, supplement),
            *shared_ownership_macros(audit, supplement),
            *scorer_sensitivity_macros(scorer_sensitivity),
            *mask_quality_macros(mask_quality),
            *graft_regime_macros(graft_regime),
            *real_fields_macros(real_fields),
            *interannotator_macros(interannotator),
            *curviness_macros(curviness),
            *depth_order_macros(depth_order),
            *real_field_metrics_macros(
                real_field_metrics,
                real_fields["fields"][0]["axis_triple"]
                ["reported_tolerance_px"]),
            *metric_panels_macros(metric_panels),
            *sensitivity_macros(sensitivity),
            *sifne_macros(sifne),
            *basu_macros(basu),
        ]
    )
    return "\n".join(lines) + "\n"


def heldout_table(held: dict) -> str:
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Group & $n$ & $F_1$ & Precision & Recall & Recovery & ARI & "
        r"VI split & VI merge & VI total \\",
        r"\midrule",
    ]
    #  VI total is printed beside its two components, not instead of them: a
    #  method that fuses everything never splits anything and scores a perfect
    #  0.000 on the split column alone, so the split column must never be the
    #  one a reader ranks by.  The total is what the comparison figures quote.
    for key, label in (("all", "All"), ("cov20", r"20\%"), ("cov30", r"30\%"),
                       ("cov40", r"40\%"), ("cov60", r"60\%")):
        row = held["summary"][key]
        total = float(row["vi_split_bits"]) + float(row["vi_merge_bits"])
        lines.append(
            f"{label} & {row['n_scored']} & {fmt(row['f1'])} & {fmt(row['precision'])} & "
            f"{fmt(row['recall'])} & {fmt(row['fragment_recovery_recovery_rate'])} & "
            f"{fmt(row['adjusted_rand_index'])} & {fmt(row['vi_split_bits'])} & "
            f"{fmt(row['vi_merge_bits'])} & {fmt(total)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def ablation_table(payload: dict, audit: dict | None = None) -> str:
    """Component ablations, plus the one formulation control that belongs here.

    The greedy per-junction matcher is a CONTROL rather than a component: it
    keeps the graph, the cost, the admissibility gate and the round schedule
    and replaces only the exact maximum-weight matching by greedy
    cheapest-first acceptance, so the difference isolates the formulation. It
    is ours by construction, which is exactly what lets it hold everything else
    fixed, and it is therefore reported here rather than in the comparator
    table, whose purpose is comparison against other people's published work.

    It is on 30 development scenes against the table's 84, so it sits in its
    own block and the caption gives both counts. Its delta is taken against the
    exact-matcher arm of the same 30 scenes, not against the 84-scene fixed
    configuration, since only the former is a paired comparison.
    """
    base = next(r for r in payload["rows"] if r["ablation"] == "frozen")
    selected = {
        "frozen",
        "no gap bridging",
        "single round",
        "no chain-extended frames",
        "no annealing",
        "no curvature term",
        "no join_px",
        "no crossing merging at all",
        "tangent term only",
    }
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Development variant & $F_1$ & $\Delta F_1$ & ARI \\",
        r"\midrule",
    ]
    for row in payload["rows"]:
        if row["ablation"] not in selected:
            continue
        delta = float(row["f1"]) - float(base["f1"])
        #  Spelled out rather than title-cased from the ablation key: "No
        #  Join\_Px" printed a parameter name at a reader, which is a draft
        #  artefact.  join_px is the length below which an arm joining two
        #  crossings makes them one node (graph.py, line 326).
        display_labels = {
            "frozen": "Fixed configuration",
            "tangent term only": "No junction chord-turn term",
            "no crossing merging at all": "No junction-cluster consolidation",
            "no join_px": "No short-connector node merging",
            "no gap bridging": "No gap bridging",
            "single round": "Single round",
            "no chain-extended frames": "No chain-extended frames",
            "no annealing": "No round schedule",
            "no curvature term": "No curvature term",
        }
        label = display_labels.get(row["ablation"], row["ablation"].title())
        lines.append(
            f"{tex_escape(label)} & {fmt(row['f1'])} & "
            f"{delta:+.3f} & {fmt(row['adjusted_rand_index'])} \\\\"
        )
    if audit:
        arms = audit["matcher_ablation"]["degraded"]["arms"]
        exact, greedy = arms["exact"], arms["greedy_junction"]
        lines.append(r"\addlinespace")
        lines.append(
            r"Greedy per-junction matcher\textsuperscript{\dag} & "
            f"{fmt(greedy['f1'])} & "
            f"{float(greedy['f1']) - float(exact['f1']):+.3f} & "
            f"{fmt(greedy['adjusted_rand_index'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


_SENS_LABELS = {
    "prune_spur_px": ("spur prune", "px", 0),
    "min_len_px": ("min fragment", "px", 0),
    "min_overlap_frac": ("overlap threshold", "", 2),
    "max_dist_px": ("fallback distance", "px", 0),
}


def _sens_label(row: dict, defaults: dict) -> str:
    diff = {k: row[k] for k in defaults if row[k] != defaults[k]}
    if not diff:
        return "published constants"
    (key, value), = diff.items()
    name, unit, nd = _SENS_LABELS[key]
    text = f"{value:.{nd}f}" if nd else f"{int(value)}"
    return f"{name} {text}{('~' + unit) if unit else ''}"


def scorer_sensitivity_table(record: dict) -> str:
    """One row per perturbed scorer constant; disclosure, not selection."""
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scorer constants & PLECTA $F_1$ & $\Delta$ greedy & $\Delta$ DNAi & "
        r"$\Delta$ GraFT \\",
        r"\midrule",
    ]
    for row in record["summary"]:
        lines.append(
            f"{_sens_label(row, record['defaults'])} & "
            f"{fmt(row['plecta_f1_mean'])} & "
            f"{fmt(row['margin_vs_greedy_swept'])} & "
            f"{fmt(row['margin_vs_dnai'])} & "
            f"{fmt(row['margin_vs_graft'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def scorer_sensitivity_macros(record: dict) -> list[str]:
    rows = record["summary"]
    f1s = [r["plecta_f1_mean"] for r in rows]
    margins = [r[f"margin_vs_{m}"] for r in rows
               for m in ("greedy_swept", "dnai", "graft")]
    return [
        macro("PlectaScorerSensitivityComboCount", str(len(rows))),
        macro("PlectaScorerSensitivityFOneLow", fmt(min(f1s))),
        macro("PlectaScorerSensitivityFOneHigh", fmt(max(f1s))),
        macro("PlectaScorerSensitivityMarginMin", fmt(min(margins))),
    ]


def real_fields_table(payload: dict) -> str:
    """Three fields, two mask sources, the five measures of Section 2.6.

    The held-out column was removed on 2026-08-25 by author decision, together
    with the results subsection that measured what training overlap costs. The
    record `plecta_real_fields.json` still carries `unet_held_out` per field,
    and `plecta_real_masks.json` still carries the contamination experiment, so
    the column can be restored without re-running anything.
    """
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Grouping} & "
        r"\multicolumn{3}{c}{Objects and crossings} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Field & Input axis & $F_1$ & Join $F_1$ & Decisions & "
        r"Det.\ $F_1$ & CF & Unpaired & Cross. \\",
        r"\midrule",
    ]
    for field in payload["fields"]:
        for key, label in (("manual", "Manual-derived"), ("unet", "U-Net")):
            c = field["conditions"][key]
            lines.append(
                f"{tex_escape(field['image'])} & {label} & "
                f"{fmt(c['pairwise_f1'])} & {fmt(c['join_f1'])} & "
                f"{c['n_decisions']:,} & {fmt(c['detection_f1'])} & "
                f"{fmt(c['crossing_fidelity'])} & "
                f"{fmt(c['crossing_fidelity_chance'])} & "
                f"{c['n_crossings']} \\\\".replace(",", r"\,")
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def real_axis_table(payload: dict) -> str:
    """What each field's U-Net axis is, as an input, before anything runs."""
    tolerance = payload["fields"][0]["axis_triple"]["reported_tolerance_px"]
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Field & Reference filaments & Completeness & Correctness & Quality"
        r" \\",
        r"\midrule",
    ]
    for field in payload["fields"]:
        triple = field["axis_triple"]
        lines.append(
            f"{tex_escape(field['image'])} & {field['n_reference']} & "
            f"{fmt(triple['completeness'])} & {fmt(triple['correctness'])} & "
            f"{fmt(triple['quality'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    assert tolerance  # the caption cites \PlectaPlacementTolerance
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons-root", type=Path, default=DEFAULT_COMPARISONS_ROOT,
        help="root of the comparisons checkout (default: the 'comparisons' "
             "directory beside this repository)")
    args = parser.parse_args()
    crossing_chance_path = args.comparisons_root / CROSSING_CHANCE_RELATIVE

    held = json.loads(HELDOUT.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    stage4 = json.loads(STAGE4.read_text(encoding="utf-8"))
    cc_baseline = json.loads(CC_BASELINE.read_text(encoding="utf-8"))
    width_validation = json.loads(WIDTH_VALIDATION.read_text(encoding="utf-8"))
    greedy = json.loads(GREEDY_BASELINE.read_text(encoding="utf-8"))
    factorial = json.loads(DENSITY_FACTORIAL.read_text(encoding="utf-8"))
    dnai = json.loads(DNAI.read_text(encoding="utf-8"))
    analysis = json.loads(COMPARATOR_ANALYSIS.read_text(encoding="utf-8"))
    swept = json.loads(GREEDY_SWEPT.read_text(encoding="utf-8"))
    graft = json.loads(GRAFT.read_text(encoding="utf-8"))
    audit = json.loads(METRICS_AUDIT.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    basu = (json.loads(BASU.read_text(encoding="utf-8"))
            if BASU.is_file() else None)
    sifne = (json.loads(SIFNE.read_text(encoding="utf-8"))
             if SIFNE.is_file() else None)
    supplement = json.loads(METRICS_SUPPLEMENT.read_text(encoding="utf-8"))
    mask_quality = json.loads(MASK_QUALITY.read_text(encoding="utf-8"))
    graft_regime = json.loads(GRAFT_REGIME.read_text(encoding="utf-8"))
    real_field_metrics = json.loads(
        REAL_FIELD_METRICS.read_text(encoding="utf-8"))
    real_fields = json.loads(REAL_FIELDS.read_text(encoding="utf-8"))
    interannotator = json.loads(
        INTERANNOTATOR.read_text(encoding="utf-8"))
    curviness = json.loads(CURVINESS.read_text(encoding="utf-8"))
    depth_order = json.loads(DEPTH_ORDER.read_text(encoding="utf-8"))
    metric_panels = json.loads(METRIC_PANELS.read_text(encoding="utf-8"))
    sensitivity = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    scorer_sensitivity = json.loads(
        SCORER_SENSITIVITY.read_text(encoding="utf-8"))
    crossing_chance = json.loads(
        crossing_chance_path.read_text(encoding="utf-8"))
    with ROBUSTNESS.open(newline="", encoding="utf-8-sig") as handle:
        robustness = list(csv.DictReader(handle))
    (RESULTS / "plecta_results.tex").write_text(
        make_macros(
            held, robustness, ablation, stage4, cc_baseline,
            width_validation, greedy, factorial, dnai, analysis, swept,
            graft, audit, supplement, mask_quality, graft_regime,
            real_field_metrics, real_fields, interannotator, curviness,
            depth_order,
            metric_panels, sensitivity,
            scorer_sensitivity, crossing_chance, basu, sifne,
        ),
        encoding="utf-8",
    )
    #  UNREFERENCED (checked 2026-08-23): no \input or \include anywhere in
    #  main.tex, supplementary.tex, sections/*.tex or results/*.tex reaches
    #  plecta_greedy_table.tex. The greedy baseline is reported instead in the
    #  four-method comparator table and in 03_results.tex's own paragraph,
    #  through \PlectaGreedy* macros. Kept emitting because the table is a
    #  correct, self-contained record and someone may want it back; delete the
    #  block and the file together if that is decided, never the file alone.
    (RESULTS / "plecta_greedy_table.tex").write_text(
        greedy_table(greedy), encoding="utf-8"
    )
    (RESULTS / "plecta_comparator_table.tex").write_text(
        comparator_table(audit, runtime, basu, sifne), encoding="utf-8"
    )
    (RESULTS / "plecta_factorial_table.tex").write_text(
        factorial_table(factorial), encoding="utf-8"
    )
    (RESULTS / "plecta_heldout_table.tex").write_text(
        heldout_table(held), encoding="utf-8"
    )
    (RESULTS / "plecta_ablation_table.tex").write_text(
        ablation_table(ablation, audit), encoding="utf-8"
    )
    (RESULTS / "plecta_real_masks_table.tex").write_text(
        real_fields_table(real_fields), encoding="utf-8"
    )
    (RESULTS / "plecta_depth_order_table.tex").write_text(
        depth_order_table(depth_order), encoding="utf-8"
    )
    #  UNREFERENCED (checked 2026-08-23), same as plecta_greedy_table.tex
    #  above. The completeness/correctness/quality triple it tabulates is
    #  reported in 03_results.tex through \PlectaMaskQuality* and
    #  \Plecta*AxisRecall macros instead.
    (RESULTS / "plecta_real_axis_table.tex").write_text(
        real_axis_table(real_fields), encoding="utf-8"
    )
    (RESULTS / "plecta_scorer_sensitivity_table.tex").write_text(
        scorer_sensitivity_table(scorer_sensitivity), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
