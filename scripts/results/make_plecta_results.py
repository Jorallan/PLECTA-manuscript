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
REAL_MASKS = RESULTS / "plecta_real_masks.json"
CC_BASELINE = RESULTS / "plecta_cc_baseline.json"
WIDTH_VALIDATION = RESULTS / "plecta_width_validation.json"
GREEDY_BASELINE = RESULTS / "plecta_greedy_baseline.json"
DENSITY_FACTORIAL = RESULTS / "plecta_density_factorial.json"
DNAI = RESULTS / "plecta_dnai_comparison.json"
COMPARATOR_ANALYSIS = RESULTS / "plecta_comparator_analysis.json"
GREEDY_SWEPT = RESULTS / "plecta_greedy_swept_paired.json"
GRAFT = RESULTS / "plecta_graft_comparison.json"
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


def factorial_macros(factorial: dict) -> list[str]:
    """Macros for the bundle-width negative control.

    Within one geometry the axis mask and the centreline reference are
    byte-identical across the three bundle widths while the thick support, and
    therefore areal coverage, varies. Any score movement across widths would
    mean something reads the silhouette.
    """
    control = factorial["width_negative_control"]
    lines = [
        macro("PlectaFactorialSceneCount", str(factorial["n_scenes"])),
        macro("PlectaFactorialGeometryCount", str(control["n_geometries"])),
        macro("PlectaFactorialZeroRangeCount",
              str(control["n_geometries_with_zero_range"])),
        macro("PlectaFactorialMaxFOneRange",
              fmt(control["max_within_geometry_f1_range"], 4)),
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


def make_macros(held: dict, robustness: list[dict], ablation: dict,
                stage4: dict, real_masks: dict, cc_baseline: dict,
                width_validation: dict, greedy: dict,
                factorial: dict, dnai: dict,
                analysis: dict, swept: dict,
                graft: dict) -> str:
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
        ]
    )

    for image in real_masks["images"]:
        stem = image["image"].replace("_", "")
        lines.extend(
            [
                macro(f"Plecta{stem}AxisRecall", fmt(image["axis_recall"], 3)),
                macro(f"Plecta{stem}AxisPrecision", fmt(image["axis_precision"], 3)),
                macro(f"Plecta{stem}ManualFOne", fmt(image["plecta_manual_axis"]["f1"], 3)),
                macro(f"Plecta{stem}UNetFOne", fmt(image["plecta_unet_axis"]["f1"], 3)),
            ]
        )
    return "\n".join(lines) + "\n"


def heldout_table(held: dict) -> str:
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Group & $n$ & $F_1$ & Precision & Recall & Recovery & ARI & VI split & VI merge \\",
        r"\midrule",
    ]
    for key, label in (("all", "All"), ("cov20", r"20\%"), ("cov30", r"30\%"),
                       ("cov40", r"40\%"), ("cov60", r"60\%")):
        row = held["summary"][key]
        lines.append(
            f"{label} & {row['n_scored']} & {fmt(row['f1'])} & {fmt(row['precision'])} & "
            f"{fmt(row['recall'])} & {fmt(row['fragment_recovery_recovery_rate'])} & "
            f"{fmt(row['adjusted_rand_index'])} & {fmt(row['vi_split_bits'])} & "
            f"{fmt(row['vi_merge_bits'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def ablation_table(payload: dict) -> str:
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
        display_labels = {
            "frozen": "Fixed configuration",
            "tangent term only": "No Junction Chord-Turn",
            "no crossing merging at all": "No Junction-Cluster Consolidation",
        }
        label = display_labels.get(row["ablation"], row["ablation"].title())
        lines.append(
            f"{tex_escape(label)} & {fmt(row['f1'])} & "
            f"{delta:+.3f} & {fmt(row['adjusted_rand_index'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def real_masks_table(payload: dict) -> str:
    lines = [
        "% Generated by scripts/results/make_plecta_results.py; do not edit.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Field & Input mask & Axis recall & Axis precision & $F_1$ & ARI \\",
        r"\midrule",
    ]
    for image in payload["images"]:
        for label, key in (("Manual-derived", "plecta_manual_axis"),
                           ("U-Net", "plecta_unet_axis")):
            values = image[key]
            axis_recall = "--" if key == "plecta_manual_axis" else fmt(image["axis_recall"])
            axis_precision = "--" if key == "plecta_manual_axis" else fmt(image["axis_precision"])
            lines.append(
                f"{tex_escape(image['image'])} & {label} & {axis_recall} & "
                f"{axis_precision} & {fmt(values['f1'])} & {fmt(values['ari'])} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    held = json.loads(HELDOUT.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    stage4 = json.loads(STAGE4.read_text(encoding="utf-8"))
    real_masks = json.loads(REAL_MASKS.read_text(encoding="utf-8"))
    cc_baseline = json.loads(CC_BASELINE.read_text(encoding="utf-8"))
    width_validation = json.loads(WIDTH_VALIDATION.read_text(encoding="utf-8"))
    greedy = json.loads(GREEDY_BASELINE.read_text(encoding="utf-8"))
    factorial = json.loads(DENSITY_FACTORIAL.read_text(encoding="utf-8"))
    dnai = json.loads(DNAI.read_text(encoding="utf-8"))
    analysis = json.loads(COMPARATOR_ANALYSIS.read_text(encoding="utf-8"))
    swept = json.loads(GREEDY_SWEPT.read_text(encoding="utf-8"))
    graft = json.loads(GRAFT.read_text(encoding="utf-8"))
    with ROBUSTNESS.open(newline="", encoding="utf-8-sig") as handle:
        robustness = list(csv.DictReader(handle))
    (RESULTS / "plecta_results.tex").write_text(
        make_macros(
            held, robustness, ablation, stage4, real_masks, cc_baseline,
            width_validation, greedy, factorial, dnai, analysis, swept,
            graft,
        ),
        encoding="utf-8",
    )
    (RESULTS / "plecta_greedy_table.tex").write_text(
        greedy_table(greedy), encoding="utf-8"
    )
    (RESULTS / "plecta_factorial_table.tex").write_text(
        factorial_table(factorial), encoding="utf-8"
    )
    (RESULTS / "plecta_heldout_table.tex").write_text(
        heldout_table(held), encoding="utf-8"
    )
    (RESULTS / "plecta_ablation_table.tex").write_text(
        ablation_table(ablation), encoding="utf-8"
    )
    (RESULTS / "plecta_real_masks_table.tex").write_text(
        real_masks_table(real_masks), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
