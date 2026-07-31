"""Generate manuscript-safe result macros from immutable evaluation JSON.

No result is inferred from prose or a missing row.  The script only summarizes
successful, finite scene-level measurements and writes ``missing`` explicitly
where the requested evidence is absent.  Bootstrap intervals are delegated to
``eval.stats_util`` so they use the same fixed seed as the evaluator.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)
import stats_util  # noqa: E402


MISSING_TEX = r"\textit{missing}"
METRICS = ("precision", "recall", "f1", "fragment_recovery_rate")
CORRECTED_METRIC_KEYS = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "fragment_recovery_rate": "fragment_recovery_recovery_rate",
}


def _load(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    """Replace unavailable numeric summaries with JSON ``null`` explicitly."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _common(row: Mapping[str, Any]) -> Mapping[str, Any]:
    scores = row.get("scores", {})
    return scores.get("common_metric", {}) if isinstance(scores, Mapping) else {}


def _metric(row: Mapping[str, Any], metric: str) -> Optional[float]:
    common = _common(row)
    key = "fragment_recovery_recovery_rate" if metric == "fragment_recovery_rate" else metric
    return _finite(common.get(key))


def _native_metric(row: Mapping[str, Any], name: str) -> Optional[float]:
    scores = row.get("scores", {})
    native = scores.get("native_diagnostics", {}) if isinstance(scores, Mapping) else {}
    paths = {
        "pairwise_precision": ("clustering", "pairwise_precision"),
        "pairwise_recall": ("clustering", "pairwise_recall"),
        "pairwise_f1": ("clustering", "pairwise_f1"),
        "instance_recovery_rate": ("instance", "recovery_rate"),
        "panoptic_quality": ("panoptic", "pq"),
    }
    value: Any = native
    for part in paths[name]:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return _finite(value)


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    valid = [value for value in (_finite(v) for v in values) if value is not None]
    return stats_util.summarize(valid).as_dict()


def _row_groups(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, "missing"))].append(row)
    return dict(sorted(out.items()))


def _success(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = _success(rows)
    return {
        "n_total": len(rows), "n_success": len(successful),
        "failure_count": sum(row.get("status") == "failed" for row in rows),
        "common": {metric: _summary(_metric(row, metric) for row in successful) for metric in METRICS},
    }


def _native_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = _success(rows)
    names = ("pairwise_precision", "pairwise_recall", "pairwise_f1",
             "instance_recovery_rate", "panoptic_quality")
    return {name: _summary(_native_metric(row, name) for row in successful) for name in names}


def _scene_key(row: Mapping[str, Any]) -> Optional[str]:
    geometry = row.get("geometry_id", row.get("scene_id"))
    if geometry is None:
        return None
    return f"{geometry}|{row.get('mask_variant', '')}"


def _unique_values(rows: Iterable[Mapping[str, Any]], metric: str) -> tuple[Dict[str, float], List[str]]:
    values: Dict[str, float] = {}
    duplicate: List[str] = []
    for row in _success(rows):
        key, value = _scene_key(row), _metric(row, metric)
        if key is None or value is None:
            continue
        if key in values:
            duplicate.append(key)
            values.pop(key, None)
        elif key not in duplicate:
            values[key] = value
    return values, sorted(set(duplicate))


def _paired(filaseg_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]],
            baseline: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for metric in METRICS:
        left, left_dup = _unique_values(filaseg_rows, metric)
        right, right_dup = _unique_values(baseline_rows, metric)
        ids = sorted(set(left) & set(right))
        result[metric] = {
            "direction": "filaseg_minus_baseline",
            "baseline": baseline,
            "n_paired_scenes": len(ids),
            "n_missing_filaseg": len(set(right) - set(left)),
            "n_missing_baseline": len(set(left) - set(right)),
            "duplicate_filaseg_scene_ids": left_dup,
            "duplicate_baseline_scene_ids": right_dup,
            "summary": stats_util.paired_summarize([left[x] for x in ids], [right[x] for x in ids]).as_dict(),
        }
    return result


def summarize_locked(document: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(document.get("per_scene", []))
    methods = _row_groups(rows, "method")
    overall = {method: _method_summary(group) for method, group in methods.items()}
    density_values = sorted({row.get("density") for row in rows}, key=lambda v: (v is None, v))
    per_density: Dict[str, Any] = {}
    for density in density_values:
        density_rows = [row for row in rows if row.get("density") == density]
        density_methods = {method: [row for row in density_rows if row.get("method") == method]
                           for method in sorted(methods)}
        density_filaseg = density_methods.get("filaseg", [])
        per_density[str(density)] = {
            "methods": {method: _method_summary(group) for method, group in density_methods.items()},
            "paired_filaseg_minus_baseline": {
                baseline: _paired(density_filaseg, group, baseline)
                for baseline, group in density_methods.items() if baseline != "filaseg"
            },
        }
    fila = methods.get("filaseg", [])
    paired = {baseline: _paired(fila, group, baseline)
              for baseline, group in methods.items() if baseline != "filaseg"}
    return {
        "locked_scene_count": len({_scene_key(row) for row in rows if _scene_key(row) is not None}),
        "row_count": len(rows), "failure_count": sum(row.get("status") == "failed" for row in rows),
        "methods": overall, "per_density": per_density, "paired_filaseg_minus_baseline": paired,
        "filaseg_native_diagnostic": _native_summary(fila),
    }


def _corrected_common(method_summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the correction report's metric names to the legacy summary API."""
    return {
        metric: dict(method_summary.get(source_key, {}))
        for metric, source_key in CORRECTED_METRIC_KEYS.items()
    }


def _corrected_method(method_summary: Mapping[str, Any], n_scenes: int) -> Dict[str, Any]:
    return {
        "n_total": n_scenes,
        "n_success": n_scenes,
        "failure_count": 0,
        "common": _corrected_common(method_summary),
    }


def _corrected_paired(
    paired_summary: Mapping[str, Any],
    baseline: str,
    n_scenes: int,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for metric, source_key in CORRECTED_METRIC_KEYS.items():
        stat = paired_summary.get(source_key, {})
        output[metric] = {
            "direction": "filaseg_minus_baseline",
            "baseline": baseline,
            "n_paired_scenes": int(stat.get("n", n_scenes)) if isinstance(stat, Mapping) else 0,
            "n_missing_filaseg": 0,
            "n_missing_baseline": 0,
            "duplicate_filaseg_scene_ids": [],
            "duplicate_baseline_scene_ids": [],
            "summary": dict(stat) if isinstance(stat, Mapping) else {},
        }
    return output


def summarize_representation(
    document: Mapping[str, Any],
    legacy_native: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Use the stored stratified summaries from the representation audit.

    No bootstrap is rerun here.  This is important because the correction report
    is the immutable evidence artifact and its overall intervals use
    within-density resampling.
    """
    aggregate = document.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        raise ValueError("representation JSON is missing aggregate summaries")
    by_method = aggregate.get("by_method", {})
    paired = aggregate.get("paired", {})
    by_density = aggregate.get("by_density", {})
    if not all(isinstance(value, Mapping) for value in (by_method, paired, by_density)):
        raise ValueError("representation JSON has malformed aggregate summaries")

    per_scene = document.get("per_scene", [])
    n_scenes = len(per_scene) if isinstance(per_scene, list) else 0
    if not n_scenes:
        primary = by_method.get("filaseg_stage3_centerline", {})
        f1 = primary.get("f1", {}) if isinstance(primary, Mapping) else {}
        n_scenes = int(f1.get("n", 0)) if isinstance(f1, Mapping) else 0

    source_to_public = {
        "connected_components_centerline": "baseline_cc",
        "minimum_turn_centerline": "baseline_skeleton",
        "filaseg_stage3_centerline": "filaseg",
        # Explicit aliases keep representation-dependent stages available
        # without changing the headline FilaSeg macros.
        "filaseg_stage3_centerline_explicit": "filaseg_stage3_centerline",
        "filaseg_stage4_centerline": "filaseg_stage4_centerline",
        "filaseg_stage4_rendered": "filaseg_stage4_rendered",
    }
    methods: Dict[str, Any] = {}
    for source, public in source_to_public.items():
        lookup = "filaseg_stage3_centerline" if source.endswith("_explicit") else source
        raw = by_method.get(lookup)
        if not isinstance(raw, Mapping):
            raise ValueError(f"representation JSON is missing aggregate.by_method.{lookup}")
        methods[public] = _corrected_method(raw, n_scenes)

    pair_map = {
        "baseline_skeleton": "primary_filaseg_minus_minimum_turn",
        "baseline_cc": "primary_filaseg_minus_connected_components",
    }
    public_paired: Dict[str, Any] = {}
    for baseline, source in pair_map.items():
        raw = paired.get(source)
        if not isinstance(raw, Mapping):
            raise ValueError(f"representation JSON is missing aggregate.paired.{source}")
        public_paired[baseline] = _corrected_paired(raw, baseline, n_scenes)

    density_output: Dict[str, Any] = {}
    for density, density_raw in sorted(by_density.items(), key=lambda item: int(item[0])):
        if not isinstance(density_raw, Mapping):
            continue
        density_n = int(density_raw.get("n_scenes", 0))
        density_methods_raw = density_raw.get("by_method", {})
        density_pairs_raw = density_raw.get("paired", {})
        density_methods: Dict[str, Any] = {}
        for source, public in source_to_public.items():
            lookup = "filaseg_stage3_centerline" if source.endswith("_explicit") else source
            raw = density_methods_raw.get(lookup, {}) if isinstance(density_methods_raw, Mapping) else {}
            density_methods[public] = _corrected_method(raw, density_n)
        density_pairs: Dict[str, Any] = {}
        for baseline, source in pair_map.items():
            raw = density_pairs_raw.get(source, {}) if isinstance(density_pairs_raw, Mapping) else {}
            density_pairs[baseline] = _corrected_paired(raw, baseline, density_n)
        density_output[str(density)] = {
            "methods": density_methods,
            "paired_filaseg_minus_baseline": density_pairs,
        }

    secondary_pair_names = {
        "stage4_rendered_minus_stage3_centerline": "FilaSegStageFourRenderedMinusStageThree",
        "stage4_centerline_minus_stage3_centerline": "FilaSegStageFourCenterlineMinusStageThree",
        "stage4_rendered_minus_stage4_centerline": "FilaSegStageFourRenderedMinusStageFourCenterline",
    }
    secondary_paired = {
        macro_prefix: _corrected_paired(paired[source], source, n_scenes)
        for source, macro_prefix in secondary_pair_names.items()
        if isinstance(paired.get(source), Mapping)
    }
    native = (
        dict(legacy_native.get("filaseg_native_diagnostic", {}))
        if isinstance(legacy_native, Mapping)
        else {}
    )
    return {
        "locked_scene_count": n_scenes,
        "row_count": n_scenes * len(methods),
        "failure_count": 0,
        "methods": methods,
        "per_density": density_output,
        "paired_filaseg_minus_baseline": public_paired,
        "secondary_paired": secondary_paired,
        "filaseg_native_diagnostic": native,
        "endpoint_source": "representation_audit_corrected",
        "bootstrap": dict(aggregate.get("bootstrap", {})),
        "primary_filaseg_method": "filaseg_stage3_centerline",
        "primary_baseline_method": "minimum_turn_centerline",
    }


def summarize_real(document: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(document.get("runs", document.get("per_scene", [])))
    variants = _row_groups(rows, "variant")
    return {
        "row_count": len(rows), "failure_count": sum(row.get("status") == "failed" for row in rows),
        "variants": {name: {"common": _method_summary(group)["common"],
                             "native_diagnostic": _native_summary(group),
                             "n_total": len(group),
                             "failure_count": sum(row.get("status") == "failed" for row in group)}
                     for name, group in variants.items()},
    }


def summarize_curvature(document: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(document.get("runs", []))
    variants = _row_groups(rows, "variant")
    output: Dict[str, Any] = {}
    for name, group in variants.items():
        by_case = _row_groups(group, "case_type")
        output[name] = {case: {"common": _method_summary(case_rows)["common"],
                               "native_diagnostic": _native_summary(case_rows),
                               "n_total": len(case_rows),
                               "failure_count": sum(row.get("status") == "failed" for row in case_rows)}
                        for case, case_rows in by_case.items()}
    return {"row_count": len(rows), "failure_count": sum(row.get("status") == "failed" for row in rows),
            "variants": output}


def _latex_escape(text: Any) -> str:
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _macro_name(text: str) -> str:
    if str(text).lower() == "filaseg":
        return "FilaSeg"
    parts = re.findall(r"[A-Za-z0-9]+", text)
    digit_words = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    converted = []
    for part in parts:
        if part.lower() == "filaseg":
            converted.append("FilaSeg")
            continue
        safe = "".join(digit_words.get(char, char) for char in part)
        converted.append(safe[:1].upper() + safe[1:])
    return "".join(converted) or "Missing"


def _number(value: Any, digits: int = 3) -> str:
    value = _finite(value)
    return MISSING_TEX if value is None else f"{value:.{digits}f}"


def _stat_tex(summary: Mapping[str, Any]) -> str:
    if not isinstance(summary, Mapping) or not summary.get("n"):
        return MISSING_TEX
    mean, sd, lo, hi = (_finite(summary.get(k)) for k in ("mean", "sd", "ci_lo", "ci_hi"))
    if mean is None:
        return MISSING_TEX
    sd_text = MISSING_TEX if sd is None else f"{sd:.3f}"
    ci_text = MISSING_TEX if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"
    return f"{mean:.3f} $\\pm$ {sd_text}; 95\\% CI {ci_text} (n={int(summary['n'])})"


def _macros(summary: Mapping[str, Any]) -> Dict[str, str]:
    locked = summary["locked"]
    macros = {
        "LockedSceneCount": str(locked["locked_scene_count"]),
        "LockedFailureCount": str(locked["failure_count"]),
    }

    def add_stat(prefix: str, stat: Optional[Mapping[str, Any]]) -> None:
        stat = stat if isinstance(stat, Mapping) else {}
        macros[prefix] = _stat_tex(stat)
        for field, suffix in (
            ("mean", "Mean"), ("sd", "SD"), ("ci_lo", "CILow"),
            ("ci_hi", "CIHigh"), ("ci_halfwidth", "CIHalfwidth"),
        ):
            macros[prefix + suffix] = _number(stat.get(field))
        n = stat.get("n")
        macros[prefix + "N"] = str(int(n)) if n is not None else "0"

    for method, method_summary in locked["methods"].items():
        method_name = _macro_name(method)
        for metric, stat in method_summary["common"].items():
            add_stat(f"Locked{method_name}Common{_macro_name(metric)}", stat)
        macros[f"Locked{method_name}FailureCount"] = str(method_summary["failure_count"])
    for density, density_summary in locked["per_density"].items():
        density_name = _macro_name(density)
        for method, method_summary in density_summary["methods"].items():
            for metric, stat in method_summary["common"].items():
                add_stat(
                    f"LockedDensity{density_name}{_macro_name(method)}Common{_macro_name(metric)}",
                    stat,
                )
        for baseline, metrics in density_summary.get(
            "paired_filaseg_minus_baseline", {}
        ).items():
            for metric, paired in metrics.items():
                add_stat(
                    f"LockedDensity{density_name}FilaSegMinus"
                    f"{_macro_name(baseline)}Common{_macro_name(metric)}",
                    paired["summary"],
                )
    for baseline, metrics in locked["paired_filaseg_minus_baseline"].items():
        for metric, paired in metrics.items():
            add_stat(
                f"FilaSegMinus{_macro_name(baseline)}Common{_macro_name(metric)}",
                paired["summary"],
            )
    for prefix, metrics in locked.get("secondary_paired", {}).items():
        for metric, paired in metrics.items():
            add_stat(
                f"{prefix}Common{_macro_name(metric)}",
                paired["summary"],
            )
    for metric, stat in locked["filaseg_native_diagnostic"].items():
        add_stat(f"LockedFilaSegNative{_macro_name(metric)}", stat)

    for variant, variant_summary in summary["real"]["variants"].items():
        prefix = f"Real{_macro_name(variant)}"
        for metric, stat in variant_summary["common"].items():
            add_stat(f"{prefix}Common{_macro_name(metric)}", stat)
        for metric, stat in variant_summary["native_diagnostic"].items():
            add_stat(f"{prefix}Native{_macro_name(metric)}", stat)
        macros[f"{prefix}FailureCount"] = str(variant_summary["failure_count"])

    for variant, cases in summary["curvature"]["variants"].items():
        for case_type, case_summary in cases.items():
            prefix = f"Curvature{_macro_name(variant)}{_macro_name(case_type)}"
            for metric, stat in case_summary["common"].items():
                add_stat(f"{prefix}Common{_macro_name(metric)}", stat)
            macros[f"{prefix}FailureCount"] = str(case_summary["failure_count"])
    return macros


def _ci_cell(stat: Mapping[str, Any]) -> str:
    mean = _number(stat.get("mean"))
    lo = _number(stat.get("ci_lo"))
    hi = _number(stat.get("ci_hi"))
    if MISSING_TEX in (mean, lo, hi):
        return MISSING_TEX
    return rf"\shortstack{{{mean}\\{{[{lo}, {hi}]}}}}"


def _label_cell(label: str) -> str:
    r"""Stack a one-line row label so it aligns with the point estimate.

    ``\shortstack`` puts its reference point on the baseline of its *last*
    line, so a single-line label in a row whose numeric cells are two-line
    ``\shortstack``s drops onto the interval line instead of the point
    estimate.  Pairing the label with an invisible interval-shaped second line
    reproduces the geometry of the numeric cells; ``\smash`` discards the
    label's own height and depth so that descenders (``FilaSeg (Stage 3)``)
    and descender-free names sit at exactly the same height.
    """
    return rf"\shortstack[l]{{\smash{{{label}}}\\\vphantom{{[0.000, 0.000]}}}}"


def _baseline_table(summary: Mapping[str, Any]) -> str:
    locked = summary["locked"]
    display = (
        ("baseline_skeleton", "Minimum-turn tracer"),
        ("filaseg", "FilaSeg (Stage 3)"),
    )
    lines = [
        r"% Generated by make_result_macros.py; entries are mean [95\% CI].",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & $P$ & $R$ & $F_1$ & \shortstack{Object-level\\recovery} \\",
        r"\midrule",
    ]
    method_rows: List[str] = []
    for method, label in display:
        data = locked["methods"].get(method)
        if not isinstance(data, Mapping):
            continue
        common = data["common"]
        method_rows.append("{} & {} & {} & {} & {} \\\\".format(
            _label_cell(_latex_escape(label)),
            _ci_cell(common["precision"]),
            _ci_cell(common["recall"]),
            _ci_cell(common["f1"]),
            _ci_cell(common["fragment_recovery_rate"]),
        ))
    # Every method row is a two-line ``\shortstack`` (point estimate above its
    # interval), so the default inter-row leading is *smaller* than the leading
    # inside a row and consecutive methods read as one run-together block.
    # ``\addlinespace`` is emitted only *between* rows, never after the last
    # one, where a rule already follows.
    for index, row in enumerate(method_rows):
        if index:
            lines.append(r"\addlinespace[4pt]")
        lines.append(row)
    paired = locked.get("paired_filaseg_minus_baseline", {}).get("baseline_skeleton", {})
    if isinstance(paired, Mapping) and paired:
        lines.append(r"\midrule")
        lines.append("{} & {} & {} & {} & {} \\\\".format(
            r"\shortstack[l]{Paired difference\\(FilaSeg $-$ minimum-turn)}",
            _ci_cell(paired.get("precision", {}).get("summary", {})),
            _ci_cell(paired.get("recall", {}).get("summary", {})),
            _ci_cell(paired.get("f1", {}).get("summary", {})),
            _ci_cell(paired.get("fragment_recovery_rate", {}).get("summary", {})),
        ))
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _sanity_table(summary: Mapping[str, Any]) -> str:
    connected = summary["locked"]["methods"].get("baseline_cc")
    lines = [
        r"% Generated by make_result_macros.py; entries are mean [95\% CI].",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Sanity-check method & $P$ & $R$ & $F_1$ & \shortstack{Object-level\\recovery} \\",
        r"\midrule",
    ]
    if isinstance(connected, Mapping):
        common = connected["common"]
        lines.append("{} & {} & {} & {} & {} \\\\".format(
            "Connected components",
            _ci_cell(common["precision"]),
            _ci_cell(common["recall"]),
            _ci_cell(common["f1"]),
            _ci_cell(common["fragment_recovery_rate"]),
        ))
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def build_results(locked_json: Path | str, curvature_json: Path | str, real_ablation_json: Path | str,
                  out_dir: Path | str = ROOT / "paper" / "results",
                  representation_json: Optional[Path | str] = None) -> Dict[str, Any]:
    """Build deterministic LaTeX/JSON result artifacts from three source reports."""
    legacy_locked = summarize_locked(_load(locked_json))
    locked = (
        summarize_representation(_load(representation_json), legacy_locked)
        if representation_json is not None
        else legacy_locked
    )
    summary = {
        "schema_version": 2 if representation_json is not None else 1,
        "locked": locked,
        "curvature": summarize_curvature(_load(curvature_json)),
        "real": summarize_real(_load(real_ablation_json)),
    }
    summary = _jsonable(summary)
    summary["macros"] = _macros(summary)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tex = ["% Generated by paper/scripts/results/make_result_macros.py. Do not edit."]
    tex.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(summary["macros"].items()))
    (out / "filaseg_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (out / "filaseg_results_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "filaseg_baseline_table.tex").write_text(_baseline_table(summary), encoding="utf-8")
    (out / "filaseg_sanity_table.tex").write_text(_sanity_table(summary), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locked-json", type=Path, required=True)
    ap.add_argument("--curvature-json", type=Path, required=True)
    ap.add_argument("--real-ablation-json", type=Path, required=True)
    ap.add_argument(
        "--representation-json",
        type=Path,
        help="corrected representation-audit report; uses its stored stratified summaries",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "results")
    args = ap.parse_args(argv)
    try:
        build_results(
            args.locked_json,
            args.curvature_json,
            args.real_ablation_json,
            args.out_dir,
            representation_json=args.representation_json,
        )
    except ValueError as exc:
        ap.error(str(exc))
    print(f"wrote result macros to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
