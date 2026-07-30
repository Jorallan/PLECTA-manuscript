"""Generate locked method-comparison and development curvature-study figures."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)
from _style import apply_style, save_fig, BLUE, ORANGE, GREEN, GRAY, thin_spines  # noqa: E402
import stats_util  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _value(row: dict, method: str | None = None) -> float | None:
    """Return a scene-level common-fragment F1 from either report schema."""
    scores = row.get("scores", {})
    value = (scores.get(method, {}) if method is not None else scores).get(
        "common_metric", {}).get("f1")
    if method is not None:
        value = scores.get(method, {}).get("f1", value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def locked_comparison(document: dict) -> None:
    """Plot the locked Stage-3 centreline comparison and paired differences."""
    apply_style()
    methods = (
        ("minimum_turn_centerline", "Minimum-turn centreline", ORANGE, "^"),
        ("filaseg_stage3_centerline", "FilaSeg Stage-3 centreline", GREEN, "o"),
    )
    rows = document.get("per_scene", [])
    if not rows:
        raise ValueError("corrected representation report has no per-scene rows")
    densities = sorted({int(row["density"]) for row in rows})
    available = set().union(*(row.get("scores", {}) for row in rows))
    required = {method for method, *_ in methods}
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "locked comparison requires corrected representation scores for "
            + ", ".join(missing)
        )

    fig, (ax, ax_delta) = plt.subplots(
        1, 2, figsize=(9.2, 4.7), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    # Simplified review mirror: plot every scene at its exact prescribed
    # density, with no horizontal jitter (see mirror README).
    scene_jitter = {
        (int(row["density"]), str(row.get("geometry_id", row.get("seed", index)))): 0.0
        for index, row in enumerate(rows)
    }
    by_density: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        density = int(row["density"])
        baseline = _value(row, "minimum_turn_centerline")
        filaseg = _value(row, "filaseg_stage3_centerline")
        if baseline is None or filaseg is None:
            continue
        key = (density, str(row.get("geometry_id", row.get("seed"))))
        by_density[density].append((scene_jitter[key], baseline, filaseg))

    for method, label, color, marker in methods:
        means, lo, hi = [], [], []
        for density in densities:
            values = [_value(row, method) for row in rows if int(row["density"]) == density]
            values = [value for value in values if value is not None]
            summary = stats_util.summarize(values)
            means.append(summary.mean)
            lo.append(summary.ci_lo)
            hi.append(summary.ci_hi)
            for jitter, baseline, filaseg in by_density[density]:
                value = baseline if method == "minimum_turn_centerline" else filaseg
                ax.scatter(density + jitter, value, s=15, alpha=0.55, color=color,
                           edgecolor="none", zorder=2)
        means_arr = np.asarray(means, float)
        lower = means_arr - np.asarray(lo, float)
        upper = np.asarray(hi, float) - means_arr
        ax.errorbar(densities, means_arr, yerr=np.vstack([lower, upper]),
                    color=color, marker=marker, lw=1.8, ms=5.5, capsize=3,
                    label=label, zorder=4)

    for density, pairs in by_density.items():
        for jitter, baseline, filaseg in pairs:
            ax.plot([density + jitter, density + jitter], [baseline, filaseg],
                    color=GRAY, lw=0.55, alpha=0.3, zorder=1)
    delta_means, delta_lo, delta_hi = [], [], []
    for density in densities:
        pairs = by_density[density]
        differences = [filaseg - baseline for _, baseline, filaseg in pairs]
        summary = stats_util.summarize(differences)
        delta_means.append(summary.mean)
        delta_lo.append(summary.ci_lo)
        delta_hi.append(summary.ci_hi)
        ax_delta.scatter([density + jitter for jitter, *_ in pairs], differences,
                         s=16, alpha=0.6, color=GREEN, edgecolor="none", zorder=2)
    delta_means_arr = np.asarray(delta_means, float)
    ax_delta.errorbar(
        densities, delta_means_arr,
        yerr=np.vstack([delta_means_arr - np.asarray(delta_lo),
                         np.asarray(delta_hi) - delta_means_arr]),
        color=GREEN, marker="o", lw=1.8, ms=5.5, capsize=3, zorder=4,
    )
    ax_delta.axhline(0, color=GRAY, lw=0.9, zorder=1)
    ax.set_xticks(densities)
    ax.set_xticklabels([f"{density}%" for density in densities])
    ax.set_xlabel("Target areal density")
    ax.set_ylabel("Common-fragment pairwise F$_1$\n(per-scene points; mean and 95% CI)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Stage-3 centreline comparison", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax_delta.set_xticks(densities)
    ax_delta.set_xticklabels([f"{density}%" for density in densities])
    ax_delta.set_xlabel("Target areal density")
    ax_delta.set_ylabel("FilaSeg Stage-3 centreline minus\nminimum-turn centreline F$_1$")
    delta_limit = max(0.05, float(np.nanmax(np.abs(np.r_[delta_lo, delta_hi]))))
    ax_delta.set_ylim(-delta_limit * 1.15, delta_limit * 1.15)
    ax_delta.set_title("Paired scene differences", loc="left")
    thin_spines(ax)
    thin_spines(ax_delta)
    fig.tight_layout()
    # Fixed metadata keeps the PDF byte-stable across repeated locked renders.
    save_fig(fig, "fig_locked_method_comparison", metadata={
        "Creator": "FilaSeg locked evaluation",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)


def curvature_study(document: dict) -> None:
    apply_style()
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in document.get("runs", []):
        if row.get("case_type") != "synthetic" or row.get("status") != "ok":
            continue
        value = _value(row)
        if value is not None:
            grouped[str(row["variant"])].append(value)
    order = ["legacy_mixed", "geometric_disabled", "geometric_qfit_0.005",
             "geometric_qfit_0.010", "geometric_qfit_0.020",
             "geometric_qfit_0.040", "geometric_qfit_0.080"]
    names = [name for name in order if name in grouped]
    labels = ["Legacy\nmixed", "Geometric\n$q_{max}$ off",
              "$q_{max}$\n0.005", "$q_{max}$\n0.010", "$q_{max}$\n0.020",
              "$q_{max}$\n0.040", "$q_{max}$\n0.080"]
    labels = [label for name, label in zip(order, labels) if name in grouped]
    summaries = [stats_util.summarize(grouped[name]) for name in names]
    means = np.asarray([item.mean for item in summaries])
    lo = np.asarray([item.ci_lo for item in summaries])
    hi = np.asarray([item.ci_hi for item in summaries])
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x, means, color=[GRAY, BLUE] + [ORANGE] * max(0, len(names) - 2),
           width=0.68, alpha=0.9, zorder=2)
    ax.errorbar(x, means, yerr=np.vstack([means - lo, hi - means]), fmt="none",
                color="#222222", capsize=4, lw=1.3, zorder=3)
    rng = np.random.default_rng(20260729)
    for i, name in enumerate(names):
        values = grouped[name]
        ax.scatter(np.full(len(values), i) + rng.uniform(-0.16, 0.16, len(values)),
                   values, s=13, alpha=0.3, color="#222222", edgecolor="none", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Development common-fragment F$_1$")
    ax.set_ylim(max(0, float(np.nanmin(lo)) - 0.12), min(1.0, float(np.nanmax(hi)) + 0.08))
    ax.set_title("Curvature formulation and fit-quality gate (development only)", loc="left")
    thin_spines(ax)
    fig.tight_layout()
    save_fig(fig, "fig_curvature_qfit")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locked-json", type=Path,
                    help="Legacy locked report path (retained for compatibility).")
    ap.add_argument("--corrected-representation-json", type=Path,
                    help="Corrected representation-audit report for Figure 4.")
    ap.add_argument("--curvature-json", type=Path, required=True)
    args = ap.parse_args()
    locked_path = args.corrected_representation_json or args.locked_json
    if locked_path is None:
        ap.error("one of --corrected-representation-json or --locked-json is required")
    locked_comparison(_read(locked_path))
    curvature_study(_read(args.curvature_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
