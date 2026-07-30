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


_METHODS = (
    ("minimum_turn_centerline", "Minimum-turn centreline", ORANGE, "^"),
    ("filaseg_stage3_centerline", "FilaSeg Stage-3 centreline", GREEN, "o"),
)


def _density_summaries(rows: list[dict], densities: list[int]) -> dict[str, dict[int, "stats_util.Summary"]]:
    """Per-method, per-density `stats_util.summarize` over scene F1 values."""
    out: dict[str, dict[int, "stats_util.Summary"]] = {}
    for method, *_ in _METHODS:
        per_density: dict[int, "stats_util.Summary"] = {}
        for density in densities:
            values = [_value(row, method) for row in rows if int(row["density"]) == density]
            values = [value for value in values if value is not None]
            per_density[density] = stats_util.summarize(values)
        out[method] = per_density
    return out


def _plot_panel(ax, rows: list[dict], densities: list[int], title: str,
                *, show_legend: bool) -> None:
    """Draw one degraded/clean panel: per-density mean, 95% CI, and the line
    connecting means across density levels. No individual scene points are
    drawn (25 scenes x 2 methods x 5 densities would bury the means).

    Identical statistical treatment and visual conventions for every caller:
    same `stats_util.summarize` bootstrap (seed and replicate count fixed in
    `eval/stats/stats_util.py`), same colours/markers, same y-limits.
    """
    summaries = _density_summaries(rows, densities)
    for method, label, color, marker in _METHODS:
        means = np.asarray([summaries[method][d].mean for d in densities], float)
        lo = np.asarray([summaries[method][d].ci_lo for d in densities], float)
        hi = np.asarray([summaries[method][d].ci_hi for d in densities], float)
        ax.errorbar(densities, means, yerr=np.vstack([means - lo, hi - means]),
                    color=color, marker=marker, lw=1.8, ms=5.5, capsize=3,
                    label=label, zorder=4)
    ax.set_xticks(densities)
    ax.set_xticklabels([f"{density}%" for density in densities])
    # "coverage", not "density": the manuscript reserves "density" for
    # centreline-length density, which is a different quantity (Section 3.2).
    ax.set_xlabel("Prescribed areal coverage")
    ax.set_title(title, loc="left")
    if show_legend:
        ax.legend(frameon=False, loc="lower left", fontsize=8)
    thin_spines(ax)


def _shared_ylim(rows: list[dict], clean_rows: list[dict], densities: list[int]) -> tuple[float, float]:
    """Tight shared y-limits spanning every plotted mean +/- CI in both panels.

    Padding is a fixed fraction of the observed CI range, then snapped
    outward to the nearest 0.05 so no error bar touches a panel edge; the
    axis is not forced to start at 0.
    """
    bounds: list[float] = []
    for data in (rows, clean_rows):
        summaries = _density_summaries(data, densities)
        for method, *_ in _METHODS:
            for density in densities:
                s = summaries[method][density]
                bounds.append(s.ci_lo)
                bounds.append(s.ci_hi)
    lo, hi = min(bounds), max(bounds)
    pad = 0.10 * (hi - lo)
    lo_padded = float(np.floor((lo - pad) / 0.05) * 0.05)
    hi_padded = float(np.ceil((hi + pad) / 0.05) * 0.05)
    return max(0.0, lo_padded), min(1.0, hi_padded)


def locked_comparison(document: dict, clean_document: dict) -> None:
    """Plot the locked Stage-3 centreline comparison: degraded vs clean masks.

    Panel (a) is the primary evaluation (`mask_w1.png`, degraded/realistic
    UNet-like input). Panel (b) is the secondary analysis on the same 125
    scenes' undegraded 1px-axis input (`mask_clean.png`). Both panels share
    identical axes, colours, markers, and bootstrap statistics -- only the
    input mask differs.
    """
    apply_style()
    rows = document.get("per_scene", [])
    clean_rows = clean_document.get("per_scene", [])
    if not rows:
        raise ValueError("corrected representation report has no per-scene rows")
    if not clean_rows:
        raise ValueError("clean-mask secondary report has no per-scene rows")

    required = {method for method, *_ in _METHODS}
    for name, data in (("degraded", rows), ("clean", clean_rows)):
        available = set().union(*(row.get("scores", {}) for row in data))
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"locked comparison requires {name}-mask scores for " + ", ".join(missing)
            )

    densities = sorted({int(row["density"]) for row in rows})
    clean_densities = sorted({int(row["density"]) for row in clean_rows})
    if densities != clean_densities:
        raise ValueError(
            f"degraded densities {densities} != clean densities {clean_densities}"
        )

    scene_ids = {str(row.get("geometry_id")) for row in rows}
    clean_scene_ids = {str(row.get("geometry_id")) for row in clean_rows}
    if scene_ids != clean_scene_ids:
        raise ValueError(
            "panels are not paired scene-by-scene: degraded and clean scene-ID "
            f"sets differ (degraded-only={sorted(scene_ids - clean_scene_ids)[:5]}, "
            f"clean-only={sorted(clean_scene_ids - scene_ids)[:5]})"
        )
    n_scenes = len(scene_ids)
    n_per_density = {
        density: sum(1 for row in rows if int(row["density"]) == density)
        for density in densities
    }
    if len(set(n_per_density.values())) != 1:
        raise ValueError(f"unequal scene counts per density: {n_per_density}")
    n_each = next(iter(n_per_density.values()))

    fig, (ax_deg, ax_clean) = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
    _plot_panel(ax_deg, rows, densities, "(a) Degraded masks", show_legend=True)
    _plot_panel(ax_clean, clean_rows, densities, "(b) Clean masks", show_legend=False)
    ylo, yhi = _shared_ylim(rows, clean_rows, densities)
    ax_deg.set_ylim(ylo, yhi)  # shared via sharey=True; set once is sufficient
    ax_deg.set_ylabel("Common-fragment pairwise F$_1$")
    fig.suptitle(
        "Stage-3 centreline comparison: degraded vs. clean input masks",
        x=0.02, ha="left", fontsize=10.5, y=0.975,
    )
    note = (
        f"Same {n_scenes} scenes in both panels ({n_each} per density level), paired scene-by-scene. "
        "Markers show the per-density mean common-fragment F$_1$ (individual scenes not plotted); "
        "error bars are 95% percentile bootstrap CIs over scenes "
        f"(seed {stats_util.BOOTSTRAP_SEED}, {stats_util.N_BOOTSTRAP} resamples). "
        "(a) mask_w1.png -- the primary, predefined input (degraded/UNet-realistic; "
        "mean foreground width ~1.8 px, area/skeleton-length). "
        "(b) mask_clean.png -- a secondary, non-predefined analysis on the undegraded, "
        "exactly-1px axis mask."
    )
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=6.3, color=GRAY,
              wrap=True)
    fig.tight_layout(rect=(0.0, 0.155, 1.0, 0.90))
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
                    help="Corrected representation-audit report for Figure 4 panel (a).")
    ap.add_argument("--clean-mask-json", type=Path,
                    default=HERE.parents[1] / "results" / "clean_mask_secondary.json",
                    help="Clean-mask secondary-analysis report for Figure 4 panel (b) "
                         "(default: <paper repo>/results/clean_mask_secondary.json).")
    # Optional: the curvature/q_max development study was cut from this draft,
    # so its figure is no longer part of the manuscript. Emitting it
    # unconditionally left an orphan fig_curvature_qfit.pdf/.png in figures/ on
    # every run. Pass --curvature-json only if you actually want that figure.
    ap.add_argument("--curvature-json", type=Path, default=None)
    args = ap.parse_args()
    locked_path = args.corrected_representation_json or args.locked_json
    if locked_path is None:
        ap.error("one of --corrected-representation-json or --locked-json is required")
    locked_comparison(_read(locked_path), _read(args.clean_mask_json))
    if args.curvature_json is not None:
        curvature_study(_read(args.curvature_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
