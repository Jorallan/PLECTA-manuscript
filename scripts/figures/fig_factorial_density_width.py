"""Development-only centreline-density x bundle-width factorial figure.

Reads the empirical Stage-3 per-scene rows merged into
``results/development_factorial_density_width.json`` (key
``empirical_stage3_per_scene``; see the module docstring of
``scripts/results/make_development_study_tables.py`` for why that key
exists) and renders:

  * a main panel of mean common-fragment F1 vs. ACHIEVED centreline-length
    density, one curve per true bundle-width level (6/11/16 px), with
    individual-scene points in the background and a 95% non-parametric
    percentile-bootstrap confidence interval on the mean;
  * a small companion panel sharing the x-axis, showing pairwise precision
    and recall pooled across widths (structurally and empirically
    width-invariant -- see below -- so the 3 width replicates are not
    independent evidence and are de-duplicated to one row per seed before
    pooling).

Central finding this figure exists to communicate honestly: because FilaSeg
Stages 1-3 read only the thin binary axis mask (identical across width
levels by construction -- see ``eval/studies/factorial_density_width.py``
in the coding repo, ``stage3_width_negative_control``), the three
bundle-width curves in the main panel are numerically IDENTICAL at every
density level (empirically confirmed: maximum within-geometry F1 range =
0.0 across all 12 (seed, density) blocks). Rather than let three identical
curves silently overplot into one and look like a single-series figure,
each width level is drawn with a small deliberate horizontal dodge and a
distinct open marker shape; the coincidence is stated in an on-figure
annotation rather than hidden or faked with perturbed y-values.

With only 4 independent seeds per (density, width) cell this is
EXPLORATORY, development-only evidence. Individual per-scene points are
always shown and the replicate count is stated on the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import GRAY, PALETTE_PRF, apply_style, save_fig, thin_spines  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed bootstrap seed / replicate count (module constants; declared exactly
# once, reused for every panel in this figure).
# ---------------------------------------------------------------------------
BOOTSTRAP_SEED = 20260730
N_BOOTSTRAP = 10000

WIDTH_LEVELS = (6.0, 11.0, 16.0)
WIDTH_COLORS = {6.0: "#553c9a", 11.0: "#b7791f", 16.0: "#2c7a7b"}  # purple/amber/teal
WIDTH_MARKERS = {6.0: "o", 11.0: "s", 16.0: "^"}
WIDTH_LABELS = {6.0: "6 px", 11.0: "11 px", 16.0: "16 px"}
# Fraction of the total achieved-density span used as the per-width dodge
# step (so curves that coincide exactly are still legible side by side).
DODGE_FRACTION = 0.028


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = N_BOOTSTRAP,
                   alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (non-parametric, resample scenes)."""
    n = values.size
    if n < 2:
        v = float(values[0]) if n == 1 else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _load_rows(report_path: Path) -> list[dict]:
    document = json.loads(report_path.read_text(encoding="utf-8"))
    rows = document.get("empirical_stage3_per_scene")
    if not rows:
        raise ValueError(
            f"{report_path} has no 'empirical_stage3_per_scene' rows -- "
            "run the empirical Stage-3 arm and merge it in before plotting"
        )
    control = document.get("empirical_stage3_width_negative_control", {})
    if control.get("status") != "pass":
        raise ValueError(
            f"empirical_stage3_width_negative_control status is "
            f"{control.get('status')!r}, not 'pass' -- refusing to render "
            "the figure's width-invariance framing against a failed control"
        )
    return rows, float(control["maximum_within_geometry_score_range"])


def _by_density_width(rows: list[dict]) -> dict[float, dict[float, list[dict]]]:
    out: dict[float, dict[float, list[dict]]] = {}
    for row in rows:
        density = float(row["target_length_density_px_per_px2"])
        width = float(row["bundle_width_level_px"])
        out.setdefault(density, {}).setdefault(width, []).append(row)
    return out


def _dedup_by_seed(rows: list[dict]) -> list[dict]:
    """One row per seed (lowest width), for pooled width-invariant panels."""
    by_seed: dict[int, dict] = {}
    for row in sorted(rows, key=lambda r: float(r["bundle_width_level_px"])):
        by_seed.setdefault(int(row["seed"]), row)
    return list(by_seed.values())


def _mean_ci(values: list[float], seed: int) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    lo, hi = _bootstrap_ci(arr, seed)
    return mean, lo, hi, int(arr.size)


def build(report_path: Path) -> dict[str, Any]:
    rows, max_range = _load_rows(report_path)
    grouped = _by_density_width(rows)
    densities = sorted(grouped)

    x_span = max(
        float(r["achieved_length_density_px_per_px2"]) for r in rows
    ) - min(float(r["achieved_length_density_px_per_px2"]) for r in rows)
    dodge_step = DODGE_FRACTION * x_span
    dodge = {6.0: -dodge_step, 11.0: 0.0, 16.0: dodge_step}

    apply_style()
    fig, (ax_f1, ax_pr) = plt.subplots(
        2, 1, figsize=(6.3, 6.4), sharex=False,
        height_ratios=(2.1, 1.0), layout="constrained",
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "figure": "fig_factorial_density_width",
        "source_report": str(report_path),
        "bootstrap": {
            "method": "non-parametric percentile bootstrap of the mean, "
                      "resampling the n=4 independent seeds within each "
                      "(density, width) cell",
            "seed": BOOTSTRAP_SEED,
            "n_boot": N_BOOTSTRAP,
        },
        "empirical_stage3_width_negative_control_max_range": max_range,
        "per_density_width": {},
        "per_density_pooled_precision_recall": {},
    }

    # ---- main panel: common-fragment F1 -----------------------------------
    density_x_center = {}
    for density in densities:
        all_x = [
            float(r["achieved_length_density_px_per_px2"])
            for width in WIDTH_LEVELS for r in grouped[density][width]
        ]
        density_x_center[density] = float(np.mean(all_x))

    for width in WIDTH_LEVELS:
        color = WIDTH_COLORS[width]
        marker = WIDTH_MARKERS[width]
        xs_mean, ys_mean, los, his = [], [], [], []
        for density in densities:
            cell_rows = grouped[density][width]
            xs_scene = [
                float(r["achieved_length_density_px_per_px2"]) + dodge[width]
                for r in cell_rows
            ]
            ys_scene = [float(r["common_f1"]) for r in cell_rows]
            ax_f1.scatter(
                xs_scene, ys_scene, s=16, color=color, alpha=0.35,
                linewidths=0, zorder=2,
            )
            mean, lo, hi, n = _mean_ci(
                ys_scene, BOOTSTRAP_SEED + int(round(density * 100000)) + int(width)
            )
            xc = density_x_center[density] + dodge[width]
            xs_mean.append(xc)
            ys_mean.append(mean)
            los.append(lo)
            his.append(hi)
            summary["per_density_width"].setdefault(str(density), {})[str(width)] = {
                "n": n, "mean_f1": mean, "ci_lo": lo, "ci_hi": hi,
            }
        ax_f1.errorbar(
            xs_mean, ys_mean, yerr=[np.array(ys_mean) - np.array(los),
                                     np.array(his) - np.array(ys_mean)],
            fmt=marker, color=color, mfc="white", mec=color, mew=1.6, ms=8.0,
            elinewidth=1.6, capsize=3.5, lw=1.8, ls="-",
            label=f"true width {WIDTH_LABELS[width]}", zorder=4,
        )

    ax_f1.set_ylabel("Common-fragment F$_1$ (Stage 3)")
    ax_f1.set_title(
        "(a) Development factorial: centreline density $\\times$ true bundle width",
        loc="left",
    )
    ax_f1.legend(loc="upper right", title="True bundle width (dodged)", ncol=1,
                 fontsize=8.6, title_fontsize=8.6)
    thin_spines(ax_f1)
    ax_f1.set_ylim(0.55, 1.0)

    # Secondary annotation: crossing density per megapixel, on a proper top
    # secondary x-axis (also exactly width-invariant; one value per density
    # level) -- kept as a real axis rather than in-plot text so it cannot
    # collide with the legend or the panel title.
    crossings = [
        float(np.mean([
            float(r["crossing_density_per_megapixel"])
            for width in WIDTH_LEVELS for r in grouped[density][width]
        ]))
        for density in densities
    ]
    xs_nodes = [density_x_center[d] for d in densities]

    def _to_crossings(x):
        return np.interp(x, xs_nodes, crossings)

    def _to_density(c):
        return np.interp(c, crossings, xs_nodes)

    secax = ax_f1.secondary_xaxis("top", functions=(_to_crossings, _to_density))
    secax.set_xticks(crossings)
    secax.set_xticklabels([f"{c:.0f}" for c in crossings], fontsize=8.0)
    secax.set_xlabel("Crossing density (per Mpx)", fontsize=8.6, color=GRAY)
    secax.tick_params(colors=GRAY, labelsize=8.0)
    secax.spines["top"].set_color(GRAY)

    ax_f1.text(
        0.02, 0.03,
        "The three width curves are exactly coincident at every density level\n"
        "(max. within-geometry $F_1$ range $=$"
        f" {max_range:.3f}); the small horizontal dodge is drawn deliberately\n"
        "so all three remain visible rather than overplotting.",
        transform=ax_f1.transAxes, ha="left", va="bottom", fontsize=7.3,
        color=GRAY, style="italic",
    )

    # ---- companion panel: pairwise precision / recall, pooled over width --
    for metric_key, label, color, ls in (
        ("common_precision", "Precision", PALETTE_PRF["precision"], "-"),
        ("common_recall", "Recall", PALETTE_PRF["recall"], "--"),
    ):
        xs_mean, ys_mean, los, his = [], [], [], []
        for density in densities:
            pooled_rows = _dedup_by_seed(
                [r for width in WIDTH_LEVELS for r in grouped[density][width]]
            )
            values = [float(r[metric_key]) for r in pooled_rows]
            mean, lo, hi, n = _mean_ci(
                values, BOOTSTRAP_SEED + int(round(density * 100000)) + 7
            )
            xs_mean.append(density_x_center[density])
            ys_mean.append(mean)
            los.append(lo)
            his.append(hi)
            summary["per_density_pooled_precision_recall"].setdefault(
                str(density), {}
            )[metric_key] = {"n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi}
        ax_pr.fill_between(xs_mean, los, his, color=color, alpha=0.18, lw=0)
        ax_pr.plot(
            xs_mean, ys_mean, ls, color=color, marker="o", ms=5.0,
            mfc="white", mec=color, mew=1.4, lw=1.8, label=label,
        )

    ax_pr.set_xlabel("Achieved centreline-length density (px / px$^2$)")
    ax_pr.set_ylabel("Pairwise score")
    ax_pr.set_title(
        "(b) Precision / recall, pooled over width (width-invariant)",
        loc="left", fontsize=10.0,
    )
    ax_pr.legend(loc="lower left", ncol=2)
    thin_spines(ax_pr)
    ax_pr.set_ylim(0.55, 1.0)
    ax_pr.set_xlim(*ax_f1.get_xlim())

    n_seeds = len({int(r["seed"]) for r in rows})
    fig.text(
        0.5, -0.02,
        f"Development-only, exploratory: n = {n_seeds} seeds per cell "
        f"(individual scenes always shown).\n"
        f"95% bootstrap CI of the mean, seed {BOOTSTRAP_SEED}, "
        f"{N_BOOTSTRAP} replicates.",
        fontsize=7.6, color=GRAY, ha="center", va="top", linespacing=1.5,
    )

    save_fig(fig, "fig_factorial_density_width", bbox_inches="tight",
             pad_inches=0.06)
    plt.close(fig)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report", type=Path,
        default=Path(__file__).resolve().parents[2] / "results" /
        "development_factorial_density_width.json",
    )
    ap.add_argument(
        "--out-json", type=Path, default=None,
        help="Optional diagnostic summary JSON (bootstrap means/CIs actually "
             "plotted). Written under the coding repo's output/ scratch tree "
             "by default -- this script owns no extra results/ file in the "
             "paper repo.",
    )
    args = ap.parse_args()
    summary = build(args.report)
    out_json = args.out_json or (
        Path(r"C:\Repos\filaments_quantification") / "output" /
        "development_factorial_density_width" /
        "fig_factorial_density_width_summary.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
