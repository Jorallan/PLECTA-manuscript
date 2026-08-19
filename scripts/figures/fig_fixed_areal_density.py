"""Development-only fixed-thick-areal-density x target-bundle-width figure.

Reads ``results/development_fixed_areal_density.json`` (raw per-scene rows,
key ``per_scene``; see the study's provenance block for how it was produced:
Stage 1-3 of ``Tools/run_full_sem_pipeline.py``, scored on the Stage-3
``3.reconnect`` centreline-instance artifact with
``eval/core/common_metric.py::score_method``, the SAME metric used by the
primary/locked evaluation and by the companion
``development_factorial_density_width`` study) and renders:

  * a main panel of mean common-fragment F1 vs. the ACTUAL measured
    thick-bundle areal density (never the calibration target), one curve per
    target bundle-width level (6/11/16 px), with individual-scene points in
    the background and a 95% non-parametric percentile-bootstrap confidence
    interval on the mean;
  * a companion panel sharing the same x-axis, showing the ACTUAL measured
    centreline-length density each width needed to reach that areal density.

Deliberate framing (preserved from the study design -- see
``results/development_fixed_areal_density.json``'s ``scope`` field): this is
NOT an isolated causal test of bundle width. At a fixed thick-bundle areal
density target, the centreline-length density (and therefore crossing
density) required to reach it is allowed to -- and does -- differ by width.
The companion panel makes that difference visible rather than hiding it; the
main panel's x position for each width is its own true measured areal
density, never displaced to force alignment across widths.

With only 5 independently seeded scenes per (width, target) cell this is
EXPLORATORY, development-only evidence. Individual per-scene points are
always shown and the replicate count is stated on the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import GRAY, apply_style, save_fig, thin_spines  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed bootstrap seed / replicate count (module constants; declared exactly
# once, reused for every panel in this figure). Deliberately a different
# constant from fig_factorial_density_width.py's BOOTSTRAP_SEED so the two
# figures' resampling draws are never accidentally coupled.
# ---------------------------------------------------------------------------
BOOTSTRAP_SEED = 20267030
N_BOOTSTRAP = 10000

WIDTH_LEVELS = (6.0, 11.0, 16.0)
# Same palette/marker family as fig_factorial_density_width.py for
# cross-figure consistency (both encode the same 3 target-bundle-width
# levels).
WIDTH_COLORS = {6.0: "#553c9a", 11.0: "#b7791f", 16.0: "#2c7a7b"}  # purple/amber/teal
WIDTH_MARKERS = {6.0: "o", 11.0: "s", 16.0: "^"}
WIDTH_LABELS = {6.0: "6 px", 11.0: "11 px", 16.0: "16 px"}


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


def _mean_ci(values: list[float], seed: int) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    lo, hi = _bootstrap_ci(arr, seed)
    return mean, lo, hi, int(arr.size)


def _load_rows(report_path: Path) -> tuple[dict, list[dict]]:
    document = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [r for r in document.get("per_scene", []) if r.get("status") == "ok"]
    if not rows:
        raise ValueError(
            f"{report_path} has no successful 'per_scene' rows to plot"
        )
    overlap = document.get("overlap_check", [])
    if not overlap:
        raise ValueError(
            f"{report_path} has no 'overlap_check' block -- refusing to "
            "render without the documented cross-width overlap evidence"
        )
    for entry in overlap:
        if not entry.get("all_three_ranges_overlap"):
            raise ValueError(
                f"target {entry.get('target_areal_density')} does not have "
                "overlapping achieved-coverage ranges across all three "
                "widths -- the study driver should have dropped it before "
                "writing the results file"
            )
    return document, rows


def _by_target_width(rows: list[dict]) -> dict[float, dict[float, list[dict]]]:
    out: dict[float, dict[float, list[dict]]] = {}
    for row in rows:
        target = float(row["target_areal_density"])
        width = float(row["bundle_width_level_px"])
        out.setdefault(target, {}).setdefault(width, []).append(row)
    return out


def build(report_path: Path) -> dict[str, Any]:
    document, rows = _load_rows(report_path)
    grouped = _by_target_width(rows)
    targets = sorted(grouped)
    n_seeds = len({int(r["seed"]) for r in rows})

    apply_style()
    fig, (ax_f1, ax_ld) = plt.subplots(
        2, 1, figsize=(6.3, 6.6), sharex=False,
        height_ratios=(2.0, 1.15), layout="constrained",
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "figure": "fig_fixed_areal_density",
        "source_report": str(report_path),
        "bootstrap": {
            "method": "non-parametric percentile bootstrap of the mean, "
                      f"resampling the n={n_seeds} independent seeds within "
                      "each (target areal density, width) cell",
            "seed": BOOTSTRAP_SEED,
            "n_boot": N_BOOTSTRAP,
        },
        "per_target_width_f1": {},
        "per_target_width_length_density": {},
    }

    # ---- main panel: common-fragment F1 vs. ACTUAL measured areal density -
    for width in WIDTH_LEVELS:
        color = WIDTH_COLORS[width]
        marker = WIDTH_MARKERS[width]
        xs_mean, ys_mean, los, his = [], [], [], []
        for target in targets:
            cell_rows = grouped[target][width]
            xs_scene = [float(r["achieved_thick_mask_coverage"]) for r in cell_rows]
            ys_scene = [float(r["common_f1"]) for r in cell_rows]
            ax_f1.scatter(
                xs_scene, ys_scene, s=20, color=color, alpha=0.4,
                linewidths=0, zorder=2,
            )
            mean_y, lo, hi, n = _mean_ci(
                ys_scene,
                BOOTSTRAP_SEED + int(round(target * 100000)) + int(width),
            )
            mean_x = float(np.mean(xs_scene))  # true measured mean, never the target
            xs_mean.append(mean_x)
            ys_mean.append(mean_y)
            los.append(lo)
            his.append(hi)
            summary["per_target_width_f1"].setdefault(str(target), {})[str(width)] = {
                "n": n, "mean_achieved_areal_density": mean_x,
                "mean_f1": mean_y, "ci_lo": lo, "ci_hi": hi,
            }
        xs_mean_arr, ys_mean_arr = np.array(xs_mean), np.array(ys_mean)
        ax_f1.errorbar(
            xs_mean_arr, ys_mean_arr,
            yerr=[ys_mean_arr - np.array(los), np.array(his) - ys_mean_arr],
            fmt=marker, color=color, mfc="white", mec=color, mew=1.6, ms=8.0,
            elinewidth=1.6, capsize=3.5, lw=1.8, ls="-",
            label=f"target width {WIDTH_LABELS[width]}", zorder=4,
        )

    ax_f1.set_ylabel("Common-fragment F$_1$ (Stage 3)")
    ax_f1.set_title(
        "(a) Fixed thick-bundle areal density: common-fragment F$_1$ by "
        "target bundle width",
        loc="left", fontsize=10.3,
    )
    ax_f1.legend(loc="upper right", title="Target bundle width", ncol=1,
                 fontsize=8.6, title_fontsize=8.6)
    thin_spines(ax_f1)

    ax_f1.text(
        0.02, 0.03,
        "NOT an isolated causal test of width: at each shared areal-density\n"
        "level the centreline-length density needed to reach it differs by\n"
        "target width (panel b). Every point sits at its true measured\n"
        "areal density -- never the calibration target, never displaced.",
        transform=ax_f1.transAxes, ha="left", va="bottom", fontsize=7.3,
        color=GRAY, style="italic",
    )

    # ---- companion panel: achieved centreline-length density --------------
    for width in WIDTH_LEVELS:
        color = WIDTH_COLORS[width]
        marker = WIDTH_MARKERS[width]
        xs_mean, ys_mean = [], []
        for target in targets:
            cell_rows = grouped[target][width]
            xs_scene = [float(r["achieved_thick_mask_coverage"]) for r in cell_rows]
            ys_scene = [float(r["achieved_length_density_px_per_px2"]) for r in cell_rows]
            ax_ld.scatter(
                xs_scene, ys_scene, s=14, color=color, alpha=0.35,
                linewidths=0, zorder=2,
            )
            mean_x = float(np.mean(xs_scene))
            mean_y = float(np.mean(ys_scene))
            xs_mean.append(mean_x)
            ys_mean.append(mean_y)
            summary["per_target_width_length_density"].setdefault(
                str(target), {}
            )[str(width)] = {
                "mean_achieved_areal_density": mean_x,
                "mean_achieved_length_density_px_per_px2": mean_y,
            }
        ax_ld.plot(
            xs_mean, ys_mean, marker=marker, color=color, mfc="white",
            mec=color, mew=1.4, ms=7.0, lw=1.6, ls="-",
        )

    ax_ld.set_xlabel("Achieved thick-bundle areal density (fraction of frame)")
    ax_ld.set_ylabel("Achieved centreline-length\ndensity (px / px$^2$)")
    ax_ld.set_title(
        "(b) Centreline density needed to reach the SAME areal density "
        "differs by target width",
        loc="left", fontsize=9.3,
    )
    thin_spines(ax_ld)
    ax_ld.set_xlim(*ax_f1.get_xlim())

    fig.text(
        0.5, -0.02,
        f"Development-only, exploratory: n = {n_seeds} seeds per cell "
        f"(individual scenes always shown).\n"
        f"95% bootstrap CI of the mean, seed {BOOTSTRAP_SEED}, "
        f"{N_BOOTSTRAP} replicates.",
        fontsize=7.6, color=GRAY, ha="center", va="top", linespacing=1.5,
    )

    save_fig(fig, "fig_fixed_areal_density", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report", type=Path,
        default=Path(__file__).resolve().parents[2] / "results" /
        "development_fixed_areal_density.json",
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
        "development_fixed_areal_density" /
        "fig_fixed_areal_density_summary.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
