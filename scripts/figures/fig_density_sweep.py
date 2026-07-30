"""Development-only density sweep: FilaSeg vs. the minimum-turn tracer.

Reads ``results/development_density_sweep.json`` and renders one panel:
common-fragment F1 against ACHIEVED centreline-length density for both
methods, with crossing density per megapixel on a secondary top axis and a
95% non-parametric percentile-bootstrap confidence interval on each mean.

Design notes
------------
* Five target length-density levels (0.01 - 0.06 px/px^2), four independent
  seeds each, at a single target bundle-width level (6 px).
* Fixing bundle width is not a simplification: Stages 1-3 read only the thin
  binary axis mask, which the generator holds byte-identical across width
  levels at fixed centreline geometry, and the empirical negative control in
  the parent factorial confirmed a maximum within-geometry F1 range of 0.000
  across widths. Width is therefore inert for this endpoint and one level is
  sufficient.
* Both methods receive the SAME degraded input mask per scene, so the two
  curves are paired scene-by-scene.
* Every value is read from the machine-generated JSON. The input-mask
  description and its measured foreground width are read from the stored
  provenance block rather than restated here, so the figure cannot drift out
  of step with the data.

With only 4 independent seeds per density this is EXPLORATORY,
development-only evidence; the replicate count and bootstrap settings are
stated on the figure. Per-scene points are not drawn -- dispersion is carried
by the bootstrap confidence interval.
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
from _style import GRAY, GREEN, ORANGE, apply_style, save_fig, thin_spines  # noqa: E402

BOOTSTRAP_SEED = 20260730
N_BOOTSTRAP = 10000

# (json method key, legend label, colour, linestyle, marker, linewidth)
# Method colours match Figure 4 (fig_locked_results.py), which plots the same
# two methods: FilaSeg green, minimum-turn orange. Keeping them identical means
# a reader flipping between the two head-to-head figures does not have to
# re-learn the legend.
METHODS = (
    ("filaseg", "FilaSeg (Stage 3)", GREEN, "-", "o", 2.1),
    ("baseline_skeleton", "Minimum-turn tracer", ORANGE, (0, (5, 2)), "s", 1.8),
)
MARKER_SIZE = 7.5


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = N_BOOTSTRAP,
                   alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (non-parametric, resample scenes)."""
    rng = np.random.default_rng(seed)
    n = values.size
    if n < 2:
        return float(values.mean()), float(values.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100.0 * alpha / 2.0)),
        float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
    )


def _load(report_path: Path) -> dict[str, Any]:
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if not document.get("per_scene"):
        raise ValueError(f"{report_path} has no 'per_scene' rows")
    return document


def build(report_path: Path) -> dict[str, Any]:
    document = _load(report_path)
    rows = document["per_scene"]
    provenance = document.get("provenance", {})

    by_density: dict[float, list[dict]] = {}
    for row in rows:
        by_density.setdefault(
            float(row["target_length_density_px_per_px2"]), []
        ).append(row)
    densities = sorted(by_density)

    apply_style()
    fig, ax = plt.subplots(figsize=(6.3, 4.1), layout="constrained")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "figure": "fig_density_sweep",
        "source_report": str(report_path),
        "bootstrap": {
            "method": "non-parametric percentile bootstrap of the mean, "
                      "resampling the independent seeds within each density",
            "seed": BOOTSTRAP_SEED,
            "n_boot": N_BOOTSTRAP,
        },
        "per_density": {},
    }

    x_center = {
        d: float(np.mean([
            float(r["achieved_length_density_px_per_px2"]) for r in by_density[d]
        ]))
        for d in densities
    }

    for key, label, color, linestyle, marker, lw in METHODS:
        xs, ys, los, his = [], [], [], []
        for density in densities:
            values = np.array(
                [float(r["methods"][key]["common_f1"]) for r in by_density[density]],
                dtype=float,
            )
            mean = float(values.mean())
            lo, hi = _bootstrap_ci(
                values, BOOTSTRAP_SEED + int(round(density * 100000)) + len(key)
            )
            xs.append(x_center[density])
            ys.append(mean)
            los.append(lo)
            his.append(hi)
            summary["per_density"].setdefault(str(density), {})[key] = {
                "n": int(values.size), "mean_f1": mean, "ci_lo": lo, "ci_hi": hi,
            }
        ax.fill_between(xs, los, his, color=color, alpha=0.15, lw=0)
        ax.plot(
            xs, ys, linestyle=linestyle, color=color, lw=lw, marker=marker,
            ms=MARKER_SIZE, mfc="white", mec=color, mew=1.6, label=label,
        )

    ax.set_xlabel("Achieved centreline-length density (px / px$^2$)")
    ax.set_ylabel("Common-fragment $F_1$")
    ax.set_ylim(0.45, 1.0)
    thin_spines(ax)

    # Input-mask provenance belongs with the legend: it is the single most
    # load-bearing fact about what these curves actually measure.
    measured = provenance.get("input_mask", {}).get("measured", {})
    width_px = measured.get("mask.png_mean_foreground_width_px")
    width_txt = f", measured mean width {width_px:.1f} px" if width_px else ""
    bundle_px = provenance.get("generator", {}).get("bundle_width_level_px")
    bundle_txt = f"; target bundle width {bundle_px:.0f} px" if bundle_px else ""
    ax.legend(
        loc="lower left", fontsize=8.6,
        title=(
            "Same degraded, nominal one-pixel axis mask\n"
            f"for both methods{width_txt}{bundle_txt}"
        ),
        title_fontsize=7.8,
    )

    crossings = [
        float(np.mean([
            float(r["crossing_density_per_megapixel"]) for r in by_density[d]
        ]))
        for d in densities
    ]
    xs_nodes = [x_center[d] for d in densities]

    def _to_crossings(x):
        return np.interp(x, xs_nodes, crossings)

    def _to_density(c):
        return np.interp(c, crossings, xs_nodes)

    secax = ax.secondary_xaxis("top", functions=(_to_crossings, _to_density))
    secax.set_xticks(crossings)
    secax.set_xticklabels([f"{c:.0f}" for c in crossings], fontsize=8.0)
    secax.set_xlabel("Crossing density (per Mpx)", fontsize=8.6, color=GRAY)
    secax.tick_params(colors=GRAY, labelsize=8.0)
    secax.spines["top"].set_color(GRAY)

    # No in-figure footer note: the development-only scope, the replicate count,
    # the pairing and the bootstrap settings are all stated in the LaTeX caption,
    # and repeating them under the axes only crowds the figure.
    summary["n_seeds"] = len({int(r["seed"]) for r in rows})

    save_fig(fig, "fig_density_sweep", bbox_inches="tight", pad_inches=0.06)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report", type=Path,
        default=Path(__file__).resolve().parents[2] / "results" /
        "development_density_sweep.json",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    summary = build(args.report)
    out_json = args.out_json or (
        Path(r"C:\Repos\filaments_quantification") / "output" /
        "development_density_sweep" / "fig_density_sweep_summary.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
