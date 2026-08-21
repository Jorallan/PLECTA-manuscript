"""Regenerate development mask-width and clean-mask sensitivity figures.

Two development-only sensitivity studies share one presentation grammar:

  * axis-mask width sweep (1 / 2 / 3 px nominal binary mask width)
  * clean vs. degraded one-pixel axis mask

Both are plotted as exactly five mean common-fragment F1 curves -- one per
prescribed areal-coverage density (20/30/40/50/60%) -- each with a paired
percentile-bootstrap 95% confidence band computed *within* that density.
Per-scene traces are not drawn -- dispersion is carried by the bootstrap
confidence band; there is no pooled "all scenes" curve. See ``_density_curve``
for the bootstrap and ``main``/``--figures`` for how the three output figures
are selected.

This script deliberately does not import ``eval/stats_util`` (the canonical
monorepo's shared stats module): this repository is a standalone mirror of
the ``paper/`` tree and does not carry the sibling ``eval/`` package. The
``summarize`` / ``paired_summarize_by_scene`` helpers below reproduce that
module's non-parametric percentile bootstrap exactly -- same algorithm, same
``BOOTSTRAP_SEED`` / ``N_BOOTSTRAP`` constants -- so the ``overall`` /
``per_density`` / ``paired`` numbers written to the output JSON are
unchanged from the canonical pipeline. Every plotted value is read from the
``results/development_*.json`` per-scene rows; nothing is hand-entered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_style, save_fig, thin_spines  # noqa: E402

# ---------------------------------------------------------------------------
# Local statistics helpers (see module docstring for why these are local
# copies of eval/stats_util rather than an import).
# ---------------------------------------------------------------------------
BOOTSTRAP_SEED = 20260729
N_BOOTSTRAP = 10000


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                   alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, over the scene axis."""
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


def summarize(values, seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOTSTRAP) -> dict:
    """Mean, sample standard deviation and 95% bootstrap CI over scenes."""
    v = np.asarray([float(x) for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_halfwidth": float("nan")}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    lo, hi = _bootstrap_ci(v, n_boot, seed)
    return {"n": int(v.size), "mean": mean, "sd": sd, "ci_lo": lo, "ci_hi": hi,
            "ci_halfwidth": float((hi - lo) / 2.0)}


def paired_summarize_by_scene(a: dict, b: dict, seed: int = BOOTSTRAP_SEED,
                               n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap paired method differences after exact scene-ID validation."""
    a_keys, b_keys = set(a), set(b)
    if a_keys != b_keys:
        missing_from_a = sorted(b_keys - a_keys)
        missing_from_b = sorted(a_keys - b_keys)
        raise ValueError(
            "scene IDs must match exactly; "
            f"missing from a={missing_from_a}, missing from b={missing_from_b}"
        )
    scene_ids = sorted(a_keys)
    av = np.asarray([a[k] for k in scene_ids], dtype=float)
    bv = np.asarray([b[k] for k in scene_ids], dtype=float)
    ok = np.isfinite(av) & np.isfinite(bv)
    return summarize(av[ok] - bv[ok], seed=seed, n_boot=n_boot)


def _rows(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    failures = [row for row in document["per_scene"] if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"{path}: {len(failures)} failed rows")
    return document["per_scene"]


def _mapping(rows: list[dict]) -> dict[str, float]:
    return {
        str(row["geometry_id"]): float(row["scores"]["common_metric"]["f1"])
        for row in rows
    }


# ---------------------------------------------------------------------------
# Five-density paired-bootstrap mean curves
# ---------------------------------------------------------------------------
DENSITIES = (20, 30, 40, 50, 60)

# Seed for the paired-bootstrap mean/CI curves drawn by this script (module
# constant, as required for reproducibility). Independent of, but numerically
# identical in method to, BOOTSTRAP_SEED above. Each density gets its own
# stream (CURVE_BOOTSTRAP_SEED + density) so per-density results do not
# depend on the order densities happen to be processed in.
CURVE_BOOTSTRAP_SEED = 20260729
CURVE_N_BOOTSTRAP = 10000

# Shared y-limits across every panel of every figure this script emits.
# Per-scene traces are not drawn, so the range no longer has to span the raw
# per-scene data: it is set to contain the whole plotted CI envelope
# (0.448-0.934 across both panels) with margin at each end, and to match the
# sibling line-plot figure fig_density_sweep.py (which uses 0.45-1.0).
Y_LIM = (0.40, 1.00)


def _density_colors() -> dict[int, tuple]:
    """Perceptually ordered (viridis) colour per density, dark->light with
    increasing density, shared between every figure this script produces."""
    cmap = matplotlib.colormaps["viridis"]
    fracs = np.linspace(0.12, 0.90, len(DENSITIES))
    return {density: cmap(frac) for density, frac in zip(DENSITIES, fracs)}


def _scene_ids_for_density(expected, density: int) -> list[str]:
    return sorted(key for key in expected if key.startswith(f"cov{density}/"))


def _density_curve(values_by_name: dict[str, dict], names: tuple[str, ...],
                    scene_ids: list[str], density: int):
    """Mean curve + 95% paired-bootstrap CI, resampling scenes within one density.

    The same bootstrap resample (the same drawn scene indices) is reused at
    every x setting for a given replicate: this is what makes the bootstrap
    *paired* rather than an independent per-x-setting bootstrap. Because the
    same scenes recur across x settings, resampling the scene (not the
    (scene, x) cell) keeps that within-scene correlation in the band and
    never averages together scientifically non-equivalent x settings.
    """
    arr = np.array(
        [[values_by_name[name][sid] for sid in scene_ids] for name in names],
        dtype=float,
    )  # shape (n_x, n_scenes)
    n = arr.shape[1]
    mean = arr.mean(axis=1)
    if n < 2:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(CURVE_BOOTSTRAP_SEED + density)
    idx = rng.integers(0, n, size=(CURVE_N_BOOTSTRAP, n))
    boot = arr[:, idx].mean(axis=2)  # (n_x, n_boot)
    lo = np.percentile(boot, 2.5, axis=1)
    hi = np.percentile(boot, 97.5, axis=1)
    return mean, lo, hi


def _plot_panel(ax, values, names, x, expected, density_colors, xticklabels,
                 xlabel: str):
    """Draw one panel: five density mean curves with shaded 95%
    paired-bootstrap CI bands. Per-scene traces are not drawn -- dispersion is
    carried by the CI band.

    Returns (handles, labels, n_by_density) for building a shared legend.
    """
    handles, labels, n_by_density = [], [], {}
    for density in DENSITIES:
        scene_ids = _scene_ids_for_density(expected, density)
        color = density_colors[density]
        mean, lo, hi = _density_curve(values, names, scene_ids, density)
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, zorder=2, lw=0)
        (line,) = ax.plot(x, mean, "-o", color=color, lw=2.4, ms=6.2,
                            mec="white", mew=0.7, zorder=4)
        handles.append(line)
        n_by_density[density] = len(scene_ids)
        labels.append(f"{density}%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylim(*Y_LIM)
    thin_spines(ax)
    return handles, labels, n_by_density


def _legend_title(n_by_density: dict[int, int]) -> str:
    """One legend title carries the scene count so it is not repeated five
    times over -- 'n = 4 scenes' if uniform across densities, else the exact
    per-density breakdown."""
    counts = set(n_by_density.values())
    if len(counts) == 1:
        n = counts.pop()
        # "areal density", distinguished from centreline-length density, a
        # different quantity (Section 3.2).
        return f"Target areal density (n = {n} scenes each)"
    parts = ", ".join(f"{d}%: n={n}" for d, n in sorted(n_by_density.items()))
    return f"Target areal density ({parts})"


def _width_panel_spec():
    x = (1, 2, 3)
    names = ("w1", "w2", "w3")
    xticklabels = ["1 px", "2 px", "3 px"]
    xlabel = "Nominal binary axis-mask width"
    return x, names, xticklabels, xlabel


def _clean_panel_spec():
    x = (0, 1)
    names = ("w1", "clean")
    xticklabels = ["Degraded\n(1 px mask)", "Clean\naxis"]
    xlabel = "Input axis-mask condition"
    return x, names, xticklabels, xlabel


def build(w1: Path, w2: Path, w3: Path, clean: Path, out_json: Path,
          figures: str = "all") -> dict:
    values = {name: _mapping(_rows(path)) for name, path in
              (("w1", w1), ("w2", w2), ("w3", w3), ("clean", clean))}
    expected = set(values["w1"])
    for name, mapping in values.items():
        if set(mapping) != expected:
            raise ValueError(f"{name}: scene IDs do not match w1")

    n_by_density = {density: len(_scene_ids_for_density(expected, density))
                     for density in DENSITIES}

    document = {
        "schema_version": 1,
        "study": "development_mask_sensitivity",
        "interpretation": "development diagnostic only",
        "reports": {name: str(path) for name, path in
                    (("w1", w1), ("w2", w2), ("w3", w3), ("clean", clean))},
        "overall": {name: summarize(mapping.values()) for name, mapping in values.items()},
        "per_density": {
            str(density): {
                name: summarize({key: value for key, value in mapping.items()
                                  if key.startswith(f"cov{density}/")}.values())
                for name, mapping in values.items()
            } for density in DENSITIES
        },
        "paired": {
            "w2_minus_w1": paired_summarize_by_scene(values["w2"], values["w1"]),
            "w3_minus_w1": paired_summarize_by_scene(values["w3"], values["w1"]),
            "clean_minus_w1": paired_summarize_by_scene(values["clean"], values["w1"]),
        },
        "curve_bootstrap": {
            "method": "non-parametric percentile bootstrap, paired by scene "
                       "within each density (the same resampled scene indices "
                       "are reused across every x setting of a panel)",
            "seed_formula": "CURVE_BOOTSTRAP_SEED + density",
            "seed": CURVE_BOOTSTRAP_SEED,
            "n_boot": CURVE_N_BOOTSTRAP,
            "n_scenes_per_density": n_by_density,
        },
        "per_scene": values,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    apply_style()
    density_colors = _density_colors()

    if figures in ("all", "standalone"):
        _build_standalone_width(values, expected, density_colors)
        _build_standalone_clean(values, expected, density_colors)
    if figures in ("all", "consolidated"):
        _build_consolidated(values, expected, density_colors)

    return document


def _shared_legend(fig, handles, labels, title, ncol):
    fig.legend(handles, labels, title=title, loc="outside lower center",
               ncol=ncol, handlelength=1.8, columnspacing=1.2,
               frameon=False, fontsize=9.0, title_fontsize=9.0)


def _build_standalone_width(values, expected, density_colors):
    x, names, xticklabels, xlabel = _width_panel_spec()
    fig, ax = plt.subplots(figsize=(6.3, 4.6), layout="constrained")
    handles, labels, n_by_density = _plot_panel(
        ax, values, names, x, expected, density_colors, xticklabels, xlabel)
    ax.set_xlim(0.8, 3.2)
    ax.set_ylabel("Development common-fragment F$_1$")
    ax.set_title("Input-mask width sensitivity (development scenes)", loc="left")
    _shared_legend(fig, handles, labels, _legend_title(n_by_density), ncol=5)
    save_fig(fig, "fig_width_sweep", subdir="archive", bbox_inches="tight")
    plt.close(fig)


def _build_standalone_clean(values, expected, density_colors):
    x, names, xticklabels, xlabel = _clean_panel_spec()
    fig, ax = plt.subplots(figsize=(6.3, 4.6), layout="constrained")
    handles, labels, n_by_density = _plot_panel(
        ax, values, names, x, expected, density_colors, xticklabels, xlabel)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Development common-fragment F$_1$")
    ax.set_title("Clean vs. degraded input mask (development scenes)", loc="left")
    _shared_legend(fig, handles, labels, _legend_title(n_by_density), ncol=5)
    save_fig(fig, "fig_clean_vs_degraded", subdir="archive", bbox_inches="tight")
    plt.close(fig)


def _build_consolidated(values, expected, density_colors):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.25, 4.0), sharey=True,
                                       layout="constrained")

    xa, names_a, ticks_a, xlabel_a = _width_panel_spec()
    handles, labels, n_by_density = _plot_panel(
        ax_a, values, names_a, xa, expected, density_colors, ticks_a, xlabel_a)
    ax_a.set_xlim(0.8, 3.2)
    ax_a.set_ylabel("Development common-fragment F$_1$")
    ax_a.set_title("(a) Axis-mask width", loc="left", fontsize=10.5)

    xb, names_b, ticks_b, xlabel_b = _clean_panel_spec()
    _plot_panel(ax_b, values, names_b, xb, expected, density_colors, ticks_b, xlabel_b)
    ax_b.set_xlim(-0.3, 1.3)
    ax_b.set_title("(b) Clean vs. degraded mask", loc="left", fontsize=10.5)
    ax_b.tick_params(labelleft=False)

    _shared_legend(fig, handles, labels, _legend_title(n_by_density), ncol=5)
    save_fig(fig, "fig_development_sensitivity", subdir="archive", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--w2", type=Path, required=True)
    ap.add_argument("--w3", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    # Default to the consolidated figure only: the two standalone figures are
    # no longer referenced by the manuscript, and emitting them by default left
    # orphaned fig_width_sweep/fig_clean_vs_degraded files in figures/.
    ap.add_argument("--figures", choices=("all", "consolidated", "standalone"),
                     default="consolidated",
                     help="Which figure(s) to emit: the consolidated two-panel "
                          "fig_development_sensitivity, the two standalone "
                          "fig_width_sweep/fig_clean_vs_degraded, or both "
                          "(default: all).")
    args = ap.parse_args()
    build(args.w1, args.w2, args.w3, args.clean, args.out_json, figures=args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
