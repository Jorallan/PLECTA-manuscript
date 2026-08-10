"""Generate the method-only PLECTA performance figure (PDF and SVG)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_BOOT = 20_000
SEED = 20_260_810


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> None:
    held = json.loads((RESULTS / "plecta_heldout.json").read_text())
    ablation = json.loads((RESULTS / "plecta_ablation_dev.json").read_text())

    densities = np.array([20, 30, 40, 60])
    means, lo, hi = [], [], []
    rows = held["per_scene"]
    for density in densities:
        values = np.asarray([r["f1"] for r in rows if r["density"] == density])
        means.append(values.mean())
        interval = bootstrap_mean(values, SEED + int(density))
        lo.append(interval[0])
        hi.append(interval[1])
    means, lo, hi = map(np.asarray, (means, lo, hi))

    base = next(r for r in ablation["rows"] if r["ablation"] == "frozen")
    variants = [
        ("Tangent only", "tangent term only"),
        ("No gap\nmatching", "no gap bridging"),
        ("No crossing\nmatching", "no crossing merging at all"),
        ("No chain\nframes", "no chain-extended frames"),
        ("Single round", "single round"),
        ("No curvature", "no curvature term"),
    ]
    deltas = [
        next(r["f1"] for r in ablation["rows"] if r["ablation"] == key)
        - base["f1"]
        for _, key in variants
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax = axes[0]
    ax.errorbar(densities, means, yerr=[means - lo, hi - means],
                color="#2166AC", marker="o", ms=4.5, lw=1.6, capsize=3)
    ax.set(xlabel="Target areal coverage (%)", ylabel="Common-fragment $F_1$",
           title="(a) Held-out performance")
    ax.set_xticks(densities)
    ax.set_ylim(0.72, 0.95)
    ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for x, y in zip(densities, means):
        ax.text(x, y + 0.012, f"{y:.3f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    y = np.arange(len(variants))
    colours = ["#B2182B" if value < -0.05 else "#EF8A62" for value in deltas]
    ax.barh(y, deltas, color=colours, height=0.68)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, [label for label, _ in variants])
    ax.invert_yaxis()
    ax.set(xlabel=r"$\Delta F_1$ from the fixed configuration",
           title="(b) One-at-a-time development ablations")
    ax.set_xlim(-0.205, 0.01)
    ax.grid(axis="x", color="#D9D9D9", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for yi, value in zip(y, deltas):
        strong = value < -0.05
        ax.text(value + 0.004 if strong else value - 0.004, yi,
                f"{value:.3f}", ha="left" if strong else "right", va="center",
                color="white" if strong else "#333333", fontsize=7)

    fig.tight_layout(w_pad=1.2)
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"fig_plecta_performance.{suffix}",
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
