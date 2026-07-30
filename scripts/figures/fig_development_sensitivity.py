"""Regenerate development mask-width and clean-mask sensitivity figures."""
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401  (restores the flat eval/ import namespace)
from _style import apply_style, save_fig, BLUE, ORANGE, GREEN, GRAY, thin_spines  # noqa: E402
import stats_util  # noqa: E402


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


def _summary(mapping: dict[str, float]) -> dict:
    return stats_util.summarize(list(mapping.values())).as_dict()


def build(w1: Path, w2: Path, w3: Path, clean: Path, out_json: Path) -> dict:
    values = {name: _mapping(_rows(path)) for name, path in
              (("w1", w1), ("w2", w2), ("w3", w3), ("clean", clean))}
    expected = set(values["w1"])
    for name, mapping in values.items():
        if set(mapping) != expected:
            raise ValueError(f"{name}: scene IDs do not match w1")
    densities = (20, 30, 40, 50, 60)
    document = {
        "schema_version": 1,
        "study": "development_mask_sensitivity",
        "interpretation": "development diagnostic only",
        "reports": {name: str(path) for name, path in
                    (("w1", w1), ("w2", w2), ("w3", w3), ("clean", clean))},
        "overall": {name: _summary(mapping) for name, mapping in values.items()},
        "per_density": {
            str(density): {
                name: _summary({key: value for key, value in mapping.items()
                                if key.startswith(f"cov{density}/")})
                for name, mapping in values.items()
            } for density in densities
        },
        "paired": {
            "w2_minus_w1": stats_util.paired_summarize_by_scene(values["w2"], values["w1"]).as_dict(),
            "w3_minus_w1": stats_util.paired_summarize_by_scene(values["w3"], values["w1"]).as_dict(),
            "clean_minus_w1": stats_util.paired_summarize_by_scene(values["clean"], values["w1"]).as_dict(),
        },
        "per_scene": values,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.45))
    names = ("w1", "w2", "w3")
    x = np.asarray((1, 2, 3))
    density_colors = {20: GREEN, 30: BLUE, 40: ORANGE, 50: "#805ad5", 60: GRAY}
    # Each trace is one fixed development geometry, so the width comparison is
    # visibly paired rather than implying independent groups at each width.
    for density in densities:
        scene_ids = sorted(key for key in expected if key.startswith(f"cov{density}/"))
        for index, scene_id in enumerate(scene_ids):
            label = f"{density}% density" if index == 0 else None
            ax.plot(x, [values[name][scene_id] for name in names], "-o",
                    color=density_colors[density], alpha=0.62, lw=1.05,
                    ms=3.7, label=label, zorder=2)
    means = np.asarray([document["overall"][name]["mean"] for name in names])
    lo = np.asarray([document["overall"][name]["ci_lo"] for name in names])
    hi = np.asarray([document["overall"][name]["ci_hi"] for name in names])
    ax.errorbar(x, means, yerr=np.vstack([means - lo, hi - means]), fmt="-D",
                color="#1a202c", mfc="white", mec="#1a202c", lw=1.8, ms=5,
                capsize=3, label="Overall mean and 95% CI", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(["1 px", "2 px", "3 px"])
    ax.set_xlim(0.8, 3.2)
    ax.set_xlabel("Degraded synthetic axis-mask width")
    ax.set_ylabel("Development common-fragment F$_1$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Input-mask width sensitivity (development scenes; n = 20)", loc="left")
    ax.legend(ncol=2, loc="lower left", handlelength=2.0, columnspacing=1.1)
    thin_spines(ax); fig.tight_layout()
    save_fig(fig, "fig_width_sweep")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(densities), figsize=(11.4, 3.85), sharey=True)
    for ax, density in zip(axes, densities):
        scene_ids = sorted(key for key in expected if key.startswith(f"cov{density}/"))
        for scene_id in scene_ids:
            ax.plot((0, 1), (values["w1"][scene_id], values["clean"][scene_id]), "-o",
                    color=density_colors[density], alpha=0.78, lw=1.25, ms=4.2,
                    zorder=2)
        degraded_mean = document["per_density"][str(density)]["w1"]["mean"]
        clean_mean = document["per_density"][str(density)]["clean"]["mean"]
        ax.plot((0, 1), (degraded_mean, clean_mean), "-D", color="#1a202c",
                mfc="white", mec="#1a202c", lw=1.8, ms=4.8, zorder=4)
        paired = stats_util.paired_summarize_by_scene(
            {key: values["clean"][key] for key in scene_ids},
            {key: values["w1"][key] for key in scene_ids},
        ).as_dict()
        ax.set_title(f"{density}% density\nn = {len(scene_ids)} scenes", fontsize=10)
        ax.text(0.5, 0.04, f"mean paired change\n{paired['mean']:+.3f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="#1a202c")
        ax.set_xticks((0, 1))
        ax.set_xticklabels(("Degraded\n1 px", "Clean\naxis"))
        ax.set_xlim(-0.22, 1.22)
        thin_spines(ax)
    axes[0].set_ylabel("Development common-fragment F$_1$")
    axes[0].set_ylim(0.35, 1.0)
    fig.suptitle("Clean versus degraded input mask on paired development scenes", x=0.5,
                 y=0.985, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.02, "Lines are individual scenes; open diamonds are per-density means.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    save_fig(fig, "fig_clean_vs_degraded")
    plt.close(fig)
    return document


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--w2", type=Path, required=True)
    ap.add_argument("--w3", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    build(args.w1, args.w2, args.w3, args.clean, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
