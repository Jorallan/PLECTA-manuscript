"""Depth-stage headline numbers: ablation grid and domain shift.

Reads exploration/depth_25d/depth_ablation_record.json (written by
scripts/results/run_depth_ablations.py over the held-out sets) and draws:

    (a) crossing-order accuracy per condition, in-domain vs shifted domain,
        with the decision rate printed inside each bar;
    (b) the same conditions split by areal coverage (20 % vs 35 %);
    (c) global structure: weighted Kendall tau and layer ARI per condition;
    (d) metric depth (span-normalised RMSE) and diameter MAE per condition.

Every value is a plain mean over scenes (n = 20 per set); exploratory, no
inference. A bar that is absent because a condition produced zero decided
crossings in that split (sharpness-only, 20 % coverage: every scene abstains
on every crossing) is labelled "no decisions" rather than left blank, so it
reads as a finding and not as a rendering gap. A bar that is present but near
zero (oracle radii trivially give ~0 diameter error; sharpness-only barely
moves ordinal/layer agreement) gets its value printed above it for the same
reason -- a bar too short to see is not the same as a bar that is not there.

Writes to figures/archive/ (no `\\includegraphics` yet).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _style import (BLUE, CHORD, FIG_W, GREEN, INK, ORANGE, PT_ANNOT,  # noqa: E402
                    PT_TITLE, plecta_style, save_fig, tagged_title,
                    thin_spines)

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RECORD = os.path.join(REPO, "exploration", "depth_25d",
                      "depth_ablation_record.json")

CONDS = ["oracle_oracle_r", "oracle_meas", "pred_meas",
         "pred_intensity", "pred_sharpness"]
LABELS = {
    "oracle_oracle_r": "oracle 2-D + oracle d",
    "oracle_meas": "oracle 2-D + measured d",
    "pred_meas": "predicted 2-D + measured d",
    "pred_intensity": "intensity channel only",
    "pred_sharpness": "sharpness channel only",
}
TAG_GAP = 0.115


def agg(rec, key, path):
    vals = []
    for scene_rec in rec["per_scene"].values():
        v = scene_rec.get("scores", {})
        for part in path:
            v = v.get(part, {}) if isinstance(v, dict) else {}
        if isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def agg_cov(rec, cov, path):
    vals = []
    for key, scene_rec in rec["per_scene"].items():
        if not key.startswith(cov):
            continue
        v = scene_rec.get("scores", {})
        for part in path:
            v = v.get(part, {}) if isinstance(v, dict) else {}
        if isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def slant_xticks(ax):
    """Full-length single-line condition labels, angled so five of them fit
    a half-width panel without the neighbouring labels running together."""
    ax.set_xticks(np.arange(len(CONDS)))
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=7.0,
                       rotation=26, ha="right", rotation_mode="anchor")


def mark_bars(ax, xs, vals, axis_max, fmt="{:.2f}", small_frac=0.07,
             missing_label=None, colour=INK, y0=None):
    """Label a bar too short to read, or state plainly that it is absent.

    `axis_max` is the axes' own y-limit, so "too short to read" is judged
    against what this panel can actually show, not a fixed pixel count.
    `missing_label` (e.g. "no decisions") is drawn at the baseline for a NaN
    value; a present-but-small value gets its number printed just above the
    bar instead of being left to disappear against the axis.
    """
    base = y0 if y0 is not None else 0.0
    for xi, v in zip(xs, vals):
        if v is None or not np.isfinite(v):
            if missing_label:
                ax.text(xi, base + 0.02 * axis_max, missing_label,
                       ha="center", va="bottom", fontsize=6.3, color=colour,
                       rotation=90, style="italic")
            continue
        if v < small_frac * axis_max:
            ax.text(xi, base + v + 0.015 * axis_max, fmt.format(v),
                   ha="center", va="bottom", fontsize=6.3, color=colour)


def main():
    plecta_style()
    record = json.load(open(RECORD, encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 0.78))
    # left is wider than a bare margin would need: the first, right-anchored,
    # rotated x-tick label of the LEFT column leans further left than its
    # own axis spine, and with too little room it clips at the canvas edge
    # rather than at the axis -- the right column never shows this because
    # the column gap (wspace) already gives its first label room to lean into.
    fig.subplots_adjust(left=0.105, right=0.925, top=0.925, bottom=0.20,
                        wspace=0.34, hspace=0.95)
    x = np.arange(len(CONDS))

    # (a) order accuracy, in-domain vs shifted --------------------------
    ax = axes[0, 0]
    for k, (set_name, colour, dx) in enumerate(
            [("heldout", GREEN, -0.19), ("heldout_shifted", ORANGE, 0.19)]):
        accs = [agg(record[f"{set_name}__{c}"], c,
                    ("crossing_order", "accuracy_decided")) for c in CONDS]
        rates = [agg(record[f"{set_name}__{c}"], c,
                     ("crossing_order", "decision_rate")) for c in CONDS]
        ax.bar(x + dx, accs, width=0.36, color=colour,
               label="in-domain" if k == 0 else "shifted rendering")
        for xi, (a, r) in zip(x + dx, zip(accs, rates)):
            if np.isfinite(a) and a > 0.12:
                ax.text(xi, a - 0.028, f"{r:.2f}", ha="center", va="top",
                        fontsize=7.0, color="white", rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color=CHORD, linewidth=0.7, linestyle=(0, (4, 3)))
    ax.text(len(CONDS) - 0.5, 0.51, "chance", ha="right", va="bottom",
            fontsize=PT_ANNOT, color=CHORD)
    slant_xticks(ax)
    ax.set_ylabel("crossing-order accuracy")
    ax.legend(loc="lower left", fontsize=PT_ANNOT)
    thin_spines(ax)
    tagged_title(ax, "a", "over/under accuracy (decision rate in bars)",
                 dy=1.04, gap=TAG_GAP)

    # (b) accuracy by coverage ------------------------------------------
    ax = axes[0, 1]
    for k, (cov, colour, dx) in enumerate(
            [("cov20", BLUE, -0.19), ("cov35", ORANGE, 0.19)]):
        accs = [agg_cov(record[f"heldout__{c}"], cov,
                        ("crossing_order", "accuracy_decided")) for c in CONDS]
        ax.bar(x + dx, accs, width=0.36, color=colour,
               label=f"{cov[3:]} % coverage")
        mark_bars(ax, x + dx, accs, 1.0, missing_label="no decisions",
                 colour=colour)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color=CHORD, linewidth=0.7, linestyle=(0, (4, 3)))
    slant_xticks(ax)
    ax.set_ylabel("crossing-order accuracy")
    ax.legend(loc="lower left", fontsize=PT_ANNOT)
    thin_spines(ax)
    tagged_title(ax, "b", "in-domain accuracy by areal coverage",
                 dy=1.04, gap=TAG_GAP)

    # (c) global structure ----------------------------------------------
    ax = axes[1, 0]
    taus = [agg(record[f"heldout__{c}"], c, ("ordinal", "kendall_tau_weighted"))
            for c in CONDS]
    aris = [agg(record[f"heldout__{c}"], c, ("layers", "layer_ari"))
            for c in CONDS]
    ax.bar(x - 0.19, taus, width=0.36, color=GREEN, label="Kendall tau")
    ax.bar(x + 0.19, aris, width=0.36, color=BLUE, label="layer ARI")
    mark_bars(ax, x - 0.19, taus, 1.0, colour=GREEN)
    mark_bars(ax, x + 0.19, aris, 1.0, colour=BLUE)
    slant_xticks(ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("agreement")
    ax.legend(loc="upper right", fontsize=PT_ANNOT)
    thin_spines(ax)
    tagged_title(ax, "c", "ordinal structure and layers (in-domain)",
                 dy=1.04, gap=TAG_GAP)

    # (d) metric z + diameter -------------------------------------------
    ax = axes[1, 1]
    rms = [agg(record[f"heldout__{c}"], c, ("metric_z", "rmse_z_span_norm"))
           for c in CONDS]
    dmae = [agg(record[f"heldout__{c}"], c, ("diameter", "diameter_mae"))
            for c in CONDS]
    rms_max = max(0.5, np.nanmax(rms) * 1.35)
    dmae_max = max(2.0, np.nanmax(dmae) * 1.35)
    ax.bar(x - 0.19, rms, width=0.36, color=GREEN,
           label="RMSE$_z$ / z span")
    ax2 = ax.twinx()
    ax2.bar(x + 0.19, dmae, width=0.36, color=ORANGE, label="diameter MAE")
    mark_bars(ax, x - 0.19, rms, rms_max, colour=GREEN)
    mark_bars(ax2, x + 0.19, dmae, dmae_max, fmt="{:.2f}", colour=ORANGE)
    slant_xticks(ax)
    ax.set_ylabel("RMSE$_z$ / z span")
    ax2.set_ylabel("diameter MAE (px)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_ylim(0, rms_max)
    ax2.set_ylim(0, dmae_max)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    thin_spines(ax)
    tagged_title(ax, "d", "metric depth and diameter", dy=1.04, gap=TAG_GAP)

    fig.text(0.075, 0.045,
             "Plain means over 20 held-out scenes per set; exploratory, no "
             "inference. Small labelled bars are real values, not missing "
             "data.",
             fontsize=PT_ANNOT, color=INK, ha="left")
    fig.text(0.075, 0.018,
             "Oracle radii are ground truth (so its diameter error is "
             "trivially ~0); measured d uses the dev-calibrated width "
             "correction.",
             fontsize=PT_ANNOT, color=INK, ha="left")
    save_fig(fig, "fig_depth_results", subdir="archive")


if __name__ == "__main__":
    main()
