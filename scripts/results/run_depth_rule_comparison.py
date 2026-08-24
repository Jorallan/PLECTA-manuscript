"""Measure the shipped depth evidence rule against the one it replaced.

Why this exists. The 2026-08-22 comparison that justified replacing the depth
stage's two-channel logistic rule with the one-channel noise-floored rule wrote
no machine-readable output. Its numbers reached the manuscript by being typed
out of prose, which put every published quantity outside the repository's own
gate. This driver recomputes them from the scenes, so
``make_depth_rule_record.py`` can read a record instead of transcribing one.

What it runs. Both rules over the same held-out set, in the ``--oracle``
condition, because the question is about the evidence rule and the oracle
removes 2-D grouping error from the comparison:

* shipped: ``parameters.yaml`` defaults, one channel, noise-floored;
* legacy:  ``scoring=winsorized_linear core_mode=radius_scaled core_px=6.0
  abstain_band=0.15``, the pre-2026-08-22 rule, which ``plecta.depth`` can
  still be asked for.

What it measures, and why each. A crossing carries a signed score whose sign
picks the upper rod and whose magnitude weights that relation in the global
ordering. So the rule is judged on two things, not one:

* **accuracy at matched coverage.** Each rule abstains at its own threshold, so
  comparing them at their own operating points confounds accuracy with how much
  each declines to answer. Sweeping the threshold gives a coverage-accuracy
  curve per rule, and the rules are compared where they answer equally often.
* **AUC of correctness against the score magnitude.** Whether the score *ranks*
  its own reliability. This is the quantity that matters for the global order,
  because the magnitude is the edge weight: a better-ordered score breaks
  contradictions at genuinely weaker relations.

Uncertainty is a scene-level bootstrap, resampling the 20 scenes with
replacement. Scenes are the independent unit; crossings within one are not.

What it cannot recover. Thirteen quantities described channels that no longer
exist in the code -- the removed fixed denominator floor and its saturation
behaviour, the sharpness channel's standalone accuracy and agreement, and the
axial channel measured inert before removal. Deleting them was the point of the
change, so they are not recomputable here. They have been withdrawn from the
manuscript rather than transcribed into it, and are kept for reference in the
untracked ``notes/depth_rule_transcribed_measurements.md``.

    python scripts/results/run_depth_rule_comparison.py [--score-only]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = r"C:\Repos\venv_cnt\Scripts\python.exe"
STUBMATCH = Path(r"C:\Repos\stubmatch")
REPO = Path(__file__).resolve().parents[2]
SCENES = Path(r"C:\Repos\filaments_quantification\input\synthetic_depth_heldout")
WORK = REPO / "exploration" / "depth_rule_comparison"
OUT = REPO / "results" / "plecta_depth_rule_measured.json"

LEGACY = ["--set", "scoring=winsorized_linear", "core_mode=radius_scaled",
          "core_px=6.0", "abstain_band=0.15"]
RULES = {"shipped": [], "legacy": LEGACY}

TOL_PX = 6.0            # crossing-location match radius, as the 2-D study uses
N_BOOT = 2000
BOOT_SEED = 20260824
COVERAGE_POINTS = (0.825, 0.80)


def predict(rule: str, extra: list) -> None:
    for scene in sorted(p for p in SCENES.glob("*/*") if (p / "gt_depth.json").is_file()):
        out = WORK / rule / f"{scene.parent.name}_{scene.name}"
        if (out / "pred_depth.json").is_file():
            continue
        cmd = [PY, "-m", "plecta.depth", "--scene", str(scene),
               "--out", str(out), "--oracle", *extra]
        res = subprocess.run(cmd, cwd=str(STUBMATCH), capture_output=True,
                             text=True)
        if res.returncode:
            print(res.stderr[-1500:], file=sys.stderr)
            raise SystemExit(f"{rule}/{out.name} failed")
    print(f"{rule}: predictions ready")


def collect(rule: str) -> list:
    """Per scene: (|score|, correct) for every matched crossing, and the GT count.

    In oracle mode the instance ids are the reference's own, so a predicted
    crossing matches a reference crossing when it names the same unordered pair
    within TOL_PX. Nothing is matched twice.
    """
    scenes = []
    for scene in sorted(p for p in SCENES.glob("*/*") if (p / "gt_depth.json").is_file()):
        gt = json.loads((scene / "gt_depth.json").read_text())
        pred_path = WORK / rule / f"{scene.parent.name}_{scene.name}" / "pred_depth.json"
        pr = json.loads(pred_path.read_text())

        index: dict = {}
        for k, g in enumerate(gt["crossings"]):
            index.setdefault(frozenset((g["i"], g["j"])), []).append(k)

        used, rows = set(), []
        for p in pr["crossings"]:
            key = frozenset((p["i"], p["j"]))
            best, bestd = None, TOL_PX
            for k in index.get(key, []):
                if k in used:
                    continue
                d = float(np.hypot(gt["crossings"][k]["x"] - p["x"],
                                   gt["crossings"][k]["y"] - p["y"]))
                if d <= bestd:
                    best, bestd = k, d
            if best is None:
                continue
            used.add(best)
            s = p.get("score")
            if s is None:
                continue
            #  The sign of the score names the upper rod; read the decision the
            #  sign implies rather than `over`, which the global pass may have
            #  flipped. The rule is being judged, not the solver.
            upper = p["i"] if s > 0 else p["j"]
            rows.append((abs(float(s)), int(upper == gt["crossings"][best]["over"])))
        scenes.append({"scene": f"{scene.parent.name}/{scene.name}",
                       "n_gt_crossings": len(gt["crossings"]), "rows": rows})
    return scenes


def curve(rows: list):
    """Coverage and accuracy as the abstention threshold sweeps over |score|."""
    if not rows:
        return np.array([]), np.array([])
    mags = np.array([r[0] for r in rows])
    ok = np.array([r[1] for r in rows], float)
    order = np.argsort(-mags)                    # most confident first
    ok = ok[order]
    n = len(ok)
    kept = np.arange(1, n + 1)
    return kept / n, np.cumsum(ok) / kept        # coverage, accuracy


def auc(rows: list) -> float:
    """Probability a correct crossing outscores an incorrect one (ties at 0.5)."""
    pos = np.array([m for m, c in rows if c == 1])
    neg = np.array([m for m, c in rows if c == 0])
    if not len(pos) or not len(neg):
        return float("nan")
    #  Rank-sum form, exact on ties.
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    #  average ranks for ties
    order = np.argsort(allv)
    sortv = allv[order]
    i = 0
    while i < len(sortv):
        j = i
        while j + 1 < len(sortv) and sortv[j + 1] == sortv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum() + len(pos)    # 1-based
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def acc_at(rows: list, coverage: float) -> float:
    cov, acc = curve(rows)
    if not len(cov):
        return float("nan")
    return float(acc[int(np.searchsorted(cov, coverage, side="left")
                         .clip(0, len(cov) - 1))])


def operating_point(rule: str) -> dict:
    """Where a rule actually sits: what it declines, and how it does on the rest.

    Read from the predictions rather than from the threshold, because a rule
    abstains for two reasons -- the score is small, or the crossing had too few
    usable flank samples -- and only the predictions know which fired.
    """
    dec = matched = correct = 0
    for scene in sorted(p for p in SCENES.glob("*/*") if (p / "gt_depth.json").is_file()):
        gt = json.loads((scene / "gt_depth.json").read_text())
        pr = json.loads((WORK / rule / f"{scene.parent.name}_{scene.name}"
                         / "pred_depth.json").read_text())
        index: dict = {}
        for k, g in enumerate(gt["crossings"]):
            index.setdefault(frozenset((g["i"], g["j"])), []).append(k)
        used = set()
        for p in pr["crossings"]:
            cand = [k for k in index.get(frozenset((p["i"], p["j"])), [])
                    if k not in used]
            if not cand:
                continue
            k = min(cand, key=lambda k: float(np.hypot(
                gt["crossings"][k]["x"] - p["x"],
                gt["crossings"][k]["y"] - p["y"])))
            if float(np.hypot(gt["crossings"][k]["x"] - p["x"],
                              gt["crossings"][k]["y"] - p["y"])) > TOL_PX:
                continue
            used.add(k)
            matched += 1
            if p.get("abstain") or p.get("score") is None:
                continue
            dec += 1
            s = float(p["score"])
            upper = p["i"] if s > 0 else p["j"]
            correct += int(upper == gt["crossings"][k]["over"])
    return {"matched": matched, "decided": dec,
            "coverage": dec / matched if matched else float("nan"),
            "accuracy": correct / dec if dec else float("nan")}


def bootstrap(shipped: list, legacy: list, stat) -> tuple:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(shipped)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = [r for i in idx for r in shipped[i]["rows"]]
        b = [r for i in idx for r in legacy[i]["rows"]]
        v = stat(a, b)
        if v == v:
            draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if not args.score_only:
        for rule, extra in RULES.items():
            predict(rule, extra)

    data = {rule: collect(rule) for rule in RULES}
    flat = {rule: [r for s in data[rule] for r in s["rows"]] for rule in RULES}
    n_gt = sum(s["n_gt_crossings"] for s in data["shipped"])

    #  Matched-coverage comparison over the coverage range both rules span.
    grid = np.linspace(0.50, 1.00, 101)
    acc_s = np.array([acc_at(flat["shipped"], c) for c in grid])
    acc_l = np.array([acc_at(flat["legacy"], c) for c in grid])
    delta_curve = acc_s - acc_l

    def d_full(a, b):
        return acc_at(a, 1.0) - acc_at(b, 1.0)

    def d_auc(a, b):
        return auc(a) - auc(b)

    #  The comparison the shipped configuration actually describes: both rules
    #  answering as often as the shipped one does. Fixed here rather than
    #  resampled, so the bootstrap varies the scenes and not the question.
    matched_cov = operating_point("shipped")["coverage"]

    def d_matched(a, b):
        return acc_at(a, matched_cov) - acc_at(b, matched_cov)

    payload = {
        "role": "the shipped depth evidence rule against the rule it replaced, "
                "recomputed from the held-out scenes",
        "computed_by": "scripts/results/run_depth_rule_comparison.py",
        "scenes": str(SCENES),
        "condition": "oracle; the 2-D grouping does not run, so the comparison "
                     "is of the evidence rule alone",
        "n_scenes": len(data["shipped"]),
        "n_gt_crossings": n_gt,
        "n_matched": {r: len(flat[r]) for r in RULES},
        "rules": {"shipped": "parameters.yaml defaults, one channel, "
                             "noise-floored",
                  "legacy": " ".join(LEGACY)},
        "bootstrap": {"draws": N_BOOT, "unit": "scene", "seed": BOOT_SEED},
        #  Where each rule actually operates, which is the comparison that
        #  describes the shipped configuration.
        "operating_point": {r: operating_point(r) for r in RULES},
        #  Forcing both rules to answer every crossing, including the ones they
        #  would decline. Reported because it is the adversarial reading, and
        #  it is the one that goes against the shipped rule: abstention is
        #  where better-ordered confidence pays, so removing it removes the
        #  advantage. It is NOT how either rule is run.
        #  Headline: both rules answering as often as the shipped one does.
        "matched_coverage": matched_cov,
        "accuracy_delta_at_matched_coverage": d_matched(flat["shipped"],
                                                        flat["legacy"]),
        "accuracy_delta_matched_ci": bootstrap(data["shipped"], data["legacy"],
                                               d_matched),
        "accuracy_all_decided": {r: acc_at(flat[r], 1.0) for r in RULES},
        "accuracy_delta_at_full_coverage": d_full(flat["shipped"], flat["legacy"]),
        "accuracy_delta_ci": bootstrap(data["shipped"], data["legacy"], d_full),
        "auc": {r: auc(flat[r]) for r in RULES},
        "auc_delta": d_auc(flat["shipped"], flat["legacy"]),
        "auc_delta_ci": bootstrap(data["shipped"], data["legacy"], d_auc),
        "accuracy_at_coverage": {
            f"{c:.3f}": {"shipped": acc_at(flat["shipped"], c),
                         "legacy": acc_at(flat["legacy"], c),
                         "delta": acc_at(flat["shipped"], c)
                                  - acc_at(flat["legacy"], c)}
            for c in COVERAGE_POINTS},
        "coverage_grid": [float(x) for x in grid],
        "delta_curve": [float(x) for x in delta_curve],
        "per_scene": data,
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"  scenes {payload['n_scenes']}  GT crossings {n_gt}  "
          f"matched {payload['n_matched']}")
    for r in RULES:
        print(f"  {r:8} accuracy {payload['accuracy_all_decided'][r]:.4f}  "
              f"AUC {payload['auc'][r]:.4f}")
    lo, hi = payload["accuracy_delta_ci"]
    print(f"  accuracy delta {payload['accuracy_delta_at_full_coverage']:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]")
    lo, hi = payload["auc_delta_ci"]
    print(f"  AUC delta      {payload['auc_delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for r in RULES:
        op = payload["operating_point"][r]
        print(f"  {r:8} operating point: coverage {op['coverage']:.3f}  "
              f"accuracy {op['accuracy']:.4f}")
    for c, v in payload["accuracy_at_coverage"].items():
        print(f"  at coverage {c}: delta {v['delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
