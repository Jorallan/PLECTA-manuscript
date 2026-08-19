"""Run the depth-stage ablation grid and collect one machine-readable record.

Conditions (all through eval/runners/run_depth.py in filaments_quantification;
each condition = one runner invocation, one depth_scores.json):

  oracle_oracle_r   oracle 2-D instances + oracle radii     (upper bound)
  oracle_meas       oracle 2-D instances + measured diameters
  pred_meas         PLECTA-predicted instances + measured diameters (full
                    post-hoc pipeline: DEPTH_MODE = posthoc)
  pred_intensity    pred_meas with the sharpness channel off  (w_sharpness=0)
  pred_sharpness    pred_meas with the intensity channel off  (w_intensity=0)

Each condition runs on every scene set passed via --sets (default: the two
held-out sets). Output: one JSON with every aggregate plus per-scene records,
written to exploration/depth_25d/ (gitignored study area) by default.

The held-out sets are touched only by this script and only to produce final
numbers; nothing here tunes on them (all calibration constants were frozen
on synthetic_depth_dev and are recorded in plecta/depth.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = r"C:\Repos\venv_cnt\Scripts\python.exe"
FQ = Path(r"C:\Repos\filaments_quantification")
HERE = Path(__file__).resolve().parents[2]
DEFAULT_OUT = HERE / "exploration" / "depth_25d"

CONDITIONS = {
    "oracle_oracle_r": ["--oracle", "--oracle-radii"],
    "oracle_meas": ["--oracle"],
    "pred_meas": [],
    "pred_intensity": ["--set", "w_sharpness=0.0"],
    "pred_sharpness": ["--set", "w_intensity=0.0"],
}

DEFAULT_SETS = {
    "heldout": FQ / "input" / "synthetic_depth_heldout",
    "heldout_shifted": FQ / "input" / "synthetic_depth_heldout_shifted",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sets", nargs="*", default=None,
                    help="name=path pairs; default: the two held-out sets")
    ap.add_argument("--conditions", nargs="*", default=sorted(CONDITIONS),
                    help=f"subset of {sorted(CONDITIONS)}")
    args = ap.parse_args()

    sets = (dict(item.split("=", 1) for item in args.sets)
            if args.sets else {k: str(v) for k, v in DEFAULT_SETS.items()})
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    combined = {}
    for set_name, scene_root in sets.items():
        for cond in args.conditions:
            run_dir = out_root / "runs" / f"{set_name}__{cond}"
            cmd = [PY, str(FQ / "eval" / "runners" / "run_depth.py"),
                   "--scenes", str(scene_root), "--out", str(run_dir),
                   *CONDITIONS[cond]]
            print(">>", set_name, cond, flush=True)
            res = subprocess.run(cmd, cwd=str(FQ), capture_output=True,
                                 text=True)
            if res.returncode != 0:
                print(res.stdout[-2000:])
                print(res.stderr[-2000:])
                raise SystemExit(f"condition {set_name}/{cond} failed")
            payload = json.loads((run_dir / "depth_scores.json").read_text(
                encoding="utf-8"))
            combined[f"{set_name}__{cond}"] = payload
            agg = payload["aggregate"]

            def g(key):
                m = agg[key]["mean"]
                return round(m, 3) if m is not None else None

            print(f"   order {g('crossing_order_accuracy')} "
                  f"decide {g('crossing_decision_rate')} "
                  f"tau {g('kendall_tau')} ARI {g('layer_ari')} "
                  f"rmse_zn {g('rmse_z_span_norm')} "
                  f"invalid {g('invalid_crossing_fraction')}", flush=True)

    dest = out_root / "depth_ablation_record.json"
    dest.write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
