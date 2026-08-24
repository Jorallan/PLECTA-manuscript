"""Run the depth-stage ablation grid and collect one machine-readable record.

Conditions (all through eval/runners/run_depth.py in filaments_quantification;
each condition = one runner invocation, one depth_scores.json):

  oracle_oracle_r    oracle 2-D instances + oracle radii     (upper bound)
  oracle_meas        oracle 2-D instances + measured diameters
  pred_meas          PLECTA-predicted instances + measured diameters, at the
                     shipped settings (DEPTH_MODE = posthoc): one channel,
                     noise-floored intensity, flat core, abstain |s| < 0.40
  pred_abstain_lo    pred_meas at abstain_score = 0.20 -- more coverage
  pred_abstain_hi    pred_meas at abstain_score = 0.60 -- more abstention
  pred_core_radius   pred_meas with core_mode = radius_scaled, so a per-rod
                     radius re-enters the evidence path
  pred_legacy_rule   the rule shipped before 2026-08-22, bit-identically:
                     scoring = winsorized_linear, core_mode = radius_scaled,
                     core_px = 6.0, abstain_band = 0.15

The grid changed with the evidence rule. Until 2026-08-22 it carried
``pred_intensity`` (w_sharpness = 0) and ``pred_sharpness`` (w_intensity = 0),
which ablated the two channels of the two-channel rule. That rule is gone: the
shipped ``scoring = "noise_floored"`` reads one channel, and w_intensity /
w_sharpness are legacy DepthParams fields consumed only by the two legacy
scorers. Under the shipped rule ``pred_intensity`` would have duplicated
``pred_meas`` exactly and ``pred_sharpness`` would have produced a degenerate
run, so both were replaced by ablations of what now decides: the abstention
threshold, the core geometry, and the previous rule as a whole.

PRECONDITION, cleared 2026-08-23. ``core_mode`` and ``scoring`` are
string-valued DepthParams fields, and ``eval/runners/run_depth.py`` used to
coerce every ``--set`` override to int or float in ``depth_params()``, so it
raised on them. It now coerces by the type of the dataclass field's default --
bool, then str, then int, then float -- mirroring ``plecta.parameters.build()``.
The guard below stays as a regression check: if that coercion is ever reverted,
the driver refuses the two affected conditions rather than writing a record with
holes in it, and ``--conditions`` still selects the numeric subset.

Each condition runs on every scene set passed via --sets (default: the two
held-out sets). Output: one JSON with every aggregate plus per-scene records,
written to exploration/depth_25d/ (gitignored study area) by default.

The held-out sets are touched only by this script and only to produce final
numbers; nothing here tunes on them (all calibration constants were frozen on
synthetic_depth_dev and are recorded in plecta/parameters.yaml).
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
    "pred_abstain_lo": ["--set", "abstain_score=0.20"],
    "pred_abstain_hi": ["--set", "abstain_score=0.60"],
    "pred_core_radius": ["--set", "core_mode=radius_scaled"],
    "pred_legacy_rule": ["--set", "scoring=winsorized_linear",
                         "core_mode=radius_scaled", "core_px=6.0",
                         "abstain_band=0.15"],
}

#  Conditions that set a string-valued DepthParams field. See PRECONDITION in
#  the module docstring.
NEEDS_STRING_OVERRIDES = ("pred_core_radius", "pred_legacy_rule")

DEFAULT_SETS = {
    "heldout": FQ / "input" / "synthetic_depth_heldout",
    "heldout_shifted": FQ / "input" / "synthetic_depth_heldout_shifted",
}


def _runner_takes_strings() -> bool:
    """True when run_depth.py can forward a string-valued override.

    Cheap source check rather than a trial run: the coercion is one line in
    ``depth_params()``, and a trial run costs a full scene set.
    """
    src = (FQ / "eval" / "runners" / "run_depth.py").read_text(
        encoding="utf-8")
    return "isinstance(default, str)" in src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sets", nargs="*", default=None,
                    help="name=path pairs; default: the two held-out sets")
    ap.add_argument("--conditions", nargs="*", default=sorted(CONDITIONS),
                    help=f"subset of {sorted(CONDITIONS)}")
    args = ap.parse_args()

    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; "
                         f"known: {sorted(CONDITIONS)}")

    blocked = [c for c in args.conditions if c in NEEDS_STRING_OVERRIDES]
    if blocked and not _runner_takes_strings():
        raise SystemExit(
            f"conditions {blocked} set a string-valued DepthParams field "
            "(core_mode / scoring), and "
            f"{FQ / 'eval' / 'runners' / 'run_depth.py'} coerces every --set "
            "override to int or float in depth_params(), so it would raise. "
            "Teach it to pass a string through when the field's default is a "
            "string, then re-run. To run the rest of the grid now: "
            "--conditions "
            + " ".join(c for c in sorted(CONDITIONS)
                       if c not in NEEDS_STRING_OVERRIDES))

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
