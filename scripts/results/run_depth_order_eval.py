"""Re-run the 12-scene depth-order study's predictions, then score them.

``exploration/depth_order_eval/score_depth.py`` scores stored ``pred_depth.json``
files; it does not produce them. Nothing in either repo recorded HOW they were
produced, so the study could not be re-measured after the evidence rule changed
on 2026-08-22. This is that missing step, recovered and written down.

Provenance check (2026-08-23). The stored predictions were reproduced from the
scene folders by running ``plecta.depth``'s own CLI at its defaults, with the
legacy overrides of the pre-2026-08-22 rule::

    --set scoring=winsorized_linear core_mode=radius_scaled \
          core_px=6.0 abstain_band=0.15

``oracle`` adds ``--oracle``; ``clean`` adds ``--mask-name mask_clean.png``
(the non-degraded axis every 3-D study standardised on, MEMO section 2). On
``cov20/synth_0000`` that reproduces the stored instance set exactly, the stored
crossing set exactly, and 39 of 41 stored oracle decisions / 34 of 37 stored
clean decisions. It is NOT bit-identical, and cannot be: the stored records
predate several unrelated changes to ``plecta/depth.py`` -- they carry no
``score`` field, no ``scoring``/``core_mode`` in ``solver_report``, and
``undecided_placement: id_rank`` where the current compact colouring reports
``coloured``. So a re-run moves the numbers for two reasons at once, the
evidence rule and that drift, and the two cannot be separated from the stored
artefacts alone.

Default settings are the SHIPPED ones (one channel, noise-floored, flat 7.5 px
core, abstain |s| < 0.40). ``--legacy`` re-runs the previous rule instead, which
is what separates rule change from code drift on demand.

Usage:
  python scripts/results/run_depth_order_eval.py [--legacy] [--score-only]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = r"C:\Repos\venv_cnt\Scripts\python.exe"
STUBMATCH = Path(r"C:\Repos\stubmatch")
REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "exploration" / "depth_order_eval"

LEGACY = ["--set", "scoring=winsorized_linear", "core_mode=radius_scaled",
          "core_px=6.0", "abstain_band=0.15"]

# condition -> extra CLI flags. Both then run at parameters.yaml defaults:
# image evidence on, grazing overlaps not cleared, undecided order compact.
CONDITIONS = {
    "oracle": ["--oracle"],
    "clean": ["--mask-name", "mask_clean.png"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", action="store_true",
                    help="re-run under the pre-2026-08-22 evidence rule")
    ap.add_argument("--score-only", action="store_true",
                    help="skip prediction, just re-score what is on disk")
    ap.add_argument("--conditions", nargs="*", default=sorted(CONDITIONS))
    args = ap.parse_args()

    if not args.score_only:
        for cond in args.conditions:
            for scene in sorted((STUDY / "scenes").glob("*/*")):
                if not (scene / "gt_depth.json").is_file():
                    continue
                out = STUDY / f"pred_{cond}" / f"{scene.parent.name}_{scene.name}"
                cmd = [PY, "-m", "plecta.depth",
                       "--scene", str(scene), "--out", str(out),
                       *CONDITIONS[cond], *(LEGACY if args.legacy else [])]
                res = subprocess.run(cmd, cwd=str(STUBMATCH),
                                     capture_output=True, text=True)
                if res.returncode != 0:
                    print(res.stdout[-2000:])
                    print(res.stderr[-2000:], file=sys.stderr)
                    raise SystemExit(f"{cond}/{out.name} failed")
                print(f"{cond:7} {res.stdout.strip()}", flush=True)

    res = subprocess.run([PY, str(STUDY / "score_depth.py")],
                         cwd=str(STUDY), capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit("score_depth.py failed")
    print(f"wrote {STUDY / 'depth_scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
