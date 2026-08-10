"""PLECTA filament instance reconstruction: binary axis mask in, instances out.

    # a directory of scene folders, each containing mask_w1.png
    python predict.py --scenes <root> --out <outdir>

    # a single mask
    python predict.py --mask <scene>/mask_w1.png --out pred.npz

The only input ever read is the binary mask.  Ground truth, `mask_clean.png` and
`sem.png` are never opened -- there is no code path here that could.

Output per scene is ``pred_multilabel.npz``, the CSR-style sparse multilabel
that ``eval/core/instance_io.load_pred_instances`` reads.  Instances may
overlap: a crossing pixel is written into every filament that runs through it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plecta.indep.linker import Params
from plecta.indep.predictor import predict, save_multilabel_npz
from plecta.indep.skelgraph import read_mask

LOCKED = "synthetic_locked_v1"

PARAMS_PATH = Path(__file__).resolve().parent / "params.json"
if not PARAMS_PATH.is_file():
    raise RuntimeError(f"frozen parameter file is missing: {PARAMS_PATH}")
TUNED: Dict[str, float] = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))


def refuse_locked(path: Path) -> None:
    """Prevent either CLI input mode from spending the archived locked set."""
    if LOCKED in str(Path(path)):
        raise SystemExit(f"refusing to run on the locked evaluation set: {path}")


def build_params(overrides: List[str] | None = None) -> Params:
    """Tuned defaults from params.json, then any command-line override.

    Types come from the dataclass *defaults*, not the annotations: this module
    uses ``from __future__ import annotations``, so field .type is the string
    "int" and an integer parameter would silently become a float.
    """
    defaults = Params()
    values = dict(TUNED)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        values[key.strip()] = raw
    out = {}
    for key, value in values.items():
        if not hasattr(defaults, key):
            raise SystemExit(f"unknown parameter '{key}'. "
                             f"known: {sorted(f.name for f in fields(Params))}")
        out[key] = int(value) if isinstance(getattr(defaults, key), int) else float(value)
    return Params(**out)


def find_scenes(root: Path, mask_name: str) -> List[Path]:
    root = Path(root)
    refuse_locked(root)
    if (root / mask_name).is_file():
        return [root]
    return [d for d in sorted(root.rglob(mask_name))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes", help="directory of scene folders")
    src.add_argument("--mask", help="a single binary mask image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-name", default="mask_w1.png")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    params = build_params(args.overrides)
    out_root = Path(args.out)

    if args.mask:
        refuse_locked(Path(args.mask))
        mask = read_mask(args.mask)
        masks = predict(mask, params)
        out_root.parent.mkdir(parents=True, exist_ok=True)
        save_multilabel_npz(out_root, masks, mask.shape)
        print(f"{len(masks)} instances -> {out_root}")
        return 0

    hits = find_scenes(Path(args.scenes), args.mask_name)
    if not hits:
        raise SystemExit(f"no {args.mask_name} found under {args.scenes}")
    out_root.mkdir(parents=True, exist_ok=True)
    for hit in hits:
        scene = hit if hit.is_dir() else hit.parent
        refuse_locked(scene)
        mask_path = scene / args.mask_name
        t0 = time.time()
        mask = read_mask(mask_path)
        masks = predict(mask, params)
        dest = out_root / f"{scene.parent.name}__{scene.name}"
        dest.mkdir(parents=True, exist_ok=True)
        save_multilabel_npz(dest / "pred_multilabel.npz", masks, mask.shape)
        if not args.quiet:
            print(f"{scene.parent.name}/{scene.name}: {len(masks)} instances  "
                  f"{time.time() - t0:.2f}s", flush=True)
    (out_root / "params_used.json").write_text(json.dumps(asdict(params), indent=1))
    print(f"{len(hits)} scenes -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
