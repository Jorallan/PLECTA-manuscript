"""Score PLECTA on the four input-mask variants of the development scenes.

The generator writes four binary masks per scene from the same geometry:
``mask_clean.png`` (an undegraded axis) and ``mask_w1/w2/w3.png`` (a 1, 2 or
3 px axis carrying the generator's degradation).  ``mask_w1`` is the evaluated
input everywhere else in the paper; this script measures what happens when
PLECTA is handed the other three instead.

Development scenes only.  Nothing here selects a parameter: the configuration
is frozen and read from ``plecta/params.json``.

Writes ``results/plecta_mask_variants.json``.

    python scripts/figures/extract_plecta_mask_variants.py \
        --plecta C:/Repos/PLECTA \
        --scenes C:/Repos/filaments_quantification/input/synthetic_thick \
        --eval   C:/Repos/filaments_quantification/eval
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "plecta_mask_variants.json"

VARIANTS = (("clean", "mask_clean.png", "clean axis"),
            ("w1", "mask_w1.png", "1 px"),
            ("w2", "mask_w2.png", "2 px"),
            ("w3", "mask_w3.png", "3 px"))
DENSITIES = (20, 30, 40, 50, 60)
N_PER_DENSITY = 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plecta", default=r"C:/Repos/PLECTA")
    ap.add_argument("--scenes",
                    default=r"C:/Repos/filaments_quantification/input/synthetic_thick")
    ap.add_argument("--eval", default=r"C:/Repos/filaments_quantification/eval")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.plecta)))
    from plecta.graph import read_mask                     # noqa: E402
    from plecta.predict import load_params, predict        # noqa: E402
    from dataclasses import asdict                         # noqa: E402

    spec = importlib.util.spec_from_file_location(
        "common_metric", Path(args.eval) / "core" / "common_metric.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)

    params = load_params()
    scenes_root = Path(args.scenes)
    rows = []
    for density in DENSITIES:
        for k in range(N_PER_DENSITY):
            scene = scenes_root / f"cov{density}" / f"synth_{k:04d}"
            if not scene.is_dir():
                continue
            for key, filename, label in VARIANTS:
                mask_path = scene / filename
                if not mask_path.is_file():
                    continue
                masks = predict(read_mask(mask_path), params)
                score = cm.score_method(mask_path, scene / "gt_labels.tif", masks)
                rows.append({
                    "scene": f"cov{density}/synth_{k:04d}",
                    "density": density,
                    "variant": key,
                    "variant_label": label,
                    "f1": float(score["f1"]),
                    "precision": float(score["precision"]),
                    "recall": float(score["recall"]),
                    "adjusted_rand_index": float(score["adjusted_rand_index"]),
                    "vi_split_bits": float(score["vi_split_bits"]),
                    "vi_merge_bits": float(score["vi_merge_bits"]),
                    "n_instances": len(masks),
                })
                print(rows[-1]["scene"], key, round(rows[-1]["f1"], 4),
                      flush=True)

    out = {
        "schema": 1,
        "role": "development-only input-mask sensitivity; exploratory, "
                "n = %d scenes per density, no inference"
                % N_PER_DENSITY,
        "scenes_root": str(scenes_root),
        "params": asdict(params),
        "variants": [{"key": k, "file": f, "label": lb} for k, f, lb in VARIANTS],
        "densities": list(DENSITIES),
        "n_per_density": N_PER_DENSITY,
        "scorer": "eval/core/common_metric.py::score_method, "
                  "gt_unassigned_policy=singleton",
        "note": "The scorer builds its common fragments from whichever mask "
                "was given to PLECTA, so each variant is scored against the "
                "fragments of its own input.",
        "per_scene": rows,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[wrote]", OUT, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
