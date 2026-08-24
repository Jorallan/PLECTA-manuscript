"""Run the depth stage on one real, manually-annotated SEM field, timed.

Scope: this is a *runtime and plumbing* demonstration, not a validated depth
result. No depth ground truth exists for real SEM data (task 15 of the depth
extension is future work), so nothing here is scored -- only reported: how
long the 2-D core and the depth stage take on a real, dense field, and what
they produce. Do not read a z value or a crossing decision off this scene as
verified; read only the counts and the timing.

Input: the "b58_skel" scene already used for every other real-data PLECTA
number in this manuscript -- `results/plecta_real_masks.json` /
`plecta_real_field_metrics.json` -- so this reuses evidence, not a new draw.
Its mask is "per-instance skeletonize then OR" of the manual annotation at
    C:/Repos/datasets/real/manual annotation/
        B58-B3-S2_100_Electron_TLD_Custom_1.66_um__manual/..._multilabel.npz
built by C:/Repos/comparisons/real_sem_study/scripts/build_scenes_v2.py.

Writes:
    results/plecta_real_depth_demo.json           summary + provenance
    results/figure_assets/real_depth_demo.npz      geometry for the figure
        (instance ids/z/layer/d/centreline CSR; crossing i/j/x/y/over/
        abstain/flipped/p_over) -- the heavy arrays a figure script needs,
        following the repository's figure_assets convention.

Usage:
    python scripts/results/make_real_depth_demo.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
ASSETS = RESULTS / "figure_assets"

SCENE = Path(r"C:\Repos\comparisons\real_sem_study\scenes_v2\b58_100\skel")
MASK_PATH = SCENE / "mask.png"
SEM_PATH = SCENE / "sem.png"
META_PATH = SCENE / "scene_meta.json"

sys.path.insert(0, r"C:\Repos\PLECTA")
from plecta.graph import read_mask  # noqa: E402
from plecta.predict import load_params, predict  # noqa: E402
from plecta import depth as D  # noqa: E402
from plecta.image.measurement import load_sem  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    mask = read_mask(MASK_PATH)

    t0 = time.time()
    params2d = load_params()
    masks, graph, matching, chains = predict(mask, params2d,
                                             return_internals=True)
    t1 = time.time()

    centrelines = D.centrelines_from_chains(graph, matching, chains)
    centrelines = {k: v for k, v in centrelines.items() if k in masks}

    image = D._read_gray(SEM_PATH)
    sem_img = load_sem(SEM_PATH)

    t2 = time.time()
    dparams = D.DepthParams()
    pred = D.run_scene(image, centrelines, radii={}, params=dparams,
                       with_metric_z=True, sem_image=sem_img)
    t3 = time.time()

    rep = pred["solver_report"]
    zs = np.array([r["z"] for r in pred["instances"] if r["z"] is not None])
    ds = np.array([r["d"] for r in pred["instances"] if r.get("d")])

    summary = {
        "role": "runtime + plumbing demonstration on one real field; "
                "NOT a validated depth result (no real-SEM depth ground "
                "truth exists) -- report timing and counts, not accuracy",
        "source_record": {
            "scene": str(SCENE).replace("\\", "/"),
            "mask_sha256": sha256(MASK_PATH),
            "sem_sha256": sha256(SEM_PATH),
            "scene_meta": meta,
            "note": "same b58_skel scene as plecta_real_masks.json / "
                    "plecta_real_field_metrics.json (1. stubmatch, frozen)",
        },
        "shape": list(mask.shape),
        "n_foreground_px": int(mask.sum()),
        "timing_s": {
            "predict_2d": round(t1 - t0, 4),
            "depth_stage": round(t3 - t2, 4),
            "total": round((t1 - t0) + (t3 - t2), 4),
        },
        "n_instances_2d": len(masks),
        "n_arms": len(graph.arms),
        "n_nodes": len(graph.nodes),
        "n_instances_with_centreline": len(centrelines),
        "depth": {
            "n_crossings": rep["n_crossings"],
            "n_abstained": rep["n_abstained"],
            "n_relations_flipped": rep["n_relations_flipped"],
            "n_precedence_cycles_raw": rep["n_precedence_cycles_raw"],
            "n_layers": rep["n_layers"],
            "metric_z_status": rep["metric_z_status"],
            "z_range_px": [round(float(zs.min()), 3), round(float(zs.max()), 3)]
                          if zs.size else None,
            "n_instances_with_diameter": int(ds.size),
            "diameter_range_px": [round(float(ds.min()), 2),
                                  round(float(ds.max()), 2)] if ds.size else None,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / "plecta_real_depth_demo.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")

    # ── geometry for the figure (figure_assets convention) ───────────────
    ids = sorted(centrelines)
    z_of = {r["id"]: r["z"] for r in pred["instances"]}
    layer_of = {r["id"]: r["layer"] for r in pred["instances"]}
    d_of = {r["id"]: r["d"] for r in pred["instances"]}

    offsets = [0]
    coords = []
    for iid in ids:
        coords.append(centrelines[iid])
        offsets.append(offsets[-1] + len(centrelines[iid]))
    coords = np.concatenate(coords, axis=0).astype(np.float32)

    cx = np.array([c["x"] for c in pred["crossings"]], dtype=np.float32)
    cy = np.array([c["y"] for c in pred["crossings"]], dtype=np.float32)
    ci = np.array([c["i"] for c in pred["crossings"]], dtype=np.int32)
    cj = np.array([c["j"] for c in pred["crossings"]], dtype=np.int32)
    cover = np.array([c["over"] if c["over"] is not None else -1
                      for c in pred["crossings"]], dtype=np.int32)
    cabstain = np.array([c["abstain"] for c in pred["crossings"]], dtype=bool)
    cflipped = np.array([c["flipped"] for c in pred["crossings"]], dtype=bool)
    cp = np.array([c["p_over"] if c["p_over"] is not None else np.nan
                   for c in pred["crossings"]], dtype=np.float32)

    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ASSETS / "real_depth_demo.npz",
        shape=np.asarray(mask.shape, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        z=np.asarray([z_of[i] if z_of[i] is not None else np.nan for i in ids],
                     dtype=np.float32),
        layer=np.asarray([layer_of[i] for i in ids], dtype=np.int32),
        d=np.asarray([d_of[i] if d_of[i] else np.nan for i in ids],
                     dtype=np.float32),
        centreline_offsets=np.asarray(offsets, dtype=np.int64),
        centreline_coords=coords,
        cross_i=ci, cross_j=cj, cross_x=cx, cross_y=cy, cross_over=cover,
        cross_abstain=cabstain, cross_flipped=cflipped, cross_p=cp,
    )
    print(f"wrote {ASSETS / 'real_depth_demo.npz'}")

    print(f"2-D predict: {summary['timing_s']['predict_2d']}s, "
          f"{summary['n_instances_2d']} instances")
    print(f"depth stage: {summary['timing_s']['depth_stage']}s, "
          f"{rep['n_crossings']} crossings, K={rep['n_layers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
