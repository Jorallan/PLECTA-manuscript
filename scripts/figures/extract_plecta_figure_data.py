"""Extract every real-pixel and real-number payload the PLECTA figures need.

This is the ONLY script in ``scripts/figures/`` that touches the pipeline
repository and the raw scene data.  It runs offline, by hand, and writes two
committed artefacts into ``results/``:

    results/plecta_figure_data.json     -- pixel payloads and measured numbers
    results/figure_assets/*.png         -- the few genuine raster images

Every ``fig_*.py`` generator then reads only those, so the figure set rebuilds
from a clean checkout of this manuscript repository with no access to
``input/``, to the pipeline source, or to any gitignored location.

Usage (paths are the defaults on the author's machine)::

    python scripts/figures/extract_plecta_figure_data.py \
        --plecta   C:/Repos/PLECTA \
        --scenes   C:/Repos/filaments_quantification/input/synthetic_thick \
        --heldout  C:/Repos/PLECTA/.local/evaluation/data \
        --eval     C:/Repos/filaments_quantification/eval

Scene and node selections are deterministic and are recorded in the
``provenance`` block of the JSON, so every figure can be traced back to the
scene, the node id and the parameter set it came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "results" / "plecta_figure_data.json"
OUT_ASSETS = REPO / "results" / "figure_assets"

# Development scenes used for every method-illustration panel.  The held-out
# set is touched only by the qualitative panel, and only to render scenes that
# were already scored -- never to choose a parameter.
CROSSING_SCENE = "cov20/synth_0001"
GAP_SCENE = "cov20/synth_0001"
ROUNDS_SCENE = "cov30/synth_0003"
PROCEDURAL_SCENE = "cov30/synth_0001"
SCORER_SCENE = "cov20/synth_0001"
UNMATCHED_SCENE = "cov40/synth_0001"
DENSITIES = ("cov20", "cov30", "cov40", "cov50", "cov60")


# ── payload encoding ───────────────────────────────────────────────────────


def pack(mask: np.ndarray) -> dict:
    """A boolean image as {h, w, idx} with idx flat row-major indices."""
    mask = np.asarray(mask, dtype=bool)
    return {"h": int(mask.shape[0]), "w": int(mask.shape[1]),
            "idx": np.flatnonzero(mask.ravel()).astype(int).tolist()}


def pack_labels(labels: np.ndarray) -> dict:
    """An int label image as {h, w, idx, lab}, background dropped."""
    labels = np.asarray(labels)
    flat = labels.ravel()
    nz = np.flatnonzero(flat)
    return {"h": int(labels.shape[0]), "w": int(labels.shape[1]),
            "idx": nz.astype(int).tolist(),
            "lab": flat[nz].astype(int).tolist()}


def crop_box(cy, cx, half, shape):
    y0 = int(np.clip(round(cy - half), 0, max(0, shape[0] - 2 * half)))
    x0 = int(np.clip(round(cx - half), 0, max(0, shape[1] - 2 * half)))
    return y0, x0, 2 * half, 2 * half


def sub(mask, box):
    y0, x0, h, w = box
    return np.asarray(mask)[y0:y0 + h, x0:x0 + w]


# ── pipeline access ────────────────────────────────────────────────────────


class Pipeline:
    def __init__(self, plecta_root: Path, eval_root: Path):
        sys.path.insert(0, str(plecta_root))
        sys.path.insert(0, str(eval_root.parent))
        from plecta.graph import read_mask, build_graph          # noqa
        from plecta.linking import (build_frames, junction_costs,  # noqa
                                    _max_weight_pairs, solve)
        from plecta.geometry import link_cost                     # noqa
        from plecta.predict import load_params, predict           # noqa
        self.read_mask = read_mask
        self.build_graph = build_graph
        self.build_frames = build_frames
        self.junction_costs = junction_costs
        self.max_weight_pairs = _max_weight_pairs
        self.solve = solve
        self.link_cost = link_cost
        self.predict = predict
        self.params = load_params()
        self.plecta_root = plecta_root
        sys.path.insert(0, str(eval_root / "core"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "common_metric", eval_root / "core" / "common_metric.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.cm = mod

    def graph_of(self, mask):
        p = self.params
        return self.build_graph(mask, spur_px=p.spur_px, bridge_px=p.bridge_px,
                                absorb_free_px=p.absorb_free_px,
                                join_px=p.join_px)

    def run(self, mask):
        """Graph, final-round frames, matching, chains and painted instances."""
        g = self.graph_of(mask)
        matching, chains = self.solve(g, self.params)
        frames = self.build_frames(g, matching, self.params.window_chain,
                                   self.params)
        masks, g2, m2, c2 = self.predict(mask, self.params,
                                         return_internals=True)
        return g, frames, matching, chains, masks


def greedy_pairs(costs, price):
    """Sequential baseline: cheapest admissible edge first, arms consumed."""
    used, out = set(), []
    for (a, b), c in sorted(costs.items(), key=lambda kv: (kv[1], kv[0])):
        if c >= 2 * price:
            break
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.append((min(a, b), max(a, b)))
    return sorted(out)


def objective(pairs, costs, stubs, price):
    """Total cost of a configuration: chosen links plus a price per free stub."""
    return (sum(costs[(min(a, b), max(a, b))] for a, b in pairs)
            + price * (len(stubs) - 2 * len(pairs)))


# ── individual payloads ────────────────────────────────────────────────────


def find_crossing_node(g, matching, min_arm=14):
    """A 4-arm node whose stubs the method paired up completely."""
    best = None
    for node in g.nodes:
        stubs = sorted(node.stubs)
        if len(stubs) != 4:
            continue
        if not all(s in matching and matching[s] in stubs for s in stubs):
            continue
        lens = [len(g.arms[g.stubs[s].aid].path) for s in stubs]
        if min(lens) < min_arm:
            continue
        if best is None or min(lens) > best[0]:
            best = (min(lens), node)
    return None if best is None else best[1]


def node_centre(node):
    pix = np.asarray(node.pixels, float)
    return float(pix[:, 0].mean()), float(pix[:, 1].mean())


def junction_case(P, g, frames, matching, node, half):
    """Everything needed to draw one junction decision from real numbers."""
    stubs = sorted(node.stubs)
    costs = P.junction_costs(g, frames, stubs, P.params)
    price = P.params.j_unmatched
    exact = sorted(P.max_weight_pairs(stubs, costs, price))
    greedy = greedy_pairs(costs, price)
    cy, cx = node_centre(node)
    box = crop_box(cy, cx, half, g.shape)
    y0, x0 = box[0], box[1]

    # every pairing that respects "no arm may close on itself", enumerated so
    # the figure can show the joint score of the alternatives, not just the winner
    def enumerate_configs(items):
        if not items:
            yield []
            return
        head, rest = items[0], items[1:]
        yield from ([*c] for c in enumerate_configs(rest))       # head unmatched
        for i, other in enumerate(rest):
            key = (min(head, other), max(head, other))
            if key not in costs:
                continue
            for c in enumerate_configs(rest[:i] + rest[i + 1:]):
                yield [key, *c]

    configs = []
    seen = set()
    for cfg in enumerate_configs(stubs):
        key = tuple(sorted(cfg))
        if key in seen:
            continue
        seen.add(key)
        if any(costs[p] >= 2 * price for p in cfg):
            continue                       # not admissible: edge refused outright
        configs.append({"pairs": [list(p) for p in cfg],
                        "total": round(objective(cfg, costs, stubs, price), 4)})
    configs.sort(key=lambda c: c["total"])

    per_stub = {}
    for s in stubs:
        f = frames[s]
        st = g.stubs[s]
        per_stub[str(s)] = {
            "aid": int(st.aid),
            "tip": [float(f.tip[0]) - y0, float(f.tip[1]) - x0],
            "tangent": [float(f.tangent[0]), float(f.tangent[1])],
            "curvature": float(f.curvature),
            "reliable": bool(f.reliable),
            "arm_px": int(len(g.arms[st.aid].path)),
            "arm": [[int(r) - y0, int(c) - x0] for r, c in g.arms[st.aid].path],
        }

    pair_rows = []
    for (a, b), c in sorted(costs.items()):
        lc = P.link_cost(frames[a], frames[b], w_turn=P.params.j_w_turn,
                         w_direct=P.params.j_w_direct, w_len=0.0,
                         chord_floor=P.params.j_chord_floor,
                         w_kappa=P.params.j_w_kappa)
        pair_rows.append({"a": int(a), "b": int(b), "cost": round(float(c), 4),
                          "theta": round(float(lc.theta), 4),
                          "phi_a": round(float(lc.phi_a), 4),
                          "phi_b": round(float(lc.phi_b), 4),
                          "d": round(float(lc.length), 3),
                          "admissible": bool(c < 2 * price)})

    node_mask = np.zeros(g.shape, bool)
    rows, cols = zip(*node.pixels)
    node_mask[list(rows), list(cols)] = True

    return {
        "nid": int(node.nid),
        "degree": len(stubs),
        "stubs": [int(s) for s in stubs],
        "origin": [y0, x0], "size": half * 2,
        "node_px": pack(sub(node_mask, box)),
        "frames": per_stub,
        "pairs": pair_rows,
        "price": float(price),
        "admissibility_limit": float(2 * price),
        "exact": [list(p) for p in exact],
        "greedy": [list(p) for p in greedy],
        "exact_total": round(objective(exact, costs, stubs, price), 4),
        "greedy_total": round(objective(greedy, costs, stubs, price), 4),
        "configurations": configs[:12],
        "n_configurations": len(configs),
    }


def divergence_by_degree(P, scene_paths):
    """How often the joint solution differs from cheapest-edge-first, by degree."""
    buckets = {"2-4": [0, 0], "5-8": [0, 0], "9+": [0, 0]}
    gaps = []
    for path in scene_paths:
        mask = P.read_mask(path)
        g = P.graph_of(mask)
        matching, _ = P.solve(g, P.params)
        frames = P.build_frames(g, matching, P.params.window_chain, P.params)
        price = P.params.j_unmatched
        for node in g.nodes:
            stubs = sorted(node.stubs)
            if len(stubs) < 2:
                continue
            costs = P.junction_costs(g, frames, stubs, P.params)
            exact = sorted(P.max_weight_pairs(stubs, costs, price))
            greedy = greedy_pairs(costs, price)
            key = "2-4" if len(stubs) <= 4 else ("5-8" if len(stubs) <= 8 else "9+")
            buckets[key][0] += 1
            if exact != greedy:
                buckets[key][1] += 1
                gaps.append(objective(greedy, costs, stubs, price)
                            - objective(exact, costs, stubs, price))
    return {
        "buckets": {k: {"n_nodes": v[0], "n_differ": v[1]} for k, v in buckets.items()},
        "n_scenes": len(scene_paths),
        "mean_objective_gap": round(float(np.mean(gaps)), 4) if gaps else 0.0,
        "median_objective_gap": round(float(np.median(gaps)), 4) if gaps else 0.0,
    }


def gap_candidates(P, g, frames, matching):
    """Every gap pair the method considered, with its gates and its verdict."""
    from scipy.spatial import cKDTree
    p = P.params
    free = [s.sid for s in g.stubs if s.sid not in matching and frames[s.sid].reliable]
    if len(free) < 2:
        return []
    tips = np.array([frames[s].tip for s in free])
    tree = cKDTree(tips)
    rows = []
    for ii, jj in tree.query_pairs(p.gap_max_len, output_type="ndarray"):
        a, b = free[int(ii)], free[int(jj)]
        sa, sb = g.stubs[a], g.stubs[b]
        if sa.aid == sb.aid:
            continue
        if sa.node is not None and sa.node == sb.node:
            continue
        lc = P.link_cost(frames[a], frames[b], w_turn=p.g_w_turn,
                         w_direct=p.g_w_direct, w_len=p.g_w_len,
                         len_scale=p.g_len_scale, chord_floor=p.g_chord_floor,
                         w_kappa=p.g_w_kappa)
        if not math.isfinite(lc.cost):
            continue
        passes = (lc.theta <= p.gap_max_theta
                  and max(lc.phi_a, lc.phi_b) <= p.gap_max_phi
                  and lc.cost < 2 * p.g_unmatched)
        rows.append({"theta": round(float(lc.theta), 4),
                     "phi_max": round(float(max(lc.phi_a, lc.phi_b)), 4),
                     "d": round(float(lc.length), 2),
                     "cost": round(float(lc.cost), 4),
                     "gated_in": bool(passes)})
    return rows


def scorer_stages(P, mask, box):
    """The scorer's four fragment-construction steps, on real pixels."""
    from skimage.morphology import skeletonize
    from skimage.measure import label as cc_label
    cm = P.cm
    skel = skeletonize(np.asarray(mask, bool))
    pruned = cm._prune_spurs(skel, 3)
    deg = cm._degree_map(pruned)
    junctions = pruned & (deg >= 3)
    non_junc = pruned & ~junctions
    lbl = cc_label(non_junc, connectivity=2)
    kept = np.zeros(mask.shape, np.int32)
    nxt = 1
    for k in range(1, int(lbl.max()) + 1):
        piece = lbl == k
        if int(piece.sum()) >= 6:
            kept[piece] = nxt
            nxt += 1
    dropped = (lbl > 0) & (kept == 0)
    return {
        "mask": pack(sub(mask, box)),
        "skeleton": pack(sub(pruned, box)),
        "junction_px": pack(sub(junctions, box)),
        "fragments": pack_labels(sub(kept, box)),
        "dropped": pack(sub(dropped, box)),
        "origin": [box[0], box[1]], "size": box[2],
    }


def layer_payload(masks, box, limit=None):
    """Painted instance layers inside a crop, kept separate so overlap is visible."""
    out = []
    for key in sorted(masks, key=lambda k: -int(masks[k].sum())):
        piece = sub(masks[key], box)
        if not piece.any():
            continue
        out.append(pack(piece))
        if limit and len(out) >= limit:
            break
    return out


def read_gt_layers(scene_dir):
    """Reference instances as separate overlapping layers."""
    data = np.load(scene_dir / "gt_multilabel.npz")
    shape = tuple(int(v) for v in data["shape"])
    ids, indptr, indices = data["ids"], data["indptr"], data["indices"]
    layers = {}
    for k, key in enumerate(ids):
        flat = indices[indptr[k]:indptr[k + 1]]
        m = np.zeros(shape[0] * shape[1], bool)
        m[flat] = True
        layers[int(key)] = m.reshape(shape)
    return layers, shape


# ── main ───────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plecta", default=r"C:/Repos/PLECTA")
    ap.add_argument("--scenes",
                    default=r"C:/Repos/filaments_quantification/input/synthetic_thick")
    ap.add_argument("--heldout",
                    default=r"C:/Repos/PLECTA/.local/evaluation/data")
    ap.add_argument("--eval", default=r"C:/Repos/filaments_quantification/eval")
    args = ap.parse_args(argv)

    P = Pipeline(Path(args.plecta), Path(args.eval))
    scenes = Path(args.scenes)
    OUT_ASSETS.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    params = asdict(P.params)

    out = {
        "schema": 1,
        "provenance": {
            "generator": "scripts/figures/extract_plecta_figure_data.py",
            "pipeline": "PLECTA (github.com/Jorallan/FilaSeg working tree)",
            "params": params,
            "params_sha256": hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()).hexdigest()[:16],
            "development_scenes": str(scenes),
            "note": "Method-illustration panels come from development scenes. "
                    "The held-out panel renders scenes that were already scored; "
                    "nothing here selects a parameter.",
        },
    }

    # ── 1. crossing: mask, arms, node, painted layers ──────────────────
    cs = scenes / CROSSING_SCENE
    mask = P.read_mask(cs / "mask_w1.png")
    g, frames, matching, chains, masks = P.run(mask)
    node = find_crossing_node(g, matching)
    cy, cx = node_centre(node)
    half = 26
    box = crop_box(cy, cx, half, g.shape)

    arm_lbl = np.zeros(g.shape, np.int32)
    for arm in g.arms:
        rows, cols = zip(*arm.pixels)
        arm_lbl[list(rows), list(cols)] = arm.aid + 1
    node_all = g.node_pixels >= 0

    out["crossing"] = {
        "scene": CROSSING_SCENE, "nid": int(node.nid),
        "origin": [box[0], box[1]], "size": box[2],
        "mask": pack(sub(mask, box)),
        "arms": pack_labels(sub(arm_lbl, box)),
        "node_px": pack(sub(node_all, box)),
        "layers": layer_payload(masks, box),
        "n_arms_scene": len(g.arms), "n_nodes_scene": len(g.nodes),
        "n_instances_scene": len(masks),
    }

    # ── 2. the junction decision, from real costs ──────────────────────
    out["junction4"] = junction_case(P, g, frames, matching, node, half)
    out["junction4"]["scene"] = CROSSING_SCENE

    us = scenes / UNMATCHED_SCENE
    umask = P.read_mask(us / "mask_w1.png")
    ug, uframes, umatching, _, _ = P.run(umask)
    target = None
    for n in ug.nodes:
        st = sorted(n.stubs)
        if len(st) != 3:
            continue
        costs = P.junction_costs(ug, uframes, st, P.params)
        if len(costs) != 3:
            continue
        exact = sorted(P.max_weight_pairs(st, costs, P.params.j_unmatched))
        if len(exact) != 1:
            continue
        free = [s for s in st if not any(s in p for p in exact)][0]
        if not any(free in p and c < 2 * P.params.j_unmatched
                   for p, c in costs.items()):
            continue
        lens = [len(ug.arms[ug.stubs[s].aid].path) for s in st]
        if target is None or min(lens) > target[0]:
            target = (min(lens), n)
    if target is not None:
        out["junction3"] = junction_case(P, ug, uframes, umatching, target[1], 22)
        out["junction3"]["scene"] = UNMATCHED_SCENE

    # ── 3. exact vs greedy, over every development node ────────────────
    all_scenes = [scenes / c / f"synth_{k:04d}" / "mask_w1.png"
                  for c in DENSITIES for k in range(4)]
    all_scenes = [p for p in all_scenes if p.is_file()]
    out["greedy_divergence"] = divergence_by_degree(P, all_scenes)

    # ── 4. gaps: gates as measured, and one bridged gap ─────────────────
    gs = scenes / GAP_SCENE
    gmask = P.read_mask(gs / "mask_w1.png")
    gg, gframes, gmatching, _, gmasks = P.run(gmask)
    junctions_only = P.solve.__wrapped__ if False else None
    # candidates are scored against the frames of the final round, with the
    # junction stage already resolved -- exactly the state gap matching sees
    jm = {}
    for a, b in gmatching.items():
        sa, sb = gg.stubs[a], gg.stubs[b]
        if sa.node is not None and sa.node == sb.node:
            jm[a] = b
    out["gap_gates"] = {
        "scene": GAP_SCENE,
        "candidates": gap_candidates(P, gg, gframes, jm),
        "max_theta": P.params.gap_max_theta,
        "max_phi": P.params.gap_max_phi,
        "max_len": P.params.gap_max_len,
        "cost_limit": round(2 * P.params.g_unmatched, 4),
    }

    best_gap = None
    for a, b in gmatching.items():
        if a >= b:
            continue
        sa, sb = gg.stubs[a], gg.stubs[b]
        if sa.node is not None and sa.node == sb.node:
            continue
        pa = np.asarray(gframes[a].tip, float)
        pb = np.asarray(gframes[b].tip, float)
        d = float(np.hypot(*(pa - pb)))
        if not (14.0 <= d <= 40.0):
            continue
        arm_len = min(len(gg.arms[sa.aid].path), len(gg.arms[sb.aid].path))
        if arm_len < 18:
            continue
        if best_gap is None or arm_len > best_gap[0]:
            best_gap = (arm_len, (pa + pb) / 2.0, pa, pb, d)
    if best_gap is not None:
        _, mid, pa, pb, dist = best_gap
        gh = int(max(24, dist * 0.95))
        gbox = crop_box(mid[0], mid[1], gh, gg.shape)
        out["gap_case"] = {
            "scene": GAP_SCENE, "d": round(dist, 2),
            "origin": [gbox[0], gbox[1]], "size": gbox[2],
            "mask": pack(sub(gmask, gbox)),
            "layers": layer_payload(gmasks, gbox),
            "tips": [[float(pa[0]) - gbox[0], float(pa[1]) - gbox[1]],
                     [float(pb[0]) - gbox[0], float(pb[1]) - gbox[1]]],
        }

    # ── 5. rounds: the same region after 1 and after 8 rounds ──────────
    from dataclasses import replace
    rs = scenes / ROUNDS_SCENE
    rmask = P.read_mask(rs / "mask_w1.png")
    rg = P.graph_of(rmask)
    big = max(rg.nodes, key=lambda n: len(n.stubs))
    rcy, rcx = node_centre(big)
    rbox = crop_box(rcy, rcx, 62, rg.shape)
    rounds = []
    for nr in (1, 2, 4, 8):
        rmasks = P.predict(rmask, replace(P.params, n_rounds=nr))
        rounds.append({"n_rounds": nr, "n_instances": len(rmasks),
                       "layers": layer_payload(rmasks, rbox)})
    out["rounds"] = {"scene": ROUNDS_SCENE,
                     "origin": [rbox[0], rbox[1]], "size": rbox[2],
                     "mask": pack(sub(rmask, rbox)),
                     "series": rounds}

    # arm-length distribution: why one arm is not enough to aim with
    lengths = []
    for path in all_scenes:
        m = P.read_mask(path)
        gg2 = P.graph_of(m)
        for arm in gg2.arms:
            pts = np.asarray(arm.path, float)
            if len(pts) < 2:
                lengths.append(0.0)
            else:
                lengths.append(float(np.hypot(*np.diff(pts, axis=0).T).sum()))
    lengths = np.asarray(lengths)
    edges = np.arange(0, 105, 5.0)
    counts, _ = np.histogram(np.clip(lengths, 0, 100), bins=edges)
    out["arm_lengths"] = {
        "n_arms": int(lengths.size), "n_scenes": len(all_scenes),
        "bin_edges": edges.tolist(), "counts": counts.astype(int).tolist(),
        "median": round(float(np.median(lengths)), 2),
        "frac_below_window_local": round(float((lengths < P.params.window_local).mean()), 4),
        "frac_below_min_quadratic": round(float((lengths < P.params.min_quadratic).mean()), 4),
        "window_local": P.params.window_local,
        "window_chain": P.params.window_chain,
        "min_quadratic": P.params.min_quadratic,
    }

    # ── 6. the scorer, on real pixels ──────────────────────────────────
    ss = scenes / SCORER_SCENE
    smask = P.read_mask(ss / "mask_w1.png")
    sg = P.graph_of(smask)
    snode = find_crossing_node(sg, P.solve(sg, P.params)[0]) or sg.nodes[0]
    scy, scx = node_centre(snode)
    sbox = crop_box(scy, scx, 30, sg.shape)
    out["scorer"] = scorer_stages(P, smask, sbox)
    out["scorer"]["scene"] = SCORER_SCENE

    # ── 7. procedural data: what the generator makes, what is evaluated ─
    ps = scenes / PROCEDURAL_SCENE
    pmask_w1 = P.read_mask(ps / "mask_w1.png")
    pmask_clean = P.read_mask(ps / "mask_clean.png")
    gt_layers, gt_shape = read_gt_layers(ps)
    meta = json.loads((ps / "gt_meta.json").read_text())
    pbox = (0, 0, gt_shape[0], gt_shape[1])
    zbox = crop_box(gt_shape[0] * 0.42, gt_shape[1] * 0.42, 40, gt_shape)
    out["procedural"] = {
        "scene": PROCEDURAL_SCENE,
        "meta": meta,
        "shape": list(gt_shape),
        "mask_clean": pack(pmask_clean),
        "mask_w1": pack(pmask_w1),
        "gt_layers": [pack(v) for v in
                      sorted(gt_layers.values(), key=lambda m: -int(m.sum()))],
        "zoom": {"origin": [zbox[0], zbox[1]], "size": zbox[2],
                 "mask_clean": pack(sub(pmask_clean, zbox)),
                 "mask_w1": pack(sub(pmask_w1, zbox))},
        "n_reference_instances": len(gt_layers),
        "ladder": [],
    }
    for cov in DENSITIES:
        d = scenes / cov / "synth_0000"
        if not (d / "mask_w1.png").is_file():
            continue
        lm = P.read_mask(d / "mask_w1.png")
        lg, _, _, _, lmasks = P.run(lm)
        out["procedural"]["ladder"].append({
            "density": int(cov[3:]), "scene": f"{cov}/synth_0000",
            "mask_w1": pack(lm), "n_instances": len(lmasks),
            "n_arms": len(lg.arms), "n_nodes": len(lg.nodes),
            "n_mask_px": int(lm.sum()),
        })

    # the one genuine raster: a crop of the simulated micrograph
    from skimage import io as skio
    sem = skio.imread(str(ps / "sem.png"))
    if sem.ndim == 3:
        sem = sem[..., :3].mean(axis=2)
    sy, sx, sh, sw = crop_box(gt_shape[0] * 0.5, gt_shape[1] * 0.5, 128, gt_shape)
    skio.imsave(str(OUT_ASSETS / "sem_crop.png"),
                sem[sy:sy + sh, sx:sx + sw].astype(np.uint8), check_contrast=False)
    out["procedural"]["sem_crop"] = {"file": "figure_assets/sem_crop.png",
                                     "origin": [sy, sx], "size": sh}
    out["procedural"]["zoom_of_sem"] = [zbox[0] - sy, zbox[1] - sx]

    # ── 8. held-out qualitative: the median scene of each density ──────
    heldout = Path(args.heldout)
    per_scene = json.loads((REPO / "results" / "plecta_heldout.json")
                           .read_text())["per_scene"]
    by_density = {}
    for row in per_scene:
        by_density.setdefault(int(row["density"]), []).append(row)
    qual = []
    for density in (20, 40, 60):
        rows = sorted(by_density[density], key=lambda r: r["f1"])
        pick = rows[len(rows) // 2]                 # the median scene, not the best
        sdir = heldout / pick["scene"]
        if not (sdir / "mask_w1.png").is_file():
            continue
        qmask = P.read_mask(sdir / "mask_w1.png")
        _, _, _, _, qmasks = P.run(qmask)
        qgt, qshape = read_gt_layers(sdir)
        frag = P.cm.common_fragments(qmask)
        pred_assign = P.cm.assign_fragments(frag, qmasks)
        gt_assign = P.cm.assign_fragments(frag, qgt)
        # a fragment agrees when its predicted group's dominant reference group
        # is its own -- the same grouping the pairwise score is computed from
        by_pred = {}
        for f, p in pred_assign.items():
            by_pred.setdefault(int(p), []).append(int(f))
        dominant = {}
        for p, frags in by_pred.items():
            counts = {}
            for f in frags:
                gid = int(gt_assign.get(f, 0))
                counts[gid] = counts.get(gid, 0) + 1
            dominant[p] = max(counts, key=lambda k: (counts[k], k)) if counts else 0
        agree = np.zeros(qshape, bool)
        differ = np.zeros(qshape, bool)
        n_ok = n_bad = 0
        for f, p in pred_assign.items():
            gid = int(gt_assign.get(f, 0))
            pix = frag == f
            if gid and dominant.get(int(p), 0) == gid:
                agree |= pix
                n_ok += 1
            else:
                differ |= pix
                n_bad += 1
        qual.append({
            "scene": pick["scene"], "density": density,
            "shape": list(qshape),
            "f1": round(float(pick["f1"]), 4),
            "precision": round(float(pick["precision"]), 4),
            "recall": round(float(pick["recall"]), 4),
            "ari": round(float(pick["adjusted_rand_index"]), 4),
            "n_gt_groups": int(pick["n_gt_groups"]),
            "n_pred_groups": int(pick["n_pred_groups"]),
            "n_fragments": int(pick["n_common_fragments"]),
            "n_fragments_agree": n_ok, "n_fragments_differ": n_bad,
            "mask": pack(qmask),
            "gt_layers": [pack(v) for v in
                          sorted(qgt.values(), key=lambda m: -int(m.sum()))],
            "pred_layers": [pack(qmasks[k]) for k in
                            sorted(qmasks, key=lambda k: -int(qmasks[k].sum()))],
            "agree": pack(agree), "differ": pack(differ),
        })
    out["qualitative"] = qual

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    size = OUT_JSON.stat().st_size / 1e6
    print(f"[wrote] {OUT_JSON}  ({size:.2f} MB)")
    print(f"[wrote] {OUT_ASSETS / 'sem_crop.png'}")
    print(json.dumps(out["greedy_divergence"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
