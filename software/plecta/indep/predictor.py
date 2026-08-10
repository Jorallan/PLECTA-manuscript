"""Mask in, overlapping instances out."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .linker import Matching, Params, solve
from .skelgraph import SkelGraph, build_graph


def instances_from_chains(graph: SkelGraph, chains: List[List[int]],
                          include_nodes: bool = True,
                          min_isolated_px: int = 0) -> Dict[int, np.ndarray]:
    """Paint each chain, sharing crossing pixels between everything through it.

    A crossing belongs to *every* filament that runs through it, so the node's
    pixels are written into every chain that has a stub there.  That is where
    the output stops being a partition and becomes a set of overlapping layers.
    """
    masks: Dict[int, np.ndarray] = {}
    for chain in list(chains):
        # A one-arm chain that touches no crossing and is shorter than the
        # scorer's 6-px fragment floor is a degradation speck, not a filament.
        # Emitting it cannot change the pairwise score (the scorer discards the
        # fragment) but it does inflate the instance count and the pixel-level
        # recovery diagnostic, so it is dropped and said so.
        if min_isolated_px and len(chain) == 1:
            arm = graph.arms[chain[0]]
            if (len(arm.pixels) < min_isolated_px
                    and all(graph.stubs[s].node is None for s in arm.stubs)):
                chains = [c for c in chains if c is not chain]
    for k, chain in enumerate(chains, start=1):
        canvas = np.zeros(graph.shape, dtype=bool)
        nodes_touched = set()
        for aid in chain:
            arm = graph.arms[aid]
            rows = [p[0] for p in arm.pixels]
            cols = [p[1] for p in arm.pixels]
            canvas[rows, cols] = True
            for sid in arm.stubs:
                node = graph.stubs[sid].node
                if node is not None:
                    nodes_touched.add(node)
        if include_nodes:
            for nid in nodes_touched:
                pix = graph.nodes[nid].pixels
                if pix:
                    canvas[[p[0] for p in pix], [p[1] for p in pix]] = True
        if canvas.any():
            masks[k] = canvas
    return masks


def predict(mask: np.ndarray, params: Optional[Params] = None,
            include_nodes: bool = True, min_isolated_px: int = 6,
            return_internals: bool = False):
    """Run the whole method on one binary mask."""
    params = params or Params()
    graph = build_graph(mask, spur_px=params.spur_px, bridge_px=params.bridge_px,
                        absorb_free_px=params.absorb_free_px,
                        join_px=params.join_px)
    matching, chains = solve(graph, params)
    masks = instances_from_chains(graph, chains, include_nodes=include_nodes,
                                  min_isolated_px=min_isolated_px)
    if return_internals:
        return masks, graph, matching, chains
    return masks


# ── serialisation compatible with instance_io.load_multilabel_npz ──────────


def save_multilabel_npz(path, masks: Dict[int, np.ndarray],
                        shape: Tuple[int, int]) -> None:
    ids = sorted(masks)
    indices: List[np.ndarray] = []
    indptr = [0]
    for i in ids:
        flat = np.flatnonzero(np.asarray(masks[i], dtype=bool).ravel())
        indices.append(flat.astype(np.int64))
        indptr.append(indptr[-1] + flat.size)
    stacked = (np.concatenate(indices) if indices
               else np.zeros(0, dtype=np.int64))
    np.savez_compressed(
        str(path),
        shape=np.asarray(shape, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=stacked,
    )

