"""Decide which arm continues into which, and stitch the results into chains.

Two decisions, both expressed as *matchings on stubs*:

* **at a crossing** -- the arms meeting at one node have to be paired up.  This
  is solved exactly and jointly per node (maximum-weight matching), not greedily
  arm-by-arm, because the arms constrain each other: pairing a with b is only
  attractive if what is left over for c and d is also cheap.
* **across a gap** -- ends left free after the crossings are resolved may be two
  halves of one filament that the mask broke.  Same machinery, one global
  matching over all free ends.

Leaving a stub unmatched is always an option and costs a fixed
``unmatched_penalty``; that is what lets a filament genuinely end, and what
stops a T-junction from inventing a continuation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .geometry import Frame, fit_frame, link_cost
from .skelgraph import SkelGraph

Matching = Dict[int, int]


@dataclass
class Params:
    # ── graph construction ────────────────────────────────────────────
    spur_px: int = 3                # dead-end hairs this short are skeleton noise
    bridge_px: int = 5              # arms this short between two crossings are
                                    # part of one bigger crossing (must be <= 5)
    absorb_free_px: int = 5         # free-ended arms this short are crossing debris
    join_px: int = 0                # arms this short merge their two crossings for
                                    # matching but keep their own identity

    # ── frames ────────────────────────────────────────────────────────
    window_local: float = 14.0      # arclength used for a first-pass tangent
    window_chain: float = 42.0      # arclength once arms have been chained
    min_quadratic: int = 9          # pixels needed before a curved fit is used
    n_rounds: int = 3               # frame/matching refinement rounds
    anneal_start: float = 1.0       # first round's acceptance limits, as a
                                    # fraction of the final ones (1.0 = off)

    # ── crossings ─────────────────────────────────────────────────────
    j_w_direct: float = 1.0         # weight on tangent-vs-tangent turn
    j_w_turn: float = 0.5           # weight on turn-onto-chord
    j_chord_floor: float = 4.0
    j_w_kappa: float = 0.0          # weight on curvature continuity
    j_unmatched: float = 0.62       # rad; pairing costs above 2x this are refused
    j_gap_relief: float = 0.0       # >0: a stub with a good gap partner waiting
                                    # pays less to decline pairing at a crossing

    # ── gaps ──────────────────────────────────────────────────────────
    # Measured on the development set: true gap links have median length 25 px
    # (p90 = 46), and 95% of *wrong* candidates turn more than 38 deg onto the
    # chord while 75% of right ones turn less than 20.  So the cut-off that
    # does the work is the turn, not the length -- the length only breaks ties.
    gap_max_len: float = 58.0       # px, longest bridge considered
    g_w_direct: float = 0.35        # weight on tangent-vs-tangent turn
    g_w_turn: float = 1.0           # weight on turn-onto-chord
    g_w_len: float = 0.30
    g_len_scale: float = 60.0
    g_chord_floor: float = 3.0
    g_w_kappa: float = 0.0
    g_unmatched: float = 0.55
    gap_max_theta: float = 0.70     # rad, hard reject on tangent disagreement
    gap_max_phi: float = 0.50       # rad, hard reject on turn-onto-chord


# ── stub frames ────────────────────────────────────────────────────────────


def walk_from_stub(graph: SkelGraph, sid: int, matching: Matching,
                   max_arclen: float) -> List[Tuple[int, int]]:
    """Pixels from a stub tip inward, following the current matching.

    Once arms are chained, the trajectory available for estimating "where is
    this filament heading" is the whole chain, not the one arm the stub sits
    on.  That matters because a third of the arms in a dense scene are under
    10 px long, which is far too short to measure a direction on.
    """
    points: List[Tuple[int, int]] = []
    total = 0.0
    cur = sid
    seen: Set[int] = set()
    while True:
        stub = graph.stubs[cur]
        if stub.aid in seen:
            break
        seen.add(stub.aid)
        arm = graph.arms[stub.aid]
        segment = arm.path if stub.end == 0 else arm.path[::-1]
        for p in segment:
            if points:
                total += math.hypot(p[0] - points[-1][0], p[1] - points[-1][1])
            points.append(p)
            if total >= max_arclen:
                return points
        far = graph.other_stub(cur)
        nxt = matching.get(far)
        if nxt is None:
            break
        cur = nxt
    return points


def build_frames(graph: SkelGraph, matching: Matching, window: float,
                 params: Params) -> Dict[int, Frame]:
    frames: Dict[int, Frame] = {}
    for stub in graph.stubs:
        pts = walk_from_stub(graph, stub.sid, matching, window)
        frames[stub.sid] = fit_frame(pts, window=window,
                                     min_quadratic=params.min_quadratic)
    return frames


# ── exact matching on a small stub set ─────────────────────────────────────


def _max_weight_pairs(stubs: Sequence[int],
                      costs: Dict[Tuple[int, int], float],
                      unmatched_penalty,
                      ) -> List[Tuple[int, int]]:
    """Minimum-total-cost partial matching, with a price for opting out.

    Maximising ``sum(P_i + P_j - c_ij)`` over the chosen pairs is identical to
    minimising ``sum(c_ij) + sum of P over the stubs left unmatched``: every pair
    chosen removes two opt-out payments and adds its own cost, and the total of
    all P is a constant.  Blossom (networkx) solves that exactly for any node
    degree, so nothing has to be greedy and nothing has to be capped.

    ``unmatched_penalty`` is either one number for every stub, or a mapping from
    stub id to its own price.  Per-stub prices are what let a stub that has a
    good continuation waiting on the far side of a mask gap decline to pair at
    the crossing.
    """
    import networkx as nx

    def price(sid: int) -> float:
        if isinstance(unmatched_penalty, dict):
            return float(unmatched_penalty[sid])
        return float(unmatched_penalty)

    graph = nx.Graph()
    graph.add_nodes_from(stubs)
    any_edge = False
    for (i, j), c in costs.items():
        limit = price(i) + price(j)
        if c < limit:
            graph.add_edge(i, j, weight=limit - c)
            any_edge = True
    if not any_edge:
        return []
    matched = nx.max_weight_matching(graph, maxcardinality=False)
    return [(int(min(a, b)), int(max(a, b))) for a, b in matched]


# ── crossings ──────────────────────────────────────────────────────────────


def junction_costs(graph: SkelGraph, frames: Dict[int, Frame],
                   stubs: Sequence[int], params: Params
                   ) -> Dict[Tuple[int, int], float]:
    costs: Dict[Tuple[int, int], float] = {}
    for ii in range(len(stubs)):
        for jj in range(ii + 1, len(stubs)):
            a, b = stubs[ii], stubs[jj]
            if graph.stubs[a].aid == graph.stubs[b].aid:
                continue  # an arm may not close on itself at one crossing
            lc = link_cost(frames[a], frames[b],
                           w_turn=params.j_w_turn,
                           w_direct=params.j_w_direct,
                           w_len=0.0,
                           chord_floor=params.j_chord_floor,
                           w_kappa=params.j_w_kappa)
            if math.isfinite(lc.cost):
                costs[(a, b)] = lc.cost
    return costs


def best_gap_option(graph: SkelGraph, frames: Dict[int, Frame],
                    params: Params) -> Dict[int, float]:
    """For every stub, the cost of the best gap link it could take instead.

    Crossings are resolved before gaps, so a stub the crossing stage pairs up is
    gone before gap bridging ever sees it -- a filament whose continuation lies
    across a mask gap can be captured by the wrong arm at the crossing and never
    released.  Knowing what each stub is giving up lets the crossing stage price
    its refusal properly instead of charging everyone the same.
    """
    from scipy.spatial import cKDTree

    ids = [s.sid for s in graph.stubs if frames[s.sid].reliable]
    if len(ids) < 2:
        return {}
    tips = np.array([frames[sid].tip for sid in ids])
    tree = cKDTree(tips)
    best: Dict[int, float] = {}
    for ii, jj in tree.query_pairs(params.gap_max_len, output_type="ndarray"):
        a, b = ids[int(ii)], ids[int(jj)]
        sa, sb = graph.stubs[a], graph.stubs[b]
        if sa.aid == sb.aid:
            continue
        if sa.node is not None and sa.node == sb.node:
            continue
        lc = link_cost(frames[a], frames[b], w_turn=params.g_w_turn,
                       w_direct=params.g_w_direct, w_len=params.g_w_len,
                       len_scale=params.g_len_scale,
                       chord_floor=params.g_chord_floor,
                       w_kappa=params.g_w_kappa)
        if not math.isfinite(lc.cost):
            continue
        if lc.theta > params.gap_max_theta:
            continue
        if max(lc.phi_a, lc.phi_b) > params.gap_max_phi:
            continue
        for sid in (a, b):
            if lc.cost < best.get(sid, math.inf):
                best[sid] = lc.cost
    return best


def resolve_junctions(graph: SkelGraph, frames: Dict[int, Frame],
                      params: Params) -> Matching:
    relief = (best_gap_option(graph, frames, params)
              if params.j_gap_relief > 0 else {})
    matching: Matching = {}
    for node in graph.nodes:
        stubs = sorted(node.stubs)
        if len(stubs) < 2:
            continue
        costs = junction_costs(graph, frames, stubs, params)
        if relief:
            prices = {sid: min(params.j_unmatched,
                               params.j_gap_relief * relief[sid])
                      if sid in relief else params.j_unmatched
                      for sid in stubs}
        else:
            prices = params.j_unmatched
        for a, b in _max_weight_pairs(stubs, costs, prices):
            matching[a] = b
            matching[b] = a
    return matching


# ── gaps ───────────────────────────────────────────────────────────────────


def bridge_gaps(graph: SkelGraph, frames: Dict[int, Frame],
                matching: Matching, params: Params) -> Matching:
    """Pair up ends the crossings left free, across empty mask.

    A free end is either a real skeleton endpoint or an arm the crossing logic
    refused to continue.  Both can be one side of a mask gap, so both take part.
    Two ends sitting on the *same* crossing are never bridged -- that decision
    was already made, and re-making it here would just undo it.
    """
    from scipy.spatial import cKDTree

    free = [s.sid for s in graph.stubs if s.sid not in matching]
    free = [sid for sid in free if frames[sid].reliable]
    if len(free) < 2:
        return dict(matching)

    tips = np.array([frames[sid].tip for sid in free])
    tree = cKDTree(tips)
    pairs = tree.query_pairs(params.gap_max_len, output_type="ndarray")

    costs: Dict[Tuple[int, int], float] = {}
    for ii, jj in pairs:
        a, b = free[int(ii)], free[int(jj)]
        sa, sb = graph.stubs[a], graph.stubs[b]
        if sa.aid == sb.aid:
            continue
        if sa.node is not None and sa.node == sb.node:
            continue
        lc = link_cost(frames[a], frames[b],
                       w_turn=params.g_w_turn,
                       w_direct=params.g_w_direct,
                       w_len=params.g_w_len,
                       len_scale=params.g_len_scale,
                       chord_floor=params.g_chord_floor,
                       w_kappa=params.g_w_kappa)
        if not math.isfinite(lc.cost):
            continue
        if lc.theta > params.gap_max_theta:
            continue
        if max(lc.phi_a, lc.phi_b) > params.gap_max_phi:
            continue
        costs[(min(a, b), max(a, b))] = lc.cost

    out = dict(matching)
    for a, b in _max_weight_pairs(free, costs, params.g_unmatched):
        out[a] = b
        out[b] = a
    return out


# ── chains ─────────────────────────────────────────────────────────────────


def matching_objective(graph: SkelGraph, matching: Matching,
                       frames: Dict[int, Frame], params: Params) -> float:
    """Total cost of a whole matching: chosen links, plus a price per free stub.

    The same quantity each per-node matching minimises locally, summed. The
    refinement loop needs it because it is not a contraction: re-estimating the
    frames can lead the next round to a *worse* configuration, and without an
    objective the loop would return whichever round happened to be last. Only
    6 of 20 development scenes reach a fixed point within the round budget, so
    this is the common case, not a corner.
    """
    total = 0.0
    seen: Set[Tuple[int, int]] = set()
    for a, b in matching.items():
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        sa, sb = graph.stubs[a], graph.stubs[b]
        at_node = sa.node is not None and sa.node == sb.node
        if at_node:
            lc = link_cost(frames[a], frames[b], w_turn=params.j_w_turn,
                           w_direct=params.j_w_direct, w_len=0.0,
                           chord_floor=params.j_chord_floor,
                           w_kappa=params.j_w_kappa)
        else:
            lc = link_cost(frames[a], frames[b], w_turn=params.g_w_turn,
                           w_direct=params.g_w_direct, w_len=params.g_w_len,
                           len_scale=params.g_len_scale,
                           chord_floor=params.g_chord_floor,
                           w_kappa=params.g_w_kappa)
        penalty = params.j_unmatched if at_node else params.g_unmatched
        total += (lc.cost if math.isfinite(lc.cost) else 4.0 * penalty) - 2.0 * penalty
    total += sum(params.j_unmatched if s.node is not None else params.g_unmatched
                 for s in graph.stubs)
    return total


def break_cycles(graph: SkelGraph, matching: Matching,
                 costs: Optional[Dict[Tuple[int, int], float]] = None) -> Matching:
    """Filaments are open curves; a matching that closes a ring is wrong.

    A ring shows up as a connected group of arms in which *every* stub is
    matched. The longest jump in the ring is dropped to open it (``costs``, if
    a caller supplies them, overrides that choice; none currently does).

    On the development set this never fires -- 0 links dropped across all 84
    scenes -- so it is a guard against a configuration the data does not
    produce, not a working part of the method.
    """
    parent = list(range(len(graph.arms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = set()
    for a, b in matching.items():
        if a < b:
            links.add((a, b))
    for a, b in sorted(links):
        ra, rb = find(graph.stubs[a].aid), find(graph.stubs[b].aid)
        if ra != rb:
            parent[rb] = ra

    groups: Dict[int, List[Tuple[int, int]]] = {}
    for a, b in links:
        groups.setdefault(find(graph.stubs[a].aid), []).append((a, b))
    members: Dict[int, Set[int]] = {}
    for aid in range(len(graph.arms)):
        members.setdefault(find(aid), set()).add(aid)

    out = dict(matching)
    for root, group_links in groups.items():
        n_arms = len(members[root])
        # a path over n arms has n-1 links; anything more closes a ring
        while len(group_links) >= n_arms:
            def jump(link: Tuple[int, int]) -> float:
                a, b = link
                if costs is not None and (min(a, b), max(a, b)) in costs:
                    return costs[(min(a, b), max(a, b))]
                pa = np.asarray(graph.stubs[a].tip, float)
                pb = np.asarray(graph.stubs[b].tip, float)
                return float(np.hypot(*(pa - pb)))
            worst = max(group_links, key=jump)
            group_links.remove(worst)
            out.pop(worst[0], None)
            out.pop(worst[1], None)
    return out


def chains_from_matching(graph: SkelGraph, matching: Matching) -> List[List[int]]:
    """Group arm ids into chains induced by the stub matching."""
    parent = list(range(len(graph.arms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in matching.items():
        if a < b:
            ra, rb = find(graph.stubs[a].aid), find(graph.stubs[b].aid)
            if ra != rb:
                parent[rb] = ra

    groups: Dict[int, List[int]] = {}
    for aid in range(len(graph.arms)):
        groups.setdefault(find(aid), []).append(aid)
    return [sorted(v) for _, v in sorted(groups.items())]


# ── top level ──────────────────────────────────────────────────────────────


def solve(graph: SkelGraph, params: Params) -> Tuple[Matching, List[List[int]]]:
    """Resolve crossings and bridge gaps, refining the frames as chains grow.

    The two decisions are re-made together on every round rather than once each
    in sequence.  They feed each other: bridging a gap lengthens a chain, which
    sharpens the direction estimate at every crossing that chain still has to
    get through, and resolving a crossing does the same for the gaps.  Round 0
    can only see one arm at a time -- a third to a half of them under 10 px --
    so its estimates are the worst ones the method ever uses.
    """
    from dataclasses import replace

    rounds = max(1, params.n_rounds)
    matching: Matching = {}
    best: Tuple[float, Matching] = (math.inf, {})
    for round_index in range(rounds):
        window = params.window_local if round_index == 0 else params.window_chain
        # Confident first: early rounds only accept joins that are clearly
        # right, so the chains they build are trustworthy enough to measure the
        # harder crossings against.  A tangle resolved from round-0 estimates on
        # 5-px arms is a coin flip; the same tangle resolved from a chain that
        # has already been assembled either side of it is not.
        frac = round_index / max(1, rounds - 1)
        strict = params.anneal_start + (1.0 - params.anneal_start) * frac
        stage = (params if strict >= 1.0 else replace(
            params,
            j_unmatched=params.j_unmatched * strict,
            g_unmatched=params.g_unmatched * strict,
            gap_max_phi=params.gap_max_phi * strict,
            gap_max_theta=params.gap_max_theta * strict,
        ))
        frames = build_frames(graph, matching, window, params)
        junctions = break_cycles(graph, resolve_junctions(graph, frames, stage))
        full = break_cycles(graph, bridge_gaps(graph, frames, junctions, stage))
        # Score the candidate against the frames it was chosen under, and keep
        # the best round rather than the last one.  Only rounds at full
        # strictness compete: an annealed round is deliberately under-linked and
        # would win on an objective that charges for every link it declined.
        if strict >= 1.0:
            value = matching_objective(graph, full, frames, params)
            if value < best[0]:
                best = (value, full)
        if full == matching and strict >= 1.0:
            break
        matching = full
    chosen = best[1] if best[0] < math.inf else matching
    return chosen, chains_from_matching(graph, chosen)

