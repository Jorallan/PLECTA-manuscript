# Authoritative stage-and-gate audit

Status: internal engineering specification, audited 2026-08-10.

> **Historical audit — its citations do not resolve against the current
> checkout.** Added 2026-08-23. This document audited `C:\Repos\stubmatch` as
> it stood on 2026-08-10, before the package was renamed and restructured.
> Every path and line number below refers to that tree:
>
> | cited then | now |
> |---|---|
> | `frozen/predict.py` | `plecta/predict.py` |
> | `frozen/indep/linker.py` | `plecta/linking.py` |
> | `frozen/indep/skelgraph.py` | `plecta/graph.py` |
> | `frozen/indep/geometry.py` | `plecta/geometry.py` |
> | `frozen/indep/predictor.py` | folded into `plecta/predict.py` |
> | `frozen/params.json` | `plecta/params.json` (and `plecta/parameters.yaml`) |
> | `semwidth/*` | `plecta/image/` |
> | `mergegate/*` | removed |
>
> Line ranges are not remapped: the files were rewritten, not merely moved, so
> a remapped range would look verified when it is not. `SPLIT.md` is now 22
> lines, so every range cited against it here lies partly or wholly past its
> end. Re-derive any citation you need to rely on.
>
> Two of the conditions this audit reports on have since changed, and the text
> below is left as written rather than edited in place:
>
> * `C:\Repos\stubmatch` **is** a Git repository now (HEAD `7d369d8`), so the
>   statements that it carries no Git metadata and offers no verifiable commit
>   are no longer true.
> * The effective configuration is now materialised in one file,
>   `plecta/parameters.yaml`, with `plecta/params.json` retained as the frozen
>   cross-check, and `evidence/` carries the seed manifest, the per-scene
>   held-out record, the release parity record and the tested environment.
>
> What remains valid is the reasoning, not the coordinates.

The code and effective frozen configuration take precedence over prose. The
working implementation is in `C:\Repos\stubmatch`. At the time of this audit
that directory carried no Git metadata, so the audit cites executable files and
stored result artifacts rather than a commit identifier; it is a Git repository
now, and a present-day audit should cite the commit.

## Publication-safe specification

The evaluated frozen method skeletonizes and prunes a binary mask, decomposes it
into arms and merged crossing nodes, alternates exact per-node continuation
matching with globally gated gap matching while re-estimating chain-supported
geometry, and paints the resulting chains as overlapping instance layers. SEM
width fitting and ribbon rendering are separate optional characterization
stages. Stage-4 silhouette absorption and the later continuation union are
default-off post-hoc grouping changes and were absent from the 128-scene
held-out result.

## Authoritative execution path

1. **Entry point and configuration.** `frozen/predict.py:32-59` constructs the
   linker's `Params` object from `frozen/params.json`; omitted JSON keys retain
   the dataclass defaults in `frozen/indep/linker.py:31-72`. The CLI loads a mask,
   calls `predict`, and writes an overlap-aware sparse multilabel NPZ
   (`frozen/predict.py:83-108`).
2. **Graph construction.** `frozen/indep/predictor.py:56-66` calls
   `frozen/indep/skelgraph.py::build_graph`. The mask is binarized at `>0`,
   skeletonized, and pruned of short endpoint hairs (`skelgraph.py:42-105,
   277-282`). Degree-3-or-higher pixels define junction regions. Nearby regions
   are merged; short connecting arms can be absorbed or retained as real arms
   while their junctions are unified (`skelgraph.py:281-358`). Remaining ordered
   components become arms with two stubs (`skelgraph.py:365-405`).
3. **Stub evidence.** Each stub obtains an outward frame containing tip,
   tangent, and signed curvature. Polynomial fitting is performed in arclength;
   a frame is reliable only when it spans at least three pixels with at least
   three samples (`frozen/indep/geometry.py:39-92`). Pair cost combines reversed
   tangent disagreement, the two turns onto the tip-to-tip chord, optional
   length, and curvature continuity (`geometry.py:111-164`).
4. **Eight coupled refinement rounds.** Round 0 fits frames from a local arm
   window; later rounds walk inward through the current matching to use assembled
   chain evidence (`linker.py:78-119,425-456`). Each round resolves crossings
   first by exact per-node partial matching, removes cycles, globally bridges
   eligible free ends across gaps, and removes cycles again
   (`linker.py:125-160,166-183,228-301,345-397,425-470`).
5. **Instances.** Connected arms in the final accepted matching form chains
   (`linker.py:400-419`). Output painting assigns arm pixels and every touched
   junction region to each chain, so crossing pixels can belong to multiple
   instances. Isolated one-arm/no-node pieces shorter than six pixels are
   omitted (`predictor.py:13-53`). Stage 1 does not draw pixels across inferred
   gaps.

## Complete effective frozen configuration

The evaluated configuration is the union of explicit JSON values and dataclass
defaults. Values marked “default” are not present in `frozen/params.json`.

| Component | Parameter | Value |
|---|---|---:|
| graph | `spur_px` | 3 |
| graph | `bridge_px` | 5 (default) |
| graph | `absorb_free_px` | 2 |
| graph | `join_px` | 14 |
| frames | `window_local` | 24 px |
| frames | `window_chain` | 55 px |
| frames | `min_quadratic` | 18 |
| refinement | `n_rounds` | 8 |
| refinement | `anneal_start` | 0.85 |
| crossing | `j_w_direct` | 0.6 |
| crossing | `j_w_turn` | 1.2 |
| crossing | `j_chord_floor` | 4 px (default) |
| crossing | `j_w_kappa` | 1.0 |
| crossing | `j_unmatched` | 0.62 |
| crossing | `j_gap_relief` | 0 (default; off) |
| gaps | `gap_max_len` | 85 px |
| gaps | `g_w_direct` | 0.2 |
| gaps | `g_w_turn` | 1.0 (default) |
| gaps | `g_w_len` | 0.15 |
| gaps | `g_len_scale` | 60 px (default) |
| gaps | `g_chord_floor` | 3 px (default) |
| gaps | `g_w_kappa` | 0.5 |
| gaps | `g_unmatched` | 0.3 |
| gaps | `gap_max_theta` | 0.4 rad |
| gaps | `gap_max_phi` | 0.28 rad |

Sources: `frozen/params.json:1-20` and
`frozen/indep/linker.py:31-72`. A publication/release snapshot must materialize
all defaults in one complete file; the current JSON alone is not a complete
freeze.

## Gate order

### Before candidate matching

1. Binarize and skeletonize the mask.
2. Iteratively prune endpoint hairs of at most 3 px.
3. Identify and cluster junction pixels.
4. Absorb crossing connectors of at most 5 px and unify their nodes.
5. Unify nodes connected by arms of at most 14 px while retaining those arms.
6. Absorb free-ended debris of at most 2 px.

These are topology-conditioning gates, not matching-cost terms.

### Crossing candidates

1. Both stubs must belong to the same merged node.
2. A stub cannot pair with the other end of its own arm.
3. The geometric frame and pair cost must be finite.
4. Tangent, chord-turn, and curvature evidence enter the cost.
5. Exact Blossom matching considers an edge only when its cost is below the
   combined unmatched price. With the frozen settings this is `cost < 1.24`.
6. Leaving a stub unmatched remains a valid decision.

There is no separate hard distance or angle threshold at a crossing.

### Gap candidates

1. Only stubs left unmatched by the crossing step are considered.
2. Both frames must be reliable.
3. Tip distance must be no more than 85 px.
4. The stubs must belong to different arms and must not occupy the same node.
5. Reversed tangent disagreement must be at most 0.4 rad.
6. Each turn onto the chord must be at most 0.28 rad.
7. Tangent, chord-turn, length, and curvature contribute to cost.
8. Exact global matching considers an edge only when `cost < 0.6`.

### After a provisional link

- Cycle breaking runs after crossing matching and after gap matching. It removes
  the longest tip jump in a cycle; documentation states it fired zero times in
  the 84 development scenes (`linker.py:345-397`, `METHOD.md:633-636`).
- The round objective is evaluated only for rounds at full strictness
  (`linker.py:459-470`). Under the frozen linear schedule, only the eighth round
  reaches strictness 1.0, so the frozen method returns the last round rather than
  selecting among several full-strictness rounds.
- The output filter removes short isolated arms only after chains are built.

## Evidence used by component

| Component | Mask topology | Tangent/chord | Gap length | Curvature | Chain history | SEM intensity | Width/brightness |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph construction | yes | no | local connector length | no | no | no | no |
| crossing matching | node membership | yes | no | yes | yes after round 0 | no | no |
| gap matching | free-end topology | yes | yes | yes | yes after round 0 | no | no |
| Stage 2 measurement | axis/node clearance | no | no | no | grouping fixed | yes | produces both |
| Stage 4 rendering | chain topology | centreline geometry | draws inferred gaps | no | fixed chain | indirectly | width used; brightness not gated |
| external continuation union | no | yes | tip distance | no | final polylines | no | no |
| external duplicate union | centreline coincidence | no | coincident length | no | final polylines | no | no |

## Stage 2 and Stage 4

The full-stack entry point is `semwidth/predict_semwidth.py`. It re-runs the
frozen grouping core, then samples perpendicular SEM profiles and attaches
width, brightness, spread, and uncertainty to each unchanged chain
(`semwidth/pipeline.py:73-128`; `semwidth/bundles.py:436-574`). Stage 2 cannot
change instance membership.

Stage 4 is opt-in through `--refine`, not a default stage. It orders existing
chains, interpolates missing gap segments, resamples, smooths with pinned
endpoints, and strokes a ribbon at the fitted width
(`semwidth/predict_semwidth.py:57-64,115-130`;
`semwidth/refine.py:52-178,253-305`). Default `absorb_thr=0`, so its default form
changes the rendered pixels but not the grouping.

### Stage-4 silhouette absorption

When explicitly enabled, Stage 4 merges two rendered instances if their
intersection divided by the smaller area exceeds the threshold and covers at
least 40 pixels. It operates largest-first and unions masks
(`semwidth/refine.py:181-217,305`). It can reverse a Stage-1 decision to keep two
chains separate, but cannot split or re-pair an incorrect chain. The executable
default is 0/off. A threshold of 0.6 appears in experimental prose but was not
used for the frozen held-out score.

## Separate post-hoc merge module

`mergegate/` is not imported by the full-stack entry point. Its default mode is
off (`mergegate/gates.py:228-265`). If enabled, it computes polyline-pair
features and applies selected instance-ID pairs as transitive unions on Stage-1
or Stage-4 masks (`mergegate/run.py:472-482`). It cannot undo an incorrect merge.

- **Continuation gate:** tangent window 12 px; tip distance at most 12 px;
  opposition at least 0.93; same-sign forward magnitude at least 0.80; line
  residual at most 6 px. An overrun case is tightened to distance 8 px, forward
  0.85, and residual 3 px (`mergegate/gates.py:244-289`).
- **Duplicate gate:** 100% of the smaller centreline within 4 px, at least 20 px
  coincident arclength, and each polyline at least 15 px
  (`mergegate/gates.py:258-295`). Continuation takes precedence.

On 84 development scenes, continuation after Stage 4 with silhouette absorption
0.6 changed mean F1 by -0.0009 on clean masks, +0.0002 after one dilation, and
+0.0105 after two dilations. The duplicate gate applied zero merges because the
Stage-4 absorption had already consumed those IDs
(`mergegate/results/sweep_dev84_stage4.json`). The one real development crop
improved by +0.0223. No official held-out run included this module. These data
support describing it as experimental and default-off, not as part of the
evaluated method.

## Absent, disabled, removed, or rejected components

- `j_gap_relief` exists but is zero/off; its development experiment was rejected.
- The SEM-based crossing-decision stage was removed after negative experiments;
  `semwidth/arms.py` remains unused (`semwidth/PRUNED_STAGE3.md`).
- Adaptive mask-thickness scaling of `window_chain` is external and not adopted.
- Thick-mask pre-smoothing and SEM-orientation conditioning were rejected or
  remain experiment-only.
- Stage-4 taper and clip-to-mask options are off; taper was negative.
- Stage-4 silhouette absorption and the external continuation/duplicate unions
  are default-off and absent from the 128-scene held-out evaluation.

## Inconsistencies requiring correction

1. The frozen “best-round” claim is not operational under the frozen annealing
   schedule because only the last round is fully strict.
2. The general round objective uses crossing unmatched prices for node stubs even
   when the selected link is a gap; this is dormant in the frozen schedule but
   invalidates the broader same-objective claim.
3. `frozen/predict.py --mask` bypasses the locked-path guard that protects the
   `--scenes` path.
4. `params.json` omits effective defaults, and a missing JSON silently falls back
   to all dataclass defaults.
5. Documentation recommends Stage 4 while the example command omits `--refine`.
6. Claims that Stage 4 never changes grouping are false when absorption is on.
7. Threshold 0.6 is alternately called adopted, experimental, and non-default;
   executable authority says default off.
8. `mergegate/results/real.json` records top-level `mode="off"` even for rows
   produced by enabled variants.
9. No Git history exists for `C:\Repos\stubmatch`, so claims of pre-committed
   prediction thresholds cannot be independently verified.

No rejected, default-off, or post-freeze component should be described as part
of the evaluated frozen algorithm.
