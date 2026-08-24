# PLECTA 2.5-D depth extension — architecture (Phase 1)

Written 2026-08-18, before any implementation, and **retro-edited in two
places on 2026-08-22**: the data-flow block below and the paragraph under it
were rewritten to describe the shipped evidence rule rather than the designed
one. Both are marked inline. Nowhere else has been touched, so apart from those
marks this is the design as written. The original wording of the edited block
was not kept anywhere and is not reconstructed here — the update at the end of
this document is the authoritative account of what changed and why, and
`sections/07_technical.tex` is the normative published account.

Scope: extend PLECTA from 2-D
instance reconstruction to depth-resolved layered reconstruction of planar
filament instances (one scalar `z_i` per instance, circular cross-section
`r_i = d_i/2`), driven by a registered grayscale image plus the binary axis
mask. The identifiability hierarchy is respected throughout:
2-D identity → crossing order → global ordinal structure → discrete layers →
metric z → 3-D tubes. Nothing claims a level the evidence does not support.

## What exists and is reused unchanged

| existing module | role in the extension |
|---|---|
| `stubmatch/plecta/{graph,geometry,linking,predict}.py` | 2-D reconstruction, untouched. `DEPTH_MODE=off` is literally "do not run the new module". `predict(..., return_internals=True)` already exposes graph/matching/chains — the depth stage consumes those. |
| `stubmatch/plecta/image/measurement.py` | background field, noise sigma, perpendicular FWHM profile fitting with refusal reasons. Reused for diameter estimation and for crossing-level appearance features. |
| `stubmatch/plecta/image/bundles.py` | ordered-centreline extraction from chains (`order_chain`), cut placement away from nodes (`collect_cuts`). Reused; crossing evidence deliberately samples the *complement* (cuts near nodes) via the same machinery. |
| `filaments_quantification/eval/generators/synth_generator.py` | `_smooth_path`, `_stamp_along`, `_densify_path`, `save_gt_multilabel`, degradation model (`degrade_to_mask`) — all imported, not copied. |
| `filaments_quantification/eval/generators/synth_thick.py` | `_smooth_path_uniform`, `place_to_coverage`-style placement, `degrade_axis` (thin masks w1/w2/w3 from one degradation draw). Imported. |
| `filaments_quantification/eval/core/common_metric.py` | unchanged; still the only source of 2-D scores. Also reused to establish the pred↔GT instance correspondence that the depth metrics need (dominant assignment over common fragments). |

## New modules (deliberately few)

1. **`filaments_quantification/eval/generators/synth_depth.py`** — latent-first
   generator. Builds the latent scene (centrelines, per-instance diameter and
   z, crossing list with over/under + contact labels), *then* renders
   observations. One file; imports the two existing generators.
2. **`filaments_quantification/eval/core/depth_metric.py`** — depth scoring:
   crossing-order accuracy/coverage/calibration with abstention, ordinal
   accuracy + Kendall tau on comparable pairs, layer ARI/VI/aligned-error,
   offset-free RMSE_z/MAE, diameter errors, penetration/validity counts,
   3-D centreline and Chamfer surface error. One file.
3. **`stubmatch/plecta/depth.py`** — the optional depth stage (posthoc mode):
   crossing identification between reconstructed instances, interpretable
   local over/under evidence, weighted precedence graph, cycle-resolving
   global ordering (exact for small graphs, greedy fallback), layer inference
   (K inferred, non-conflicting instances share layers), optional metric z
   under an explicit contact/spacing assumption, diameter fusion, tube sweep +
   PLY/OBJ export. `plecta/predict.py` and `plecta/linking.py` are not
   imported *by* it beyond public functions and are not modified.
4. *(Phase 8 only, if evidence supports it)* a `joint` hook: an optional
   `evidence` callable threaded into `linking.junction_costs`; default `None`
   reproduces current behaviour byte-identically. Not implemented until the
   posthoc baseline exists and the ablation says image evidence helps at
   ambiguous junctions (margin `ΔS(v) = S2 − S1` small).

   > **Outcome, recorded 2026-08-23.** This was built —
   > `C:\Repos\stubmatch\plecta\joint.py` and `tests/test_joint.py`, both
   > 2026-08-19 — and measured on 2026-08-20. The ablation said no:
   > **-0.0129 F1 as shipped**, **+0.0005 F1 repaired**, failing its own AUC
   > kill criterion. It is deliberately not shipped. The gate above did its
   > job; the negative result is the finding.

Paper-repo additions: `scripts/figures/fig_depth_*.py` (sharing `_style.py`)
and `scripts/results/make_depth_*.py`, only once results exist. Development
experiments live in `exploration/depth_25d/` (gitignored) per repo convention.

## Latent scene model (generator)

- Centrelines: correlated-curvature walks (`_smooth_path_uniform`), placed to
  a target *thick* areal coverage as in `synth_thick`.
- Diameter: per-instance `d_i` drawn from a scene-level range; optionally a
  slow multiplicative modulation `d_i(s)` (Gaussian-smoothed, bounded ±15 %).
  Constant-`d` is the default first model.
- Crossings: detected on the latent geometry by exact polyline segment
  intersection of centreline pairs (not raster overlap), each with its (x, y),
  local crossing angle, and the pair (i, j).
- z assignment: instances are assigned z sequentially in random order. A new
  instance must satisfy `|z_i − z_j| ≥ r_i + r_j` against every already-placed
  crossing partner. With probability `p_contact`, one such constraint is made
  an equality (`contact`) against an already-placed partner — references only
  earlier instances, so contact constraints form a forest and can always be
  satisfied exactly. A feasibility re-draw (bounded retries) handles dense
  scenes; unplaceable instances are dropped and recorded, never silently.
  After assembly the whole scene is *verified*: every crossing pair re-checked
  against the inequality; violation aborts generation (it is a bug, not data).
- True layers: derived from z by sweeping: instances whose z-intervals
  `[z_i − r_i, z_i + r_i]` can be flattened without violating any crossing
  constraint share a layer; the minimal-K assignment is stored, plus raw z.

## Ground truth stored per scene

`gt_depth.json`: per instance {id, z, layer, d (or d(s) samples), centreline},
per crossing {i, j, xy, angle, over-id, contact flag, latent cue deltas
(Δblur σ, Δbrightness, occlusion opacity)}; generator params + seed.
Rasters: existing outputs unchanged in format — `gt_labels.tif`,
`gt_multilabel.npz`, `mask_clean.png`, `mask_w1/2/3.png`, `sem.png` — plus
`depth_map.tif` (z of topmost instance per pixel, float32, NaN background) and
`instance_depth.png` (visualisation). The pipeline input contract is untouched:
PLECTA still reads only `mask_w*.png`; the depth stage additionally reads
`sem.png`.

## Rendering model (forward, randomised nuisances)

`I = R({C_i, z_i, r_i, α_i}, Ψ) + ε`. Instances are composited in z order
(bottom first). Cues, each independently randomised so no single deterministic
rule (e.g. higher = brighter) identifies depth:

- defocus: per-instance PSF σ from `|z_i − z_focus|` with randomised focal
  plane (may sit above, inside, or below the stack) and DoF scale;
- occlusion: the upper instance overwrites the lower within its silhouette
  with randomised opacity (1.0 = hard SEM-like occlusion, <1 = translucent);
- intrinsic brightness `α_i` independent of z; background field, grain,
  contrast as in `make_sem`;
- observed width = physical `d_i` convolved with the instance's PSF — so
  `w_obs ≠ d_physical` by construction and diameter recovery must invert a
  calibrated forward model.

Rendering domains (presets, train/config on some, test on others):
`ideal` (no blur, no noise, hard occlusion), `microscopy` (strong defocus
cue), `semlike` (weak defocus, hard occlusion, grain), `shifted`
(out-of-distribution: inverted contrast, brightness *correlated* with z to
bait spurious cues, altered PSF law). Deliberately unidentifiable crossings
(matched d, α, blur, symmetric geometry) are generated at a controlled rate
and flagged in GT; the correct output there is abstention.

## Depth stage (posthoc) data flow

```
mask ──predict()──▶ chains/instances ──┐
                                       ├─▶ crossings between instances (graph nodes
sem ──load_sem()──▶ background field ──┘    traversed by ≥2 chains + gap-adjacent overlaps)
     ─▶ per-crossing evidence: ONE channel, flank-vs-core median intensity,
        normalised by the crossing's own noise   [RETRO-EDITED 2026-08-22:
        these three lines describe the SHIPPED rule, not the Phase-1 design,
        which was a multi-feature probability p_c with an abstention band on p.
        See the update at the end of this document.]
        ─▶ score s; sign(s) = who is on top, |s| = weight, abstain |s| < 0.40
     ─▶ weighted precedence graph ─▶ max-weight acyclic subgraph (exact ILP small,
        greedy FAS fallback; corrections recorded) ─▶ topological layers, minimal K,
        non-conflicting instances share a layer
     ─▶ optional metric z: least squares on layer spacing with |Δz| ≥ r_i + r_j
        (equality only under an explicit, ablated contact assumption); min z = 0
     ─▶ diameter: FWHM cuts (existing) + PSF deconvolution calibrated per domain
     ─▶ tubes: circular sweep along (x_i(s), y_i(s), z_i); PLY/OBJ export
```

Both raw local scores and globally corrected relations are stored; the
difference is itself an analysis result. (The stored primitive is the score
`s`; the probability `p` in the record is a monotone relabelling of it, kept
for schema compatibility, and is not the decision quantity. RETRO-EDITED
2026-08-22: this paragraph said "probabilities", which was the design's
primitive, not the shipped one.)

## Datasets and seeds

Seed series already burned (never reuse): 20260720, 20270101, 20280101,
20290101, 20310101, 20330101, 20350101. New series:

- `20400101…` depth development (tune here only);
- `20410101…` depth held-out (touch only for final numbers);
- `20420101…` depth locked (untouched until the manuscript freeze).

`input/synthetic_locked_v1` is never used for any depth work. Datasets are
reproducible from `gt_meta.json`/manifest; failed placements, abstentions,
cycle corrections, optimisation status and runtime are all recorded.

## Invariants that must not move

- `DEPTH_MODE=off` ≡ current PLECTA, byte-identical outputs (guarded by the
  existing core-contract tests plus a new parity test).
- 2-D scores keep coming from `common_metric.py` only; depth metrics are
  reported separately, never collapsed into one composite number.
- The frozen `software/plecta` snapshot in the paper repo is not touched.
- No hand-written numbers in prose; every table/figure from scripts.
- Preserve `\Igor{}` / `\igorreply{}` byte-identical.
- Manuscript edits wait until figures/validations exist (author request,
  2026-08-18).

## Phase order (each validated before the next)

1. this document;
2. generator + sanity figures + interpenetration proof;
3. oracle depth reconstruction (GT 2-D instances → order/layers/z);
4. PLECTA-predicted instances → depth (quantify loss vs oracle);
5. image evidence features/predictor;
6. global ordering incl. adversarial contradictory graphs;
7. diameter + tubes;
8. joint mode (only if ablation justifies) — built 2026-08-19, measured
   2026-08-20, negative (-0.0129 F1 as shipped, +0.0005 repaired, failed its
   AUC kill criterion), not shipped;
9. experiments + figures; 10. manuscript.

## Update, 2026-08-22 — the per-crossing evidence rule was replaced

> **Normative account: `sections/07_technical.tex`.** The rule change is
> written out in four places — that section, this note,
> `docs/internal/DEPTH_ARCHITECTURE.md` and
> `exploration/depth_order_eval/MEMO.md` — each carrying the same ~20
> measured constants. They agree as of 2026-08-23. If they ever disagree, the
> published section wins, and its numbers come from
> `results/plecta_depth_rule.tex`, generated by
> `scripts/results/make_depth_rule_record.py`: correct them there, not here.
> What is unique to this copy is what it is worth keeping for.

This document is the Phase-1 design, written before implementation, apart
from the two blocks marked RETRO-EDITED above. The
per-crossing evidence it envisaged (a multi-feature probability `p_c` with an
abstention band) is **not** what shipped. What ships is one channel, scored and
thresholded directly:

    s      = 2 * F_inf
    F_inf  = (|m_c - m_j| - |m_c - m_i|) / max(|m_i - m_j|, 2 * sigma_hat)

`m_i`, `m_j` are the two filaments' flank median intensities, `m_c` the pooled
core median, and `sigma_hat = 1.4826 * MAD` of the within-rod flank residuals —
a per-crossing *noise* estimate (each filament's flank samples centred on their
own median, the two residual sets pooled). `sign(s)` names the upper filament,
`|s|` is the edge weight in the global ordering, and the crossing abstains when
`|s| < 0.40`: a threshold in score units, not a probability band. There is no
gradient image, no sharpness channel, no logistic and no channel weights. The
core arc is a flat 7.5 px rather than sized from the two radii, so **no per-rod
radius enters the evidence path at all** — radii re-enter only downstream, in
the non-interpenetration constraints on metric z. A probability `p` is still
written to the per-crossing output so the schema is unchanged, but it is a
monotone relabelling of `s`, not a calibrated confidence and not the decision
quantity.

Two measured defects forced the change:

1. The old intensity feature divided by `max(|m_i - m_j|, 0.02)`. Because the
   numerator is bounded by `|m_i - m_j|` (reverse triangle inequality), the
   feature reached ±1 for *any* separation above 0.02 grey levels — far below
   the image noise — and `|2*F| >= 1.98` then cleared any abstention threshold.
   The rule was structurally unable to abstain on such crossings. 32.3 % of
   synthetic and 61.5 % of real crossings were saturated this way; the saturated
   ones whose separation fell below the noise scored ~0.70 accuracy against
   ~0.93 for those above it.
2. The sharpness channel contributed nothing measurable: it decided 0.2 % of
   real crossings, agreed with intensity at chance (kappa ~ +0.09), and scored
   0.54–0.59 standalone. Its centreline ± r samples also fall inside the *other*
   filament's stroke within the core, so it was contaminated exactly where it
   was needed.

Held-out evidence (20 optically rendered synthetic scenes, 1355 crossings,
seeds disjoint from every set used to develop either rule and to choose
between them; the constants are a weaker claim, since core_px = 7.5 was
settled by a sweep on the clean set and the abstention threshold's flatness
was checked there, so the set is not held out with respect to those two
values -- neither was invented there (7.5 px is exactly what the previous
rule's radius-free fallback already computed) and both sit on a plateau): same
accuracy at the same coverage (+0.0025, 95 % CI [-0.0042, +0.0079]) with
better-ordered confidence (AUC of correctness against `|s|` 0.7772 -> 0.8698,
+0.0926, CI [+0.0563, +0.1321]); +0.0188 accuracy at 82.5 % coverage and
+0.0240 at 80 %. The claim is *the same accuracy with better-ordered confidence
from a simpler evidence path* — **not** a more accurate rule. A second, axial
continuity channel was built and measured inert (-0.0008, CI [-0.0059,
+0.0048]) and is deliberately excluded.

Two design invariants above are unaffected and still hold: raw local relations
and globally corrected ones are both stored, and no composite depth score
exists. The "Phase 5: image evidence features/predictor" milestone should now be
read as delivered by this one-channel rule rather than by a learned predictor.
