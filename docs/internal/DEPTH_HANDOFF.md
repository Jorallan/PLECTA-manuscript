# PLECTA 2.5-D depth extension — technical handoff

Written 2026-08-18. Companion to `DEPTH_ARCHITECTURE.md` (the design, written
before implementation). This file records what was actually built, where it
lives, what the development numbers are, and what remains open. Manuscript
text has deliberately NOT been touched (author request: figures and
validations first).

## What was built, file by file

### filaments_quantification

| file | role |
|---|---|
| `eval/generators/synth_depth.py` | latent-first 2.5-D generator: planar rods with one z each, circular cross-section, exact non-interpenetration (`verify_scene` aborts generation on any violation), contact crossings satisfy the bound with equality, per-crossing over/under + cue-delta ground truth, randomised renderer with four domains (`ideal`, `microscopy`, `semlike`, `shifted`), thin masks w1/2/3 + clean axis through the *existing* degradation code. |
| `eval/core/depth_metric.py` | depth scoring, one identifiability level at a time: crossing order (+abstention/calibration), ordinal (direct/transitive/Kendall), layers (ARI/VI/shift-aligned error), offset-free metric z, diameter, tube-surface Chamfer/Hausdorff, physical validity. `match_instances` maps predicted→GT instances through the common fragments (same units as every 2-D score). |
| `eval/runners/run_depth.py` | one command per condition: runs the depth stage over a scene set, scores it, writes `depth_scores.json` (per-scene + aggregate + full condition record). |
| `tests/generators/test_synth_depth.py` | 8 tests: crossing detection (incl. double crossing), contact exactness, interpenetration proof, minimal-K layering, determinism. |
| `tests/core/test_depth_metric.py` | 15 hand-checkable metric tests. |

New datasets (seed series disjoint from every earlier series):

| set | seeds | contents | status |
|---|---|---|---|
| `input/synthetic_depth_dev` | 20400101+ | 12 scenes, microscopy, cov 20/35 | tuning allowed |
| `input/synthetic_depth_heldout` | 20410101+ | 20 scenes, microscopy | final numbers only |
| `input/synthetic_depth_heldout_shifted` | 20415001+ | 20 scenes, shifted rendering (brightness↔z bait, altered blur law, contrast inversion) | final numbers only |
| (reserved) | 20420101+ | locked set — NOT generated yet, on purpose | untouched |

### stubmatch

| file | role |
|---|---|
| `plecta/depth.py` | the optional post-hoc stage (`DEPTH_MODE=posthoc` ≡ run this after `predict`; `off` ≡ don't). Crossing identification (same segment-intersection construction as the generator), occlusion-continuity evidence (ONE channel, noise-floored flank-vs-core median intensity; score `s`, abstain `|s| < 0.40`)[^oldrule], global order per component (exact subset-DP ≤14 nodes, greedy insertion + adjacent swaps above), raw AND corrected relations stored, minimal-K layers from the DAG, metric z under an explicit compact-stack prior (SLSQP + exact bottom-up repair sweep ⇒ zero interpenetration by construction), FWHM diameter via the existing `plecta.image` cut machinery + dev-calibrated width→diameter correction, tube meshes + PLY/OBJ export. CLI: `python -m plecta.depth --scene … --out … [--oracle] [--oracle-radii] [--tubes]`. |
| `tests/test_depth.py` | 10 tests: consistent chains, the A>B>C>A contradictory cycle (weakest relation flipped, raw kept), exact-vs-greedy agreement, layers, compact stack touching + clearing, abstained pairs still clearing, evidence on synthetic occlusion images, unidentifiable crossing abstains. |

[^oldrule]: That is the rule shipped from 2026-08-22. Every number in the
    tables below was measured under the previous one, an intensity +
    edge-sharpness pair combined through a logistic; see the 2026-08-22 note at
    the end of this document, and `sections/07_technical.tex` for the normative
    account.

`plecta/{graph,geometry,linking,predict}.py` are untouched;
`tests.test_core_contract` still passes (checked 2026-08-18). The paper
repo's frozen `software/plecta` snapshot is untouched.

### paper repo

| file | role |
|---|---|
| `docs/internal/DEPTH_ARCHITECTURE.md` | Phase-1 design document |
| `scripts/figures/fig_depth_scene.py` | generator sanity figure (archive) |
| `scripts/figures/fig_depth_reconstruction.py` | 3-D truth vs reconstruction (archive) |
| `scripts/figures/fig_depth_results.py` | ablation/domain-shift headline figure (archive) |
| `scripts/results/run_depth_ablations.py` | the ablation grid driver |
| `exploration/depth_25d/` | run outputs + `depth_ablation_record.json` (gitignored study area) |

All figures write to `figures/archive/` because no `\includegraphics`
reaches them yet, per repository rule.

## Dev-set calibration constants (all frozen before held-out was touched)

- evidence combination: `w_intensity=2.446`, `w_sharpness=1.576` — logistic
  fit, 830 matched crossings, monotone calibration 0.59→0.98 across
  confidence bins on dev; `abstain_band=0.15`.
  **Superseded 2026-08-22:** the logistic, both channel weights and the
  probability band are gone. The shipped rule is one channel with a noise
  floor and `abstain |s| < 0.40` in score units. See the note at the end.
- width→diameter: `d = 1.176·w_obs + 2.559` (340 oracle-centreline
  instances; residual RMS 1.44 px; residual bias −0.93 px at σ>2 px).
  This is DOMAIN calibration and expected to degrade under `shifted`.

## Development results (12 dev scenes, oracle 2-D instances, measured d)

order accuracy 0.835 at decision rate 0.86 (ambiguous crossings abstain at
~5× the rate of ordinary ones), ordinal direct 0.81, Kendall tau 0.40,
layer ARI 0.43, span-normalised RMSE_z 0.26 under the compact-stack prior,
diameter MAE 1.07 px (9.5 % relative), interpenetration exactly zero.
Predicted 2-D instances lose ~2 points of order accuracy and ~21 % of GT
crossings to 2-D identity/coverage errors.

## Held-out results (20 scenes per set, plain means, no inference)

From `exploration/depth_25d/depth_ablation_record.json` (2026-08-18):

| set / condition | order acc | decide | tau | layer ARI | RMSE_z/span | invalid | d MAE px |
|---|---|---|---|---|---|---|---|
| in-domain, oracle 2-D + oracle d | 0.855 | 0.866 | 0.366 | 0.346 | 0.277 | 0.000 | — |
| in-domain, oracle 2-D + measured d | 0.859 | 0.866 | 0.384 | 0.336 | 0.277 | 0.000 | 1.27 |
| in-domain, predicted 2-D + measured d | 0.849 | 0.862 | 0.309 | 0.272 | 0.275 | 0.110 | 1.32 |
| in-domain, intensity channel only | 0.851 | 0.865 | 0.295 | 0.275 | 0.278 | 0.105 | 1.32 |
| in-domain, sharpness channel only | (1.0) | 0.007 | 0.033 | 0.004 | 0.369 | 0.118 | 1.32 |
| shifted, oracle 2-D + oracle d | 0.809 | 0.835 | 0.365 | 0.442 | 0.291 | 0.000 | — |
| shifted, oracle 2-D + measured d | 0.801 | 0.841 | 0.368 | 0.474 | 0.275 | 0.185 | 1.29 |
| shifted, predicted 2-D + measured d | 0.783 | 0.851 | 0.309 | 0.336 | 0.300 | 0.235 | 1.36 |

Readings: (i) the sharpness channel alone almost never clears the abstention
band (decides 0.7 % of crossings) — the occlusion-intensity cue carries the
method; (ii) the shifted rendering (brightness-vs-z bait, altered blur law,
contrast inversion) costs ~5–7 points of order accuracy but stays far above
chance, and the abstention machinery holds; (iii) the nonzero "invalid"
fractions occur exactly where the *scorer* uses true radii while the solver
had only estimated or default radii — with oracle radii the reconstruction
is exactly interpenetration-free everywhere, by construction; (iv) predicted
2-D instances recover ~79 % of GT crossings (the rest are lost to mask
degradation and 2-D grouping errors), so `coverage_matched`, not order
accuracy, is where 2-D quality bites hardest.

## Deliberate scientific choices

- The identifiability hierarchy is enforced in the outputs: crossing order,
  layers and metric z are separate blocks; metric z is labelled with its
  compact-stack assumption in `pred_depth.json` itself.
- Ambiguous-by-construction crossings must be *abstained*, and abstention on
  them is scored as its own virtue, never blended into accuracy.
- Raw local relations and globally corrected ones are both stored; flips and
  raw cycle counts are part of every output (analysis material, not noise).
- The generator refuses to create an invalid scene (hard assertion), and the
  reconstruction cannot emit one (repair sweep is exact on the DAG).
- No composite "depth score" exists anywhere.

## Known limitations / open items

1. **Joint mode was built, measured, and failed — it is deliberately not
   shipped.** `DEPTH_MODE=joint` (image evidence entering
   `linking.junction_costs` at low-margin junctions) is implemented in
   `C:\Repos\stubmatch\plecta\joint.py` with `tests/test_joint.py` beside
   it, both dated 2026-08-19, and it was measured on 2026-08-20: **-0.0129 F1
   as shipped** and **+0.0005 F1 repaired**, and it fails its own AUC kill
   criterion. The geometry-only special case therefore remains the only 2-D
   path. Do not claim any 2-D improvement from image evidence — not because
   the experiment is outstanding, but because it was run and came back
   negative. (Corrected 2026-08-23; this item previously read "not
   implemented ... has not been started", which was wrong.)
2. Evidence features are V1: occlusion continuity only. Blur magnitude is
   deliberately unused for *order* (it is symmetric about the focal plane);
   it could sharpen metric z via |z−z_focus| but only with the two-fold
   side ambiguity resolved.
3. Calibration ECE per scene is ~0.23 (small per-scene bins); pooled
   calibration is monotone. A held-out reliability figure should pool.
4. Diameter correction is linear in w_obs, per domain; no per-instance blur
   deconvolution. Under `shifted` rendering the diameter numbers are
   expected to (and do) degrade — that is a reported result, not a bug.
5. `d_vary` (slowly varying diameter) defaults to 0 in the generated sets;
   the generator supports it but no experiment uses it yet.
6. Real-data depth validation (spacer-height monofilaments etc., task 15)
   is untouched; nothing here should be claimed for real SEM depth.
7. Scale normalisation of the 2-D core's pixel constants (task 14) was not
   attempted — explicitly deferred so the validated core stays stable.
8. The locked depth set (seed 20420101+) is intentionally not generated
   until the method is final.

## How to reproduce everything

```powershell
# datasets
C:\Repos\venv_cnt\Scripts\python.exe C:\Repos\filaments_quantification\eval\generators\synth_depth.py --out C:\Repos\filaments_quantification\input\synthetic_depth_dev --n 6 --coverages 0.20,0.35 --domain microscopy --seed0 20400101 --preview
# unit tests
cd C:\Repos\filaments_quantification; C:\Repos\venv_cnt\Scripts\python.exe -m unittest tests.generators.test_synth_depth tests.core.test_depth_metric
cd C:\Repos\stubmatch;               C:\Repos\venv_cnt\Scripts\python.exe -m unittest tests.test_depth tests.test_core_contract
# ablation grid + figures
C:\Repos\venv_cnt\Scripts\python.exe scripts\results\run_depth_ablations.py
C:\Repos\venv_cnt\Scripts\python.exe scripts\figures\fig_depth_scene.py
C:\Repos\venv_cnt\Scripts\python.exe scripts\figures\fig_depth_reconstruction.py
C:\Repos\venv_cnt\Scripts\python.exe scripts\figures\fig_depth_results.py
```

## Note, 2026-08-22 — the per-crossing evidence rule was replaced

> **Normative account: `sections/07_technical.tex`.** The rule change is
> written out in four places — that section, this note,
> `docs/internal/DEPTH_ARCHITECTURE.md` and
> `exploration/depth_order_eval/MEMO.md` — each carrying the same ~20
> measured constants. They agree as of 2026-08-23. If they ever disagree, the
> published section wins, and its numbers come from
> `results/plecta_depth_rule.tex`, generated by
> `scripts/results/make_depth_rule_record.py`: correct them there, not here.
> What is unique to this copy is what it is worth keeping for.

Everything measured above — the dev results, the held-out table, and reading
(i) in particular — was produced under the **previous** two-channel rule. The
rule has since been replaced on measured evidence, so the numbers above are a
record of what that version did, not of what the code does now.

**Shipped rule.** One channel, no gradient image, no per-rod radius in the
evidence path:

    s      = 2 * F_inf
    F_inf  = (|m_c - m_j| - |m_c - m_i|) / max(|m_i - m_j|, 2 * sigma_hat)

`m_i`, `m_j` are the two filaments' flank median intensities, `m_c` the pooled
core median, `sigma_hat = 1.4826 * MAD` of the within-rod flank residuals (each
filament's flank samples centred on their own median, the two residual sets
pooled) — a per-crossing noise estimate. `sign(s)` decides, `|s|` is the weight
carried into the global ordering, and the crossing abstains when `|s| < 0.40`,
a threshold in score units rather than a band around a probability. The core arc
is a flat 7.5 px, not sized from the two radii. A probability `p` is still
written to the output for schema compatibility, but it is a monotone
relabelling of `s` — not a calibrated confidence, and not the decision quantity.

**Why.** Two measured defects of the old rule:

1. It divided by `max(|m_i - m_j|, 0.02)`. The numerator is bounded by
   `|m_i - m_j|` (reverse triangle inequality), so the feature reached ±1 for
   *any* separation above 0.02 grey levels, far below the image noise, and
   `|2*F| >= 1.98` then cleared any abstention threshold — the rule was
   structurally unable to abstain on such crossings. 32.3 % of synthetic and
   61.5 % of real crossings were saturated this way, and the saturated crossings
   whose separation fell below the noise scored ~0.70 accuracy against ~0.93 for
   those above it. `2*sigma_hat` makes "confident" mean "separated by more than
   the noise".
2. The sharpness channel measured at chance. It decided 0.2 % of real crossings,
   agreed with intensity at chance (kappa ~ +0.09), and scored 0.54–0.59
   standalone — which is the same finding as reading (i) above, now acted on.
   Its centreline ± r samples also fall inside the *other* filament's stroke
   within the core, so the quantity was contaminated exactly where it was
   needed. Removed, with the gradient image, the two channel weights and the
   logistic.

**Held-out evidence for the replacement** (20 optically rendered synthetic
scenes, 1355 crossings, seeds disjoint from every set used to develop either rule and to choose
between them; the constants are a weaker claim, since core_px = 7.5 was
settled by a sweep on the clean set and the abstention threshold's flatness
was checked there, so the set is not held out with respect to those two
values -- neither was invented there (7.5 px is exactly what the previous
rule's radius-free fallback already computed) and both sit on a plateau): same accuracy at the same coverage, difference +0.0025, 95 % CI
[-0.0042, +0.0079]; better-ordered confidence, AUC of correctness against `|s|`
rising 0.7772 -> 0.8698 (+0.0926, CI [+0.0563, +0.1321]); +0.0188 accuracy at
82.5 % coverage and +0.0240 at 80 %. Because `|s|` is the edge weight,
better-ordered confidence means contradictions are broken at the genuinely
weakest relations. **The claim is the same accuracy with better-ordered
confidence from a simpler evidence path — not a more accurate rule.** A second,
axial continuity channel was built and measured inert on the same held-out set
(-0.0008, CI [-0.0059, +0.0048]) and is deliberately not included.

**Still open after this change.** Limitation 2 above ("evidence features are
V1") is superseded in its detail but not in kind: the evidence is still
occlusion continuity only, and blur magnitude is still deliberately unused for
order. Limitation 3 (calibration ECE) is now moot for the decision — nothing is
decided on a probability — but a reliability figure would still describe the
reported `p`. Every table in this file needs re-measuring before it is quoted.

**Manuscript status.** `sections/02_materials_methods.tex`,
`sections/07_technical.tex`, `sections/01_introduction.tex` and
`sections/03_results.tex` were updated to the new rule on 2026-08-22. The
`\PlectaDepth*` macros in `results/plecta_results.tex` were **not** touched:
they still carry old-rule numbers, and every block of prose quoting them now
carries a `% TODO(depth-rerun)` marker.
