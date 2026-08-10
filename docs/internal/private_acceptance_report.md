# Private replacement-method acceptance report

Status: internal acceptance test. This comparison and the superseded method must
not appear in the publication narrative, figures, tables, captions, conclusions,
or publication-facing filenames.

## Decision

**Pass.** On a purpose-built 50-scene synthetic set, the frozen replacement core
outperforms the superseded implementation under identical masks, ground truth,
representation normalization, and scoring. The paired mean common-fragment F1
difference is **+0.0902** with a density-stratified bootstrap 95% confidence
interval of **[+0.0747, +0.1071]**. The replacement is better in 47 of 50 scenes.
The same conclusion holds for the clean-mask condition.

This result licenses revision around the replacement method. It does not license
a claim of universal superiority, and it will remain private.

## Study used

Source directory:
`C:\Repos\comparisons\filaseg_stubmatch_strandface`.

Primary data are 50 independently generated 512-by-512 scenes, 10 at each of
20%, 30%, 40%, 50%, and 60% nominal coverage. Seeds begin at 20400101. The set
was generated for the comparison after the compared methods were frozen. Every
method received the same `mask_w1.png` in the degraded condition and the same
`mask_clean.png` in the clean condition. All outputs were normalized to
overlap-aware instance masks and scored through the same common-fragment path.

Artifacts:

- protocol: `README.md` in the source directory;
- per-scene degraded results: `results/testcmp10_per_scene.csv`;
- per-scene clean results: `results/testcmp10_per_scene_clean.csv`;
- frozen prediction runners: `scripts/run_methods.py`;
- parity guard: `scripts/parity_check.py`;
- shared scorer: `scripts/score_all.py`;
- timing verification: `scripts/time_methods.py`.

The parity guard reproduces the established locked-set F1, ARI, VI split, and VI
merge values before admitting the comparison. The FilaSeg representation trap
is handled by loading its overlap-aware Stage-3 multilabel NPZ, not its lossy
single-label preview. The primary 50-scene set reproduces the two established
anchors closely: predecessor F1 0.7316 versus 0.734 on its locked set, and the
fixed minimum-turn baseline 0.6702 versus 0.663.

## Primary degraded-mask results

Plain scene means, n=50:

| Method | F1 | Precision | Recall | Recovery | ARI | VI split | VI merge | VI total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen replacement core | 0.8218 | 0.8306 | 0.8144 | 0.7651 | 0.8175 | 0.3015 | 0.2710 | 0.5726 |
| Superseded implementation | 0.7316 | 0.7509 | 0.7176 | 0.5933 | 0.7250 | 0.4839 | 0.3760 | 0.8599 |

Paired replacement-minus-superseded differences use a deterministic
density-stratified nonparametric bootstrap with 20,000 resamples and seed
20260810 (metric-specific seed offsets prevent shared resample artefacts):

| Metric | Mean difference | 95% CI | Replacement wins |
|---|---:|---:|---:|
| F1 | +0.0902 | [+0.0747, +0.1071] | 47/50 |
| precision | +0.0797 | [+0.0603, +0.1011] | 44/50 |
| recall | +0.0968 | [+0.0806, +0.1144] | 46/50 (2 ties) |
| recovery | +0.1718 | [+0.1479, +0.1951] | 49/50 |
| ARI | +0.0924 | [+0.0762, +0.1099] | 47/50 |
| VI split | -0.1824 | [-0.2075, -0.1584] | lower in 50/50 |
| VI merge | -0.1050 | [-0.1323, -0.0792] | lower in 42/50 |
| VI total | -0.2874 | [-0.3315, -0.2460] | lower in 46/50 |

## Clean-mask robustness

| Method | Degraded F1 | Clean F1 | Change |
|---|---:|---:|---:|
| Frozen replacement core | 0.8218 | 0.8442 | +0.0224 |
| Superseded implementation | 0.7316 | 0.7389 | +0.0073 |

On clean masks, the paired F1 difference is +0.1053 (95% CI +0.0929 to
+0.1179), and the replacement is better in all 50 scenes. The replacement's
absolute degradation sensitivity is larger, but its degraded-mask performance
remains higher at every tested density.

## Performance by density

Replacement F1 on the degraded masks is 0.894, 0.857, 0.807, 0.796, and 0.753
from 20% through 60% coverage. Superseded F1 is 0.787, 0.799, 0.745, 0.687, and
0.640. The replacement therefore passes at every density, including the sparse
regime where the earlier implementation historically struggled.

## Runtime and failures

All 200 degraded result rows (four methods by 50 scenes) and all 200 clean rows
have status `ok`. No scene was dropped.

The replacement's verified warm, in-memory predict call has mean 1.581 s,
median 0.617 s, 90th percentile 4.326 s, and maximum 12.847 s. Stored prediction
fingerprints match in all 50 scenes. The predecessor timing is not directly
comparable because it includes CLI startup and multi-stage disk I/O; its reported
mean wall time is 13.53 s and is treated as an upper bound rather than an
algorithm-only speed claim. Memory was not instrumented and remains unresolved.

## Interpretation and acceptance limits

- The performance gain is supported across F1, precision, recall, recovery, ARI,
  and both VI components, rather than selected from one favourable metric.
- The paired analysis uses scenes as the sampling unit and preserves density
  strata.
- The data come from one procedural generator. They establish improvement on the
  controlled instance-grouping problem, not on every microscope or morphology.
- The replacement deliberately aligns its arm decomposition to the scorer's
  common-fragment decomposition. That is valid but means part of the gain is
  representation alignment, not solely better geometric reasoning.
- The 50-scene set is an acceptance set, not the method's declared 128-scene
  held-out set. No tuning was performed after seeing it.
- Real evidence remains too small and asymmetric for a universal method ranking.

No further threshold tuning is justified by this acceptance test.
