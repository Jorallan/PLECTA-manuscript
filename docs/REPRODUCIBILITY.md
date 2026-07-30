# Reproducibility record

Date: 2026-07-29

## Immutable checkpoints

- Locked source tag: `locked-evaluation-v1-2026-07-29`
- Locked source commit: `2ffd6cf`
- Manuscript checkpoint tag: `manuscript-checkpoint-2026-07-29`
- Original locked report: `output/locked_evaluation/results.json`
- Original locked report SHA-256:
  `4A4A7022FA0052791B8A2F0948F0BD05BA350027217B78C220C4CCFF388DD5EC`
- Corrected representation audit:
  `results/locked_v1_representation_audit_corrected.json`
- Corrected audit SHA-256:
  `0AA2F8BBCBA902279B550EA341218AE5C2DDE82E47F0C1D355486CD03050E903`
- Hashed evaluation manifest:
  `reproducibility/evaluation_manifest.csv`
- Manifest SHA-256:
  `A8833B2D6E736E30EC06011B58F763FCBAD837149089C83E0D97A509BC0133B3`
- Development hybrid report SHA-256:
  `CADED26206F59078E6A167F900D44311B030E55ED60E875FEAAE552F883D092E`
- Development density-width report SHA-256:
  `30574090DAAE18A4CF437EA4D22718CA60A3A3B3E368B2C59C2381AB43CE8C4A`

The original locked report is preserved unchanged. The corrected report is a
new derived artifact. It reads stored outputs, selects equivalent per-instance
centreline representations, retains ground-truth-unassigned common fragments as
singletons, and uses density-stratified scene bootstrap intervals. It does not
rerun or tune any method. As an integrity check, it reproduces all 125 stored
legacy Stage-4 F1 values with zero numerical drift under the legacy exclusion
policy.

## Environment

- Python 3.12.9
- NumPy 2.4.4
- SciPy 1.16.3
- scikit-image 0.25.2
- OpenCV 4.13.0
- tifffile 2025.10.16
- PyYAML 6.0.3

Bootstrap resampling uses independent scenes, seed 20260729, resampling within
each density, and identical paired-scene indices for method differences.

## Locked data

The locked set is `input/synthetic_locked_v1`, with 25 scenes at each of 20%,
30%, 40%, 50% and 60% prescribed thick-mask coverage. Seeds follow
`20260801 + density_percent*100 + sample_index`. All 125 scene directories were
complete before evaluation. No development-only hybrid or factorial study reads
or writes this directory.

The manifest records a SHA-256 hash for every input mask, flattened ground
truth, selected overlap-aware ground truth, prediction artifact and source
report. The overlap-aware multilabel ground truth is the artifact actually used
by the corrected evaluation.

## Commands

Original locked execution:

```text
python eval/runners/experiment_runner.py
  --samples input/synthetic_locked_v1
  --out output/locked_evaluation/results.json
  --mask-name mask_w1.png
  --methods baseline_cc,baseline_skeleton
  --filaseg run
  --filaseg-root output/locked_evaluation/filaseg
  --reconnect-config 3.reconnect/reconnect_config.yaml
  --cc-min-area 1
  --max-gap-px 28
  --max-turn-deg 20
  --configuration-locked
```

Non-overwriting representation audit:

```text
python eval/studies/representation_audit.py
  --source-json output/locked_evaluation/results.json
  --samples-root input/synthetic_locked_v1
  --out output/evaluation_corrections/locked_v1_representation_audit_corrected.json
```

Hashed development and locked manifest:

```text
python eval/manifests/build_manifest.py
  --locked-results output/locked_evaluation/results.json
  --development-results output/final_development_w1/results.json
  --out output/evaluation_corrections/evaluation_manifest_with_hashes.csv
```

Machine-generated result macros and tables:

```text
python paper/scripts/results/make_result_macros.py
  --locked-json output/locked_evaluation/results.json
  --representation-json output/evaluation_corrections/locked_v1_representation_audit_corrected.json
  --curvature-json output/development_curvature_study/curvature_study.json
  --real-ablation-json output/final_real_ablation/real_ablation_results.json
  --out-dir paper/results
```

Locked quantitative and qualitative figures:

```text
python paper/scripts/figures/fig_locked_results.py
  --corrected-representation-json output/evaluation_corrections/locked_v1_representation_audit_corrected.json
  --curvature-json output/development_curvature_study/curvature_study.json

python paper/scripts/figures/fig_paired_locked_examples.py
  --audit-json output/evaluation_corrections/locked_v1_representation_audit_corrected.json
  --output-dir paper/figures
```

Development sensitivity and real descriptive figures:

```text
python paper/scripts/figures/fig_development_sensitivity.py
python paper/scripts/figures/fig_qualitative_real.py
```

Development-only redesign studies and their generated supplementary tables:

```text
python eval/studies/hybrid_grouping_factorial.py

python eval/generators/synth_factorial.py
  --out output/development_factorial_density_width/scenes

python eval/studies/factorial_density_width.py
  --scenes output/development_factorial_density_width/scenes
  --out output/development_factorial_density_width/factorial_report.json

python paper/scripts/results/make_development_study_tables.py
  --hybrid-json paper/results/development_hybrid_grouping_factorial.json
  --factorial-json paper/results/development_factorial_density_width.json
  --out-dir paper/results
```

## Primary corrected result

- FilaSeg Stage-3 centreline F1: 0.734, 95% CI [0.722, 0.745]
- Minimum-turn centreline F1: 0.663, 95% CI [0.651, 0.675]
- Paired FilaSeg minus minimum-turn F1: 0.071, 95% CI [0.059, 0.082]
- Stage-4 rendered minus Stage-3 centreline F1: -0.053,
  95% CI [-0.059, -0.047]

The primary endpoint is grouping on equivalent centreline representations.
Stage-4 rendering is secondary. Connected components are a supplementary sanity
check.

## Expected paper artifacts

- `results/locked_evaluation.json` and `.csv`, retained as historical sources
- `results/locked_v1_representation_audit_corrected.json`
- `results/filaseg_results_summary.json`
- `results/filaseg_results.tex`
- `results/filaseg_baseline_table.tex`
- `results/filaseg_sanity_table.tex`
- `results/development_hybrid_grouping_factorial.json`
- `results/development_factorial_density_width.json`
- `results/development_hybrid_table.tex`
- `results/development_factorial_table.tex`
- `reproducibility/evaluation_manifest.csv`
- `figures/fig_locked_method_comparison.pdf` and `.png`
- `figures/fig_paired_locked_wins.pdf` and `.png`
- `figures/fig_paired_locked_losses.pdf` and `.png`
- `figures/fig_paired_locked_ties.pdf` and `.png`
- `figures/fig_width_sweep.pdf` and `.png`
- `figures/fig_clean_vs_degraded.pdf` and `.png`
- `figures/fig_qualitative_real.pdf` and `.png`
- `main.pdf`

A final DOI or archival release identifier remains an author action.
