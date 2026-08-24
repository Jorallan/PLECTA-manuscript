# Leakage and fairness audit

Status: internal audit, 2026-08-10.

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
>
> It also predates the 2026-08-22 depth evidence-rule replacement and says
> nothing about the 20-scene / 1355-crossing optical set on which the two
> depth rules were compared. Those seeds are disjoint from the material either
> rule was developed on, but the flat core half-size (`core_px = 7.5`) was
> settled by a sweep run on that same set, and the abstention threshold's
> flatness was checked there, so the depth-rule comparison is in-sample with
> respect to those two constants. `sections/07_technical.tex` now says so;
> it previously claimed the seeds were disjoint from every set used to choose
> the rule *or its constants*, which was stronger than the truth.
>
> What remains valid is the reasoning, not the coordinates.


## Verdict

There is no direct evidence that the frozen replacement method was tuned on its
declared 128-scene held-out set. Development and held-out seeds are disjoint, the
tuning/oracle entry points do not expose the held-out override, and the stored
held-out artifact reports all 128 scenes with no failures. However, the freeze is
procedural rather than cryptographically immutable, the held-out set is an
independent sample from the same generator rather than a new distribution, and
several documentation and guard defects prevent a publication-grade provenance
claim without repair.

## Declared splits and seed audit

- Development: 84 scenes: 20 supplied `synthetic_thick`, 48 `dev_ext`, and 16
  generated coverage-40 scenes (`C:\Repos\stubmatch\SPLIT.md:16-33`).
- Held-out: 128 scenes: 96 `val` scenes and 32 generated coverage-40 scenes
  (`SPLIT.md:35-47`).
- The held-out set is explicitly an independent same-generator sample, not an
  independent distribution (`SPLIT.md:44-47`).
- The stronger `synthetic_locked_v1` set was not used by the replacement method
  (`SPLIT.md:65-69`).

Direct inspection of `gt_meta.json` gives these disjoint seed ranges:

| Split source | Count | Seed range |
|---|---:|---:|
| `dev_ext` | 48 | 20272101-20276116 |
| `gen_dev40` | 16 | 20294101-20294116 |
| `val` | 96 | 20282101-20286132 |
| `gen_test40` | 32 | 20314101-20314132 |

All inspected seeds are unique and no development/held-out overlap exists.

## Frozen held-out evidence

`results/heldout.json` contains 128 per-scene rows, records the held-out override,
and reports no failures. Mean common-fragment F1 is 0.8545, precision 0.8676,
recall 0.8430, recovery 0.7994, and ARI 0.8508. Density F1 is 0.9112, 0.8736,
0.8582, and 0.7749 at coverage 20, 30, 40, and 60. The development mean is
0.8671. Stored per-scene rows permit a post-hoc confidence interval without
rerunning or tuning.

The only comparator on the same 128 scenes is the one-instance-per-connected-
component floor (F1 0.0855). No predecessor or conventional tracer was run on
these 128 scenes.

## Guard assessment

- `scripts/_common.py:127-148` refuses declared held-out paths unless an explicit
  flag is passed, and it checks scene-ID uniqueness.
- Tuning, ablation, and oracle entry points omit the override.
- The guard is based on path strings, not seed or content hashes. Copying or
  renaming data can evade it.
- `frozen/predict.py --scenes` applies a locked-set path guard, but direct
  `--mask` prediction bypasses it (`frozen/predict.py:62-68,86-101`).
- `SPLIT.md:52-54` says only `evaluate.py` exposes `--allow-holdout`, but
  `scripts/baseline_cc.py:35-44` exposes it too. Its held-out JSON does not record
  the override.

These defects do not demonstrate leakage, but they invalidate stronger claims
that access is impossible by construction.

## Freeze and provenance assessment

- `C:\Repos\stubmatch` has no `.git` directory. There is no independently
  verifiable commit, tag, or pre-registration order.
- `frozen/params.json` omits five effective defaults. The full effective
  configuration is present in result prose/artifacts but not in one executable
  freeze file.
- Source modification times postdate `results/heldout.json`; documentation
  reports bit-identical development/held-out parity after later default-off
  edits, but the earlier source bytes are unavailable.
- Raw held-out predictions are absent. Only per-scene metric rows remain, so
  output equivalence requires rerunning.
- `data/seed_manifest.json` omits `gen_dev40` and `gen_test40`, despite
  `SPLIT.md:9-10` claiming every seed is recorded. The referenced verifier does
  not exist in this directory.

Required repair: materialize all effective parameters, snapshot source and
dependency versions, record SHA-256 hashes of code/data/results, complete the
seed manifest and verifier, and preserve raw predictions for a release run.

## Acceptance-comparison fairness

The private 50-scene acceptance study is like-for-like on masks, ground truth,
common fragments, scorer, and edge-case policies. Outputs are normalized before
scoring. A parity gate reproduces established F1, ARI, and VI components. All
scenes and failures are retained. Clean and degraded masks from the same geometry
are analyzed separately because they induce different common-fragment
populations. Runtime is comparable only among warm in-process calls; the
predecessor's multi-process CLI wall time is labelled an upper bound. Memory was
not recorded.

The acceptance data were generated after the methods were frozen. The study is
exploratory (10 scenes per density), but the total paired sample of 50 supports
the reported stratified bootstrap interval. It is an internal acceptance test,
not a publication comparator.

## Ablation fairness

Ablations are development-only and one-at-a-time. Largest F1 losses are removal
of the chord-turn term (-0.1858), all gap bridging (-0.1394), and both junction-
merge mechanisms (-0.0634). Removing spur pruning changes F1 by +0.0017 but the
operation is retained for representation parity. The stored ablation artifact
contains aggregate/per-density summaries rather than per-scene paired rows, so
paired uncertainty cannot be reconstructed. Interaction ablations remain
missing.

Post-hoc Stage-4 and continuation merges have only development/one-real-crop
evidence and were absent from the held-out run. They cannot be promoted into the
frozen algorithm without a new, explicitly versioned freeze and a deliberate
evaluation.

## Real-mask comparison

`C:\Repos\datasets\scratchfolder` contains two distinct full SEM fields (280 and
430 annotated filaments); an earlier 512-pixel crop is a subset, not a third
independent sample. Only the B58_110 field is clean with respect to the inspected
nnU-Net training set. B58_100 was in training.

The manual-skeleton condition is derived from the same annotation used for
evaluation and is therefore an oracle-input/ceiling condition. The predicted and
manual-derived masks create different common-fragment populations, so their
pairwise F1 values are not strictly like-for-like. Cross-mask claims should rely
on axis precision/recall, detection/absence, and within-mask grouping metrics,
with the ceiling status made explicit.

Only the following publication-safe conclusion is currently supported: input
mask quality materially limits downstream instance reconstruction. The current
evidence does not support broad generalization from real SEM, segmenter
sensitivity across specimens, or a statistically powered method ranking.

## Publication claim limits

Permitted with caveats:

- same-generator held-out performance of the frozen mask-only core;
- density dependence within that held-out sample;
- development-only component ablations, clearly labelled;
- sensitivity to clean versus degraded synthetic masks;
- exploratory CNT SEM demonstration;
- input-mask sensitivity as a limitation.

Not permitted from current evidence:

- independent-distribution generalization;
- a held-out contribution from Stage 2, Stage 4, silhouette absorption, or the
  continuation merge;
- a claim that the post-hoc merge is validated;
- population-level real-SEM performance;
- a reproducible immutable release or pre-registration ordering;
- memory-efficiency claims.

## Priority unresolved experiments

1. Freeze a versioned final snapshot and run it once on untouched
   `synthetic_locked_v1`, with all intended external comparators under one scorer.
2. Add at least three independent annotated real fields, preferably across
   specimens/acquisition sessions, using a training-clean mask extractor.
3. Run a fixed mask-source by instance-method factorial on the same real fields
   with a fixed filament universe.
4. Preserve per-scene ablation rows and test the important gap-by-crossing and
   node-join-by-absorption interactions.
5. Complete provenance hashes, the seed verifier, dependency lock, and raw
   prediction archive.
