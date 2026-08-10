# Revision length and novelty audit

Date: 2026-08-10

## Length decision

The predecessor PDF had 28 pages. The first PLECTA-only rewrite had 13 pages,
after removing the obsolete FilaSeg/Hough pipeline, predecessor comparisons,
duplicated editorial discussion, and superseded figures. That removal was
directionally correct, but the 13-page version also omitted material needed to
understand and audit PLECTA.

The present revision compiles to 17 pages. The restored material is technical,
not narrative padding:

- procedural-generator settings and split scope;
- frame, cost, gating, matching, and pair-count equations;
- a complete materialised parameter table;
- operational common-fragment assignment and statistical handling;
- geometric-evidence and exact-matching schematic;
- rule-selected held-out qualitative examples with a provenance manifest;
- same-input connected-component diagnostic floor;
- synthetic apparent-width validation; and
- corrected ablation names and interpretations.

This is the appropriate scale for the current evidence. Returning to 28 pages
would mainly restore superseded method history rather than strengthen PLECTA.

## Targeted novelty search

The search covered graph, optimization, skeleton, crossing-resolution, gap
linking, and filament-instance methods in microscopy and fibre imaging. Primary
papers reviewed include:

- Basu, Liu and Rohde (2015), integer-programming combination of centreline
  fragments: https://doi.org/10.1109/TCBB.2014.2372783
- DeFiNe (2015), optimization-based filament covers of weighted graphs:
  https://doi.org/10.1038/srep18267
- De et al. (2016), directed-graph crossover resolution:
  https://doi.org/10.1109/TMI.2015.2465962
- SIFNE (2017), terminus linking by direction, continuity, and gap distance:
  https://doi.org/10.1091/mbc.E16-06-0421
- orientation-aware terminus pairing (2019):
  https://doi.org/10.1109/CVPRW.2019.00021
- microplastic-fibre instance segmentation (2020), including a skeleton
  continuity baseline and learned tangled-fibre grouping:
  https://doi.org/10.1109/WACV45572.2020.9093352
- GraFT (2026), graph-based tracing and tracking of actin structures:
  https://doi.org/10.1126/sciadv.adz4132
- DNAi (2026), weighted local endpoint pairing after junction removal:
  https://doi.org/10.1093/nar/gkag335

The general problem and the individual ingredients are prior art. The manuscript
therefore makes no "first" claim for skeleton graphs, geometric continuation,
distance gates, optimization, endpoint pairing, or overlap-aware filament
outputs.

The defensible PLECTA contribution is the specific evaluated combination:
binary-mask graph preconditioning; exact per-junction partial matching with an
unmatched option; global gated matching across gaps; iterative chain-supported
frame re-estimation; and overlap-aware output layers. This is a targeted
technical literature audit, not an exhaustive systematic review or patent
search.

## Evidence boundary

The main quantitative result remains 128 unseen-seed scenes from the same
procedural generator, not an independent image distribution. Ablations and
synthetic width validation remain development-only. Real SEM evidence remains
an exploratory two-field case study, with only one training-clean U-Net field.
The connected-component result is a diagnostic floor, not a competitive
state-of-the-art comparison.

## Remaining author inputs

Submission completion still requires author-supplied specimen/acquisition
metadata, upstream model details, ORCIDs/CRediT roles, and an archival release
or DOI. These facts must not be inferred.
