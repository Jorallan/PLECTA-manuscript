# Reconstruction plan

Date: 2026-08-11

## Why this branch exists

The 17-page revision at `402e441` was rejected by the author. This branch
restarts the manuscript from `1dc9d35` and rebuilds it, porting forward only
material that survives an explicit justification test. Nothing was deleted and
no public history was rewritten.

## Archived state

| What | Where |
|---|---|
| Rejected revision, branch | `archive/ai-revision-402e441` (pushed to origin) |
| Rejected revision, tag | `archive-402e441-ai-expanded` (pushed to origin) |
| Reconstruction branch | `reconstruction`, branched from `1dc9d35` |

`review-draft` is untouched and still points at `402e441`.

## The three versions

Source words across `main.tex` and `sections/`:

| Commit | Pages | Words | What it is |
|---|---|---|---|
| `1bf048f` | 28 | 10,252 | Legacy manuscript, superseded name and pipeline |
| `1dc9d35` | 13 | 4,163 | First rewrite around the current method |
| `402e441` | 17 | 5,506 | Rejected expansion |

The 13-page cut removed 59% of the source. The expansion added back 32%, of
which **87% went into Materials and methods** — not Results, not Discussion,
not the limitations.

## Defects in the rejected revision

Verified directly, not taken on trust:

1. **The revision colour carries no information.** 96.6% of body words are
   inside `\rev{}` or a `revisioncyan` block, so the compiled PDF is entirely
   cyan including the title, headings and table bodies. A reader learns nothing
   about what changed.
2. **Six equations, three figures and the parameter table are never
   referenced.** 35 labels are defined against 7 `\ref` calls. `fig:gates`,
   `fig:plecta-geometry`, `fig:plecta-real-mask-source`, `tab:plecta-parameters`
   and every numbered equation float unanchored: the reader meets Figure 3 with
   nothing in the text pointing at it.
3. **Methods transcribes code into English.** The fragment-assignment tie-break
   rule, the cycle guard (followed by the admission that no cycle ever occurred
   in 84 scenes) and the metric edge-case handling (in a run reported one page
   later as having zero failures) are implementation notes, not method.
4. **`\needed{}` markers were removed.** The preamble deliberately made them
   independent of the `\ifaq` switch so a missing-metadata blocker stays visible;
   the SEM acquisition blocker was converted to ordinary prose at `5199bee`.
5. **A raw code identifier reaches a publication table.** `No Join\_Px` in
   `results/plecta_ablation_table.tex`.

## Material that must come back from `1bf048f`

The genuinely damaging loss is the **width x centreline-length-density
factorial** deleted at `a9a8704`. Within each (seed, length-density) block the
mask, clean axis and reference are byte-identical across bundle widths while
achieved areal density varies more than twofold, and the within-geometry F1
range across widths was zero in all twelve blocks. That is a structural
negative control: it shows areal coverage cannot causally affect centreline
grouping, and that **crossing density is the real difficulty axis**.

The rejected revision nonetheless stratifies every table and figure by target
areal coverage — the axis the legacy manuscript had already shown to be
confounded. This is a scientific defect, not a presentation one.

The supporting data are all still tracked, so this is a re-run against the
current method rather than a reconstruction:
`results/development_factorial_density_width.json`,
`results/development_factorial_table.tex`,
`results/development_density_sweep.json`, `figures/fig_density_sweep.*`.

Secondary loss: the physical-scale adaptation rule. The rejected revision
hard-codes roughly twenty pixel-valued parameters with no rescaling rule, which
documents the method as valid only at the one pixel size used here.

## Material to port forward from `402e441`

Each of these was checked against the machine-readable record before being
accepted:

- **Corrected ablation labels.** `Tangent Term Only` was renamed to
  `No Junction Chord-Turn` and `No Crossing Merging At All` to
  `No Junction-Cluster Consolidation`, with values unchanged. `1dc9d35`
  publishes the wrong interpretation in four places. Verified by diffing the
  generated table at both commits.
- **Evidence records** added at `6b634b8`: width validation, connected-component
  diagnostic, and the qualitative selection manifest.
- **The named prior-method citations** added to `references.bib` at `adabba2`.

## Decisions taken

| Decision | Choice |
|---|---|
| Baseline | `1dc9d35`, with explicit ports from both `1bf048f` and `402e441` |
| Name | Keep PLECTA and the one-sentence *plectere* etymology; drop the backronym |
| Revision colour | Cyan marks material change against `1bf048f`, the version last read by a supervisor — not against the intermediate rewrites |
| External comparison | Not SOAX. See below. |

## External comparison

`C:\Repos\comparisons` already holds an audited study. SOAX 3.7.0 was run from
the vendor binary on 125 locked scenes and scores F1 0.516; DeFiNe was run from
the authors' own source and is a structural scope mismatch, able to represent
only 33-52% of ground-truth filaments on a degraded mask and unable to run above
30% areal density.

Neither is worth reporting as a matched comparator. Beating a method that is not
designed for this input does not support a superiority claim, and a reviewer
would say so. Both are cited instead. The comparator search is directed at the
closer prior methods — Basu, SIFNE, Wegmayr, DNAi, GraFT — with an honest
"could not be run fairly, for this specific reason" recorded wherever that is
the outcome.

## Evidence boundaries that constrain every claim

Unchanged from the previous audit and restated here because the rebuild must
respect them: same-generator held-out scenes are not distribution transfer;
ablations are development-only; synthetic width recovery is not physical CNT
diameter validation; manual-derived masks are an oracle-like input; the real
evidence is two fields, only one of which is training-clean; mask-specific
fragment universes prevent paired cross-mask comparison; 2-D overlap-aware
identity is not 3-D depth ordering; and no superiority claim is available
without a fair comparator.
