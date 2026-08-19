# FilaSeg manuscript — review draft

This repository is a **shortened working draft** of the FilaSeg manuscript.
It is not the submission manuscript. It exists to give a quicker read while
detailed sections are still being filled in.

Canonical (full) manuscript:
https://github.com/Jorallan/Paper---FilaSeg-Geometry-Driven-Instance-Extraction-of-thin-filaments

FilaSeg is a deterministic geometry-based method that converts a binary
filament-centreline mask and its paired SEM image into separated filament
instances. The primary evaluation endpoint stops before SEM-based width
rendering and compares equivalent per-instance centrelines against a
minimum-turn skeleton baseline.

## What is different from the canonical manuscript

- Several Materials-and-methods subsections (Stages 1-4, data and reference
  annotations, evaluation metrics, configuration/baselines/ablation,
  implementation) are replaced by a short paragraph and a visible `\note{...}`
  marker; Figures 2 and 3 are kept.
- The Discussion subsections "What the comparison establishes" and
  "Development findings and failure modes" are shortened the same way.
- "Relation to prior work" (Discussion) and "Scope of the comparison"
  (Results) are removed; their unique content overlaps with material kept
  elsewhere (Introduction related work, Discussion limitations).
- The curvature/$q_{\max}$ development-study, connected-component sanity
  check, near-tie qualitative examples and hybrid-grouping-factorial material
  are removed; the independent density-width factorial is kept and reported
  as a table in the Results section. The dataset-identity record and the
  reproduction commands live in `reproducibility/evaluation_manifest.csv` and
  `docs/REPRODUCIBILITY.md` rather than in a manuscript appendix.
- Figure 4 (method comparison) is re-plotted with no per-scene horizontal
  jitter, and the abstract, introduction and terminology are simplified.
  See the `\note{...}` markers in the PDF for open points, and the CHANGES
  section of this README continues below.

Unknown experimental or authorship facts are visibly marked
`[Needed before submission: ...]` in the manuscript source itself. Open
questions and points still to add are marked `[To do: ...]` or with a `Note:`
paragraph. Names and author order are preserved as Oday Allan, Stefan Luding
and Igor Ostanin; Igor Ostanin is the corresponding author.

Machine-generated numerical macros live in `results/plecta_results.tex`.
Do not edit that file manually.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`latexmk -pdf main.tex` is equivalent where Perl is available.

## Project map

- `main.tex`: master source, title block, back matter, and the `\note{...}`,
  `\todo{...}` and `\needed{...}` command definitions
- `references.bib`: bibliography
- `sections/`: article sections (simplified)
- `figures/`: only the figures used by this simplified manuscript
- `results/`: numerical macros and the tables this manuscript still uses
- `scripts/figures/`, `scripts/results/`: generators for the retained figures
  and tables (see the canonical repository for the full script set)
- `reproducibility/`: `evaluation_manifest.csv`, scene/method provenance and hashes
- `docs/`: `REPRODUCIBILITY.md`, `DEFINE_FEASIBILITY.md`, `refs_notes.md`

## Feedback loop

This draft is not synced back into the canonical manuscript automatically.
Once we agree on the open points, apply the relevant edits by hand to the
canonical repository.
