# FilaSeg manuscript — supervisor-review mirror

This repository is a **shortened, supervisor-review mirror** of the canonical
FilaSeg manuscript. It is not the submission manuscript. It exists to give a
supervisor a quicker read before detailed sections are restored.

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
  implementation) are replaced by a short paragraph and a visible
  `\reviewnote{...}` marker; Figures 2 and 3 are kept.
- The Discussion subsections "What the locked comparison establishes" and
  "Development findings and failure modes" are shortened the same way.
- "Relation to prior work" (Discussion) and "Scope of the comparison"
  (Results) are removed; their unique content overlaps with material kept
  elsewhere (Introduction related work, Discussion limitations).
- The curvature/$q_{\max}$ development-study, connected-component sanity
  check, near-tie qualitative examples and hybrid-grouping-factorial
  appendices are removed; the independent density-width factorial, dataset
  identity and reproduction appendices are kept.
- The locked-comparison figure (Figure 4) is re-plotted with no connected-component
  series and no per-scene horizontal jitter, and the abstract, introduction
  and terminology are simplified. See `\reviewnote{...}` markers in the PDF
  for open questions, and the CHANGES section of this README continues below.

Unknown experimental or authorship facts are visibly marked
`[ACTION REQUIRED: ...]` or `[MISSING INFORMATION: ...]` in the manuscript
source itself, unchanged from the canonical manuscript. Names and author
order are preserved as Oday Allan, Stefan Luding and Igor Ostanin; Igor
Ostanin is the corresponding author.

Machine-generated numerical macros live in `results/filaseg_results.tex`.
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

- `main.tex`: master source, title block, back matter, and the
  `\reviewnote{...}` command definition
- `references.bib`: bibliography
- `sections/`: article and appendix sections (simplified)
- `figures/`: only the figures used by this simplified manuscript
- `results/`: numerical macros and the tables this manuscript still uses
- `scripts/figures/`, `scripts/results/`: generators for the retained figures
  and tables (see the canonical repository for the full script set)
- `reproducibility/`: `evaluation_manifest.csv`, scene/method provenance and hashes
- `docs/`: `REPRODUCIBILITY.md`, `DEFINE_FEASIBILITY.md`, `refs_notes.md`

## Feedback loop

This mirror is not synced back into the canonical manuscript automatically.
Once supervisor feedback is collected, the relevant edits should be
applied by hand to the canonical repository.
