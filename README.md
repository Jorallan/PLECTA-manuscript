# PLECTA manuscript

**This repository is the canonical manuscript source.** It holds the live
article, the supplementary material, the generators behind every number and
figure, and the internal records. Work happens here.

The method was called FilaSeg until mid-2026. That name survives only in this
repository's own directory name and in a few old study folders, and should not
be used in new text.

`Paper---FilaSeg-Geometry-Driven-Instance-Extraction-of-thin-filaments`
(https://github.com/Jorallan/Paper---FilaSeg-Geometry-Driven-Instance-Extraction-of-thin-filaments)
is the **superseded FilaSeg-era predecessor**, frozen at 2026-07-30 with a
disjoint history. It is kept read-only for reference. Nothing is edited there,
and nothing from here is mirrored into it.

PLECTA is a deterministic geometry-based method that converts a binary
filament-axis mask -- and nothing else -- into separated filament instances.
An optional post-hoc depth stage additionally reads the paired greyscale
micrograph to decide which filament passes in front at each crossing; the
greyscale enters nowhere else. The primary evaluation endpoint stops before
SEM-based width rendering and compares equivalent per-instance centrelines
against the baselines described in the Results.

Names and author order are Oday Allan, Stefan Luding and Igor Ostanin; Igor
Ostanin is the corresponding author.

Machine-generated numerical macros live in `results/plecta_results.tex`,
`results/plecta_parameters.tex`, `results/plecta_depth_rule.tex` and
`results/runtime_comparison.tex`. Do not edit those files manually -- change
the generator in `scripts/results/` and re-run it.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

and the same four commands for `supplementary.tex`. `latexmk -pdf main.tex` is
equivalent where Perl is available.

## Project map

- `main.tex`: master source of the article, title block, preamble and back
  matter
- `supplementary.tex`: master source of the supplementary material, which
  carries the technical sections and the appendix
- `sections/`: article and supplementary sections, `\input` by the two masters
- `references.bib`: bibliography
- `figures/`: exactly the figures the manuscript includes;
  `figures/archive/` holds rendered output no `\includegraphics` reaches
- `results/`: the JSON records, the generated macro files and the generated
  tables
- `scripts/figures/`, `scripts/results/`: the generators for those figures,
  tables and macros, plus `check_type_scale.py`
- `reproducibility/`: `evaluation_manifest.csv`, scene/method provenance and
  hashes
- `docs/`: `REPRODUCIBILITY.md`, `DEFINE_FEASIBILITY.md`, `refs_notes.md`
- `docs/internal/`: working records not written for the reader -- the depth
  architecture and handoff notes, the leakage/fairness and stage-gate audits,
  and the literature matrix
- `exploration/`: self-contained exploratory studies (gitignored), each with
  its own README and memo; nothing here is a build input
- `software/`: the released `plecta` package and its tests, as shipped with
  the manuscript
- `slides/`: the overview deck and its build script

## Conventions

`CLAUDE.md` in this directory carries the working rules that matter most:
never hand-edit a generated file, never write a measured number into the prose
as a literal, and never tune on the locked evaluation set.
