# Editorial comment inventory

This inventory preserves uncertain comments until their authorship and resolution
are known. Git blame identifies the committer, not necessarily the author of the
comment text, so no comment is attributed to Igor solely from repository history.

## Comment mechanisms before revision

- `main.tex` defines orange, switchable `\todo{}` comments.
- `main.tex` defines always-visible red `\needed{}` comments.
- `main.tex` defines generic red `\note{}` comments without authorship.
- No Igor- or supervisor-specific macro existed before this checkpoint.

## Added mechanisms

- `\rev{}` marks added or materially rewritten manuscript text in cyan
  (`#00A6C8`) and is reversible by changing one macro.
- `\supervisorcomment{}` provides an explicit label for comments whose
  supervisor authorship is confirmed. No existing comment is converted without
  that confirmation.

## Unresolved comment groups

| Location | Subject | Current handling |
|---|---|---|
| `main.tex`, after the abstract | Whether the upstream CycleGAN/U-Net workflow and SEM-assisted rendering belong in the paper | Preserve until the revised scope implements the prompt's decision; do not assign authorship |
| `sections/01_introduction.tex`, opening block | General-to-specific framing, literature, and contribution structure | Preserve as uncertain during the technical audit |
| `sections/02_materials_methods.tex` | Missing generator, morphology, reconnection, Stage-4, metric, and SEM metadata | Resolve from code/evidence where possible; retain precise cyan placeholders for author-only facts |
| `main.tex`, author/back-matter blocks | ORCIDs, CRediT roles, and archived release | Retain; these require author input or release authorization |

## Resolution rule

Delete a red/orange comment only when its requested change is implemented and
verified, or when its author confirms removal. Preserve comments of uncertain
authorship. Record every deletion or conversion in the final comment-resolution
table.
