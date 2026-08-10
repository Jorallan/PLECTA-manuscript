# PLECTA frozen core

PLECTA converts a thin binary filament-axis mask into overlap-aware 2-D
filament instances. This directory is a versioned snapshot of the exact
mask-only core that produced the stored 128-scene held-out result. The provisional
publication name does not change the algorithm.

## Scope

Included:

- skeletonization and short-spur pruning;
- crossing-node and arm/stub construction;
- local and chain-supported tangent/curvature frames;
- exact per-node continuation matching;
- globally gated gap matching;
- eight refinement rounds; and
- sparse overlap-aware multilabel output.

Not included in this frozen core:

- SEM-based width or brightness measurement;
- ribbon rendering;
- silhouette absorption;
- the experimental continuation/duplicate merge gate; or
- any ground-truth-dependent operation.

Those components did not contribute to the stored held-out result.

## Run

From the repository root, using the recorded scientific-Python environment:

```powershell
C:\Repos\venv_cnt\Scripts\python.exe software\plecta\predict.py `
  --mask path\to\mask.png `
  --out path\to\pred_multilabel.npz
```

For a tree of scene directories containing `mask_w1.png`:

```powershell
C:\Repos\venv_cnt\Scripts\python.exe software\plecta\predict.py `
  --scenes path\to\scenes `
  --out path\to\predictions
```

The output NPZ stores flattened pixel indices and offsets for each instance.
Crossing pixels may occur in more than one instance.

## Freeze contract

`params.json` materializes every effective parameter, including values that were
implicit dataclass defaults in the working implementation. Missing parameters
are not silently accepted. The direct-mask and multi-scene CLI paths both refuse
the archived `synthetic_locked_v1` set.

The matching behavior is otherwise unchanged from the evaluated core. In
particular, the frozen annealing schedule has only one fully strict round, so the
last round is returned. That behavior is retained for result parity and is
documented in `docs/internal/stage_gate_audit.md` rather than silently changed.

## Dependencies and licensing

The code requires Python, NumPy, SciPy, scikit-image, NetworkX, and an image
reader supported by scikit-image. Exact environment versions and source hashes
belong in the clean-checkout reproducibility record.

No standalone software licence has been approved for this snapshot. Repository
publication or reuse terms must be decided by the authors before an archived
software release.
