# DeFiNe feasibility assessment

Date assessed: 2026-07-29

## Conclusion

**Feasible in principle, but blocked for the present benchmark by a legacy
runtime and by the need for a predeclared graph adapter.**

DeFiNe is not included as a scored baseline in the full degraded-mask
evaluation. This is not a citation-only substitute for an experiment. It is a
documented exclusion based on the method's input contract and the reproducible
software available at the time of evaluation.

## Input and output compatibility

DeFiNe consumes a weighted, undirected geometric graph in GML format. Nodes
store positions; edges represent filament segments; edge weights represent
local intensity or thickness. It does not consume a binary centreline mask
directly. Its output is a GML graph whose edge colours encode filament
identities, together with a CSV of per-filament measures.

A restricted same-scene comparison is conceptually possible:

1. skeletonize the shared binary centreline mask;
2. split it at endpoints and junctions;
3. export a fixed weighted GML graph;
4. run DeFiNe in its overlap-enabled mode; and
5. rasterize every returned path as a separate instance layer for the common
   fragment metric.

The overlap-enabled formulation is required because it permits a graph edge to
belong to more than one path. The exact-cover option does not represent shared
overlap edges.

## Why it is not run in the locked full-task benchmark

Two unresolved adapter choices would materially change the experiment:

- A binary centreline mask supplies no natural nonconstant intensity or
  thickness edge weights. Sampling weights from the paired grayscale image
  could be defensible, but that would be a separately specified input protocol,
  not the same binary-mask-only contract used by the two mandatory baselines.
- DeFiNe decomposes the graph it is given. It does not create missing graph
  edges across breaks in a degraded mask. Adding virtual gap edges would
  introduce a new reconnection algorithm and would no longer evaluate the
  published method alone.

The available source also targets Python 2.7.3 with PyGTK, SciPy, NumPy,
NetworkX, GLPK, and CVXOPT. The authors' page reports binaries tested on
32-bit Ubuntu 12.04/Mint 13 and Windows 7, and was last updated in 2014. No
validated environment or bitwise-specified mask-to-GML adapter exists in this
repository. Reconstructing that environment and protocol after seeing locked
results would violate the configuration-lock requirement.

Accordingly, the fair future experiment is a separately declared
connected-network subtask using a frozen graph exporter and weight definition.
It must not be presented as a direct baseline for FilaSeg's gap-reconnection
task.

## Primary sources

- Breuer, D. and Nikoloski, Z. (2015), “DeFiNe: an optimisation-based method
  for robust disentangling of filamentous networks,” *Scientific Reports* 5,
  18267. <https://doi.org/10.1038/srep18267>
- Authors' software and dependency page:
  <https://mathbiol.mpimp-golm.mpg.de/DeFiNe/>
