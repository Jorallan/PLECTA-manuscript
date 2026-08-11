# Prior-method matrix, novelty test, and comparator feasibility

Date: 2026-08-11. Targeted technical review, not a systematic or patent search.

## What each method does

| | Input | Junctions | Direction estimate | Optimiser | Unmatched allowed | Bridges gaps | Re-estimates | Overlap in output |
|---|---|---|---|---|---|---|---|---|
| **Basu 2015** | intensity | deletes bifurcation nodes | spline over merged chain | binary integer program, chains ≤3 segments | yes, singleton variables | yes | yes, refit and re-solve to convergence | no |
| **DeFiNe 2015** | pre-extracted weighted graph | graph nodes | edge deflection angle | binary LP filament cover | n/a | within graph | no | yes, `over` mode |
| **SIFNE 2017** | intensity (LOFT) | excises crossing pixel + neighbourhood | terminus to centre of mass, fixed radius | greedy priority-ranked pairing | yes, becomes filament end | yes | no | no |
| **Wegmayr 2020** | intensity | shared "in principle" | branch angle (baseline) | mean-shift over learned embeddings | n/a | no | no | assigned arbitrarily |
| **DNAi 2026** | mask (post-segmentation) | hit-and-miss, single-linkage cluster, erase disk | angle over a 15 px trace | exhaustive enumeration, degree 3 or 4 only | parity leftover, wired to centre | no | no | no, hard partition |
| **GraFT 2026** | intensity (Frangi) | nodes at highest curvature | per-component angle threshold | constrained DFS + greedy longest-first cover | n/a | no, explicitly | no | no, hard edge partition |
| **Liu 2019** | mask | orientation-bin split | trained 6-branch orientation network | greedy lexicographic terminus pairing | — | yes | implicit in merge order | yes |
| **PLECTA** | mask | cluster, keep, never erase | least-squares frame over chain | exact max-weight partial matching, any degree | yes, priced | yes, gated | yes, 8 rounds | yes |

## Novelty test, ingredient by ingredient

| Ingredient | Held already by | Verdict |
|---|---|---|
| Binary-mask graph preconditioning | Liu 2019, Wegmayr 2020, DNAi 2026, GraFT 2026 | not novel |
| Exact partial matching with priced null | Basu 2015 (global ILP with singletons), DeFiNe 2015, DNAi 2026 up to degree 4, Jaqaman 2008 (priced null in a LAP) | **partially anticipated** |
| Matching across disconnected gaps | Basu 2015, SIFNE 2017, Liu 2019 | not novel |
| Iterative re-estimation after linking | Basu 2015 (to convergence), Liu 2019 (greedy), SOAX | anticipated in principle |
| Overlap-aware output | DeFiNe 2015, Liu 2019 | not novel |

**The five-ingredient combination is unclaimed, but only just.** Liu et al.
2019 holds four of the five and is missing only exact matching. Basu holds three.

**The one defensible individual claim** is a degree-general exact maximum-weight
*partial* matching with a *priced* unmatched option, against DNAi's parity-forced
leftover, degree-four cap and saturating cost terms, all verifiable in its
public source.

Do not claim novelty for skeleton graphs, geometric continuation, angular or
distance gates, endpoint pairing, optimisation, or filament-level output.

## Comparator feasibility, ranked

| Rank | Method | Runnable? | Effort | Blocker to disclose |
|---|---|---|---|---|
| 1 | **GraFT** | yes | 1--2 d | MIT Python, CPU, no training. Inject the mask at the skeleton-to-graph step. Hard edge partition can never match overlap ground truth at a crossing; no gap bridging. |
| 2 | **DNAi junction resolver** | yes, separable | 0.5--1.5 d | Two blockers found in source: the final `category != "single"` filter returns **empty output** on a single-colour binary mask (one-line patch, must be disclosed); and `if n < 2 or n > 4: continue` leaves degree ≥5 junctions unresolved (do **not** patch, report). Colour term inert, so run as a geometry-only configuration. |
| 3 | **SIFNE** | only with MATLAB | 2--4 d | BSD-2. Tip pairing is binary-native; the LOFT front end degenerates on a mask. Parameters are in 20 nm/px SMLM units and must be re-derived on development seeds. |
| 4 | **Wegmayr** | **no** | --- | No code released. The stated rationale for its embeddings is that junction analysis "would benefit from information on the image context around a junction" --- a binary axis mask contains none. Its *continuity baseline* (group branches by lowest mutual angle) is ~2 h to reimplement and is the correct greedy ablation. |
| 5 | **Basu** | **no** | --- | No code. Its likelihood is the product of raw intensity and a normalised cross-correlation, so on a binary mask the likelihood collapses to the mask itself and the orientation field degenerates. Only a "Basu-style ILP linker" ablation is possible, which is not the published method. |

Also worth considering: **Pick-and-Trace** (Liu et al., MICCAI 2023) consumes a
binary segmentation and ships a synthetic-filament generator, so it can be
trained on development seeds and tested on held-out scenes without unfairness.

SOAX and DeFiNe were run in an earlier study (`C:\Repos\comparisons`). SOAX
scores far below PLECTA and is not designed for this input; DeFiNe can represent
only 33--52% of the ground-truth filaments on a degraded mask and cannot run
above 30% areal density. Both are cited, neither is reported as a comparator.

## Techniques worth importing

| | Idea | Source | Effort | Failure it addresses |
|---|---|---|---|---|
| **T5** | **Margin-based confidence and abstention.** Exact matching yields a second-best matching cost for free; second-best minus chosen, plus gap count and frame residual, gives a per-instance confidence and a precision/recall curve. | DNAi ships a *learned* error detector; this is the non-learned analogue | 2--3 d | Nothing currently tells a user which instances to trust. Greedy methods have no second-best, so this makes exact matching pay for itself. Answers SIFNE's compounding-error objection directly. |
| **T1** | **Chain-consistent scoring.** Price chains rather than single pairs, so decisions at both ends of an arm are jointly constrained. Start with a two-junction lookahead. | Basu 2015 | 1--3 d | Per-junction matching cannot see that it is swapping short arms between two nearby crossings. Measure round-1 vs round-8 disagreement first; annealing may already fix most of it. |
| **T3** | **Curvature-adaptive frame support.** | GraFT's reverse Visvalingam--Whyatt node keeping | 2--4 d | A single long window blurs genuine high-curvature turns exactly where filaments bend into crossings. |
| **T8** | **Max-deflection guard.** An L-infinity limit alongside the summed-curvature term. | DeFiNe | hours | A single 70-degree kink has low *summed* curvature but is implausible. |
| **T9** | **Single vs tangled reporting split**, and dilation-tolerant mIoU. | Wegmayr 2020 | hours | Pre-empts "how much of the F1 comes from isolated filaments?" |
| **T11** | **False-split / false-merge reporting.** | FISBe, Mais et al. 2024 | hours | Cross-checks the VI decomposition already reported. |

**Refused transfers, and why:** Wegmayr's pixel embeddings need intensity
context and labelled patches; DeFiNe's roughness needs intensity- or
thickness-weighted edges; SIFNE's centre-of-mass tip direction is weaker than a
least-squares frame, and its residual re-extraction has nothing to re-detect
from a mask; GraFT's greedy longest-first cover; DNAi's saturating costs,
parity-forced null and degree-four cap.

## Highest value per unit effort

The **greedy angular baseline** (Wegmayr's continuity rule, ~2 h, no third-party
code) directly tests the claim the narrowed contribution now rests on: that
exact matching beats greedy pairing on the same masks with the same scorer. It
should be a row in the results table before any third-party comparator is run.
