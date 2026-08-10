# Publication naming study

Status: recommendation recorded 2026-08-10. The choice remains provisional
pending author approval; manuscript uses must therefore be marked with `\rev{}`.

## Method concept that the name must represent

The frozen core does more than semantic segmentation. It decomposes a binary
filament-axis mask into arms and crossing nodes, pairs arm-end stubs using
explicit geometric evidence, re-estimates frames from assembled chains, and
returns overlap-aware 2-D filament instances. It does not recover depth and does
not use SEM intensity in the grouping decision.

## Search procedure and limits

Searches were performed on 2026-08-10 for each candidate together with terms
such as “software,” “algorithm,” “image segmentation,” “filament,” and
“reconstruction,” with targeted PyPI and GitHub queries for the finalists. This
is a practical collision screen, not a trademark opinion or an exhaustive legal
search. Absence of a result means no prominent exact-name collision was found in
the queried sources on that date.

## Ten candidates

| Candidate | Optional expansion or origin | Connection | Pronunciation | Main strength | Main concern | Search finding | Title reading | Repository reading |
|---|---|---|---|---|---|---|---|---|
| **PLECTA** | Mnemonic: *Pairwise Linking with Evidence from Chain Tangents and Assignment*; name also evokes Latin *plectere*, to weave or twist | Reweaves locally ambiguous arms into chain-consistent instances | PLEK-tah | Distinctive, compact, mechanism-relevant, scientifically credible | *Ochropleura plecta* is a moth species; the full expansion is best treated as a mnemonic, not forced into every use | Exact software/segmentation and PyPI/GitHub searches found no prominent method collision; the biological species is documented by the [EPPO Global Database](https://gd.eppo.int/taxon/OKHRPL); the Latin root is documented as “to weave; to twist” by [Wiktionary](https://en.wiktionary.org/wiki/plectere) | “PLECTA: Chain-informed reconstruction of dense filament instances” reads cleanly | `plecta` is short and package-friendly |
| **StubWeave** | Descriptive compound; no acronym | Explicitly names the stub-pairing and chain-weaving mechanism | STUB-weev | Transparent and memorable; almost impossible to misread as 3-D | Slightly informal; “stub” may sound implementation-specific | Exact-name software/algorithm searches returned no prominent collision | “StubWeave: Overlap-aware instance reconstruction…” is clear but less formal | `stubweave` is distinctive and searchable |
| **CAIRN** | *Chain-Aware Instance Reconstruction of Networks* | Chain evidence guides a path through ambiguous topology, like a cairn marks a route | cairn | Natural acronym and restrained metaphor | Major and growing collision load | A 2026 topology-aware 3-D scene method already uses CAIRN ([arXiv:2607.06534](https://arxiv.org/abs/2607.06534)); active AI/security software also uses the name ([GitHub](https://github.com/oritera/Cairn)) | Strong title sound, but ambiguous in search | `cairn` is already crowded |
| **CIRCA** | *Chain-Informed Reconstruction by Crossing Assignment* | Captures the crossing-assignment and chain-refinement loop | SIR-kah | Natural-sounding and technically accurate | Acronym is widely reused | CIRCA already names a medical-imaging system ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12663097/)) and a clustering method ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7900870/)) | Reads well but needs domain qualifier everywhere | `circa` is generic and already packaged |
| **PACER** | *Pairing Arms with Chain Evidence for Reconstruction* | Directly describes the key mechanism | PAY-ser | Excellent natural acronym and sense of iterative progress | Direct reconstruction-software collision | PaCER is established electrode-trajectory reconstruction software ([official documentation](https://adhusch.github.io/PaCER/stable/index.html)); `pacer` also exists on [PyPI](https://pypi.org/project/pacer/0.16.0/) | Attractive but collision-prone | Package name unavailable/confusing |
| **CATENA** | *Chain-Aware Topological Extraction and Network Assembly*; Latin for chain | Emphasizes chain construction and topology | kuh-TEE-nah | Genuine meaning and polished scientific tone | Numerous software and scientific uses; acronym feels mildly engineered | Catena already names automated connectome-reconstruction software ([Janelia OSSI](https://ossi.janelia.org/projects/)) and an active media-service standard ([GitHub](https://github.com/rossvideo/Catena)) | Credible but not distinctive | `catena` is crowded |
| **CLASP** | *Chain-Link Assignment through Stub Pairing* | Captures pairing and attachment | clasp | Vivid, concise, easy to pronounce | Direct segmentation collision and broad acronym reuse | CLASP already names a 2025 image-segmentation method ([CVF paper](https://openaccess.thecvf.com/content/ICCV2025W/CVAM/papers/Curie_CLASP_Adaptive_Spectral_Clustering_for_Unsupervised_Per-Image_Segmentation_ICCVW_2025_paper.pdf)) and several unrelated algorithms | Good prose, poor searchability | `clasp` is heavily reused |
| **INSTAR** | *Instance Network Stitching by Tangent-Aware Reconstruction* | “Instar” contains “instance” while the expansion describes tangent-guided stitching | IN-star | Memorable and concise | Existing segmentation/software terminology; biological meaning is unrelated | INSTAR appears in the NITRC software glossary under segmentation ([NITRC](https://www.nitrc.org/include/glossary.php)) | Strong title sound but collision risk | `instar` is not distinctive enough |
| **PACT** | *Pairing Arms with Chain Tangents* | Very compact description of the core matching evidence | pact | Short, natural, mechanism-exact | Dominant established imaging acronym | PACT is standard terminology for photoacoustic computed tomography and its reconstruction algorithms ([PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10013779/)) | Misleading in an imaging paper | Package/search collision is severe |
| **RAVEL** | Mnemonic: *Refined Arm-Vector Evidence Linking* | Repeatedly links arm vectors; “ravel” also means tangle or disentangle | RAV-el | Memorable relation to tangled filaments | Expansion is forced and name is already commercial software terminology | RAVEL is active Cadence PCB-rule software terminology ([Cadence](https://community.cadence.com/cadence_blogs_8/b/pcb/posts/debugging-ravel-rules-from-silent-failures-to-visual-proof)); many unrelated uses make search noisy | Evocative but ambiguous | `ravel` is crowded |

## Shortlist

### 1. PLECTA — recommended

- **Conceptual fit:** weaving is a restrained metaphor for reconstructing
  identity through crossings; the mnemonic names the actual chain-tangent and
  assignment machinery.
- **Pronunciation:** PLEK-tah.
- **Strengths:** distinctive, five/six letters, non-generic, works in prose and
  code, makes no 3-D or learning claim.
- **Possible confusion:** a species epithet and other minor lexical uses, not a
  prominent algorithm or software product in the screened domain.
- **Paper title:** *PLECTA: Chain-informed reconstruction of dense filament
  instances from binary axis masks*.
- **Repository/package:** `plecta`; historical artifact mapping can retain the
  old working directory name internally.
- **Acronym quality:** acceptable if presented as a mnemonic once; the name also
  stands without the expansion, avoiding strained repeated prose.

### 2. StubWeave — fallback

- **Conceptual fit:** literal pairing of stubs into chain-consistent instances.
- **Pronunciation:** STUB-weev.
- **Strengths:** highly searchable, understandable without expansion, no
  misleading dimensional or learning claim.
- **Possible confusion:** none prominent in the exact-name screen.
- **Paper title:** *StubWeave: Chain-informed reconstruction of dense filament
  instances*.
- **Repository/package:** `stubweave`.
- **Name quality:** natural compound rather than an acronym; slightly less formal
  but maximally transparent.

### 3. CAIRN — strong language, rejected unless collisions are accepted

- **Conceptual fit:** chain-aware reconstruction and route-finding through an
  ambiguous network.
- **Pronunciation:** cairn.
- **Strengths:** excellent natural acronym and memorable metaphor.
- **Collision:** a current topology-aware 3-D scene method and active software
  already use it. Searchability is no longer acceptable.

### 4. CIRCA — technically clean, rejected unless collisions are accepted

- **Conceptual fit:** chain-informed crossing assignment is exactly the core
  operation.
- **Pronunciation:** SIR-kah.
- **Strengths:** concise and paper-friendly.
- **Collision:** established medical-imaging and clustering uses, plus generic
  lexical noise.

## Recommendation

Use **PLECTA** provisionally and mark the name in cyan until the authors approve
it. Use **StubWeave** if the authors prefer a fully descriptive name without a
Latin root or mnemonic expansion.

The recommended title is:

> **PLECTA: Chain-informed reconstruction of dense filament instances from
> binary axis masks**

The explanation in the paper should be one restrained sentence: “The name
PLECTA evokes *plectere*, to weave or entwine, reflecting the reconstruction of
filament identity through locally ambiguous crossings.” Do not imply mechanics,
depth recovery, or a physical weaving model that the code does not implement.
