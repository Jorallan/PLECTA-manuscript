# Reference notes

One line per citation key: what the work says, and what claim in the paper it can support.
All 45 entries in `references.bib` were verified via web search and/or Crossref lookup
(title, authors, venue, volume/pages, and DOI/URL cross-checked against at least one
primary source such as the publisher page, PubMed, or Crossref metadata).

## 1. CNT / nanofiber / nanowire microscopy image analysis & tools

- **hotaling2015diameterj** — DiameterJ: ImageJ/FIJI plugin that segments SEM micrographs of nanofibers with 24 algorithms and extracts diameter distributions in ~30s/image. Supports the "automated quantification tools exist for nanofiber diameter, but not for instance-level bundle/filament topology" framing, and is a natural comparison point for the postprocess width-measurement stage.
- **stein2008fire** — FIRE: distance-transform + nucleation-point network extraction algorithm for 3D collagen gel confocal images, validated against known network mechanics. Prior art for "vectorize a binary/intensity mask into a fiber network" — directly analogous to this pipeline's stringart/vectorization stage, but for a different modality.
- **bredfeldt2014ctfire** — CT-FIRE: combines curvelet-transform denoising with the FIRE fiber-tracking algorithm to extract individual collagen fibers (length, width, angle, curvature) from SHG microscopy images. Supports the claim that curvelet/Hough-style transforms are an established route to converting fibrous image intensity into per-fiber primitives before tracking.
- **xu2015soax** — SOAX: open-source tool using Stretching Open Active Contours (SOACs) to extract centerlines/junctions of 2D/3D biopolymer networks, explicitly designed to resolve crossings/junctions. Strong precedent for this pipeline's "reconnect" stage — SOACs merging/reconfiguring at junctions is the closest published analogue to the tip-tangent/collinearity reconnect logic.
- **dong2022detecting** — Applies topological data analysis (persistent-homology barcodes) to SEM images to quantify CNT bundle orientation/alignment fraction, validated against Raman/XRD Herman's orientation factor (R²=0.97). Supports claims about why CNT orientation/alignment quantification from SEM is materially important and methodologically nontrivial.
- **imtiaz2023facile** — 2D-FFT + total-variation image decomposition method to estimate CNT film alignment (nematic order parameter) directly from SEM/optical images. Another concrete precedent for image-based CNT alignment quantification, useful as a lighter-weight baseline contrast to this pipeline's explicit instance-level approach.
- **hajilounezhad2021predicting** — CNTNet: physics-based simulated CNT forest SEM images + deep learning (classification + regression) to predict density/diameter/growth attributes and mechanical stiffness/buckling load. Supports both the CNT-materials-context claim (image attributes → mechanical properties) and the synthetic-ground-truth claim (simulated imagery as a substitute for exhaustive manual annotation).
- **nguyen2023cntsegnet** — CNTSegNet: self-supervised, orientation-guided U-Net-style network for segmenting individual CNTs in dense, crossing/occluding CNT forest SEM imagery. Directly relevant prior art for "instance-level extraction of a single CNT/bundle from a crossing-dense SEM image" — the deep-learning alternative to this pipeline's Hough+reconnect approach.
- **trujillo2017segmentation** — Two classical-CV + ANN pipelines (matched-filter-bank and Perona-Malik/Otsu preprocessing) for segmenting SEM images of carbon nanotubes. Earlier precedent (pre-U-Net era) for CNT-specific image segmentation, useful for framing the evolution of the field.
- **dedonato2025deep** — U-Net-based segmentation of SEM images of ZnO nanoparticles and graphene-oxide nanosheets, with a purpose-built annotated dataset; >95% accuracy/recall. Supports the general claim that deep segmentation is now routinely applied across diverse nanomaterial SEM modalities, not just CNTs.

## 2. Filament / curvilinear structure extraction (general)

- **frangi1998vesselness** — The canonical Hessian-eigenvalue "vesselness" filter for enhancing tubular/curvilinear structures in medical images. Standard citation for any Hessian-based ridge/tube enhancement discussion, and a natural point of contrast with this pipeline's Hough-based line primitive extraction.
- **steger1998unbiased** — Subpixel-accurate curvilinear line detector that explicitly models line width/asymmetry to avoid positional bias. Relevant to justify/contextualize any subpixel centerline or width-measurement claims in the postprocess stage.
- **sato1998linefilter** — Independent, contemporaneous Hessian-eigenvalue multiscale line filter for 3D curvilinear structures (vessels, bronchi). Companion citation alongside Frangi for "Hessian/ridge-based" tubular structure enhancement.

## 3. Hough transform

- **hough1962patent** — Original Hough transform patent (machine recognition of complex line patterns in particle-track photographs). Historical/foundational citation for the Hough transform itself.
- **duda1972hough** — Duda & Hart's angle-radius (ρ,θ) reformulation of the Hough transform for line/curve detection — the version actually used in modern implementations. Primary citation for "Hough transform" as used in the stringart/vectorization stage.
- **kiryati1991probabilistic** — Introduces randomly subsampling edge points before voting (probabilistic Hough transform), showing large speedups with little accuracy loss. Precursor to PPHT; supports discussion of why a probabilistic/tiled Hough variant is used instead of the full deterministic transform.
- **matas2000ppht**  — Progressive Probabilistic Hough Transform (PPHT): interleaves voting and detection, minimizing computation and directly outputting line segments (not just infinite lines) — this is the OpenCV `HoughLinesP` algorithm. Primary citation for the "probabilistic Hough transform" used to vectorize the segmentation mask into oriented line/branch primitives.

## 4. Skeletonization / thinning / medial axis

- **blum1967transformation** — Introduces the medial axis transform (grassfire analogy) as a shape descriptor — the foundational concept behind skeletonization. Root citation for "skeleton"/"medial axis" terminology used in the postprocess stage.
- **zhang1984thinning** — Classic two-subiteration parallel thinning algorithm reducing a binary shape to a unit-width skeleton while preserving topology/endpoints. Standard citation for 2D skeletonization as used to derive filament centerlines.
- **lee1994skeleton** — 3D parallel medial-surface/medial-axis thinning algorithm with a topology-preserving Euler-characteristic table; this is the algorithm behind `skimage.morphology.skeletonize` (Lee's method) commonly used in Python image-processing pipelines. Direct citation if the postprocess skeletonization step uses this specific method.

## 5. Deep learning segmentation used in microscopy

- **ronneberger2015unet** — U-Net: encoder-decoder CNN with skip connections for biomedical image segmentation from few annotated images; foundational architecture for essentially all subsequent biomedical segmentation work. Baseline citation for any discussion of deep-learning alternatives to the classical CV pipeline described in the paper.
- **isensee2021nnunet** — nnU-Net: self-configuring U-Net variant that automatically adapts preprocessing/architecture/training to a new dataset and outperforms specialized methods across 23 segmentation challenges. Supports discussion of state-of-the-art deep segmentation as an alternative/complement to the binary-mask input this pipeline consumes.
- **stringer2021cellpose** — Cellpose: generalist deep instance-segmentation model for cells, trained on >70,000 varied annotated objects, requiring no retraining across image types. Relevant comparison for "generalist instance segmentation" approaches versus this pipeline's geometry-driven instance formation (reconnect stage).
- **schmidt2018stardist** — StarDist: predicts star-convex polygons per pixel for direct instance segmentation of round/convex objects (nuclei), avoiding a separate detection+refinement step. Useful contrast case: illustrates why a star-convex/blob-shaped instance representation is unsuited to elongated, crossing filament instances (motivating this pipeline's different approach).
- **kirillov2023sam** — Segment Anything Model (SAM): promptable, zero-shot foundation model for image segmentation trained on >1B masks. Relevant for discussing whether/why general-purpose foundation segmentation models are (not) well suited to thin, crossing, low-SNR filament instances in SEM.

## 6. Instance segmentation & tracing of overlapping/crossing curvilinear objects

- **brown2011diadem** — The DIADEM data sets: benchmark light-microscopy neuron images released to drive automation of digital axon/dendrite reconstruction. Precedent for benchmark curation methodology in a curvilinear-tracing domain structurally similar to CNT bundle tracing.
- **gillette2011diademmetric** — The DIADEM metric: compares two tree reconstructions of the same neuron by matching bifurcation/termination topology, tolerant to different numbers of points along a path. Directly relevant prior art for topology-aware evaluation of traced curvilinear instances — a domain-specific alternative/complement to the pairwise-clustering metrics (ARI/VI/PQ) used in this paper's eval subsystem.
- **xiao2013app2** — APP2: automatic 3D neuron tracing via hierarchical (long-segment-first) pruning of a gray-weighted image distance tree, built on a fast-marching distance transform; implemented as an open-source Vaa3D plugin. Relevant precedent for tree/graph-pruning-based tracing of branching curvilinear structures from a distance transform, conceptually related to skeleton pruning in the postprocess stage.
- **pound2013rootnav** — RootNav: semi-automated tool that fits a root-system model to top-down plant images via an EM-based likelihood approach plus interactive path refinement ("satnav for roots"). Another domain example of instance-level tracing of a branching/crossing curvilinear network from 2D imagery.
- **yang2018crack** — Fully convolutional network for pixel-level crack detection AND measurement (width/length) from images, evaluated with precision/recall/F1. Relevant for the postprocess width-measurement claims — an independent domain (civil infrastructure) doing SEM-analogous curvilinear-defect segmentation + geometric measurement.

## 7. Evaluation metrics

- **rand1971objective** — Introduces the Rand index: pairwise-agreement measure between two partitions of the same set. Base citation before introducing the chance-corrected Adjusted Rand Index used in the eval subsystem.
- **hubert1985comparing** — Introduces/derives the Adjusted Rand Index (ARI), correcting the Rand index for chance agreement. Primary citation for the ARI metric used in the paper's fragment-clustering evaluation.
- **meila2007comparing** — Introduces Variation of Information (VI): an information-theoretic metric (true distance metric on the space of clusterings) for comparing two clusterings/partitions. Primary citation for the VI metric used in the eval subsystem.
- **unnikrishnan2007toward** — Surveys/formalizes the problem of objectively evaluating image segmentations against multiple human ground truths, given that segmentation is inherently ill-posed (no unique correct answer). Supports methodological framing for why multiple complementary metrics (F1, ARI, VI, PQ) are reported rather than a single accuracy number.
- **martin2001database** — Introduces the Berkeley Segmentation Dataset (BSDS) and an error measure tolerant of differing segmentation granularity. Companion citation to Unnikrishnan et al. for the broader "how do we evaluate segmentation against human annotation" literature.
- **kirillov2019panoptic** — Introduces Panoptic Quality (PQ) = Segmentation Quality × Recognition Quality, unifying detection and segmentation-quality evaluation for instance+semantic segmentation. Primary citation for the PQ metric used in the paper's evaluation.
- **amigo2009comparison** — Defines formal constraints (cluster homogeneity/completeness, rag bag, size-vs-quantity) that a good clustering-evaluation metric should satisfy, and shows BCubed is the only common metric satisfying all of them; also covers pairwise F1 in this context. Relevant citation for the pairwise-F1 / precision-recall decomposition (over-merge vs. under-merge) used as the paper's primary fragment-clustering metric.

## 8. Curve fitting / splines / smoothing

- **dierckx1993curve** — The reference monograph for the FITPACK B-spline fitting library (smoothing splines, automatic/adaptive knot placement), underlying `scipy.interpolate.splprep`/`UnivariateSpline`. Primary citation if the postprocess centerline-smoothing stage uses FITPACK-derived spline fitting.
- **savitzky1964smoothing** — Introduces Savitzky-Golay filtering: local polynomial least-squares smoothing/differentiation via convolution. Primary citation if centerline smoothing or derivative (tangent/curvature) estimation uses Savitzky-Golay filters.

## 9. CNT materials context (why geometry matters; DEM simulation)

- **devolder2013cnt** — Broad review of CNT commercial applications (yarns, sheets, composites, electronics), summarizing how synthesis/processing controls bulk CNT material performance. Supports the introduction's motivation for why extracting individual bundle/filament geometry from SEM matters for downstream materials applications.
- **coleman2006small** — Review of mechanical reinforcement of polymers by CNTs, and how dispersion/alignment/aspect ratio govern composite mechanical properties. Supports the specific claim that bundle geometry and alignment (not just CNT content) control mechanical performance — directly motivates why per-instance geometry extraction (not just areal density) is the right measurement target.
- **volkov2010structural** — Mesoscopic (bead-and-spring / distinct-element) simulation showing bending buckling governs the structural stability of low-density CNT films. Example of a DEM-type simulation that consumes network/bundle geometry as input — supports the "this pipeline exports geometry for DEM" claim.
- **ostanin2013distinct** — Develops a distinct element method (DEM) for large-scale simulation of CNT assemblies, explicitly built to take realistic CNT network/bundle geometry as simulation input. Directly supports the DEM-export motivation for this pipeline: DEM simulations require exactly the per-instance geometric primitives (position, orientation, length, connectivity) this pipeline produces.
- **wittmaack2018mesoscopic** — Mesoscopic DEM modeling of the self-organization of CNTs into vertically aligned bundle networks, generating computational samples with controllable alignment/bundling statistics. Further supports the DEM-simulation motivation and the more general claim that CNT bundle formation/geometry is an active simulation research topic that benefits from measured (not just simulated) geometric input.

## 10. Synthetic ground-truth generation for validating microscopy image analysis

- **wiesmann2017using** — Presents a method to simulate realistic fluorescence cell micrographs with known ground truth, validated (via observer study and matched segmentation-pipeline performance) as a substitute for laborious manual annotation when benchmarking segmentation algorithms. Primary supporting citation for this paper's synthetic ground-truth generator used alongside real manually-annotated SEM crops.
  - (See also **hajilounezhad2021predicting** above, which uses physics-simulated CNT forest imagery with known ground truth for the same purpose in the CNT domain specifically.)

---

## References sought but NOT verified (omitted from references.bib)

The following were searched for because they are natural fits for the requested topic areas,
but I could not confirm a single, specific, correctly-attributed paper/tool with enough
confidence to cite. Rather than guess authors/venues/years, they were left out entirely.

- **A DiameterJ-equivalent tool specifically for nanowire (as opposed to nanofiber/collagen)
  network quantification.** Searches turned up many individual studies that use ImageJ/generic
  tools to measure nanowire dimensions from SEM, and general percolation/transparency-based
  characterization of silver nanowire networks, but no single canonical, widely-cited
  "nanowire network quantification algorithm" paper analogous to DiameterJ/FIRE/SOAX for
  fibers. If your paper needs this specific claim, it likely needs a narrower, more targeted
  search than time allowed here.
- **A specific, single canonical paper for Monte-Carlo/DEM simulation of CNT network electrical
  percolation that takes measured (not simulated) network geometry as input.** Multiple
  relevant papers on CNT percolation simulation appeared (Monte Carlo stick/rectangle models,
  3D KD-tree-based conductive path search), but none stood out as *the* standard reference,
  and I could not confirm authorship/venue details to my verification bar in the time
  available. The DEM-mechanics side of this topic is well covered instead by
  volkov2010structural / ostanin2013distinct / wittmaack2018mesoscopic.
- **RootNav 2.0** (Yasrab et al., "RootNav 2.0: Deep learning for automatic navigation of
  complex plant root architectures," GigaScience, 2019) — this one I *did* find and could
  likely verify with one more lookup, but chose to omit it as redundant with pound2013rootnav
  for the purposes of this bibliography (both are root-tracing citations; only one was needed
  per the requested scope). Not a fabrication risk — just left out for parsimony. Mentioning
  here in case the deep-learning-era RootNav update is specifically wanted.
