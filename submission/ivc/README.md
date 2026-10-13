# Submission package: Image and Vision Computing

Prepared 2026-10-13. Everything here is generated or written for the IVC
submission; the manuscript sources stay in the repository root.

| file | what it is | how it was made |
|---|---|---|
| `highlights.docx` | 5 bullets, 66–72 characters each (limit 85) | written; limit asserted in the script |
| `graphical_abstract.{pdf,tif,png}` | 13 × 5.2 cm (2.5 : 1), 3070 × 1228 px | `python scripts/figures/fig_graphical_abstract.py --ivc submission/ivc` |
| `figures/Figure_N.pdf` | the 13 article figures, one file each, numbered as in the PDF | `python scripts/submission/export_figures.py` (build `main.tex` first) |
| `declaration_of_interests.docx` | the "no competing interests" declaration | mirrors the text of Elsevier's declarations tool |
| `cover_letter.docx` | cover letter from the corresponding author | written; two DOIs still to insert |

Upload alongside these: `main.tex`, `sections/*.tex`, `results/*.tex`,
`figures/fig_pipeline_*.tex`, `references.bib` (or `main.bbl`) and
`supplementary.pdf`.

## Still open before submitting

1. **CRediT roles and competing interests are entered in the submission
   portal**, not in the manuscript (both sections were removed from
   `main.tex` on 2026-10-13). Upload `declaration_of_interests.docx`, or the
   file Elsevier's own tool generates (https://declarations.elsevier.com).
   If the portal does not ask for CRediT roles, the statement must go back
   into the manuscript.
2. **Affiliation postal address.** IVC wants the full postal address on the
   title page; `main.tex` has only the city.
3. **Generative-AI declaration.** Required only if AI tools were used in
   preparing the manuscript; the authors decide. It goes in its own section
   before the references.
4. **Code: archived already.** Zenodo version DOI `10.5281/zenodo.22688093`
   (v0.1.1, 2026-09-10; concept DOI `10.5281/zenodo.22687992`). The 14 later
   commits add only opt-in options, so v0.1.1 reproduces the paper. Cite the
   *version* DOI, or cut a new release first and cite that one.
5. **Data: `4tu_deposit_plecta/` is ready to upload except for one
   question.** Its README's last comment asks who supplied and imaged the CNT
   material and whether that party agrees to CC BY 4.0 for the micrographs.
   Resolve that, deposit, reserve the DOI, and then put both DOIs into the
   Data availability section, `references.bib` (as `[dataset]` and
   `[software]` entries) and the cover letter.

## 4TU.ResearchData: the manual annotations

**Prepare the package first.** Nothing in it may identify annotator B, who
asked to remain unnamed.

- One folder per field, named as in the paper (`B58_100`, `B58_110`,
  `B58_300`), each with `annotator_A/` and `annotator_B/`. The sources are
  under `C:\Repos\datasets\real\manual annotation` (annotator A) and its
  named child folder (annotator B). The raw folders use the instrument names
  (`B58-B3-S2_100`, `S2_110`, `S3_100`); confirm the mapping to the paper's
  names before renaming.
- Per annotator and field: the scorer-readable `*_multilabel.npz` with its
  `*_multilabel_ids.json`, and `*_centerlines.json`.
- **Exclude** `*_history.csv` (an editing log that may carry user names and
  timestamps), `_plecta_tune_tiles/`, `plecta_tuned.yaml` and `nnunet/`.
  Search every remaining file, JSON contents included, for annotator B's
  name.
- The SEM images (`*_original.tif`): include them only if whoever owns the
  samples agrees, because the annotations are much less useful without them.
- A `README.txt`: what each file is, how to load it with NumPy, the pixel
  size (1.08 nm/px; 1.66 µm horizontal field width), the instrument and
  settings from Section 3.2, the licence, and how to cite the dataset.
- Consider depositing the locked synthetic set and the per-scene records in
  the same deposit, or a second one. IVC's data policy (Option C) prefers
  deposit to "available on request".

**Deposit:**

1. Go to https://data.4tu.nl, log in, and choose University of Twente
   (institutional login).
2. Start a new dataset from your dashboard.
3. Metadata: title, both authors with ORCID, description, keywords,
   licence (CC BY 4.0 suggested), funding (TKI-HTSM / PPS-I, 24PPS125).
   Add the related publication once it has a DOI.
4. Upload the files, zipped per field if convenient.
5. **Reserve the DOI before publishing**, so the manuscript can cite it now.
6. Submit. 4TU's data stewards review a deposit before it goes public, so
   allow a few working days. If the data must stay private until
   acceptance, ask 4TU support about an embargo or a reviewer link before
   submitting. The UT library's data stewards can also help.

## Zenodo: the code

1. Make sure `Jorallan/PLECTA` is public and that the commit you release
   carries the frozen `plecta/params.json` the paper's numbers were produced
   with. `CITATION.cff` and `LICENSE` are already in the repository.
2. At https://zenodo.org, log in with GitHub.
3. In Zenodo, go to Account → GitHub, press *Sync*, and switch
   `Jorallan/PLECTA` on.
4. On GitHub, create a release, for example tag `v1.0.0`, titled "PLECTA
   1.0.0, as submitted". Zenodo archives it and mints a DOI within minutes.
5. Open the new Zenodo record and complete the metadata: both authors with
   ORCID, description, licence, the grant in the description.
6. Cite the **version DOI**, not the concept DOI, which always resolves to the
   newest release.
7. Optionally repeat for `PLECTA_APP` and `ImageAnnotator`, which the Data
   availability section also lists.
