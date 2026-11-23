# Submission package: Pattern Recognition

Prepared 2026-11-23 against the journal's Guide for Authors. The manuscript
sources stay in the repository root; `main.tex` is already in the journal's
layout (single column, double spaced, Times 10 pt, 8 pt captions and title
block, margins 4.3/4.8/4.3/4.8 cm, numbered pages). In that layout the article
is 33 pages; the limit is 20–35 including figures, tables, references and
biographies.

| file | what it is | how it was made |
|---|---|---|
| `cover_letter.docx` | answers the journal's three required questions (state of the art, public datasets, measures) | `python scripts/submission/make_pr_cover_letter.py` |
| highlights, graphical abstract, declaration | reuse `../ivc/highlights.docx`, `../ivc/graphical_abstract.*`, `../ivc/declaration_of_interests.docx`; all meet this journal's limits | — |

## Still open before submitting

1. **Author biographies** (about 100 words each) replace the red placeholders
   at the end of `main.tex`.
2. **Pattern Recognition references.** The journal asks for recent work from
   the pattern-recognition field; the article cites none from this journal.
   Keep the total within 35–55 (currently 54).
3. **Research data** must be deposited and linked at submission (the journal
   applies Elsevier's Option C). The 4TU deposit and its DOI are still open;
   the DOI also goes into the cover letter.
4. **Title page:** full postal address of the affiliation.
5. **Generative-AI declaration**, before the references, if AI tools were used
   beyond grammar and spelling checks.
6. **CRediT roles and the declarations-tool document** in the submission
   system.
7. Date the cover letter.
