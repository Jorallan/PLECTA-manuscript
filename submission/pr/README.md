# Submission package: Pattern Recognition

Prepared 2026-11-23 against the journal's Guide for Authors. The manuscript
sources stay in the repository root; `main.tex` is already in the journal's
layout (single column, double spaced, Times 10 pt, 8 pt captions and title
block, margins 4.3/4.8/4.3/4.8 cm, numbered pages). In that layout the article
is 33 pages; the limit is 20–35 including figures, tables, references and
any biographies. Author biographies are not required at submission (the guide
mentions them only in the page limit); add them if the editor asks.

| file | what it is | how it was made |
|---|---|---|
| `cover_letter.docx` | answers the journal's three required questions (state of the art, public datasets, measures) | `python scripts/submission/make_pr_cover_letter.py` |
| highlights, graphical abstract, declaration | reuse `../ivc/highlights.docx`, `../ivc/graphical_abstract.*`, `../ivc/declaration_of_interests.docx`; all meet this journal's limits | — |

## Still open before submitting

References: six recent Pattern Recognition papers are cited in Section 2,
and the article cites 50 references (the journal asks for 35–55).

1. **Research data:** the article and the cover letter state that the manual
   annotations will be deposited in 4TU.ResearchData upon acceptance (the
   journal's Option C allows a statement instead of a link). Deposit them on
   acceptance and replace the sentence with the DOI.
2. **CRediT roles and the declarations-tool document** in the submission
   system.
