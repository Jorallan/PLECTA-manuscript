"""Write every figure the manuscript includes as a separate, numbered file.

Elsevier asks for each figure as its own file, named in the order the article
numbers them.  The numbers are read from ``main.aux`` and the files from the
``figure`` environments in ``sections/``, so the mapping cannot drift from the
built PDF.  Build ``main.tex`` first (latexmk writes ``build/main.aux``).

Matplotlib figures are copied as they are.  The two TikZ structure diagrams
are compiled standalone, with the ``pk`` styles cut verbatim from
``main.tex``'s preamble, so they render exactly as in the article.

    python scripts/submission/export_figures.py [OUTDIR]

OUTDIR defaults to ``submission/ivc/figures``.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECTIONS = ["01_introduction", "01b_related_work", "02_materials_methods",
            "03_results", "04_discussion"]


def figure_sources():
    """label -> ('pdf', path) or ('tikz', path) for every figure environment."""
    out = {}
    for name in SECTIONS:
        text = open(os.path.join(ROOT, "sections", name + ".tex"),
                    encoding="utf8").read()
        for env in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S):
            label = re.search(r"\\label\{([^}]*)\}", env).group(1)
            g = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", env)
            t = re.search(r"\\input\{(figures/[^}]*)\}", env)
            if g:
                out[label] = ("pdf", g.group(1))
            elif t:
                out[label] = ("tikz", t.group(1) + ".tex")
    return out


def figure_numbers():
    aux = open(os.path.join(ROOT, "build", "main.aux"), encoding="utf8").read()
    return {m.group(1): int(m.group(2)) for m in
            re.finditer(r"\\newlabel\{(fig:[^}]*)\}\{\{(\d+)\}", aux)}


def tikz_preamble():
    """The colour and style definitions the diagrams are drawn with."""
    main = open(os.path.join(ROOT, "main.tex"), encoding="utf8").read()
    start = main.index(r"\definecolor{pkink}")
    end = main.index(r"\newcounter{igorcomment}")
    return main[start:end]


def compile_tikz(src, dest):
    body = (r"\documentclass[11pt,border=2pt]{standalone}" "\n"
            r"\usepackage[T1]{fontenc}\usepackage{lmodern}" "\n"
            r"\usepackage{amsmath,amssymb}\usepackage{xcolor}\usepackage{tikz}" "\n"
            r"\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}" "\n"
            r"\newcommand{\citep}[1]{}" "\n"
            + tikz_preamble() +
            "\\begin{document}\n\\input{" + os.path.join(ROOT, src).replace("\\", "/")
            + "}\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as tmp:
        tex = os.path.join(tmp, "fig.tex")
        open(tex, "w", encoding="utf8").write(body)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "fig.tex"], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL)
        shutil.copyfile(os.path.join(tmp, "fig.pdf"), dest)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    outdir = argv[0] if argv else os.path.join(ROOT, "submission", "ivc", "figures")
    os.makedirs(outdir, exist_ok=True)
    sources, numbers = figure_sources(), figure_numbers()
    missing = set(numbers) ^ set(sources)
    if missing:
        raise SystemExit(f"labels without a source or a number: {sorted(missing)}")
    for label, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        kind, src = sources[label]
        dest = os.path.join(outdir, f"Figure_{n}.pdf")
        if kind == "pdf":
            shutil.copyfile(os.path.join(ROOT, src), dest)
        else:
            compile_tikz(src, dest)
        print(f"Figure_{n}.pdf  <- {src}  ({os.path.getsize(dest) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
