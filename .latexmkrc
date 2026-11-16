# Local MiKTeX builds only. Overleaf and other TeX Live systems also read this
# file, and the BibTeX option below exists only in MiKTeX, so everything is
# guarded and other systems build with their defaults.
if (`bibtex --version 2>&1` =~ /MiKTeX/) {
    # By-products (.aux, .log, .bbl, ...) go to build/, which .gitignore
    # excludes; the PDFs stay beside their sources.
    $aux_dir = 'build';
    $out_dir = '.';
    # MiKTeX's BibTeX looks beside the .aux file, not here, and would otherwise
    # pick up a same-named references.bib shipped with a MiKTeX package.
    use Cwd;
    my $root = getcwd();
    $bibtex = "bibtex --include-directory=\"$root\" %O %B";
}

# Overleaf shows the PDF of its main document only. After every successful
# build of main.tex, also build the supplement as supplementary-preview.pdf,
# which Overleaf lists under "Logs and output files" -> "Other files". It reads
# the article's numbers from the .aux just written (%R, Overleaf's job name).
# PLECTA_NESTED stops the nested build from starting another one. Not done
# under MiKTeX, where both documents are built by hand.
if (!$ENV{PLECTA_NESTED} && `bibtex --version 2>&1` !~ /MiKTeX/) {
    $success_cmd = 'case %T in *main.tex*) PLECTA_NESTED=1 latexmk -pdf '
        . '-interaction=nonstopmode -jobname=supplementary-preview '
        . '-usepretex="\\def\\plectamainaux{%R}" supplementary.tex '
        . '>/dev/null 2>&1 || true;; esac';
}

# supplementary.tex reads the article's section numbers from main.aux (xr).
# Overleaf compiles one root file at a time, so main.aux may not exist when the
# supplement is built; this rule builds main.tex first whenever it is missing.
# The root check stops the article's own build from recursing.
add_cus_dep('tex', 'aux', 0, 'makeexternaldocument');
sub makeexternaldocument {
    if (!($root_filename eq $_[0])) {
        system("latexmk -cd -pdf \"$_[0]\"");
    }
}
