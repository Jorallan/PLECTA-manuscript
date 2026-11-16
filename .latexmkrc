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
