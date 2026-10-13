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
