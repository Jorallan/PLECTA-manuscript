# Build by-products (.aux, .log, .bbl, ...) go to build/, which .gitignore
# excludes; the PDFs stay beside their sources. supplementary.tex reads the
# article's section numbers from build/main.aux, so build main.tex first.
$aux_dir = 'build';
$out_dir = '.';
# MiKTeX's BibTeX looks beside the .aux file, not here, and would otherwise
# pick up a same-named references.bib shipped with a MiKTeX package.
use Cwd;
my $root = getcwd();
$bibtex = "bibtex --include-directory=\"$root\" %O %B";
