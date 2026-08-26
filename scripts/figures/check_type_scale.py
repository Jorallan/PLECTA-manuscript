"""Verify the shared type scale, from the glyphs actually written to the PDF.

Every generator is run, and then the PDF each one wrote is opened, its content
streams inflated, and every ``/F<n> <size> Tf`` operator read.  That is the
size a glyph is *set at on the page*, which is the only measurement that
catches the failure this check exists for: matplotlib mathtext draws a sub- or
superscript at a fraction of the base size, so a label asked for at 7.5 pt can
put a 5.25 pt glyph on the page.  An earlier version of this script asked the
``Text`` artists what size they had been given, and reported the whole set
clean while eight of the ten figures were carrying 5.25 pt glyphs.

Three things are checked, and any one of them failing exits non-zero:

  * no glyph is written below ``MIN_PT`` -- read from the PDF, authoritative;
  * the page is exactly ``FIG_W`` wide -- read from the PDF ``/MediaBox``, so
    the LaTeX include scale is 1.0 and a label set at n pt prints at n pt;
  * the figure object agrees, as a second opinion that can also *name* the
    offending label, which a content stream cannot.

    python scripts/figures/check_type_scale.py
    python scripts/figures/check_type_scale.py --self-test

``--self-test`` puts the defect back (mathtext's stock 0.70 script ratio),
renders into a scratch directory rather than ``figures/``, and passes only if
this script fails.  A gate nobody has watched fail is not a gate.
"""
from __future__ import annotations

import importlib
import io
import os
import re
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

import _style
from _style import FIG_W, MIN_PT, SCRIPT_SHRINK

#: The figures the manuscript includes, in the order it includes them.
#: ``fig_metric_panels`` used to write two of them, the degraded and the clean
#: mask condition; they are now the two rows of one figure, which is why the
#: numbering below closes up after it.
SHIPPED = [
    ("fig_pipeline_scope", 1),
    ("fig_plecta_frame_cost", 2),
    ("fig_plecta_exact_matching", 3),
    ("fig_depth_evidence", 4),
    ("fig_depth_order", 5),
    ("fig_f1_example", 6),
    ("fig_metric_panels", 7),
    ("fig_plecta_examples", 8),
    ("fig_real_sem_axis", 9),
    ("fig_real_sem", 10),
    ("fig_pipeline_upstream", 11),
    ("fig_resem_synth", 12),
    ("fig_resem_real", 13),
]


#: Generators no ``\includegraphics`` reaches any more, kept because each was
#: once a figure and may be wanted again: fig_comparators drew pairwise F1
#: alone, which fig_metric_panels now draws beside the other two measures, and
#: fig_graft_regime a second generator under two of them; fig_plecta_density and
#: fig_plecta_sensitivity both drew a coverage response now reported in the
#: text, the second of them a null; fig_comparators_density carried a third of
#: what fig_comparators carried; fig_plecta_control and fig_plecta_performance
#: both drew the width negative control; fig_plecta_task predates
#: fig_common_fragment_metric; and fig_real_sem_overlay is the three-panel
#: centreline variant of fig_real_sem_axis, keeping the annotation beside the
#: two reconstructions, which at this size is near-indistinguishable from
#: PLECTA on the manual-derived axis.  They are still rendered and still gated, so a
#: shared-style change cannot quietly break one of them, but they are numbered
#: apart so the table above never suggests the manuscript has fifteen figures.
SUPERSEDED = [
    ("fig_common_fragment_metric", 100),
    ("fig_plecta_gates", 105),
    ("fig_plecta_task", 101),
    ("fig_comparators_density", 102),
    ("fig_plecta_performance", 103),
    ("fig_plecta_control", 104),
    ("fig_comparators", 106),
    ("fig_graft_regime", 107),
    ("fig_plecta_density", 108),
    ("fig_plecta_sensitivity", 109),
    ("fig_real_sem_overlay", 110),
]

MODULES = SHIPPED + SUPERSEDED

TF = re.compile(rb"/[A-Za-z0-9]+\s+([0-9]*\.?[0-9]+)\s+Tf")
STREAM = re.compile(rb"stream\r?\n(.*?)endstream", re.S)
MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([-0-9.]+)\s+([-0-9.]+)\s+"
                      rb"([-0-9.]+)\s+([-0-9.]+)\s*\]")
SCRIPT = re.compile(r"[_^]")
TOL = 0.01


# ── the authoritative measurement: what is written into the PDF ────────────


def pdf_glyph_sizes(path):
    """Every font size set by a ``Tf`` operator in the file, in points."""
    data = open(path, "rb").read()
    sizes = []
    for raw in STREAM.findall(data):
        try:
            body = zlib.decompress(raw)
        except zlib.error:
            body = raw
        sizes += [round(float(m), 2) for m in TF.findall(body)]
    return sizes


def pdf_width_in(path):
    """Page width in inches, from ``/MediaBox`` (PDF units are 1/72 in)."""
    data = open(path, "rb").read()
    m = MEDIABOX.search(data)
    if m is None:
        return None
    x0, _, x1, _ = (float(v) for v in m.groups())
    return (x1 - x0) / 72.0


# ── the second opinion: what the figure object was asked for ───────────────


def artist_sizes(fig):
    """(effective pt, text) for every Text artist that will put ink down.

    ``effective`` folds in the mathtext script ratio, so a label written
    ``$C_{ab}$`` at 7.5 pt is reported at the size its subscript is really set
    at.  This is what lets a violation be traced back to a label; the content
    stream knows the size but not the string.
    """
    fig.canvas.draw()
    found = []
    for artist in fig.findobj(Text):
        if not artist.get_visible():
            continue
        text = artist.get_text()
        if not text or not text.strip():
            continue
        size = float(artist.get_fontsize())
        if "$" in text and SCRIPT.search(text):
            size *= _current_script_shrink()
        found.append((round(size, 2), text.strip()))
    return found


def _current_script_shrink():
    import matplotlib._mathtext as _mathtext
    return float(getattr(_mathtext, "SHRINK_FACTOR", SCRIPT_SHRINK))


# ── the gate ───────────────────────────────────────────────────────────────


def run_gate(modules=MODULES, quiet=False) -> int:
    captured = {}
    original = Figure.savefig

    def spy(self, fname, *a, **kw):
        result = original(self, fname, *a, **kw)
        if isinstance(fname, str) and fname.endswith(".pdf"):
            captured[os.path.basename(fname)[:-4]] = (
                artist_sizes(self), float(self.get_size_inches()[0]), fname)
        return result

    Figure.savefig = spy
    rows, failures = [], []
    try:
        for name, number in modules:
            captured.clear()
            module = importlib.import_module(name)
            buf, err = io.StringIO(), io.StringIO()
            keep = (sys.stdout, sys.stderr)
            sys.stdout, sys.stderr = buf, err
            try:
                module.main([]) if _takes_argv(module) else module.main()
            finally:
                sys.stdout, sys.stderr = keep
                plt.close("all")

            for figname, (found, fig_w, path) in captured.items():
                written = pdf_glyph_sizes(path)
                page_w = pdf_width_in(path)
                smallest = min(written) if written else float("nan")
                rows.append((number, figname, smallest, page_w, fig_w,
                             sorted(set(written))))

                if not written:
                    failures.append((figname, "no text found in the PDF"))
                elif smallest < MIN_PT - TOL:
                    failures.append(
                        (figname, "%.2f pt glyph written; the floor is %.1f pt"
                         % (smallest, MIN_PT)))
                    blame = sorted({(s, t) for s, t in found
                                    if s < MIN_PT - TOL})
                    for size, text in blame[:6]:
                        failures.append((figname, "    %.2f pt  %r"
                                         % (size, text)))
                    if not blame:
                        failures.append(
                            (figname, "    no artist accounts for it; look "
                                      "for a mathtext script or an embedded "
                                      "image"))
                if page_w is None:
                    failures.append((figname, "no /MediaBox in the PDF"))
                elif abs(page_w - FIG_W) > 1e-3:
                    failures.append(
                        (figname, "page is %.4f in wide, \\linewidth is "
                                  "%.4f in; the include scale is not 1.0"
                         % (page_w, FIG_W)))
                if abs(fig_w - FIG_W) > 1e-3:
                    failures.append(
                        (figname, "figure object is %.4f in wide" % fig_w))
    finally:
        Figure.savefig = original

    if not quiet:
        print(f"{'fig':>4}  {'name':<28} {'min pt':>7} {'page in':>8} "
              f"{'fig in':>7}   sizes written (pt)")
        for number, figname, smallest, page_w, fig_w, written in sorted(rows):
            used = "  ".join(f"{s:g}" for s in written)
            print(f"{number:>4}  {figname:<28} {smallest:>7.2f} "
                  f"{(page_w if page_w else float('nan')):>8.4f} "
                  f"{fig_w:>7.4f}   {used}")

        print(f"\nfloor {MIN_PT} pt, measured from the PDF content streams;  "
              f"mathtext script ratio {_current_script_shrink():.3f} "
              f"(7.5 -> {7.5 * _current_script_shrink():.2f} pt, 8.5 -> "
              f"{8.5 * _current_script_shrink():.2f} pt);  \\linewidth = "
              f"{FIG_W:.4f} in = {FIG_W * 72:.2f} bp, and every page is that "
              f"wide, so width=\\linewidth includes at scale 1.0 and n pt "
              f"prints as n pt")

        if failures:
            print("\nFAILED:")
            for figname, message in failures:
                print(f"  {figname}: {message}")
        else:
            print("\nevery glyph written to every figure clears the floor")
    return 1 if failures else 0


# ── proving the gate can fail ──────────────────────────────────────────────


def self_test() -> int:
    """Put the defect back and require this script to catch it."""
    scratch = tempfile.mkdtemp(prefix="type_scale_selftest_")
    figures_dir, pin = _style.FIGURES_DIR, _style._pin_mathtext_script_size
    import matplotlib._mathtext as _mathtext
    try:
        _style.FIGURES_DIR = scratch          # never touch figures/
        _style._pin_mathtext_script_size = lambda: setattr(
            _mathtext, "SHRINK_FACTOR", 0.70)  # the original defect, exactly
        print("self-test: mathtext script ratio forced back to 0.70, "
              f"rendering into {scratch}")
        status = run_gate([("fig_plecta_frame_cost", 3),
                           ("fig_plecta_gates", 4)])
    finally:
        _style.FIGURES_DIR = figures_dir
        _style._pin_mathtext_script_size = pin
        pin()
    if status == 0:
        print("\nSELF-TEST FAILED: the 5.25 pt defect was not caught")
        return 1
    print("\nself-test passed: the gate exits non-zero on a 5.25 pt glyph")
    return 0


def _takes_argv(module):
    import inspect
    try:
        return bool(inspect.signature(module.main).parameters)
    except (TypeError, ValueError):
        return False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return self_test()
    return run_gate()


if __name__ == "__main__":
    raise SystemExit(main())
