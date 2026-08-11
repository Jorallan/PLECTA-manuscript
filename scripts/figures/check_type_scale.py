"""Verify the shared type scale, by measurement rather than by assertion.

Each figure generator is imported and run.  ``Figure.savefig`` is intercepted,
so the *figure object that is actually written* is inspected: every ``Text``
artist that carries visible characters is asked for its point size, and tick
labels are read after a draw so that the size matplotlib resolved -- not the
size the script asked for -- is the one reported.

Two things are checked:

  * the figure is exactly ``FIG_W`` wide, so the LaTeX include scale is 1.0 and
    a label specified at n pt prints at n pt;
  * no drawn character is below ``MIN_PT``.

    python scripts/figures/check_type_scale.py
"""
from __future__ import annotations

import importlib
import io
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

from _style import FIG_W, MIN_PT

MODULES = [
    ("fig_pipeline_scope", 1),
    ("fig_plecta_task", 2),
    ("fig_plecta_frame_cost", 3),
    ("fig_plecta_gates", 4),
    ("fig_plecta_exact_matching", 5),
    ("fig_plecta_density", 6),
    ("fig_comparators_density", 7),
    ("fig_plecta_examples", 8),
    ("fig_plecta_sensitivity", 9),
    ("fig_plecta_performance", 10),
]


def sizes(fig):
    """Point size of every Text artist that will put ink on the page."""
    fig.canvas.draw()
    found = []
    for artist in fig.findobj(Text):
        if not artist.get_visible():
            continue
        text = artist.get_text()
        if not text or not text.strip():
            continue
        found.append((round(float(artist.get_fontsize()), 2), text.strip()))
    return found


def main() -> int:
    captured = {}
    original = Figure.savefig

    def spy(self, fname, *a, **kw):
        if isinstance(fname, str) and fname.endswith(".pdf"):
            captured[os.path.basename(fname)[:-4]] = (
                sizes(self), float(self.get_size_inches()[0]))
        return original(self, fname, *a, **kw)

    Figure.savefig = spy
    worst = 0.0
    failures = []
    rows = []
    try:
        for name, number in MODULES:
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
            for figname, (found, width_in) in captured.items():
                counts = Counter(s for s, _ in found)
                smallest = min(counts) if counts else float("nan")
                rows.append((number, figname, smallest, width_in,
                             sorted(counts.items(), reverse=True)))
                if counts and smallest < MIN_PT - 1e-9:
                    below = sorted({(s, t) for s, t in found if s < MIN_PT})
                    failures.append((figname, below[:6]))
                if abs(width_in - FIG_W) > 1e-3:
                    failures.append((figname,
                                     [(width_in, "figure width, inches")]))
                worst = max(worst, 0.0)
    finally:
        Figure.savefig = original

    print(f"{'fig':>4}  {'name':<28} {'min pt':>7} {'width in':>9}   "
          f"sizes used (pt x count)")
    for number, figname, smallest, width_in, counts in sorted(rows):
        used = "  ".join(f"{s:g}x{n}" for s, n in counts)
        print(f"{number:>4}  {figname:<28} {smallest:>7.2f} {width_in:>9.4f}"
              f"   {used}")

    print(f"\nfloor {MIN_PT} pt;  \\linewidth = {FIG_W:.4f} in = "
          f"{FIG_W * 72:.2f} bp;  every figure is drawn at that width, so "
          f"width=\\linewidth includes at scale 1.0 and n pt prints as n pt")
    if failures:
        print("\nBELOW THE FLOOR:")
        for figname, below in failures:
            for size, text in below:
                print(f"  {figname}: {size} pt  {text!r}")
        return 1
    print("\nall figures clear the floor")
    return 0


def _takes_argv(module):
    import inspect
    try:
        return bool(inspect.signature(module.main).parameters)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
