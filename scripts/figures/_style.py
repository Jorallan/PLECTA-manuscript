"""Shared matplotlib style for paper figures.

Import and call `apply_style()` once per script before creating figures.
Provides a shared colorblind-safe palette and a `save_fig` helper that
writes both PDF (vector) and PNG (300 dpi) into paper/figures/.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette (fixed and semantic; do not cycle or reassign per chart)
# ---------------------------------------------------------------------------
BLUE = "#2b6cb0"    # precision / over-merge
ORANGE = "#dd6b20"  # recall / under-merge
GREEN = "#2f855a"   # F1
GRAY = "#718096"    # neutral / reference
LIGHT_GRAY = "#a0aec0"
DARK = "#1a202c"

PALETTE_PRF = {"precision": BLUE, "recall": ORANGE, "f1": GREEN}

FIGURES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "figures")
)


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.grid": False,
            "figure.dpi": 150,
        }
    )


def save_fig(fig, name, **kwargs):
    """Save fig as both PDF and PNG (300dpi) into paper/figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(pdf_path, **kwargs)
    fig.savefig(png_path, dpi=300, **kwargs)
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")
    return pdf_path, png_path


def thin_spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(0.9)
