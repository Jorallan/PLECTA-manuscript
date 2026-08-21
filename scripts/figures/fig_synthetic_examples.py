"""FIG: thick-bundle / thin-axis synthetic examples at two areal densities.

Columns: thick-bundle GT | synthetic SEM (thick) | degraded thin-axis mask (w=2)
| clean 1px axis.  Rows: low density (cov20) and high density (cov60).
Uses the new sandbox thick-bundle dataset under input/synthetic_thick/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from skimage.color import label2rgb

sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, save_fig

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_gray(path):
    import imageio.v2 as iio
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def colorize(tif_path):
    lab = tifffile.imread(tif_path)
    if lab.ndim == 3:
        lab = lab[0]
    return label2rgb(lab, bg_label=0, bg_color=(0, 0, 0))


def main():
    apply_style()
    rows = [("20% areal density", "input/synthetic_thick/cov20/synth_0000"),
            ("60% areal density", "input/synthetic_thick/cov60/synth_0000")]
    cols = [("Bundle ground truth", "gt_labels.tif", "label"),
            ("Synthetic SEM", "sem.png", "gray"),
            ("Thin-axis mask (w=2)", "mask_w2.png", "gray"),
            ("Clean 1px axis", "mask_clean.png", "gray")]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.7))
    for ri, (rlabel, reldir) in enumerate(rows):
        d = os.path.join(ROOT, reldir)
        for ci, (title, fn, kind) in enumerate(cols):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                print("MISSING", p); continue
            if kind == "label":
                ax.imshow(colorize(p))
            else:
                ax.imshow(load_gray(p), cmap="gray", vmin=0, vmax=255)
            if ri == 0:
                # 8.5 pt rendered at 0.585 page scale.
                ax.set_title(title, fontsize=14.5, fontweight="bold")
            if ci == 0:
                # 8.0 pt rendered at 0.585 page scale; the row label stays the
                # smaller of the two, as in _style.py (title 11, label 10.5).
                ax.set_ylabel(rlabel, fontsize=13.7, fontweight="bold")
    # 9.5 pt rendered at 0.585 page scale (one step above the column titles).
    fig.suptitle("Thick-bundle / thin-axis synthetic scenes with exact ground truth",
                 fontsize=16.2, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig_synthetic_examples", subdir="archive", bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    main()
