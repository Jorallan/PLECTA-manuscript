"""The two input axes, as PLECTA is handed them.

Four panels: the manual-derived axis and the nnU-Net axis of each held-out
field, drawn over the micrograph. These are the binary masks themselves and not
any reconstruction, so there is nothing to tell apart within a panel and one
colour is the right number. What the figure is for is the comparison *between*
the two axes of a field, which is where the difference between the two
conditions of Table 5 comes from: the same filaments, read a second time by a
network rather than by a person, arrive broken into more pieces and displaced by
a pixel or two almost everywhere.

It is placed before the reconstruction figure because it shows the input to it.

    python scripts/figures/fig_real_sem_axis.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _style import (FIG_W, INK, PLECTA, PT_TITLE, blank_rgb,      # noqa: E402
                    framed, paint, plecta_style, save_fig, tagged_title)
from fig_real_sem import (ASSET, DRAW_DILATION, dilate,           # noqa: E402
                          field_label, micrograph_ground)


def draw_axis(ax, sem: np.ndarray, mask: np.ndarray) -> None:
    """One binary axis over the micrograph, in a single colour."""
    image = micrograph_ground(sem)
    paint(image, dilate(mask, DRAW_DILATION), PLECTA)
    ax.imshow(image, interpolation="nearest")


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    #  One field, not both. What this figure has to show is the difference
    #  between two mask sources, which one field demonstrates; carrying the
    #  second doubles a tall figure to repeat a point Table 5 already makes
    #  numerically for every field. The reconstruction figure is where two
    #  fields earn their space, because there the claim is that the result is
    #  not one lucky field.
    fields = list(meta["fields"])[:1]

    #  Two columns per field would give four across the page; instead the fields
    #  are rows, so each panel keeps the width that makes a one-pixel axis
    #  readable and the two axes of a field sit side by side, which is the
    #  comparison the figure exists to make.
    pad, gap, row_gap_in, label_frac = 0.005, 0.020, 0.075, 0.016
    left = pad + label_frac
    panel = (1.0 - pad - left - gap) / 2.0
    panel_in = panel * FIG_W
    crop_h, crop_w = data[f"{fields[0]}__sem"].shape[:2]
    panel_h_in = panel_in * crop_h / crop_w
    fig_h = len(fields) * panel_h_in + (len(fields) - 1) * row_gap_in + 0.19
    fig = plt.figure(figsize=(FIG_W, fig_h))

    for r, field in enumerate(fields):
        sem = data[f"{field}__sem"]
        bottom_in = (len(fields) - 1 - r) * (panel_h_in + row_gap_in)
        for column, (tag, title, key) in enumerate(
                (("a", "Manual-derived axis", "mask_manual"),
                 ("b", "nnU-Net axis", "mask_unet"))):
            ax = fig.add_axes([left + column * (panel + gap),
                               bottom_in / fig_h, panel, panel_h_in / fig_h])
            framed(ax)
            draw_axis(ax, sem, data[f"{field}__{key}"])
            if r == 0:
                #  A bold "(a)" at 8.5 pt is about 0.21 in, which is 0.07 of a
                #  3.0 in panel; below that the tag prints on the first word.
                tagged_title(ax, tag, title, dy=1.02, gap=0.078)
        fig.text(left - 0.008, (bottom_in + panel_h_in / 2) / fig_h,
                 field_label(field), rotation=90, ha="right", va="center",
                 fontsize=PT_TITLE, color=INK)

    save_fig(fig, "fig_real_sem_axis")
    plt.close(fig)

    for field in fields:
        block = meta["per_field"][field]
        print("[fig_real_sem_axis] %s crop row %d col %d, %dx%d px"
              % (field, block["crop_row"], block["crop_col"],
                 block["crop_w"], block["crop_h"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
