"""The three-panel centreline overlay: annotation, and PLECTA from each axis.

The same three things ``fig_real_sem`` draws in its panels (a), (c) and (e), but
as centrelines rather than at width -- which is the form every score in the
paper is actually computed on, and is why this is kept. ``fig_real_sem`` draws
them at width because a region compared against a hairline is not a comparison;
that choice is right for the shipped figure, but it does hide the object the
numbers were measured from.

Kept runnable and gated, per the repository convention for a generator no
``\\includegraphics`` reaches: it writes to ``figures/archive/``.  It is the
version to come back to if a reader or reviewer asks to see the grouping as the
scorer sees it.

    python scripts/figures/fig_real_sem_overlay.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _style import FIG_W, plecta_style, save_fig                  # noqa: E402
from fig_real_sem import (ASSET, COL_GAP, DRAW_DILATION, PAD,     # noqa: E402
                          RASTER_DPI, TITLE_H, add_panel,
                          colour_of_reference, field_label, instances,
                          show, unpack)


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    unmatched = int(meta["unmatched_label"])
    field = list(meta["fields"])[0]

    sem = data[f"{field}__sem"]
    shape = sem.shape[:2]
    reference = unpack(data, field, "reference")
    colours = colour_of_reference(shape, *reference)

    panel_w = (FIG_W - 2 * PAD - 2 * COL_GAP) / 3.0
    panel_h = panel_w * shape[0] / shape[1]
    fig_h = TITLE_H + panel_h
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=RASTER_DPI)

    panels = (("a", "Manual annotation", reference),
              ("b", "PLECTA, manual-derived axis", unpack(data, field, "manual")),
              ("c", "PLECTA, nnU-Net axis", unpack(data, field, "unet")))
    for column, (tag, title, layer) in enumerate(panels):
        ax = add_panel(fig, fig_h, PAD + column * (panel_w + COL_GAP), TITLE_H,
                       panel_w, panel_h, tag, title)
        show(ax, instances(sem, *layer, colours, unmatched,
                           dilation=DRAW_DILATION))

    save_fig(fig, "fig_real_sem_overlay", subdir="archive")
    plt.close(fig)
    print("[fig_real_sem_overlay] %s whole field %dx%d px, centrelines"
          % (field_label(field), shape[1], shape[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
