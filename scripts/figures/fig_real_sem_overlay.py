"""The three-panel overlay: annotation, and PLECTA from each axis, over the SEM.

Superseded by ``fig_real_sem``, which drops the annotation panel because it is
near-indistinguishable from PLECTA on the manual-derived axis at this size --
that near-identity is the result, but Table 5 states it numerically and two
almost identical pictures state it weakly.

Kept runnable and gated, per the repository convention for a generator no
``\\includegraphics`` reaches: it writes to ``figures/archive/``.  It is the
version to come back to if a reader or reviewer asks to see the reference drawn
beside the reconstructions rather than inferred from the instance colours.

    python scripts/figures/fig_real_sem_overlay.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _style import plecta_style, save_fig                         # noqa: E402
from fig_real_sem import (ASSET, DRAW_DILATION,                   # noqa: E402
                          colour_of_reference, panel_row)


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))

    fig = plt.figure()
    panel_row(fig, data["sem"],
              (("a", "Manual annotation", data["reference"]),
               ("b", "PLECTA, manual-derived axis", data["manual"]),
               ("c", "PLECTA, nnU-Net axis", data["unet"])),
              colour_of_reference(data["reference"]),
              int(meta["unmatched_label"]),
              dilation=DRAW_DILATION, gap_frac=0.118)

    save_fig(fig, "fig_real_sem_overlay", subdir="archive")
    plt.close(fig)
    print("[fig_real_sem_overlay] %s crop at row %d col %d, %dx%d px"
          % (meta["field"], meta["crop_row"], meta["crop_col"],
             meta["crop_w"], meta["crop_h"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
