"""What the two mask sources cost, on the centrelines the scores are computed on.

Two panels across ``\\linewidth``: PLECTA's instances from the manual-derived
axis and from the nnU-Net axis, over the micrograph, as one-pixel centrelines
rather than width-rendered ribbons. This is the form every number in Table 5 is
computed on -- the width layer of ``fig_real_sem`` runs afterwards and is not
evaluated -- and it is the form in which the difference between the two masks is
visible: the same filaments arrive in more pieces on the right.

Two panels rather than three: a panel of the manual annotation is
near-indistinguishable from PLECTA on the manual-derived axis at this size,
which is the result but is one Table 5 states numerically and two almost
identical pictures state weakly. The three-panel variant is
``fig_real_sem_overlay``, which writes to ``figures/archive/``.

    python scripts/figures/fig_real_sem_axis.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _style import plecta_style, save_fig                         # noqa: E402
from fig_real_sem import (ASSET, DRAW_DILATION, colour_of_reference,  # noqa: E402
                          panel_row)


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    colours = colour_of_reference(data["reference"])

    fig = plt.figure()
    panel_row(fig, data["sem"],
              (("a", "PLECTA, manual-derived axis", data["manual"]),
               ("b", "PLECTA, nnU-Net axis", data["unet"])),
              colours, int(meta["unmatched_label"]),
              #  A bold "(a)" at 8.5 pt is about 0.21 in, which is 0.07 of a
              #  3.0 in panel; below that the tag prints on the first word.
              dilation=DRAW_DILATION, gap_frac=0.078)
    save_fig(fig, "fig_real_sem_axis")
    plt.close(fig)

    print("[fig_real_sem_axis] %s crop row %d col %d, %dx%d px; local F1 %s"
          % (meta["field"], meta["crop_row"], meta["crop_col"],
             meta["crop_w"], meta["crop_h"],
             ", ".join("%s %.3f" % kv
                       for kv in sorted(meta["local_f1"].items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
