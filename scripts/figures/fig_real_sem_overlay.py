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
from fig_real_sem import (ASSET, DRAW_DILATION, colour_of_reference,  # noqa: E402
                          field_label, label_image, panel_grid, unpack)


def main() -> int:
    plecta_style()
    data = np.load(ASSET)
    meta = json.loads(str(data["meta"]))
    fields = list(meta["fields"])

    rows = []
    for field in fields:
        sem = data[f"{field}__sem"]
        reference = unpack(data, field, "reference")
        rows.append((
            field_label(field), sem,
            colour_of_reference(label_image(sem.shape[:2], *reference)),
            (("a", "Manual annotation", reference),
             ("b", "PLECTA, manual-derived axis", unpack(data, field, "manual")),
             ("c", "PLECTA, nnU-Net axis", unpack(data, field, "unet"))),
        ))

    fig = plt.figure()
    panel_grid(fig, rows, int(meta["unmatched_label"]),
               dilation=DRAW_DILATION, gap_frac=0.118, label_frac=0.016)

    save_fig(fig, "fig_real_sem_overlay", subdir="archive")
    plt.close(fig)
    for field in fields:
        block = meta["per_field"][field]
        print("[fig_real_sem_overlay] %s crop row %d col %d, %dx%d px"
              % (field, block["crop_row"], block["crop_col"],
                 block["crop_w"], block["crop_h"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
