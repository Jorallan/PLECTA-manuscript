"""Figure: what the common-fragment score is worth at one crossing.

Section 2.6 says the primary endpoint counts *pairs of fragments*: the input
mask is cut into arms by a rule that reads neither the reference nor any
prediction, and the score asks how many pairs of those arms a method groups as
the reference groups them.  That is an abstract sentence, and this makes it
concrete on the smallest scene that contains the whole question -- two strands
crossing once.

The scorer (``eval/core/common_metric.py``) skeletonises the mask, deletes the
skeleton pixels with three or more 8-neighbours -- drawn in red -- and labels
what is left.  Here that leaves four arms, so there are six arm pairs to get
right or wrong.  Each panel is one way of grouping those four arms; the arms of
one predicted instance carry one colour, and a group that is one of the two
reference strands keeps that strand's colour, so a wrong group is visible as a
colour the reference never had:

  (a) the reference pairing itself;
  (b) one strand left unlinked at the crossing -- the split failure;
  (c) both strands taken as one instance -- the merge failure;
  (d) the arms paired the wrong way round -- the swap, which no
      object-detection score can see;
  (e) every arm its own instance, which leaves the prediction with no pairs at
      all: recall is 0, precision has no denominator, and F_1 is undefined.
      The paper quotes the adjusted Rand index and variation of information
      beside the pairwise score partly for this reason.

Nothing here is hand-computed.  The mask is built once, the fragments come from
``common_metric.common_fragments``, both sides are assigned by
``common_metric.assign_fragments``, and every printed number is a field of
``common_metric.pairwise_scores`` -- the same call every published number in the
manuscript goes through.

    python scripts/figures/fig_f1_example.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _style import (FIG_W, FIL_A, FIL_B, FONE, GRAY, INK, JUNCTION, PT_ANNOT,
                    PT_TITLE, ROSE, VIOLET, blank_rgb, paint, pixel_axes,
                    plecta_style, save_fig)

#: The scorer itself.  eval/ is a flat namespace, not a package, so both it and
#: eval/core have to be reachable by bare name (common_metric.py does the same
#: insertion for its own siblings).
EVAL = r"C:\Repos\filaments_quantification\eval"

N = 27              # scene side, px.  Small on purpose: the reader should be
                    #  able to count the pixels the metric works on.
HALF = 1            # stroke half-width -> a 3 px strand
MASK_GREY = "#d5dbe3"

#: A predicted group that is not one of the reference strands has to be drawn
#: in a colour the reference never used, so that "this group is wrong" is
#: legible without reading the numbers.  Green is not among them: in this set
#: green means PLECTA and nothing else.
EXTRA = (VIOLET, ROSE, "#8A6A3A", "#4CA3C7")

#: Arms are named by the quadrant they occupy, and the two reference strands
#: are the two diagonals: A runs top-left to bottom-right, B top-right to
#: bottom-left.
REFERENCE = (("A_TL", "A_BR"), ("B_TR", "B_BL"))
REF_COLOUR = (FIL_A, FIL_B)

GROUPINGS = (
    ("a", "matches the\nreference",
     (("A_TL", "A_BR"), ("B_TR", "B_BL"))),
    ("b", "one strand\nnot linked",
     (("A_TL",), ("A_BR",), ("B_TR", "B_BL"))),
    ("c", "both strands\nmerged",
     (("A_TL", "A_BR", "B_TR", "B_BL"),)),
    ("d", "arms\nmis-paired",
     (("A_TL", "B_TR"), ("A_BR", "B_BL"))),
    ("e", "every arm\nseparate",
     (("A_TL",), ("A_BR",), ("B_TR",), ("B_BL",))),
)


# ── the scene, and the scorer's own view of it ────────────────────────────


def scene():
    """Two crossing strands, the four arms they leave, and the fragment set.

    The arms are halves of the strand masks, split at the row through the
    crossing, so a "prediction" here is a union of real mask pixels rather than
    a hand-written assignment vector: it is put through the same
    ``assign_fragments`` call the reference is.
    """
    centre = (N - 1) // 2

    def strand(sign):
        m = np.zeros((N, N), dtype=bool)
        for t in range(-centre, centre + 1):
            r, c = centre + t, centre + sign * t
            m[max(0, r - HALF):r + HALF + 1,
              max(0, c - HALF):c + HALF + 1] = True
        return m

    a, b = strand(+1), strand(-1)
    rows = np.repeat(np.arange(N)[:, None], N, axis=1)
    arms = {"A_TL": a & (rows <= centre), "A_BR": a & (rows >= centre),
            "B_TR": b & (rows <= centre), "B_BL": b & (rows >= centre)}
    return {"mask": a | b, "strands": {1: a, 2: b}, "arms": arms,
            "centre": centre}


def measure(sc):
    """Fragments, the junction pixels deleted to make them, and every score."""
    sys.path.insert(0, os.path.join(EVAL, "core"))
    sys.path.insert(0, EVAL)
    import common_metric as CM
    from skimage.morphology import skeletonize

    frag = CM.common_fragments(sc["mask"])
    ids = [int(k) for k in np.unique(frag) if int(k) != 0]
    if len(ids) != 4:
        raise SystemExit("one crossing must leave four arms, got %r" % (ids,))

    #  Exactly the pixels common_fragments removes, re-derived through its own
    #  helpers so the red pixels are the deleted ones and not a redrawing.
    skel = CM._prune_spurs(skeletonize(sc["mask"]), 3)
    junction = skel & (CM._degree_map(skel) >= 3)

    #  Fragment id -> arm name, by which quadrant the fragment sits in.  The
    #  labelling order is skimage's and is not relied on.
    centre, arm_of = sc["centre"], {}
    for k in ids:
        r, c = np.nonzero(frag == k)
        arm_of[k] = ("A_TL" if c.mean() < centre else "B_TR") if r.mean() < centre \
            else ("B_BL" if c.mean() < centre else "A_BR")
    if sorted(arm_of.values()) != sorted(sc["arms"]):
        raise SystemExit("arms did not land one per quadrant: %r" % (arm_of,))

    gt_assign = CM.assign_fragments(frag, sc["strands"])
    for group, ref_id in zip(REFERENCE, (1, 2)):
        if {gt_assign[k] for k in ids if arm_of[k] in group} != {ref_id}:
            raise SystemExit("the reference did not take the two diagonals: %r"
                             % (gt_assign,))

    scored = []
    for letter, title, groups in GROUPINGS:
        pred = {i + 1: np.logical_or.reduce([sc["arms"][name] for name in g])
                for i, g in enumerate(groups)}
        s = CM.pairwise_scores(gt_assign, CM.assign_fragments(frag, pred))
        scored.append({"letter": letter, "title": title, "groups": groups,
                       "scores": s})
    return {"frag": frag, "arm_of": arm_of, "junction": junction,
            "gt_assign": gt_assign, "panels": scored}


def colour_of(groups):
    """One colour per predicted group; a reference strand keeps its own."""
    spare = iter(EXTRA)
    out = {}
    for g in groups:
        match = [c for ref, c in zip(REFERENCE, REF_COLOUR)
                 if set(ref) == set(g)]
        out[g] = match[0] if match else next(spare)
    return out


# ── the page ───────────────────────────────────────────────────────────────


def panel(fig, rect, sc, d, groups):
    ax = pixel_axes(fig, rect, N)
    img = blank_rgb(sc["mask"].shape)
    paint(img, sc["mask"], MASK_GREY)
    colour = colour_of(groups)
    by_arm = {d["arm_of"][k]: k for k in d["arm_of"]}
    for g in groups:
        for name in g:
            paint(img, d["frag"] == by_arm[name], colour[g])
    ax.imshow(img, interpolation="nearest", zorder=2)
    #  The deleted set is drawn as one smooth disc rather than as its six
    #  individual pixels. The pixels are jagged and read as an artefact of
    #  this rasterisation; what the panel is about is that a neighbourhood
    #  around the crossing is removed before any pair is scored. The disc is
    #  centred on the deleted pixels and sized to cover them, so it stands
    #  for the real set rather than replacing it with a different one.
    ys, xs = np.nonzero(d["junction"])
    cx, cy = xs.mean(), ys.mean()
    r = max(np.hypot(xs - cx, ys - cy).max() + 0.5, 1.4)
    ax.add_patch(Circle((cx, cy), r, facecolor=JUNCTION, edgecolor="white",
                        lw=0.7, alpha=0.95, zorder=3))
    return ax


def main() -> int:
    plecta_style()
    sc = scene()
    d = measure(sc)

    PANEL_IN = 1.00
    BAND = (("top", 0.06), ("tag", 0.34), ("row", PANEL_IN), ("f1", 0.26),
            ("pr", 0.21), ("gap", 0.06), ("prose", 0.52), ("bottom", 0.03))
    fig_h = sum(v for _k, v in BAND)
    at, run = {}, 0.0
    for key, height in BAND:                 # distance from the top, inches
        at[key] = run
        run += height

    fig = plt.figure(figsize=(FIG_W, fig_h))
    IN = 1.0 / fig_h                         # inches -> figure fraction
    w, h = PANEL_IN / FIG_W, PANEL_IN * IN
    #  The right edge stops short of the prose margin: the widest number line,
    #  "P undefined", is a little wider than its own panel and would otherwise
    #  hang off the page.
    left, right = 0.030, 0.970
    step = (right - left - w) / (len(GROUPINGS) - 1)
    row_y = 1.0 - (at["row"] + PANEL_IN) * IN

    for i, p in enumerate(d["panels"]):
        x = left + i * step
        panel(fig, [x, row_y, w, h], sc, d, p["groups"])
        fig.text(x + w / 2.0, row_y + h + 0.012,
                 "(%s) %s" % (p["letter"], p["title"]), fontsize=PT_TITLE,
                 fontweight="bold", color=INK, ha="center", va="bottom",
                 linespacing=1.30)

        s = p["scores"]
        finite = np.isfinite(s["f1"])
        fig.text(x + w / 2.0, 1.0 - (at["f1"] + 0.19) * IN,
                 (FONE + " = %.3f" % s["f1"]) if finite
                 else (FONE + " undefined"),
                 fontsize=PT_TITLE, fontweight="bold",
                 color=INK if finite else GRAY, ha="center", va="baseline")
        fig.text(x + w / 2.0, 1.0 - (at["pr"] + 0.155) * IN,
                 ("P %.2f   R %.2f" % (s["precision"], s["recall"])) if finite
                 else "P undefined   R %.2f" % s["recall"],
                 fontsize=PT_ANNOT, color=GRAY, ha="center", va="baseline")

    n_pairs = 6
    fig.text(left, 1.0 - (at["prose"] + 0.02) * IN,
             ("Two strands cross.  The scorer deletes the junction pixels "
              "(red) and keeps the four arms they separate, then scores\n"
              "the %d pairs of arms: a pair is right when the grouping joins "
              "it, or leaves it apart, as the reference does.  The\n"
              "reference is the pairing in (a); P and R are the precision and "
              "recall of those %d pair decisions." % (n_pairs, n_pairs)),
             fontsize=PT_ANNOT, color=INK, ha="left", va="top",
             linespacing=1.50)

    save_fig(fig, "fig_f1_example", bbox_inches=None)
    plt.close(fig)

    print("scene %dx%d  mask %d px  junction %d px  fragments %s"
          % (N, N, int(sc["mask"].sum()), int(d["junction"].sum()),
             {d["arm_of"][k]: int((d["frag"] == k).sum())
              for k in sorted(d["arm_of"])}))
    print("reference assignment (fragment -> instance):", d["gt_assign"])
    for p in d["panels"]:
        s = p["scores"]
        print("  (%s) %-22s F1=%-6s P=%-6s R=%-6s ARI=%+.3f  "
              "VI merge %.3f / split %.3f  tp=%g fp=%g fn=%g  [%s]"
              % (p["letter"], p["title"].replace("\n", " "),
                 "%.3f" % s["f1"] if np.isfinite(s["f1"]) else "nan",
                 "%.3f" % s["precision"] if np.isfinite(s["precision"]) else "nan",
                 "%.3f" % s["recall"] if np.isfinite(s["recall"]) else "nan",
                 s["adjusted_rand_index"], s["vi_merge_bits"],
                 s["vi_split_bits"], s["tp"], s["fp"], s["fn"],
                 s["pair_evidence_reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
