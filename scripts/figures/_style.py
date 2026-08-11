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


# ===========================================================================
#  PLECTA figure set
#  ------------------------------------------------------------------------
#  The vocabulary is inherited from the existing figures in this repository
#  (fig_filaseg_scope, fig_reconnect_gates, fig_locked_method_comparison,
#  fig_paired_locked_examples, fig_development_sensitivity) rather than
#  invented: same three semantic hues, same slate/dark neutrals, same
#  rounded-box and legend idiom.
#
#      BLUE      filament / fragment A.  In the pipeline figure -- which has
#                its own legend and contains no filaments -- a learned
#                component.
#      ORANGE    filament / fragment B.  In the pipeline figure, a
#                dependency on the grayscale image.
#      GREEN     PLECTA itself: its geometric stages, the quantities it
#                measures (theta, phi, kappa, w, q(d)), the links it
#                accepts, and its series in every chart.
#      GRAY      the tip-to-tip chord and the separation d; a candidate link
#                not taken (always dashed); muted annotation.
#      DARK      raw mask pixels, data artifacts, body text, axes.
#      JUNCTION  junction: skeleton pixels of degree >= 3 and the node
#                clusters formed from them.
#      FALSE_INST / MISSED_INST   an instance the method invents, and a
#                reference instance it never recovers (paired examples).
#      DENSITY_CYCLE   target areal coverage, 20% to 60% (viridis, as in
#                fig_development_sensitivity).
#      INSTANCE_CYCLE  instance identity *within one panel*, nothing else.
# ===========================================================================

FIL_A = BLUE           # filament / instance A: its arm, stub, tangent, layer
FIL_B = ORANGE         # filament / instance B: its arm, stub, tangent, layer
PLECTA = GREEN         # a PLECTA stage, a quantity it measures, a link it takes
GEOM = GREEN           # alias: the measured geometry is PLECTA's own
CHORD = GRAY           # tip-to-tip chord and d; a candidate not taken
INK = DARK             # raw mask pixels; data artifacts; body text; axes
JUNCTION = "#D1495B"   # junction: degree >= 3 skeleton pixels and their clusters
AGREE = GREEN          # grouped as the reference groups it
DIFFER = "#C2185B"     # an instance the method invents
FALSE_INST = "#C2185B"
MISSED_INST = "#E8B10A"

BLUE_TINT = "#eaf1f8"
GREEN_TINT = "#eaf4ee"
ORANGE_TINT = "#fdf1e7"

#: Target areal coverage, 20 % to 60 %, as in fig_development_sensitivity.
DENSITY_CYCLE = ("#440154", "#3B528B", "#21918C", "#5EC962", "#C7E020")

#: Categorical ramp used only where many instances must be told apart.  The
#: colours carry identity *within a panel* and nothing else.
INSTANCE_CYCLE = (
    "#1F6FB2", "#E07A1F", "#2E9E6B", "#B03A6A", "#7C5CC4",
    "#8A6A3A", "#D45087", "#4CA3C7", "#9A9F1F", "#C0392B",
    "#3B7A57", "#8E44AD",
)

#: \linewidth of the manuscript (11pt article, a4paper, margin=1in) is
#: 451.3 TeX pt = 449.61 bp = 6.2446 in.  Figures are drawn at exactly this
#: width and included with ``width=\linewidth``, so the include scale is 1.0
#: and a label specified at n pt prints at n pt.  Never draw below 7.0.
FIG_W = 451.3 / 72.27
MIN_PT = 7.0

# ---------------------------------------------------------------------------
#  THE TYPE SCALE.  Two sizes, five roles.
#  ------------------------------------------------------------------------
#  Every figure in the set uses these and nothing else.  An earlier round drew
#  panel titles at 8.0, 9.0 and 9.6 pt in different figures and annotation at
#  6.9, 7.0 and 7.2, which is what made the set look uneven; the fix is not a
#  wider scale but a narrower one.  Bold weight and the (a)/(b) tags carry the
#  hierarchy, so the title needs only a small step above body text.
#
#      8.5 bold   panel title
#      7.5        axis label, tick label, in-figure annotation, legend
#      7.0        hard floor, reserved for dense numeric annotation
#
#  ``check_type_scale.py`` re-derives the smallest size actually drawn in each
#  figure from the figure object itself, so this block is a claim that is
#  checked rather than asserted.
# ---------------------------------------------------------------------------
PT_TITLE = 8.5       # panel title (always bold)
PT_AXIS = 7.5        # axis label
PT_TICK = 7.5        # tick label
PT_ANNOT = 7.5       # in-figure annotation
PT_LEGEND = 7.5      # legend entry
PT_MIN = 7.0         # the floor; dense numeric annotation only

#: Names the earlier round used.  Kept so nothing has to be renamed twice, but
#: they now resolve into the scale above rather than to three separate sizes.
PT_BODY = PT_AXIS
PT_LABEL = PT_TICK
PT_TINY = PT_ANNOT


def plecta_style():
    """rcParams for the PLECTA set.  Call once, before creating a figure."""
    apply_style()
    plt.rcParams.update(
        {
            "font.size": PT_AXIS,
            "axes.titlesize": PT_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": PT_AXIS,
            "legend.fontsize": PT_LEGEND,
            "xtick.labelsize": PT_TICK,
            "ytick.labelsize": PT_TICK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.7,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ── payload helpers (results/plecta_figure_data.json) ──────────────────────


def load_figure_data():
    """The committed extract every PLECTA figure reads."""
    import json
    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results",
                     "plecta_figure_data.json"))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def unpack(payload):
    """{h, w, idx} -> boolean image."""
    import numpy as np
    out = np.zeros(payload["h"] * payload["w"], dtype=bool)
    out[payload["idx"]] = True
    return out.reshape(payload["h"], payload["w"])


def unpack_labels(payload):
    """{h, w, idx, lab} -> int label image."""
    import numpy as np
    out = np.zeros(payload["h"] * payload["w"], dtype=int)
    out[payload["idx"]] = payload["lab"]
    return out.reshape(payload["h"], payload["w"])


def blank_rgb(shape):
    import numpy as np
    img = np.ones((shape[0], shape[1], 3), dtype=float)
    return img


def paint(img, mask, colour):
    """Write a colour into an RGB float image wherever mask is True."""
    import matplotlib.colors as mcolors
    import numpy as np
    rgb = np.asarray(mcolors.to_rgb(colour))
    img[np.asarray(mask, dtype=bool)] = rgb
    return img


def bare(ax):
    """A panel that shows pixels: no ticks, no spines, square, y down."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")
    return ax


def framed(ax, colour=None, lw=0.6):
    """A pixel panel with a hairline border, so white content has an edge."""
    bare(ax)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(colour or CHORD)
        ax.spines[side].set_linewidth(lw)
    return ax


def panel_tag(ax, letter, dx=0.0, dy=1.0, **kw):
    """The (a), (b), ... tag, placed in axes coordinates above the panel."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PT_TITLE, fontweight="bold",
            color=INK, **kw)


def panel_title(ax, text, dx=0.055, dy=1.0, **kw):
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PT_TITLE, fontweight="bold", color=INK, **kw)


# ── page layout ────────────────────────────────────────────────────────────
#
#  Every PLECTA figure is laid out top-down in *inches* and converted to
#  figure fractions once, at the end.  Doing the arithmetic in fractions is
#  what lets a block of prose silently land on the next row's title when
#  either one changes length.

LINE_IN = PT_ANNOT * 1.55 / 72.0        # height of one 7 pt line, inches


def stack(rows, top=0.06, bottom=0.06):
    """Lay rows out top-down and return their y positions plus the height.

    ``rows`` is a sequence of ``(key, panel_height_in, n_prose_lines)``.  The
    result maps each key to ``{"title", "panel", "prose"}`` as *fractions of
    the returned figure height*, so a caller does::

        rows, fig_h = stack([("r1", 1.2, 3), ("r2", 1.4, 4)])
        fig = plt.figure(figsize=(FIG_W, fig_h))
        ax = fig.add_axes([0.05, rows["r1"]["panel"], w, 1.2 / fig_h])
    """
    y = top
    marks = []
    for key, panel_in, n_prose in rows:
        y += 0.115
        title_y = y
        y += 0.055
        panel_y = y + panel_in
        y += panel_in + 0.075
        prose_y = y
        y += n_prose * LINE_IN + 0.10
        marks.append((key, title_y, panel_y, prose_y))
    height = y + bottom - 0.10
    out = {key: {"title": 1.0 - t / height,
                 "panel": 1.0 - p / height,
                 "prose": 1.0 - q / height}
           for key, t, p, q in marks}
    return out, height


def fig_title(fig, x, y, letter, text, gap=0.038):
    fig.text(x, y, f"({letter})", fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")
    fig.text(x + gap, y, text, fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")


def fig_prose(fig, x, y, text, colour=INK):
    fig.text(x, y, text, fontsize=PT_ANNOT, color=colour, ha="left", va="top",
             linespacing=1.55)


def pixel_axes(fig, rect, n_rows, n_cols=None):
    """A panel that shows pixels one-for-one, row 0 at the top."""
    n_cols = n_cols if n_cols is not None else n_rows
    ax = fig.add_axes(rect)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(CHORD)
        s.set_linewidth(0.5)
    return ax


# ── the rounded-box schematic idiom, shared with fig_filaseg_scope ────────


def rbox(ax, xc, yc, w, h, title, edge, face="white", lw=1.6, fontsize=PT_ANNOT,
         fontcolor=None, ls="solid", bold=True, rounding=0.10, zorder=3,
         linespacing=1.3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, linestyle=ls, edgecolor=edge, facecolor=face,
        zorder=zorder))
    ax.text(xc, yc, title, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=fontcolor or DARK, zorder=zorder + 1,
            linespacing=linespacing)


def stage_box(ax, xc, yc, w, h, tag, name, colour, fontsize=PT_ANNOT,
              tag_fontsize=PT_ANNOT):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.8, edgecolor=colour, facecolor="white", zorder=3))
    ax.text(xc, yc + h / 2 - 0.20, tag, ha="center", va="center",
            fontsize=tag_fontsize, fontweight="bold", color=colour, zorder=4)
    ax.text(xc, yc - 0.12, name, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=DARK, zorder=4, linespacing=1.35)


def harrow(ax, x0, y0, x1, y1, colour=None, lw=1.5, ls="solid", mscale=12,
           zorder=2):
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                          mutation_scale=mscale, linewidth=lw, linestyle=ls,
                          color=colour or GRAY, zorder=zorder)
    ax.add_patch(arr)
    return arr


def elbow(ax, pts, colour=None, lw=1.4, ls=(0, (4, 2.5)), zorder=2, mscale=12):
    """Rectilinear connector through pts; the final leg gets an arrowhead."""
    from matplotlib.patches import FancyArrowPatch
    colour = colour or GRAY
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=colour, linewidth=lw, linestyle=ls, zorder=zorder,
                solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=mscale, linewidth=lw,
                                 linestyle=ls, color=colour, zorder=zorder))


def swatch_legend(ax, items, x_positions, y, box_w=0.34, box_h=0.18,
                  fontsize=PT_LEGEND, gap=0.46, lw=1.4):
    """The four-item outlined-swatch legend used by fig_filaseg_scope."""
    from matplotlib.patches import FancyBboxPatch
    for (colour, label), lx in zip(items, x_positions):
        ax.add_patch(FancyBboxPatch(
            (lx, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=lw, edgecolor=colour, facecolor="white", zorder=4))
        ax.text(lx + gap, y, label, ha="left", va="center", fontsize=fontsize,
                color=DARK, zorder=4)


def show_layers(ax, layers, colours, background=None, bg_colour=None):
    """Draw boolean layers as an RGB image, later layers on top.

    Used for whole-scene panels, where drawing a rectangle per pixel would
    put a hundred thousand vector patches into the PDF.
    """
    import numpy as np
    shape = layers[0].shape if layers else background.shape
    img = blank_rgb(shape)
    if background is not None:
        paint(img, background, bg_colour or LIGHT_GRAY)
    for layer, colour in zip(layers, colours):
        paint(img, layer, colour)
    ax.imshow(img, interpolation="nearest", zorder=2)
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    return img
