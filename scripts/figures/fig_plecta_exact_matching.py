"""Figure: the pairing at a crossing is solved jointly and exactly.

Every number here is measured, not illustrative: the pixels, the six pairwise
costs, the joint score of each admissible configuration, the
declined-but-admissible edge at a real T-junction, and the node-by-node
comparison against a sequential cheapest-edge-first rule.  All of it is read
from ``results/plecta_figure_data.json``.

    python scripts/figures/fig_plecta_exact_matching.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from _style import (FIG_W, PT_BODY, PT_LABEL, PT_TINY, PT_TITLE,
                    FIL_A, FIL_B, JUNCTION, CHORD, INK,
                    load_figure_data, plecta_style, save_fig, unpack,
                    unpack_labels)

FIG_H = 3.00
LINE_IN = 7.0 * 1.55 / 72.0          # one 7 pt line, inches
LETTERS = "abcd"
PANEL = 1.12                       # side of a square pixel panel, inches


def fx(inches):
    return inches / FIG_W


def fy(inches):
    return inches / FIG_H


def title(fig, x, y, letter, text):
    fig.text(x, y, f"({letter})", fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")
    fig.text(x + 0.038, y, text, fontsize=PT_TITLE, fontweight="bold",
             color=INK, ha="left", va="bottom")


def prose(fig, x, y, text):
    fig.text(x, y, text, fontsize=PT_TINY, color=INK, ha="left", va="top",
             linespacing=1.55)


def pixel_axes(fig, rect, n):
    ax = fig.add_axes(rect)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)          # row 0 at the top, as in the image
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(CHORD)
        s.set_linewidth(0.5)
    return ax


def draw_pixels(ax, mask, colour, z=2):
    for r, c in zip(*np.nonzero(mask)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colour,
                               edgecolor="none", zorder=z))


def draw_split_pixels(ax, mask, first, second, z=4):
    """A pixel owned by two instances, drawn in both of their colours."""
    for r, c in zip(*np.nonzero(mask)):
        x0, y0 = c - 0.5, r - 0.5
        ax.add_patch(Polygon([(x0, y0), (x0 + 1, y0), (x0, y0 + 1)],
                             facecolor=first, edgecolor="none", zorder=z))
        ax.add_patch(Polygon([(x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)],
                             facecolor=second, edgecolor="none", zorder=z))


def letter_map(frames, stubs):
    """Name the stubs a, b, c, ... anticlockwise from due east.

    Display bearing, so the letters in every panel run the same way round as
    the arms in the pixel panel.
    """
    bear = {}
    for sid in stubs:
        dr, dc = frames[str(sid)]["tangent"]
        bear[sid] = math.degrees(math.atan2(-dr, dc)) % 360.0
    ordered = sorted(stubs, key=lambda s: bear[s])
    return {s: LETTERS[i] for i, s in enumerate(ordered)}, bear


# ── panels ────────────────────────────────────────────────────────────────


def panel_before(ax, data, names, bear):
    j, cr = data["junction4"], data["crossing"]
    draw_pixels(ax, unpack(cr["mask"]), INK)
    draw_pixels(ax, unpack(cr["node_px"]), JUNCTION, z=3)
    for sid, name in names.items():
        tip = j["frames"][str(sid)]["tip"]
        ang = math.radians(bear[sid])
        ax.text(tip[1] + 17.0 * math.cos(ang), tip[0] - 17.0 * math.sin(ang),
                name, fontsize=PT_LABEL, fontweight="bold", color=INK,
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=CHORD,
                          lw=0.5))


def panel_after(ax, data, names, first_pair):
    """Colour the layer that contains the first chosen pair as filament A."""
    cr = data["crossing"]
    arms = unpack_labels(cr["arms"])
    layers = [unpack(p) for p in cr["layers"]]
    aid_first = data["junction4"]["frames"][str(first_pair[0])]["aid"]
    key = np.argmax([int((lay & (arms == aid_first + 1)).sum())
                     for lay in layers])
    order = [layers[key]] + [lay for i, lay in enumerate(layers) if i != key]

    shared = order[0] & order[1] if len(order) >= 2 else np.zeros_like(order[0])
    draw_pixels(ax, order[0] & ~shared, FIL_A)
    if len(order) >= 2:
        draw_pixels(ax, order[1] & ~shared, FIL_B)
    draw_split_pixels(ax, shared, FIL_A, FIL_B)
    return int(shared.sum())


def panel_costs(ax, data, names):
    j = data["junction4"]
    rows = sorted(j["pairs"], key=lambda r: r["cost"])
    limit = j["admissibility_limit"]
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r["cost"], height=0.60,
                color=INK if r["admissible"] else "white",
                edgecolor=INK, linewidth=0.7, zorder=3)
        ax.text(r["cost"] + 0.10, yi, f"{r['cost']:.2f}", va="center",
                ha="left", fontsize=PT_TINY, color=INK, zorder=4)
    ax.axvline(limit, color=JUNCTION, lw=1.1, zorder=5)
    ax.text(limit + 0.06, len(rows) - 0.42, "$2p_i$", color=JUNCTION,
            fontsize=PT_TINY, ha="left", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names[r['a']]}–{names[r['b']]}" for r in rows],
                       fontsize=PT_LABEL)
    ax.set_xlim(0, 4.15)
    ax.set_ylim(-0.62, len(rows) - 0.30)
    ax.set_xlabel("pairing cost $C_{ab}$", fontsize=PT_LABEL, labelpad=0.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.5)


def config_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-3.05, 2.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def config_diagram(ax, stubs, names, bear, pairs, total, chosen):
    """One configuration: which stubs are joined, which pay to stay free."""
    pos = {s: np.array([math.cos(math.radians(bear[s])),
                        math.sin(math.radians(bear[s]))]) for s in stubs}
    matched = {s for p in pairs for s in p}
    # Two arms can leave a node in almost the same direction.  Fan those
    # labels apart along the circle instead of letting them overprint; a
    # radial stagger would push one of them onto the score printed below.
    lab_deg, placed = {}, []
    for s in sorted(stubs, key=lambda s: bear[s]):
        angle = bear[s]
        while any(min(abs(angle - q), 360 - abs(angle - q)) < 26 for q in placed):
            angle += 13.0
        lab_deg[s] = angle
        placed.append(angle)
    for s in stubs:
        ax.plot(*np.array([pos[s] * 0.26, pos[s]]).T, color=CHORD, lw=0.8,
                zorder=2)
        lab = np.array([math.cos(math.radians(lab_deg[s])),
                        math.sin(math.radians(lab_deg[s]))]) * 1.50
        ax.text(*lab, names[s], fontsize=PT_TINY,
                fontweight="bold", ha="center", va="center",
                color=INK if s in matched else CHORD, zorder=6)
        if s not in matched:
            ax.plot(*pos[s], "o", ms=3.6, mfc="white", mec=JUNCTION, mew=1.0,
                    zorder=5)
    for k, (a, b) in enumerate(pairs):
        ax.plot(*np.array([pos[a], pos[b]]).T, color=(FIL_A, FIL_B)[k % 2],
                lw=2.0, solid_capstyle="round", zorder=4)
    if chosen:
        ax.add_patch(Rectangle((-1.78, -1.78), 3.56, 3.56, fill=False,
                               ec=INK, lw=1.0, zorder=1))
    ax.text(0, -2.96, f"{total:.2f}", ha="center", va="bottom",
            fontsize=PT_LABEL, fontweight="bold" if chosen else "normal",
            color=INK if chosen else CHORD)


def panel_degrees(ax, data):
    div = data["greedy_divergence"]
    order = ["2-4", "5-8", "9+"]
    frac = [100.0 * div["buckets"][k]["n_differ"]
            / max(1, div["buckets"][k]["n_nodes"]) for k in order]
    ns = [div["buckets"][k]["n_nodes"] for k in order]
    x = np.arange(3)
    ax.bar(x, frac, width=0.55, color=INK, zorder=3)
    for xi, value in zip(x, frac):
        ax.text(xi, value + 3.0, f"{value:.0f}%", ha="center", va="bottom",
                fontsize=PT_TINY, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["2–4", "5–8", "≥9"], fontsize=PT_LABEL)
    ax.set_xlabel("stubs at the node", fontsize=PT_LABEL, labelpad=2.0)
    ax.set_ylabel("nodes differing (%)", fontsize=PT_LABEL, labelpad=1.0)
    ax.set_ylim(0, 88)
    ax.set_yticks([0, 25, 50, 75])
    ax.tick_params(axis="x", length=0, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)



def _rows():
    """Vertical layout, laid out top-down in inches then converted once.

    Working in inches and converting at the end is what keeps a row of prose
    from landing on the next row's title when either one changes length.
    """
    out, y = {}, 0.06
    for key, n_prose in (("r1", 0), ("r2", 0)):
        y += 0.115
        out[key] = {"title": 1.0 - fy(y)}
        y += 0.055
        out[key]["panel"] = 1.0 - fy(y + PANEL)
        y += PANEL + 0.075
        out[key]["prose"] = 1.0 - fy(y)
        y += n_prose * LINE_IN + 0.10
    out["total_in"] = y
    return out


ROWS = _rows()


def main() -> int:
    plecta_style()
    data = load_figure_data()
    j4, j3 = data["junction4"], data["junction3"]
    names4, bear4 = letter_map(j4["frames"], j4["stubs"])
    names3, bear3 = letter_map(j3["frames"], j3["stubs"])
    div = data["greedy_divergence"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    w_sq, h_sq = fx(PANEL), fy(PANEL)

    # ── row 1: one real crossing, decided ──────────────────────────────
    r1 = ROWS["r1"]
    ax_a = pixel_axes(fig, [0.045, r1["panel"], w_sq, h_sq],
                      data["crossing"]["size"])
    panel_before(ax_a, data, names4, bear4)

    ax_b = fig.add_axes([0.300, r1["panel"] + 0.055, 0.215, h_sq - 0.055])
    panel_costs(ax_b, data, names4)

    for i, cfg in enumerate(j4["configurations"]):
        ax = config_axes(fig, [0.578 + i * 0.100, r1["panel"], 0.096, h_sq])
        config_diagram(ax, j4["stubs"], names4, bear4,
                       [tuple(p) for p in cfg["pairs"]], cfg["total"],
                       chosen=(i == 0))

    title(fig, 0.030, r1["title"], "a", "Four stubs meet")
    title(fig, 0.285, r1["title"], "b", "All six pairings")
    title(fig, 0.563, r1["title"], "c", "Every admissible configuration")


    # ── row 2: the free option, and what solving jointly buys ──────────
    r2 = ROWS["r2"]
    ax_d = pixel_axes(fig, [0.045, r2["panel"], w_sq, h_sq],
                      data["crossing"]["size"])
    n_shared = panel_after(ax_d, data, names4, tuple(j4["exact"][0]))

    for i, cfg in enumerate(j3["configurations"]):
        ax = config_axes(fig, [0.300 + i * 0.100, r2["panel"], 0.096, h_sq])
        config_diagram(ax, j3["stubs"], names3, bear3,
                       [tuple(p) for p in cfg["pairs"]], cfg["total"],
                       chosen=(i == 0))

    ax_f = fig.add_axes([0.760, r2["panel"] + 0.075, 0.205, h_sq - 0.115])
    panel_degrees(ax_f, data)

    title(fig, 0.030, r2["title"], "d", "The output")
    title(fig, 0.285, r2["title"], "e", "Leaving a stub free")
    title(fig, 0.700, r2["title"], "f", "Joint vs. sequential")

    declined = max(p["cost"] for p in j3["pairs"] if p["admissible"])

    save_fig(fig, "fig_plecta_exact_matching", bbox_inches=None)
    plt.close(fig)
    print("shared px:", n_shared, "names:", names4, "declined:", declined,
          "needed height (in):", round(ROWS["total_in"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

