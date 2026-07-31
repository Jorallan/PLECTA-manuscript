"""Render audit-selected, correspondence-coloured locked-evaluation panels.

A single merged figure is produced, stacked as two bands: the top band holds
the scenes with FilaSeg's largest paired advantage, the bottom band the scenes
with the minimum-turn tracer's largest paired advantage.  Each band shows
`--rows-per-band` scenes, taken in stored order from the audit's
`qualitative_selection`, and the two bands are separated by a horizontal rule.
The input audit is the sole source of scene selection.  Within a scene, each
prediction is matched independently to overlap-aware ground truth by a maximum
sum one-to-one assignment of binary-mask IoU.  Matches with zero IoU are
discarded.  Thus a matched prediction inherits its ground-truth colour, while
unmatched predictions are false instances and unmatched ground-truth instances
are shown as missed dashed outlines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io as skio
from skimage.morphology import binary_dilation, disk, footprint_rectangle, skeletonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
import _evalpath  # noqa: F401,E402
import instance_io as IIO  # noqa: E402
from _style import GRAY, apply_style  # noqa: E402


FALSE_COLOR = "#D81B60"
MISSED_COLOR = "#FDE725"
#: The two bands of the merged figure, top to bottom: the `qualitative_selection`
#: key holding that band's scenes, and the title printed above the band.  The
#: keys are stored data field names, not free labels.  The audit's `two_near_ties`
#: category is deliberately absent -- the manuscript uses only the two extremes,
#: so the ties are never rendered and no ties file is written.
SELECTIONS = (
    ("three_largest_filaseg_wins", "Largest FilaSeg wins"),
    ("three_largest_minimum_turn_wins", "Largest minimum-turn wins"),
)

#: Scenes drawn per band, i.e. how many of each category's stored rows are
#: taken (in stored order, never re-sorted).  Two per band keeps the merged
#: figure to four rows, which is what the manuscript shows.
DEFAULT_ROWS_PER_BAND = 2

#: Display-only centreline thickening (see `_thicken_for_display`).  This
#: never touches results/ data, matching, or scoring -- it only changes the
#: array handed to `imshow` for the two centreline columns, immediately
#: before rendering.  2 px (diamond footprint) was compared against 3 px
#: (square footprint) on both figures.  Per-instance colours stayed crisp
#: and distinguishable at both widths, even in the densest crossing regions
#: (nearest-neighbour interpolation, no anti-aliased colour blending).  3 px
#: was chosen because most centreline segments run diagonally, where the
#: diamond footprint under-fills relative to the cardinal directions and
#: still reads as faint/broken at 2 px; the square footprint fills those
#: diagonal segments and turn points uniformly, which is the difference
#: that decides legibility in print.
DEFAULT_CENTRELINE_DILATION_PX = 3

# Method names are the manuscript's own prose names, identical to the legends
# of Figures 4 and 8, so one name per method appears everywhere.
COLUMN_HEADERS = ("Degraded input mask", "Overlap-aware GT", "Minimum-turn tracer",
                  "FilaSeg (Stage 3)")

# Layout, in inches on the unscaled canvas, laid out explicitly rather than by
# `tight_layout` so the band titles and the separating rule can be given
# guaranteed clearance.  The panel grid is inset from the left edge to leave
# room for the three-line row labels.
FIG_WIDTH_IN = 9.9
GRID_LEFT = 0.078          # figure fraction; ~0.77 in for the rotated row label
GRID_RIGHT = 0.985
COL_WSPACE = 0.05          # gridspec spacing, as a fraction of a panel
ROW_HSPACE = 0.05
PAD_TOP_IN = 0.06
BAND_TITLE_IN = 0.30       # band title text plus its trailing gap
COLUMN_HEADER_IN = 0.30    # reserved above the top band for the column titles
BAND_GAP_IN = 0.88         # between the bands: rule, band-2 title, clearance
RULE_OFFSET_IN = 0.30      # rule position below the last row of the top band
BAND_TITLE_CLEAR_IN = 0.14  # clear space under a band title before its panels
PAD_BOTTOM_IN = 0.42       # clears the legend, which straddles the figure edge
RULE_X = (0.010, 0.990)


def read_selection(
    audit: Mapping[str, Any], *, rows_per_band: int = DEFAULT_ROWS_PER_BAND
) -> dict[str, list[dict[str, Any]]]:
    """Return the report's stored selection, without recomputing or hand-picking.

    Only the two categories actually rendered are read, and each keeps its
    stored order: the first `rows_per_band` rows are taken as they stand.
    """
    if rows_per_band < 1:
        raise ValueError("rows_per_band must be at least 1")
    selection = audit.get("qualitative_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("audit has no qualitative_selection")
    result: dict[str, list[dict[str, Any]]] = {}
    for key, _title in SELECTIONS:
        rows = selection.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"audit qualitative_selection lacks {key}")
        if len(rows) < rows_per_band:
            raise ValueError(
                f"audit qualitative_selection {key} holds {len(rows)} rows, "
                f"fewer than the {rows_per_band} requested per band"
            )
        result[key] = [dict(row) for row in rows[:rows_per_band]]
    return result


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum()) / union if union else 0.0


def _gt_color(instance_id: int) -> tuple[float, float, float]:
    """Return a deterministic high-contrast colour without a short repeating cycle."""
    hue = (0.618033988749895 * int(instance_id)) % 1.0
    return tuple(matplotlib.colors.hsv_to_rgb((hue, 0.78, 0.92)))


def one_to_one_matches(
    ground_truth: IIO.InstanceSet, prediction: IIO.InstanceSet
) -> dict[int, int]:
    """Map prediction IDs to GT IDs by deterministic maximum-total-IoU matching."""
    gt_ids, pred_ids = sorted(ground_truth.masks), sorted(prediction.masks)
    if not gt_ids or not pred_ids:
        return {}
    scores = np.array([
        [_iou(ground_truth.masks[gid], prediction.masks[pid]) for pid in pred_ids]
        for gid in gt_ids
    ])
    rows, cols = linear_sum_assignment(-scores)
    return {
        pred_ids[col]: gt_ids[row]
        for row, col in zip(rows, cols)
        if scores[row, col] > 0.0
    }


def _centreline(instances: IIO.InstanceSet) -> IIO.InstanceSet:
    masks = {iid: skeletonize(mask) for iid, mask in instances.masks.items()}
    return IIO.InstanceSet(instances.shape, {iid: mask for iid, mask in masks.items()
                                             if mask.any()}, source=instances.source)


def _display_footprint(width_px: int) -> np.ndarray | None:
    """Return the structuring element used to thicken a 1 px centreline for display.

    2 px uses a diamond (4-connected) footprint; 3 px uses a full 3x3 square,
    which fills the corners a diamond leaves open and so reads as a uniform
    width regardless of the local line orientation.  Both footprints are
    odd-sized and centred, so dilation never shifts the rendered line.
    """
    if width_px <= 1:
        return None
    if width_px == 2:
        return disk(1)
    if width_px == 3:
        return footprint_rectangle((3, 3))
    return disk(max(1, width_px // 2))


def _thicken_for_display(instances: IIO.InstanceSet, width_px: int) -> IIO.InstanceSet:
    """Dilate each instance mask independently, for rendering only.

    This must never be used for matching, scoring, or anything that reads
    back into results/ -- it exists solely to make 1 px skeleton centrelines
    legible in print.  Dilating per-instance mask (rather than a combined
    label array) guarantees each instance keeps exactly its own colour: a
    shared-pixel tie between two nearby dilated instances is resolved the
    same deterministic way (later `iid` wins) as the undilated renderer
    already uses in `_rgb_instances`, so instance identity is never
    ambiguous or overwritten incorrectly.
    """
    footprint = _display_footprint(width_px)
    if footprint is None:
        return instances
    masks = {iid: binary_dilation(mask, footprint) for iid, mask in instances.masks.items()}
    return IIO.InstanceSet(instances.shape, masks, source=instances.source)


def _rgb_instances(instances: IIO.InstanceSet, match: Mapping[int, int], *, gt: bool = False) -> np.ndarray:
    image = np.zeros((*instances.shape, 3), dtype=float)
    for iid in sorted(instances.masks):
        gid = iid if gt else match.get(iid)
        color = _gt_color(int(gid)) if gid is not None else FALSE_COLOR
        image[instances.masks[iid]] = matplotlib.colors.to_rgb(color)
    return image


def _missed_ids(gt: IIO.InstanceSet, match: Mapping[int, int]) -> set[int]:
    return set(gt.masks).difference(match.values())


def _draw_instances(ax: Any, instances: IIO.InstanceSet, match: Mapping[int, int], gt: IIO.InstanceSet) -> None:
    ax.imshow(_rgb_instances(instances, match))
    for gid in _missed_ids(gt, match):
        ax.contour(gt.masks[gid].astype(float), levels=[0.5], colors=MISSED_COLOR,
                   linestyles="--", linewidths=0.7)


def _path(audit_row: Mapping[str, Any], name: str) -> Path:
    try:
        return Path(str(audit_row["artifacts"][name]["path"]))
    except KeyError as exc:
        raise ValueError(f"audit row has no {name} artifact") from exc


def _load_mask(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    return np.asarray(image[..., 0] if image.ndim == 3 else image) > 0


def _row_for_scene(audit: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    for row in audit.get("per_scene", []):
        if str(row.get("geometry_id")) == scene_id:
            return row
    raise ValueError(f"selected scene is absent from audit per_scene: {scene_id}")


def _row_label(chosen: Mapping[str, Any]) -> str:
    """Return the three-line row label: manifest key, areal density, difference.

    Line 1 is the stored `geometry_id` verbatim: it is the key this scene is
    looked up by in the evaluation manifest, so it is never renamed, split or
    prettified.  Line 2 spells that key's `covNN` prefix out in words, read
    from the row's own stored `density` field rather than by slicing the id;
    this axis is called areal density throughout the manuscript.  Line 3 is
    the paired difference, upright F with subscript 1 to match \\Fone in
    main.tex.
    """
    delta = float(chosen["filaseg_minus_minimum_turn_f1"])
    return "\n".join((
        str(chosen["geometry_id"]),
        f"{int(chosen['density'])}% areal density",
        f"$\\Delta$F$_1$ = {delta:+.3f}",
    ))


def _draw_scene(
    audit: Mapping[str, Any],
    chosen: Mapping[str, Any],
    panels: Sequence[Any],
    *,
    centreline_dilation_px: int,
    with_headers: bool,
) -> None:
    """Draw one scene's four panels into an already-created row of axes."""
    row = _row_for_scene(audit, str(chosen["geometry_id"]))
    mask = _load_mask(_path(row, "input_mask"))
    gt, _ = IIO.load_gt(_path(row, "ground_truth_selected"), shape=mask.shape)
    minimum_turn = _centreline(IIO.load_pred_instances(_path(row, "minimum_turn")))
    filaseg = _centreline(IIO.load_pred_instances(_path(row, "filaseg_stage3")))
    # Matching/scoring runs on the true 1 px skeletons above; only the
    # arrays handed to imshow below are thickened, and only for display.
    mt_match, fs_match = one_to_one_matches(gt, minimum_turn), one_to_one_matches(gt, filaseg)
    minimum_turn_display = _thicken_for_display(minimum_turn, centreline_dilation_px)
    filaseg_display = _thicken_for_display(filaseg, centreline_dilation_px)
    panels[0].imshow(mask, cmap="gray", vmin=0, vmax=1)
    panels[1].imshow(_rgb_instances(gt, {}, gt=True))
    _draw_instances(panels[2], minimum_turn_display, mt_match, gt)
    _draw_instances(panels[3], filaseg_display, fs_match, gt)
    # 10.6 pt, ~7 pt rendered at the ~0.66 page scale: one step down from the
    # two-line label it replaces, so three rotated lines clear the row height.
    panels[0].set_ylabel(_row_label(chosen), fontsize=10.6, linespacing=1.4)
    for column, ax in enumerate(panels):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if with_headers:
            # 8.5 pt rendered at 0.660 page scale.
            ax.set_title(COLUMN_HEADERS[column], fontsize=12.9, pad=6)


def render_figure(
    audit: Mapping[str, Any],
    bands: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    output_base: Path,
    *,
    centreline_dilation_px: int = DEFAULT_CENTRELINE_DILATION_PX,
) -> None:
    """Render every band into one figure and save it as PDF and PNG."""
    # Same shared rcParams (frameless legends, bold axes titles, 11 pt base)
    # as every other paper figure; must run before the figure is created.
    apply_style()
    n_cols = len(COLUMN_HEADERS)
    rows_per_band = max(len(rows) for _title, rows in bands)
    # Panels are square (the scenes are), so a row is as tall as a column is
    # wide; deriving the height this way leaves no dead space inside a band.
    panel_in = ((GRID_RIGHT - GRID_LEFT) * FIG_WIDTH_IN
                / (n_cols + COL_WSPACE * (n_cols - 1)))
    band_in = panel_in * (rows_per_band + ROW_HSPACE * (rows_per_band - 1))
    header_in = PAD_TOP_IN + BAND_TITLE_IN + COLUMN_HEADER_IN
    fig_height_in = (header_in + band_in + BAND_GAP_IN + band_in + PAD_BOTTOM_IN)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in))

    def _y(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / fig_height_in

    for band_index, (title, selected) in enumerate(bands):
        top_in = header_in + band_index * (band_in + BAND_GAP_IN)
        grid = fig.add_gridspec(
            len(selected), n_cols, left=GRID_LEFT, right=GRID_RIGHT,
            top=_y(top_in), bottom=_y(top_in + band_in),
            wspace=COL_WSPACE, hspace=ROW_HSPACE,
        )
        # Band title, left-aligned on the panel grid.  9.5 pt rendered at
        # 0.660 page scale (one step above the column titles).
        title_top_in = (PAD_TOP_IN if band_index == 0
                        else top_in - BAND_TITLE_IN - BAND_TITLE_CLEAR_IN)
        fig.text(GRID_LEFT, _y(title_top_in), title, ha="left", va="top",
                 fontsize=14.4, fontweight="bold")
        if band_index:
            # Thin neutral rule in the gap above this band, so the two bands
            # read as separate blocks rather than one six-deep grid.
            rule_y = _y(top_in - BAND_GAP_IN + RULE_OFFSET_IN)
            fig.add_artist(Line2D(RULE_X, (rule_y, rule_y), transform=fig.transFigure,
                                  color=GRAY, lw=0.8, solid_capstyle="butt"))
        for row_index, chosen in enumerate(selected):
            panels = [fig.add_subplot(grid[row_index, column]) for column in range(n_cols)]
            _draw_scene(audit, chosen, panels,
                        centreline_dilation_px=centreline_dilation_px,
                        with_headers=band_index == 0 and row_index == 0)
    handles = [
        Line2D([], [], color=FALSE_COLOR, lw=4, label="False instance"),
        Line2D([], [], color=MISSED_COLOR, lw=1.2, ls="--", label="Missed GT instance"),
    ]
    # 7.5 pt rendered at 0.660 page scale.
    fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               fontsize=11.4)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rows-per-band",
        type=int,
        default=DEFAULT_ROWS_PER_BAND,
        help=(
            "Scenes shown per band, taken in stored order from the head of "
            "each audit category (default: "
            f"{DEFAULT_ROWS_PER_BAND})."
        ),
    )
    parser.add_argument(
        "--centreline-dilation-px",
        type=int,
        default=DEFAULT_CENTRELINE_DILATION_PX,
        help=(
            "Display-only dilation width (pixels) applied to the rendered "
            "minimum-turn and FilaSeg centreline panels. Does not affect "
            "matching, scoring, or any stored result (default: "
            f"{DEFAULT_CENTRELINE_DILATION_PX})."
        ),
    )
    args = parser.parse_args(argv)
    if args.rows_per_band < 1:
        parser.error("--rows-per-band must be at least 1")
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    selected = read_selection(audit, rows_per_band=args.rows_per_band)
    bands = [(title, selected[key]) for key, title in SELECTIONS]
    render_figure(
        audit, bands, args.output_dir / "fig_paired_locked_examples",
        centreline_dilation_px=args.centreline_dilation_px,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
