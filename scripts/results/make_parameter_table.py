"""Emit the released parameters as tables and macros, from the released file.

Source: ``plecta/parameters.yaml`` in the PLECTA repository -- the single
run-time source of truth, which carries every tunable of both stages grouped by
the stage it belongs to. ``plecta/params.json``, the immutable flat record of
the 2-D configuration that produced every published number, is read alongside
it as a cross-check: this script asserts that every key in the frozen record
agrees with ``grouping_2d``, the same assertion ``plecta.parameters
.verify_against_frozen()`` makes inside the package. If the two ever disagree
the script stops rather than publishing a number that no released file holds.

The supplementary material's technical section quotes these values, and quoting
them by hand would let the paper and the method drift apart silently: a
parameter could be retuned in the code and the manuscript would go on reporting
the old number with no error anywhere. Generating them means the manuscript
cannot disagree with the configuration it describes.

Three products:

* ``plecta_parameter_table.tex`` -- the 2-D cost weights and scales, junction
  set beside gap set, since the two differ and the difference is the point;
* ``plecta_depth_parameter_table.tex`` -- the depth stage's own entries, so the
  half of the released file the paper claims to publish is actually shown;
* ``plecta_parameters.tex`` -- macros for the individual thresholds the prose
  names in passing, from both stages.

Override the two source paths with ``PLECTA_PARAMS_YAML`` and ``PLECTA_PARAMS``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results"
PARAMS = Path(os.environ.get(
    "PLECTA_PARAMS_YAML", r"C:\Repos\PLECTA\plecta\parameters.yaml"))
FROZEN = Path(os.environ.get(
    "PLECTA_PARAMS", r"C:\Repos\PLECTA\plecta\params.json"))

#  (symbol, description, junction key, gap key). None means the term does not
#  apply to that link type rather than being zero by coincidence.
ROWS = (
    (r"$w_d$",      "reversal angle $\\theta$",           "j_w_direct",   "g_w_direct"),
    (r"$w_t$",      "chord turns $\\varphi_a+\\varphi_b$", "j_w_turn",     "g_w_turn"),
    (r"$w_\kappa$", "curvature continuity",               "j_w_kappa",    "g_w_kappa"),
    (r"$w_\ell$",   "length $d/\\ell_0$",                  None,           "g_w_len"),
    (r"$\ell_0$",   "length scale (px)",                  None,           "g_len_scale"),
    (r"$d_0$",      "chord-fade scale (px)",              "j_chord_floor", "g_chord_floor"),
    (r"$p_i$",      "price of leaving a stub free",       "j_unmatched",  "g_unmatched"),
)

MACROS = (
    ("PlectaParamSpurPx",        "spur_px",        "%.0f"),
    ("PlectaParamAbsorbPx",      "bridge_px",      "%.0f"),
    ("PlectaParamFreeDebrisPx",  "absorb_free_px", "%.0f"),
    ("PlectaParamJoinPx",        "join_px",        "%.0f"),
    ("PlectaParamWindowLocal",   "window_local",   "%.0f"),
    ("PlectaParamWindowChain",   "window_chain",   "%.0f"),
    ("PlectaParamMinQuadratic",  "min_quadratic",  "%.0f"),
    ("PlectaParamRounds",        "n_rounds",       "%.0f"),
    ("PlectaParamAnnealStart",   "anneal_start",   "%.2f"),
    ("PlectaParamKappaScale",    "kappa_scale",    "%.0f"),
    ("PlectaParamGapMaxLen",     "gap_max_len",    "%.0f"),
    ("PlectaParamGapMaxTheta",   "gap_max_theta",  "%.2f"),
    ("PlectaParamGapMaxPhi",     "gap_max_phi",    "%.2f"),
    ("PlectaParamJunctionPrice", "j_unmatched",    "%.2f"),
    ("PlectaParamGapPrice",      "g_unmatched",    "%.2f"),
)

#  The depth stage, from the same released file. (macro, dotted path under
#  depth_3d, printf format, symbol as the prose sets it, what it does.)
DEPTH_ROWS = (
    ("PlectaParamCorePx", "crossing_detection.core_px", "%g",
     "core (px)",
     "half-length of the shared crossing arc; flat at every crossing, so no "
     "per-rod radius enters the evidence path"),
    ("PlectaParamWindowPx", "crossing_detection.window_px", "%g",
     "window (px)",
     "arclength sampled either side of a crossing for evidence"),
    ("PlectaParamMinFlankPx", "crossing_detection.min_flank_px", "%g",
     "min.\\ flank",
     "fewer usable flank samples and the crossing abstains"),
    ("PlectaParamAbstainScore", "evidence_weights.abstain_score", "%.2f",
     "abstain $\\lvert s\\rvert$",
     "abstention threshold, in score units, not a probability band"),
    ("PlectaParamExactMaxNodes", "solver.exact_max_nodes", "%g",
     "exact limit",
     "components to this many nodes are ordered exactly, larger ones "
     "greedily"),
)


def _num(value: float) -> str:
    """Trailing zeros are noise in a parameter table."""
    text = f"{float(value):g}"
    return text


def _flatten(tree: dict) -> dict:
    """parameters.yaml groups by concern; params.json and the prose do not."""
    flat: dict = {}
    for group in tree.values():
        if not isinstance(group, dict):
            continue
        for key, value in group.items():
            if key in flat and flat[key] != value:
                raise SystemExit(
                    f"{PARAMS}: key {key!r} appears twice with different "
                    f"values ({flat[key]!r}, {value!r})")
            flat[key] = value
    return flat


def _dig(tree: dict, dotted: str):
    node = tree
    for part in dotted.split("."):
        if part not in node:
            raise SystemExit(f"{PARAMS}: no key {dotted!r} (missing {part!r})")
        node = node[part]
    return node


def main() -> int:
    if not PARAMS.is_file():
        raise SystemExit(
            f"released parameter file not found: {PARAMS}. Set "
            "PLECTA_PARAMS_YAML to the parameters.yaml of the PLECTA "
            "checkout.")
    if not FROZEN.is_file():
        raise SystemExit(
            f"frozen parameter file not found: {FROZEN}. Set PLECTA_PARAMS to "
            "the params.json of the PLECTA checkout.")
    tree = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    p = _flatten(tree["grouping_2d"])

    #  The frozen record is the cross-check, not the source. Any disagreement
    #  means the published numbers and the released file have parted company.
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    drift = {k: (v, p.get(k)) for k, v in frozen.items() if p.get(k) != v}
    if drift:
        raise SystemExit(
            f"{PARAMS} disagrees with the frozen record {FROZEN} on "
            f"{sorted(drift)}: {drift}. Published numbers were produced under "
            "the frozen values; resolve this before regenerating.")

    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Symbol & Term & Junction & Gap \\", r"\midrule"]
    for sym, term, jkey, gkey in ROWS:
        j = _num(p[jkey]) if jkey else "---"
        g = _num(p[gkey]) if gkey else "---"
        lines.append(f"{sym} & {term} & {j} & {g} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "plecta_parameter_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    depth = tree["depth_3d"]
    #  A paragraph column: the descriptions are sentences, and a bare `l`
    #  column ran the table 32 pt past \linewidth.
    dlines = [r"\begin{tabular}{lc>{\raggedright\arraybackslash}p{0.58\linewidth}}",
              r"\toprule",
              r"Quantity & Value & What it sets \\", r"\midrule"]
    for _macro, dotted, _fmt, label, what in DEPTH_ROWS:
        dlines.append(f"{label} & {_num(_dig(depth, dotted))} & {what} " + r"\\")
    dlines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "plecta_depth_parameter_table.tex").write_text(
        "\n".join(dlines) + "\n", encoding="utf-8")

    macro_lines = ["% Generated by scripts/results/make_parameter_table.py;"
                   " do not edit.",
                   "% Source: " + str(PARAMS).replace("\\", "/")
                   + ", cross-checked against "
                   + str(FROZEN).replace("\\", "/") + "."]
    for name, key, fmt in MACROS:
        macro_lines.append(f"\\newcommand{{\\{name}}}{{{fmt % p[key]}}}")
    # The admissibility limits follow from the prices and are quoted as such.
    macro_lines.append(
        f"\\newcommand{{\\PlectaParamJunctionLimit}}"
        f"{{{2 * p['j_unmatched']:.2f}}}")
    macro_lines.append(
        f"\\newcommand{{\\PlectaParamGapLimit}}{{{2 * p['g_unmatched']:.2f}}}")
    for name, dotted, fmt, _label, _what in DEPTH_ROWS:
        macro_lines.append(
            f"\\newcommand{{\\{name}}}{{{fmt % _dig(depth, dotted)}}}")
    (OUT / "plecta_parameters.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT / 'plecta_parameter_table.tex'}")
    print(f"wrote {OUT / 'plecta_depth_parameter_table.tex'} "
          f"({len(DEPTH_ROWS)} depth-stage rows)")
    print(f"wrote {OUT / 'plecta_parameters.tex'} "
          f"({len(MACROS) + 2 + len(DEPTH_ROWS)} macros, from {PARAMS}, "
          f"cross-checked against {FROZEN})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
