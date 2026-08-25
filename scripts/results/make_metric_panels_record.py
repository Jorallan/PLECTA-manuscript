"""The three reported measures against coverage, for the comparison figure.

Section 3.3 reports one endpoint per method per coverage stratum.  Three
measures are now reported instead of one, because they answer different
questions and none of them answers all three:

  * **common-fragment pairwise F1** counts *pairs* of fragments, so it sees an
    identity swap at a crossing and is quadratic in fragmentation;
  * **detection F1 at IoU 0.1 with a 2 px placement tolerance** counts
    *objects*, so it is the fairest reading of a fragmented mask and is blind to
    an arm swap -- it scores an inverted reconstruction 1.000 where pairwise
    scores 0.000;
  * **Crossing Fidelity** counts *crossings*: the fraction of reference
    crossings whose incident arms the prediction partitions exactly as the
    reference does.  It states the paper's claim directly, and it is
    mechanistically distinct from the other two because a crossing carries no
    information about a *gap* -- which is why its correlation with pairwise F1
    falls from 0.876 on clean masks to 0.747 on degraded ones.

For PLECTA, GraFT and DNAi all three come from one already-computed study,
``metrics_study``, which ran them on the same 50 scenes through one scorer.
SIFNE and Basu were run by their own studies and are read from those: nothing
is recomputed here either way.  This aggregates per coverage stratum and writes
the committed extract the figure generator plots.

**Five methods, and two of them are qualified.**  SIFNE is run from its own
released source, but at parameters we selected on development scenes -- the
dagger in ``tab:plecta-comparators``.  Basu is OUR REIMPLEMENTATION of Stage B,
because no implementation of that paper has ever been released -- the star.
Both qualifications travel with the curves; a legend that drops them claims two
external results the study does not have.

**Crossing Fidelity carries a chance level, and it is not the one the
manuscript used to print.**  0.709 is ``null_pair_accuracy``, the chance level
of *pair accuracy* -- a different statistic, and the one that has been dropped.
The do-nothing prediction that leaves every arm unpaired resolves 0.088 of the
degraded set's crossings exactly and 0.080 of the clean set's, measured by
``metrics_study/scripts/crossing_fidelity_chance.py`` by running the stored
scorer with an empty prediction.  Quoting 0.709 beside a fidelity of 0.809
would have understated the result by an order of magnitude in the free term.

**Parity.**  The aggregation is gated on reproducing the per-density means the
manuscript already prints from ``plecta_dnai_comparison.json`` and
``plecta_graft_comparison.json`` -- PLECTA, DNAi and GraFT, both mask
conditions, every stratum -- and the overall means of
``plecta_sifne_comparison.json``, ``plecta_basu_reimplementation.json`` and the
Basu study's own ``eval_testcmp10.json``, before anything new it emits is
believed.  A regrouped record yields plausible-looking numbers when it is
wrong, and this is what catches that.  The two upstream records that had to be
computed for this figure carry their own gates: SIFNE's detection F1 will not
be written unless PLECTA's detection column reproduces through the same call,
and Basu's per-scene cells will not be written unless every recomputed pairwise
F1 reproduces ``eval_testcmp10.json`` exactly.

**What is aggregated how.**  Every panel value is a plain mean of per-scene
values, so the three panels are the same kind of average.  GraFT did not finish
3 of the 50 degraded scenes and 1 of the clean ones inside its wall-clock
budget; those are excluded from its means, they are the hardest scenes of the
densest stratum, and every affected cell carries its own n.  Neither SIFNE nor
Basu was censored: every one of their cells is required to be a full ten
scenes, and the gate fails if it is not.

    python scripts/results/make_metric_panels_record.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
STUDY = Path("C:/Repos/comparisons/metrics_study")
PER_SCENE = STUDY / "results" / "per_scene_methods.csv"
DETECTION = STUDY / "results" / "detection_tolerance_sweep.csv"
CHANCE = STUDY / "results" / "crossing_fidelity_chance.json"

#: SIFNE and Basu were run by their own studies, not by ``metrics_study``, so
#: their per-scene rows come from those studies' own records rather than from
#: the shared CSV.  Neither record carried detection F1 when this was written:
#: ``sifne_comparison/scripts/detection_testcmp10.py`` computes it from the
#: pixel lists SIFNE already wrote, and
#: ``basu_comparison/scripts/make_panel_cells.py`` computes all three measures
#: per scene at the frozen tuned parameters.  Both are gated on reproducing a
#: number their own study already stored before anything new is believed.
SIFNE = Path("C:/Repos/comparisons/sifne_comparison")
SIFNE_PER_SCENE = {c: SIFNE / "results" / ("per_scene_testcmp10_%s.json" % c)
                   for c in ("degraded", "clean")}
SIFNE_DETECTION = SIFNE / "results" / "detection_testcmp10.json"
BASU = Path("C:/Repos/comparisons/basu_comparison")
BASU_PANEL_CELLS = BASU / "results" / "panel_cells_testcmp10.json"
BASU_EVAL = BASU / "results" / "eval_testcmp10.json"

DENSITIES = (20, 30, 40, 50, 60)
CONDITIONS = ("degraded", "clean")
#: Ordered by overall F1 on the degraded axis, descending -- the order
#: ``tab:plecta-comparators`` prints and the order the figure's legend uses.
METHODS = (("plecta", "PLECTA"), ("sifne", "SIFNE"), ("basu", "Basu"),
           ("graft", "GraFT"), ("dnai", "DNAi"))
#: The three ``metrics_study`` ran itself, and so the three its shared CSVs
#: carry.  SIFNE and Basu are read from their own studies below.
CSV_METHODS = ("plecta", "graft", "dnai")

#: DNAi's own shipped IoU threshold, and the placement tolerance that makes an
#: object IoU usable between two independently read one-pixel centrelines.
DETECTION_COLUMN = "det_0.1_t2"
IOU_THRESHOLD = 0.1
TOLERANCE_PX = 2

PARITY_TOL = 1e-5


def num(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    kept = [v for v in values if not math.isnan(v)]
    return sum(kept) / len(kept) if kept else float("nan")


def aggregate() -> dict:
    rows = list(csv.DictReader(PER_SCENE.open(encoding="utf-8")))
    det = list(csv.DictReader(DETECTION.open(encoding="utf-8")))

    out = {}
    for condition in CONDITIONS:
        per_method = {}
        for key in CSV_METHODS:
            cells = {}
            for density in DENSITIES:
                scenes = [r for r in rows
                          if r["mask_variant"] == condition
                          and r["method"] == key
                          and int(r["density"]) == density]
                done = [r for r in scenes if r["status"] == "ok"]
                dscenes = [r for r in det
                           if r["variant"] == condition and r["method"] == key
                           and r["scene"].startswith("cov%d/" % density)]
                cells[str(density)] = {
                    "f1": mean([num(r["f1"]) for r in done]),
                    "detection_f1": mean([num(r[DETECTION_COLUMN])
                                          for r in dscenes]),
                    "crossing_fidelity": mean([num(r["jr_exact_rate"])
                                               for r in done]),
                    "ari": mean([num(r["adjusted_rand_index"]) for r in done]),
                    "vi_split_bits": mean([num(r["vi_split_bits"])
                                           for r in done]),
                    "vi_merge_bits": mean([num(r["vi_merge_bits"])
                                           for r in done]),
                    "vi_total_bits": mean([num(r["vi_split_bits"])
                                           + num(r["vi_merge_bits"])
                                           for r in done]),
                    "n": len(done),
                    "n_attempted": len(scenes),
                    "n_detection": len(dscenes),
                }
            per_method[key] = cells
        per_method["sifne"] = sifne_cells(condition)
        per_method["basu"] = basu_cells(condition)
        out[condition] = per_method
    return out


def sifne_cells(condition: str) -> dict:
    """SIFNE's cells: its own per-scene record, plus the detection record.

    ``run_sifne.py`` writes pairwise F1, ARI, VI and Crossing Fidelity per
    scene, rounded to four places; detection F1 is not in it and comes from
    ``detection_testcmp10.py``, which recomputed it from the same stored pixel
    lists and would not write unless PLECTA's detection column reproduced.
    """
    rows = json.loads(SIFNE_PER_SCENE[condition].read_text(encoding="utf-8"))
    det = json.loads(SIFNE_DETECTION.read_text(encoding="utf-8"))["per_scene"]
    cells = {}
    for density in DENSITIES:
        cov = "cov%d" % density
        done = [r for r in rows if r["cov"] == cov and r["status"] == "ok"]
        dscenes = [r for r in det if r["variant"] == condition
                   and r["scene"].startswith(cov + "/")]
        cells[str(density)] = {
            "f1": mean([float(r["f1"]) for r in done]),
            "detection_f1": mean([float(r["detection_f1"]) for r in dscenes]),
            "crossing_fidelity": mean([float(r["cf_exact_rate"])
                                       for r in done]),
            "ari": mean([float(r["adjusted_rand_index"]) for r in done]),
            "vi_total_bits": mean([float(r["vi_total_bits"]) for r in done]),
            "n": len(done),
            "n_attempted": len([r for r in rows if r["cov"] == cov]),
            "n_detection": len(dscenes),
        }
    return cells


def basu_cells(condition: str) -> dict:
    """Basu's cells, from the per-scene record of the reimplementation study."""
    rows = json.loads(BASU_PANEL_CELLS.read_text(
        encoding="utf-8"))["per_scene"]
    cells = {}
    for density in DENSITIES:
        cov = "cov%d" % density
        done = [r for r in rows
                if r["variant"] == condition and r["cov"] == cov]
        cells[str(density)] = {
            "f1": mean([float(r["f1"]) for r in done]),
            "detection_f1": mean([float(r["detection_f1"]) for r in done]),
            "crossing_fidelity": mean([float(r["crossing_fidelity"])
                                       for r in done]),
            "ari": mean([float(r["adjusted_rand_index"]) for r in done]),
            "vi_total_bits": mean([float(r["vi_total_bits"]) for r in done]),
            "n": len(done),
            "n_attempted": len(done),
            "n_detection": len(done),
        }
    return cells


def chance_levels() -> dict:
    """What the do-nothing prediction scores on Crossing Fidelity."""
    payload = json.loads(CHANCE.read_text(encoding="utf-8"))
    return {
        condition: {
            "pooled": block["null_exact_rate_pooled"],
            "n_crossings": block["n_junctions"],
            "by_density": {d: cell["null_exact_rate_pooled"]
                           for d, cell in block["by_density"].items()},
        }
        for condition, block in payload["summary"].items()
    }


def check_parity(panels: dict) -> list[str]:
    """Reproduce the per-density means the manuscript already prints."""
    failures = []

    def compare(what, want, got):
        if math.isnan(want) or math.isnan(got) or abs(want - got) > PARITY_TOL:
            failures.append("%s: stored %.6f, recomputed %.6f"
                            % (what, want, got))

    dnai = json.loads((RESULTS / "plecta_dnai_comparison.json")
                      .read_text(encoding="utf-8"))
    graft = json.loads((RESULTS / "plecta_graft_comparison.json")
                       .read_text(encoding="utf-8"))
    for condition in CONDITIONS:
        stored = dnai["per_density"][condition]
        for key, arm in (("plecta", "plecta"), ("dnai", "dnai_tuned")):
            for density in DENSITIES:
                compare("%s %s %d%% F1" % (condition, key, density),
                        stored[arm][str(density)]["f1"]["mean"],
                        panels[condition][key][str(density)]["f1"])
        gb = graft["arms"]["graft|%s" % condition]["by_density"]
        for density in DENSITIES:
            cell = panels[condition]["graft"][str(density)]
            compare("%s graft %d%% F1" % (condition, density),
                    gb[str(density)]["f1"], cell["f1"])
            if gb[str(density)]["n"] != cell["n"]:
                failures.append("%s graft %d%%: stored n=%d, recomputed n=%d"
                                % (condition, density, gb[str(density)]["n"],
                                   cell["n"]))

    #  SIFNE and Basu have no per-density record in the manuscript, so the
    #  gate is their overall mean: the five strata are ten scenes each, so a
    #  plain mean of the five cells is the mean over the fifty scenes, and it
    #  must reproduce what each study's own committed record states.  Every
    #  cell must also be a full ten scenes: neither method was censored, and a
    #  silently short stratum is exactly what a regrouped record produces.
    def overall(key, condition):
        return mean([panels[condition][key][str(d)]["f1"] for d in DENSITIES])

    for key in ("sifne", "basu"):
        for condition in CONDITIONS:
            for density in DENSITIES:
                n = panels[condition][key][str(density)]["n"]
                if n != 10:
                    failures.append("%s %s %d%%: n=%d, expected 10"
                                    % (condition, key, density, n))

    #  SIFNE's per-scene file is rounded to four places, so its mean lands
    #  2e-6 from the record's full-precision overall -- inside PARITY_TOL, and
    #  the reason this is a mean rather than an identity.
    sifne = json.loads((RESULTS / "plecta_sifne_comparison.json")
                       .read_text(encoding="utf-8"))
    compare("degraded sifne overall F1", sifne["overall"]["f1"],
            overall("sifne", "degraded"))
    basu = json.loads((RESULTS / "plecta_basu_reimplementation.json")
                      .read_text(encoding="utf-8"))
    compare("degraded basu overall F1", basu["overall"]["f1"],
            overall("basu", "degraded"))
    basu_eval = json.loads(BASU_EVAL.read_text(encoding="utf-8"))["means"]
    for condition in CONDITIONS:
        compare("%s basu overall F1" % condition,
                basu_eval["tuned_%s" % condition], overall("basu", condition))
    return failures


def main() -> int:
    panels = aggregate()
    failures = check_parity(panels)
    if failures:
        raise SystemExit("PARITY FAILED, refusing to write:\n  "
                         + "\n  ".join(failures))

    payload = {
        "role": ("the three reported measures against target areal coverage, "
                 "for the two-row comparison figure of Section 3.3; five "
                 "methods, both mask conditions"),
        "scenes": ("50 scenes, ten at each of 20, 30, 40, 50 and 60 % target "
                   "areal coverage, identical input masks for every method, "
                   "one scorer"),
        "aggregation": ("plain mean of per-scene values in every panel; no "
                        "interval is drawn and none is claimed"),
        "detection_metric": ("DNAi's greedy-IoU object detection F1 at its "
                             "shipped IoU threshold of %g, with a %d px "
                             "symmetric placement tolerance; without the "
                             "tolerance an object IoU between two "
                             "independently read one-pixel centrelines "
                             "measures placement rather than shape"
                             % (IOU_THRESHOLD, TOLERANCE_PX)),
        "crossing_fidelity": ("the fraction of reference crossings whose "
                              "incident arms the prediction partitions exactly "
                              "as the reference does, per scene, then "
                              "averaged. Only junctions the reference itself "
                              "treats as a crossing are scored: a node whose "
                              "arms all belong to one filament is a "
                              "skeletonisation artefact and scoring it would "
                              "reward doing nothing"),
        "crossing_fidelity_chance": ("the do-nothing prediction that leaves "
                                     "every arm unpaired, scored through the "
                                     "same code; it is not the 0.709 the "
                                     "manuscript used to print, which is the "
                                     "chance level of pair accuracy, a "
                                     "different statistic"),
        "censoring": ("GraFT exceeded its per-scene wall-clock budget on 3 of "
                      "the 50 degraded scenes and 1 of the clean ones, all in "
                      "the densest stratum; they are excluded from its means, "
                      "they are the hardest scenes of the set, and so its "
                      "curve is optimistic"),
        "iou_threshold": IOU_THRESHOLD,
        "tolerance_px": TOLERANCE_PX,
        "densities": list(DENSITIES),
        "parity": ("reproduces every per-density mean already printed in "
                   "results/plecta_dnai_comparison.json and "
                   "results/plecta_graft_comparison.json, both conditions, "
                   "within %g, including GraFT's censored counts; and the "
                   "overall means of results/plecta_sifne_comparison.json, "
                   "results/plecta_basu_reimplementation.json and the Basu "
                   "study's own eval_testcmp10.json, with every SIFNE and "
                   "Basu stratum required to be a full ten scenes"
                   % PARITY_TOL),
        "comparator_provenance": ("SIFNE is run from its own released source "
                                  "at parameters we selected on development "
                                  "scenes, marked with a dagger in the "
                                  "manuscript; Basu is OUR REIMPLEMENTATION "
                                  "of Stage B, because no implementation of "
                                  "that paper was ever released, marked with "
                                  "a star. Neither qualification may be "
                                  "dropped when these curves are shown"),
        "source_record": {
            "study": str(STUDY).replace("\\", "/"),
            "per_scene": {"file": "results/per_scene_methods.csv",
                          "sha256": sha256(PER_SCENE)},
            "detection": {"file": "results/detection_tolerance_sweep.csv",
                          "sha256": sha256(DETECTION)},
            "chance": {"file": "results/crossing_fidelity_chance.json",
                       "sha256": sha256(CHANCE)},
            "sifne": {
                "study": str(SIFNE).replace("\\", "/"),
                "per_scene_degraded": {
                    "file": "results/per_scene_testcmp10_degraded.json",
                    "sha256": sha256(SIFNE_PER_SCENE["degraded"])},
                "per_scene_clean": {
                    "file": "results/per_scene_testcmp10_clean.json",
                    "sha256": sha256(SIFNE_PER_SCENE["clean"])},
                "detection": {"file": "results/detection_testcmp10.json",
                              "sha256": sha256(SIFNE_DETECTION)},
            },
            "basu": {
                "study": str(BASU).replace("\\", "/"),
                "panel_cells": {"file": "results/panel_cells_testcmp10.json",
                                "sha256": sha256(BASU_PANEL_CELLS)},
                "eval": {"file": "results/eval_testcmp10.json",
                         "sha256": sha256(BASU_EVAL)},
            },
        },
        "chance": chance_levels(),
        "panels": panels,
    }

    destination = RESULTS / "plecta_metric_panels.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print("wrote", destination)
    for condition in CONDITIONS:
        print("\n--", condition)
        print("%-8s %5s %7s %7s %7s %7s %7s %5s"
              % ("method", "cov", "F1", "det", "CF", "ARI", "VI tot", "n"))
        for key, label in METHODS:
            for density in DENSITIES:
                c = panels[condition][key][str(density)]
                print("%-8s %4d%% %7.3f %7.3f %7.3f %7.3f %7.3f %5d"
                      % (label, density, c["f1"], c["detection_f1"],
                         c["crossing_fidelity"], c["ari"], c["vi_total_bits"],
                         c["n"]))
        print("   chance for CF: %.3f over %d crossings"
              % (payload["chance"][condition]["pooled"],
                 payload["chance"][condition]["n_crossings"]))
    print("\nparity: every per-density mean reproduces the committed records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
