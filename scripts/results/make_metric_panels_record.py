"""The three reported measures against coverage, for the two comparison figures.

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

All three come from one already-computed study, ``metrics_study``, which ran
every method on the same 50 scenes through one scorer.  Nothing is recomputed
here: this reads its two per-scene CSVs, aggregates them per coverage stratum,
and writes the committed extract the two figure generators plot.

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
conditions, every stratum -- before anything new it emits is believed.  A
regrouped CSV yields plausible-looking numbers when it is wrong, and this is
what catches that.

**What is aggregated how.**  Every panel value is a plain mean of per-scene
values, so the three panels are the same kind of average.  GraFT did not finish
3 of the 50 degraded scenes and 1 of the clean ones inside its wall-clock
budget; those are excluded from its means, they are the hardest scenes of the
densest stratum, and every affected cell carries its own n.

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

DENSITIES = (20, 30, 40, 50, 60)
CONDITIONS = ("degraded", "clean")
METHODS = (("plecta", "PLECTA"), ("graft", "GraFT"), ("dnai", "DNAi"))

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
        for key, _label in METHODS:
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
        out[condition] = per_method
    return out


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
    return failures


def main() -> int:
    panels = aggregate()
    failures = check_parity(panels)
    if failures:
        raise SystemExit("PARITY FAILED, refusing to write:\n  "
                         + "\n  ".join(failures))

    payload = {
        "role": ("the three reported measures against target areal coverage, "
                 "for the two comparison figures of Section 3.3; one mask "
                 "condition each"),
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
                   "within %g, including GraFT's censored counts" % PARITY_TOL),
        "source_record": {
            "study": str(STUDY).replace("\\", "/"),
            "per_scene": {"file": "results/per_scene_methods.csv",
                          "sha256": sha256(PER_SCENE)},
            "detection": {"file": "results/detection_tolerance_sweep.csv",
                          "sha256": sha256(DETECTION)},
            "chance": {"file": "results/crossing_fidelity_chance.json",
                       "sha256": sha256(CHANCE)},
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
