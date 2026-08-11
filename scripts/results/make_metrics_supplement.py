"""Derive the two manuscript-facing quantities the metrics-study export omitted.

`results/plecta_metrics_audit.json` is the manuscript record exported by
`C:/Repos/comparisons/metrics_study/scripts/export_manuscript.py`. It carries
shared-ownership precision and recall but not the pixel counts they are formed
from, and it carries the GraFT-frame scores but not the provenance of the frame
they were computed on. Both are needed in the prose, so this script copies them
out of the study's own artefacts into a second record.

Nothing is computed here that the study did not already compute: the shared
pixel counts are per-scene columns of its `per_scene_methods.csv`, averaged over
scenes, and the frame provenance is copied verbatim. Run only when the study is
present; the emitted record is what the manuscript build reads.

    python scripts/results/make_metrics_supplement.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
STUDY = Path(r"C:/Repos/comparisons/metrics_study")
PER_SCENE = STUDY / "results" / "per_scene_methods.csv"
FRAME_PROVENANCE = STUDY / "data" / "graft_native" / "provenance.json"
OUT = RESULTS / "plecta_metrics_supplement.json"


def shared_ownership(rows: list[dict]) -> dict:
    """Mean shared-pixel counts per scene, reference against prediction.

    Scenes a method did not complete are excluded, exactly as they are from its
    scores; the reference mean is therefore stated per method so that the ratio
    is formed on the scenes that method actually contributed.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        grouped[(row["mask_variant"], row["method"])].append(row)

    out: dict[str, dict] = {}
    for (variant, method), scenes in sorted(grouped.items()):
        reference = sum(float(r["shared_gt_px"]) for r in scenes) / len(scenes)
        predicted = sum(float(r["shared_pred_px"]) for r in scenes) / len(scenes)
        out.setdefault(variant, {})[method] = {
            "n_scenes": len(scenes),
            "reference_shared_px_mean": reference,
            "predicted_shared_px_mean": predicted,
            "predicted_over_reference": predicted / reference,
        }
    return out


def main() -> None:
    with PER_SCENE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    frame = json.loads(FRAME_PROVENANCE.read_text(encoding="utf-8"))

    payload = {
        "description": (
            "Two quantities derived from the metrics study's own artefacts and "
            "not present in results/plecta_metrics_audit.json: the shared-pixel "
            "counts behind its shared-ownership precision and recall, and the "
            "provenance of GraFT's bundled synthetic frame."
        ),
        "provenance": {
            "study": str(STUDY),
            "per_scene_methods": str(PER_SCENE),
            "frame_provenance": str(FRAME_PROVENANCE),
            "generator": "scripts/results/make_metrics_supplement.py",
        },
        "shared_ownership_pixels": shared_ownership(rows),
        "graft_native_frame": {
            "n_instances": frame["n_instances"],
            "side_px": frame["shape"][0],
            "reference_matches_vendor_nonoise_image": frame[
                "gt_union_matches_nonoise_image"
            ],
            "licence": frame["licence"],
            "source": frame["source"],
            "note": frame["note"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
