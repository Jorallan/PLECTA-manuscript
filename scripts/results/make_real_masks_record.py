"""Rebuild the real-SEM record from the study that produced it.

Every other number in this paper traces to a stored evaluation record. These six
did not: `results/plecta_real_masks.json` carried the values with no
`source_record`, and the algorithm repository's own report reads them back out
of the manuscript, so the manuscript cited the repository and the repository
cited the manuscript.

Both sources exist and are named here, so the record is now derived rather than
transcribed:

  reconstruction scores and filament counts
      real_sem_study/results.json, cells/<scene>/"1. stubmatch, frozen"
  mask quality of the upstream axes
      real_sem_study/comparison_figures/nnunet_vs_annotation/metrics.json

The manual annotations these ultimately rest on are held outside any repository,
at C:/Repos/datasets/real/manual annotation, one directory per field, each
carrying the per-instance centrelines, the overlap-aware multilabel and the
original micrograph.

Usage:
  python scripts/results/make_real_masks_record.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
STUDY = Path("C:/Repos/comparisons/real_sem_study")
SCORES = STUDY / "results.json"
MASK_QUALITY = (STUDY / "comparison_figures" / "nnunet_vs_annotation"
                / "metrics.json")
ANNOTATIONS = Path("C:/Repos/datasets/real/manual annotation")

FROZEN = "1. stubmatch, frozen"

# manuscript name -> (scene key for the manual axis, scene key for the U-Net
# axis, whether the field was in the upstream segmenter's training data)
FIELDS = {
    "B58_100": ("b58_skel", "b58_nnunet", "image present in training data"),
    "B58_110": ("b110_skel", "b110_nnunet",
                "verified absent from relevant training set"),
}

METRICS = ("f1", "precision", "recall")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(scores: dict, scene: str) -> dict:
    block = scores["cells"][scene][FROZEN]
    out = {m: round(float(block[m]), 4) for m in METRICS if m in block}
    for alias in ("ari", "adjusted_rand_index"):
        if alias in block:
            out["ari"] = round(float(block[alias]), 4)
            break
    return out


def main() -> None:
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    quality = json.loads(MASK_QUALITY.read_text(encoding="utf-8"))
    # The mask-quality file is keyed by the manuscript's own image names and
    # carries the filament counts as well, so it is the anchor.
    by_image = {row["image"]: row for row in quality}

    images = []
    for name, (manual_key, unet_key, training) in FIELDS.items():
        mask = by_image[name]
        images.append({
            "image": name,
            "n_filaments": int(mask["n_filaments"]),
            "segmenter_training_status": training,
            "segmenter_training_note": mask.get("note", "").strip(),
            "axis_recall": mask["axis_recall"],
            "axis_precision": mask["mask_precision"],
            "filaments_lost": mask["filaments_lost"],
            "pieces_per_covered_filament": mask["pieces_per_covered"],
            "plecta_manual_axis": cell(scores, manual_key),
            "plecta_unet_axis": cell(scores, unet_key),
        })

    payload = {
        "role": "two-image exploratory mask-source study; no inference",
        "source_record": {
            "study": str(STUDY).replace("\\", "/"),
            "scores": {
                "file": str(SCORES.relative_to(STUDY)).replace("\\", "/"),
                "sha256": sha256(SCORES),
                "path_in_file": f'cells/<scene>/"{FROZEN}"',
            },
            "mask_quality": {
                "file": str(MASK_QUALITY.relative_to(STUDY)).replace("\\", "/"),
                "sha256": sha256(MASK_QUALITY),
            },
            "annotations": {
                "root": str(ANNOTATIONS).replace("\\", "/"),
                "fields": sorted(p.name for p in ANNOTATIONS.iterdir()
                                 if p.is_dir()) if ANNOTATIONS.is_dir() else [],
                "note": ("held outside any repository; each field carries "
                         "per-instance centrelines, an overlap-aware "
                         "multilabel and the original micrograph"),
            },
        },
        "images": images,
    }

    destination = RESULTS / "plecta_real_masks.json"
    previous = json.loads(destination.read_text(encoding="utf-8"))
    destination.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {destination}")

    # Every value the manuscript already prints must survive unchanged.
    old = {i["image"]: i for i in previous["images"]}
    for image in payload["images"]:
        before = old.get(image["image"], {})
        for key, value in image.items():
            if key in before and before[key] != value:
                print(f"  CHANGED {image['image']}.{key}: "
                      f"{before[key]} -> {value}")
        print(f"  {image['image']}: manual f1 {image['plecta_manual_axis']['f1']}, "
              f"U-Net f1 {image['plecta_unet_axis']['f1']}")


if __name__ == "__main__":
    main()
