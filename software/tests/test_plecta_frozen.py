"""Contract tests for the versioned PLECTA core."""
from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = ROOT / "software"
if str(SOFTWARE) not in sys.path:
    sys.path.insert(0, str(SOFTWARE))

from plecta.predict import PARAMS_PATH, build_params, refuse_locked  # noqa: E402


class FrozenContractTests(unittest.TestCase):
    def test_json_materializes_every_effective_parameter(self) -> None:
        recorded = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(recorded, asdict(build_params()))

    def test_direct_mask_mode_cannot_spend_locked_set(self) -> None:
        with self.assertRaises(SystemExit):
            refuse_locked(Path("somewhere/synthetic_locked_v1/cov20/mask.png"))

    def test_unrelated_path_is_allowed(self) -> None:
        refuse_locked(Path("new_data/cov20/mask.png"))


if __name__ == "__main__":
    unittest.main()
