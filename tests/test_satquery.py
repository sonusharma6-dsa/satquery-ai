import tempfile
import unittest
from pathlib import Path

import numpy as np

from satquery.controller import plan_query, run_analysis
from satquery.report import build_report


def asset(array, modality="unknown", registered=True):
    return {
        "array": array.astype("float32"),
        "bands": array.shape[2],
        "width": array.shape[1],
        "height": array.shape[0],
        "extension": ".png",
        "modality": modality,
        "registered": registered,
    }


class SatQueryCoreTests(unittest.TestCase):
    def test_change_query_plans_bitemporal_analysis(self):
        plan = plan_query("What changed between before and after?", 2)
        self.assertIn("change_analysis", plan.tasks)

    def test_change_analysis_requires_registration(self):
        images = [
            asset(np.zeros((4, 4, 3)), registered=False),
            asset(np.ones((4, 4, 3)), registered=True),
        ]
        with self.assertRaisesRegex(ValueError, "co-registration"):
            run_analysis("what changed", images)

    def test_change_analysis_writes_evidence_and_report(self):
        with tempfile.TemporaryDirectory() as output_dir:
            images = [asset(np.zeros((4, 4, 3))), asset(np.ones((4, 4, 3)))]
            result = run_analysis(
                "what changed",
                images,
                {"change_threshold": 0.18, "output_dir": output_dir},
            )
            evidence = result["results"][0].evidence_paths[0]
            self.assertTrue(Path(evidence).is_file())
            self.assertIn("100.0%", result["answer"])
            self.assertIn("SatQuery AI Analysis Report", build_report(result))

    def test_optical_sar_requires_both_modalities(self):
        images = [asset(np.zeros((4, 4, 3)), modality="optical"), asset(np.ones((4, 4, 3)), modality="unknown")]
        with self.assertRaisesRegex(ValueError, "one optical image and one SAR image"):
            run_analysis("fuse optical and sar", images)

    def test_empty_query_is_rejected_by_core_api(self):
        with self.assertRaisesRegex(ValueError, "Ask a question"):
            run_analysis("", [asset(np.zeros((4, 4, 3)))])


if __name__ == "__main__":
    unittest.main()
