import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_spatial_model
from vision_baseline.fashion_multiseed import main, run_suite, summarize_reports


def report(seed, accuracy, ece):
    return {"dataset": {"manifest_sha256": "fixed", "split_sizes": {"test": 10}},
            "protocol": {"seed": seed, "epochs": 2},
            "test": {"macro_f1": accuracy,
                     "calibrated": {"accuracy": accuracy, "ece": ece, "nll": 0.5},
                     "uncalibrated": {"accuracy": accuracy, "ece": 0.2, "nll": 0.6}}}


class MultiseedTests(unittest.TestCase):
    def test_known_sample_statistics_and_regressions(self):
        reports = [report(1, 0.5, 0.1), report(2, 0.75, 0.3), report(3, 1.0, 0.2)]
        result = summarize_reports(reports)
        self.assertEqual(result["metrics"]["accuracy"]["mean"], 0.75)
        self.assertEqual(result["metrics"]["accuracy"]["sample_stddev"], 0.25)
        self.assertEqual(result["calibration_regressions"]["ece_increased"], 1)
        self.assertEqual(result["calibration_regressions"]["nll_increased"], 0)
        self.assertEqual(result["seeds"], [1, 2, 3])

    def test_mismatched_protocol_dataset_and_seed_rejected(self):
        first = report(1, 0.5, 0.1)
        for field, key, value in (("protocol", "epochs", 3),
                                  ("dataset", "manifest_sha256", "other"),
                                  ("protocol", "seed", 1)):
            second = report(2, 0.7, 0.2)
            second[field][key] = value
            with self.assertRaises(ValueError):
                summarize_reports([first, second])
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            second = report(2, value, 0.2)
            with self.assertRaises(ValueError):
                summarize_reports([first, second])

    def test_invalid_seed_plan_rejected_before_loading(self):
        for seeds in ((), (1,), (1, 1), (True, 2), (-1, 2), (1, 2**32), tuple(range(11))):
            with self.assertRaises(ValueError):
                run_suite("missing.json", "missing", seeds=seeds)

    def test_manifest_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            def mutate(*args, **kwargs):
                path.write_text('{"changed":true}', encoding="utf-8")
                return report(kwargs["seed"], 0.5, 0.1)
            with patch("vision_baseline.fashion_multiseed.run_experiment", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    run_suite(path, directory, seeds=(1, 2))

    def test_real_fixture_pipeline_cli_and_byte_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = test_spatial_model.SpatialConvNetTests().make_fixture(root)
            first = run_suite(manifest, root, seeds=(31, 32), channels=2,
                              grid_size=2, epochs=2, batch_size=10)
            self.assertFalse(first["dataset"]["official_source_verified"])
            self.assertEqual(first["aggregate"]["run_count"], 2)
            self.assertEqual(first["aggregate"]["seeds"], [31, 32])
            for run in first["runs"]:
                self.assertEqual(sum(map(sum, run["test"]["confusion_matrix"])), 10)
            before = copy.deepcopy(first)
            summarize_reports(first["runs"])
            self.assertEqual(first, before)
            outputs = [root / "a.json", root / "b.json"]
            for output in outputs:
                self.assertEqual(main([str(manifest), str(root), "--seeds", "31", "32",
                                       "--channels", "2", "--grid-size", "2", "--epochs", "2",
                                       "--batch-size", "10", "--output", str(output)]), 0)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(json.loads(outputs[0].read_bytes()), first)


if __name__ == "__main__":
    unittest.main()
