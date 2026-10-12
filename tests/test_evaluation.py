import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_baseline.evaluate import (
    calibration_metrics,
    corrupt_images,
    evaluate_model,
    fit_temperature,
    main,
    probabilities_from_logits,
    selective_metrics,
)
from vision_baseline.model import TinyConvNet, make_dataset


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_reliability_bins_are_well_formed(self):
        probabilities = np.array([[.8, .2], [.25, .75], [.4, .6], [.9, .1]])
        labels = np.array([0, 1, 1, 1])
        metrics = calibration_metrics(probabilities, labels, bins=5)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(sum(row["count"] for row in metrics["reliability_bins"]), 4)
        self.assertGreater(metrics["nll"], 0.0)
        self.assertGreater(metrics["brier"], 0.0)
        self.assertGreaterEqual(metrics["ece"], 0.0)

    def test_temperature_fit_does_not_worsen_calibration_nll(self):
        logits = np.tile(np.array([[2.0, 0.0]]), (4, 1))
        labels = np.array([0, 0, 0, 1])
        temperature = fit_temperature(logits, labels)
        before = calibration_metrics(probabilities_from_logits(logits), labels)["nll"]
        after = calibration_metrics(probabilities_from_logits(logits, temperature), labels)["nll"]
        self.assertGreater(temperature, 1.0)
        self.assertLessEqual(after, before)

    def test_report_and_selective_results_are_deterministic(self):
        calibration_images, calibration_labels = make_dataset(3, seed=31, size=8)
        test_images, test_labels = make_dataset(4, seed=37, size=8)
        model = TinyConvNet(channels=3, seed=41)
        first = evaluate_model(model, calibration_images, calibration_labels,
                               test_images, test_labels, bins=4)
        second = evaluate_model(model, calibration_images, calibration_labels,
                                test_images, test_labels, bins=4)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["test"]["samples"], 12)
        self.assertEqual(first["test"]["selective"][-1]["retained"], 12)

    def test_corruption_is_bounded_and_deterministic(self):
        images, _ = make_dataset(2, seed=7, size=8)
        first = corrupt_images(images, seed=9)
        np.testing.assert_array_equal(first, corrupt_images(images, seed=9))
        self.assertTrue(np.all((first >= 0.0) & (first <= 1.0)))
        self.assertFalse(np.array_equal(first, images))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            probabilities_from_logits(np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            probabilities_from_logits(np.zeros((2, 2)), temperature=0.0)
        with self.assertRaises(ValueError):
            calibration_metrics(np.array([[.6, .6]]), np.array([0]))
        with self.assertRaises(ValueError):
            selective_metrics(np.array([[.5, .5]]), np.array([0]), (0.0,))
        with self.assertRaises(ValueError):
            corrupt_images(np.zeros((1, 4, 4)), seed=1, noise_std=-0.1)

    def test_cli_artifact_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            self.assertEqual(main(["--output", str(first)]), 0)
            self.assertEqual(main(["--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            report = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(report["test"]["samples"], 90)


if __name__ == "__main__":
    unittest.main()
