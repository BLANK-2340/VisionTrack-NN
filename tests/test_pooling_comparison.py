import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import test_spatial_model
from test_multiseed import report
from vision_baseline.fashion_pooling_comparison import (
    compare_suites, main, matched_global_channels,
)
from vision_baseline.model import TinyConvNet
from vision_baseline.spatial_model import SpatialConvNet


def suites():
    grid = {"dataset": {"manifest_sha256": "fixed"}, "evaluation": {"ece_bins": 10},
            "runs": [report(1, 0.8, 0.1), report(2, 0.9, 0.2)]}
    for row in grid["runs"]:
        row["protocol"].update(channels=2, grid_size=2, parameter_count=110)
    global_pool = copy.deepcopy(grid)
    for row, accuracy in zip(global_pool["runs"], (0.5, 0.7)):
        row["protocol"].update(channels=5, grid_size=1)
        row["test"]["calibrated"]["accuracy"] = accuracy
        row["test"]["macro_f1"] = accuracy
    return grid, global_pool


class PoolingComparisonTests(unittest.TestCase):
    def test_parameter_formula_matches_actual_models(self):
        for channels, grid in ((2, 2), (8, 4)):
            global_channels = matched_global_channels(channels, grid)
            first = SpatialConvNet(channels=channels, grid_size=grid)
            second = SpatialConvNet(channels=global_channels, grid_size=1)
            self.assertEqual(sum(p.size for p in first.parameters().values()),
                             sum(p.size for p in second.parameters().values()))
        self.assertEqual(matched_global_channels(8, 4), 68)
        for channels, grid in ((3, 2), (0, 4), (8, 1), (128, 4), (True, 2)):
            with self.assertRaises(ValueError):
                matched_global_channels(channels, grid)

    def test_grid_one_equals_global_pool_forward_and_gradients(self):
        rng = np.random.default_rng(9)
        images = rng.uniform(0.1, 1, (3, 8, 8))
        labels = np.array([0, 1, 2])
        spatial = SpatialConvNet(channels=3, classes=3, grid_size=1, seed=7)
        global_pool = TinyConvNet(channels=3, classes=3, seed=7)
        np.testing.assert_allclose(spatial.predict_logits(images), global_pool.predict_logits(images),
                                   rtol=1e-12, atol=1e-12)
        first_loss, first_gradients = spatial.loss_and_gradients(images, labels)
        second_loss, second_gradients = global_pool.loss_and_gradients(images, labels)
        self.assertAlmostEqual(first_loss, second_loss, places=12)
        for key in first_gradients:
            np.testing.assert_allclose(first_gradients[key], second_gradients[key],
                                       rtol=1e-12, atol=1e-12)

    def test_paired_statistics_and_mismatches(self):
        grid, global_pool = suites()
        result = compare_suites(grid, global_pool)
        delta = result["paired_grid_minus_global"]["aggregate"]["accuracy_percentage_points"]
        self.assertAlmostEqual(delta["mean"], 25)
        self.assertAlmostEqual(delta["sample_stddev"], np.sqrt(50))
        for key, value in (("parameter_count", 111), ("epochs", 3), ("channels", 4)):
            invalid = copy.deepcopy(global_pool)
            for row in invalid["runs"]:
                row["protocol"][key] = value
            with self.assertRaises(ValueError):
                compare_suites(grid, invalid)
        invalid = copy.deepcopy(global_pool)
        invalid["dataset"]["manifest_sha256"] = "different"
        with self.assertRaises(ValueError):
            compare_suites(grid, invalid)

    def test_fixture_cli_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = test_spatial_model.SpatialConvNetTests().make_fixture(root)
            outputs = [root / "a.json", root / "b.json"]
            for output in outputs:
                self.assertEqual(main([str(manifest), str(root), "--channels", "2",
                    "--grid-size", "2", "--seeds", "31", "32", "--epochs", "2",
                    "--batch-size", "10", "--output", str(output)]), 0)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            result = json.loads(outputs[0].read_bytes())
            for arm in result["arms"].values():
                self.assertFalse(arm["dataset"]["official_source_verified"])
                self.assertEqual(arm["aggregate"]["run_count"], 2)
                self.assertEqual(arm["runs"][0]["protocol"]["parameter_count"], 110)


if __name__ == "__main__":
    unittest.main()
