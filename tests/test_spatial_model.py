import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_baseline.fashion_mnist import build_manifest
from vision_baseline.fashion_spatial_experiment import run_experiment
from vision_baseline.spatial_model import SpatialConvNet


class SpatialConvNetTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        rng = np.random.default_rng(3)
        for prefix, samples in (("train", 3), ("t10k", 1)):
            labels = np.repeat(np.arange(10, dtype=np.uint8), samples)
            images = rng.integers(0, 32, (len(labels), 28, 28), dtype=np.uint8)
            for index, label in enumerate(labels):
                images[index, 2 * int(label):2 * int(label) + 3, :] += 192
            image_raw = struct.pack(">IIII", 2051, len(labels), 28, 28) + images.tobytes()
            label_raw = struct.pack(">II", 2049, len(labels)) + labels.tobytes()
            (root / f"{prefix}-images-idx3-ubyte.gz").write_bytes(
                gzip.compress(image_raw, mtime=0)
            )
            (root / f"{prefix}-labels-idx1-ubyte.gz").write_bytes(
                gzip.compress(label_raw, mtime=0)
            )
        manifest = build_manifest(root, 2, 1, 1, seed=23, require_official=False)
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return path

    def test_selected_gradients_match_central_differences(self):
        rng = np.random.default_rng(5)
        images = rng.uniform(0.1, 1.0, (4, 7, 7))
        labels = np.array([0, 1, 2, 1], dtype=np.int64)
        model = SpatialConvNet(channels=2, classes=3, grid_size=2, seed=11)
        _, gradients = model.loss_and_gradients(images, labels)
        epsilon = 1.0e-6
        checks = (("kernel", (0, 1, 2)), ("conv_bias", (1,)),
                  ("weight", (3, 2)), ("bias", (0,)))
        for name, index in checks:
            parameter = model.parameters()[name]
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss_and_gradients(images, labels)[0]
            parameter[index] = original - epsilon
            minus = model.loss_and_gradients(images, labels)[0]
            parameter[index] = original
            numeric = (plus - minus) / (2.0 * epsilon)
            self.assertAlmostEqual(numeric, gradients[name][index], places=6)

    def test_training_is_deterministic_and_reduces_loss(self):
        rng = np.random.default_rng(13)
        images = rng.uniform(0.0, 1.0, (12, 8, 8))
        labels = np.repeat(np.arange(3, dtype=np.int64), 4)
        first = SpatialConvNet(channels=3, classes=3, grid_size=2, seed=17)
        second = SpatialConvNet(channels=3, classes=3, grid_size=2, seed=17)
        first_history = first.train(images, labels, epochs=5, batch_size=4,
                                    learning_rate=0.1, seed=19)
        second_history = second.train(images, labels, epochs=5, batch_size=4,
                                      learning_rate=0.1, seed=19)
        self.assertEqual(first_history, second_history)
        self.assertLess(first_history[-1], first_history[0])
        for name in first.parameters():
            np.testing.assert_array_equal(first.parameters()[name], second.parameters()[name])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            SpatialConvNet(grid_size=0)
        model = SpatialConvNet(grid_size=4)
        with self.assertRaisesRegex(ValueError, "pooling grid"):
            model.predict(np.ones((1, 5, 5)))
        with self.assertRaises(ValueError):
            model.loss_and_gradients(np.ones((2, 8, 8)), np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            model.train(np.ones((2, 8, 8)), np.array([0, 1]), epochs=0)
        with self.assertRaises(ValueError):
            model.train(np.ones((2, 8, 8)), np.array([0, 1]), learning_rate=float("nan"))

    def test_manifest_experiment_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_fixture(root)
            first = run_experiment(manifest, root, channels=2, grid_size=2,
                                   epochs=2, batch_size=10, seed=31)
            second = run_experiment(manifest, root, channels=2, grid_size=2,
                                    epochs=2, batch_size=10, seed=31)
            self.assertEqual(json.dumps(first, sort_keys=True),
                             json.dumps(second, sort_keys=True))
            self.assertEqual(first["dataset"]["split_sizes"],
                             {"train": 20, "validation": 10, "test": 10})
            self.assertEqual(sum(sum(row) for row in first["test"]["confusion_matrix"]), 10)


if __name__ == "__main__":
    unittest.main()
