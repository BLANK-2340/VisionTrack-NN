import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_baseline.fashion_experiment import classification_metrics, main, run_experiment
from vision_baseline.fashion_mnist import build_manifest


class FashionExperimentTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        rng = np.random.default_rng(7)
        for prefix, samples in (("train", 4), ("t10k", 2)):
            labels = np.repeat(np.arange(10, dtype=np.uint8), samples)
            images = np.zeros((len(labels), 28, 28), dtype=np.uint8)
            for index, label in enumerate(labels):
                images[index, 2 + int(label) * 2:4 + int(label) * 2, :] = 220
                images[index] += rng.integers(0, 20, (28, 28), dtype=np.uint8)
            image_raw = struct.pack(">IIII", 2051, len(labels), 28, 28) + images.tobytes()
            label_raw = struct.pack(">II", 2049, len(labels)) + labels.tobytes()
            (root / f"{prefix}-images-idx3-ubyte.gz").write_bytes(
                gzip.compress(image_raw, mtime=0)
            )
            (root / f"{prefix}-labels-idx1-ubyte.gz").write_bytes(
                gzip.compress(label_raw, mtime=0)
            )
        manifest = build_manifest(root, 2, 1, 1, seed=17, require_official=False)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return root, manifest_path

    def test_classification_metrics_use_true_rows(self):
        labels = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        predictions = np.array([0, 1, 1, 1, 0], dtype=np.int64)
        report = classification_metrics(predictions, labels, 3)
        self.assertEqual(report["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
        self.assertEqual([row["support"] for row in report["per_class"]], [2, 2, 1])

    def test_experiment_is_deterministic_and_keeps_test_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.make_fixture(Path(directory))
            first, first_model = run_experiment(
                manifest, root, channels=2, epochs=2, batch_size=10, seed=23
            )
            second, second_model = run_experiment(
                manifest, root, channels=2, epochs=2, batch_size=10, seed=23
            )
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
            self.assertEqual(first["dataset"]["split_sizes"],
                             {"train": 20, "validation": 10, "test": 10})
            self.assertFalse(first["dataset"]["official_source_verified"])
            self.assertEqual(len(first["test"]["confusion_matrix"]), 10)
            self.assertEqual(sum(sum(row) for row in first["test"]["confusion_matrix"]), 10)
            for name in first_model.parameters():
                np.testing.assert_array_equal(first_model.parameters()[name],
                                              second_model.parameters()[name])

    def test_cli_artifact_and_checkpoint_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.make_fixture(Path(directory))
            outputs = [root / "a.json", root / "b.json"]
            checkpoints = [root / "a.vtnn", root / "b.vtnn"]
            for output, checkpoint in zip(outputs, checkpoints):
                arguments = [str(manifest), str(root), "--output", str(output),
                             "--checkpoint", str(checkpoint), "--channels", "2",
                             "--epochs", "2", "--batch-size", "10", "--seed", "29"]
                self.assertEqual(main(arguments), 0)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(checkpoints[0].read_bytes(), checkpoints[1].read_bytes())

    def test_invalid_hyperparameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.make_fixture(Path(directory))
            for keyword, value in (("channels", 0), ("epochs", 0),
                                   ("batch_size", 0), ("learning_rate", float("nan")),
                                   ("bins", 0)):
                with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                    run_experiment(manifest, root, **{keyword: value})


if __name__ == "__main__":
    unittest.main()
