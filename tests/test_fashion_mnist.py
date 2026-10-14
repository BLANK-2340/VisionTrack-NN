import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_baseline.fashion_mnist import FILES, build_manifest, load_idx_pair, load_split, main


class FashionMnistTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        labels = np.repeat(np.arange(10, dtype=np.uint8), 3)
        test_labels = np.repeat(np.arange(10, dtype=np.uint8), 2)
        for prefix, values in (("train", labels), ("t10k", test_labels)):
            images = np.arange(len(values) * 28 * 28, dtype=np.uint8)
            image_raw = struct.pack(">IIII", 2051, len(values), 28, 28) + images.tobytes()
            label_raw = struct.pack(">II", 2049, len(values)) + values.tobytes()
            (root / f"{prefix}-images-idx3-ubyte.gz").write_bytes(gzip.compress(image_raw, mtime=0))
            (root / f"{prefix}-labels-idx1-ubyte.gz").write_bytes(gzip.compress(label_raw, mtime=0))
        return root

    def test_idx_loading_and_normalized_manifest_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_fixture(Path(directory))
            images, labels = load_idx_pair(root / FILES["train_images"][0],
                                           root / FILES["train_labels"][0])
            self.assertEqual(images.shape, (30, 28, 28))
            np.testing.assert_array_equal(np.bincount(labels), np.full(10, 3))
            manifest = build_manifest(root, 1, 1, 1, seed=17, require_official=False)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            split_images, split_labels = load_split(path, root, "validation")
            self.assertEqual(split_images.shape, (10, 28, 28))
            self.assertEqual(split_images.dtype, np.float64)
            self.assertTrue(np.all((split_images >= 0.0) & (split_images <= 1.0)))
            np.testing.assert_array_equal(np.bincount(split_labels), np.ones(10, dtype=int))

    def test_manifest_is_deterministic_balanced_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_fixture(Path(directory))
            first = build_manifest(root, 1, 1, 1, seed=23, require_official=False)
            second = build_manifest(root, 1, 1, 1, seed=23, require_official=False)
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
            self.assertEqual(first["splits"]["train"]["class_counts"], [1] * 10)
            self.assertEqual(first["splits"]["validation"]["class_counts"], [1] * 10)
            self.assertTrue(set(first["splits"]["train"]["indices"]).isdisjoint(
                first["splits"]["validation"]["indices"]
            ))
            changed = build_manifest(root, 1, 1, 1, seed=24, require_official=False)
            self.assertNotEqual(first["splits"]["train"]["indices"],
                                changed["splits"]["train"]["indices"])

    def test_official_checksum_and_changed_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "official MD5 mismatch"):
                build_manifest(root, 1, 1, 1)
            manifest = build_manifest(root, 1, 1, 1, require_official=False)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            image_path = root / FILES["test_images"][0]
            image_path.write_bytes(image_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_split(path, root, "test")

    def test_malformed_idx_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images.gz"
            labels = root / "labels.gz"
            labels.write_bytes(gzip.compress(struct.pack(">II", 2049, 1) + b"\x00", mtime=0))
            invalid_images = [
                b"short", struct.pack(">IIII", 99, 1, 28, 28) + bytes(784),
                struct.pack(">IIII", 2051, 1, 27, 28) + bytes(756),
                struct.pack(">IIII", 2051, 1, 28, 28) + bytes(783),
            ]
            for raw in invalid_images:
                images.write_bytes(gzip.compress(raw, mtime=0))
                with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                    load_idx_pair(images, labels)

    def test_cli_writes_byte_identical_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_fixture(Path(directory))
            first, second = root / "first.json", root / "second.json"
            common = [str(root), "--train-per-class", "1", "--validation-per-class", "1",
                      "--test-per-class", "1", "--seed", "31", "--allow-nonofficial-files"]
            self.assertEqual(main(common + ["--output", str(first)]), 0)
            self.assertEqual(main(common + ["--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
