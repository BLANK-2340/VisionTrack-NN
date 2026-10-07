import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_baseline import TinyConvNet, dumps_model, load_model, loads_model, save_model
from vision_baseline.infer import predict_file
from vision_baseline.pgm import loads_pgm


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_is_deterministic_and_exact(self):
        model = TinyConvNet(channels=4, classes=3, seed=23)
        first = dumps_model(model)
        self.assertEqual(first, dumps_model(model))
        restored = loads_model(first)
        for name, parameter in model.parameters().items():
            np.testing.assert_array_equal(restored.parameters()[name], parameter)
        images = np.random.default_rng(9).random((5, 8, 8))
        np.testing.assert_array_equal(restored.predict(images), model.predict(images))
        self.assertEqual(dumps_model(restored), first)

    def test_file_api_and_corruption_rejection(self):
        model = TinyConvNet(seed=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.vtnn"
            save_model(model, path)
            loaded = load_model(path)
            np.testing.assert_array_equal(loaded.kernel, model.kernel)
        encoded = dumps_model(model)
        variants = [b"", b"wrong" + encoded[5:], encoded[:-1], encoded + b"x"]
        damaged = bytearray(encoded)
        damaged[-33] ^= 1
        variants.append(bytes(damaged))
        for value in variants:
            with self.subTest(size=len(value)):
                with self.assertRaises(ValueError):
                    loads_model(value)

    def test_metadata_and_nonfinite_values_are_rejected(self):
        model = TinyConvNet(seed=1)
        model.bias[0] = np.nan
        with self.assertRaises(ValueError):
            dumps_model(model)
        model.bias[0] = 0.0
        encoded = dumps_model(model)
        metadata_size = struct.unpack_from("<I", encoded, 8)[0]
        start, end = 12, 12 + metadata_size
        metadata = json.loads(encoded[start:end])
        variants = []
        metadata["version"] = 99
        variants.append(metadata)
        oversized = json.loads(encoded[start:end])
        oversized["arrays"][0][1] = [10**12, 3, 3]
        variants.append(oversized)
        for variant in variants:
            replacement = json.dumps(variant, sort_keys=True, separators=(",", ":")).encode("ascii")
            body = encoded[:8] + struct.pack("<I", len(replacement)) + replacement + encoded[end:-32]
            invalid = body + hashlib.sha256(body).digest()
            with self.assertRaises(ValueError):
                loads_model(invalid)


class PgmTests(unittest.TestCase):
    def test_ascii_binary_and_sixteen_bit_pgm(self):
        ascii_image = loads_pgm(b"P2\n# example\n3 2\n15\n0 3 6 9 12 15\n")
        np.testing.assert_allclose(ascii_image, np.array([[0, .2, .4], [.6, .8, 1.0]]))
        binary = loads_pgm(b"P5\n3 1\n255\n" + bytes([0, 127, 255]))
        np.testing.assert_allclose(binary, np.array([[0, 127 / 255, 1.0]]))
        sixteen = loads_pgm(b"P5\n2 1\n1023\n" + bytes([0, 0, 3, 255]))
        np.testing.assert_allclose(sixteen, np.array([[0, 1.0]]))

    def test_malformed_pgm_is_rejected(self):
        invalid = [
            b"", b"P6\n1 1\n255\n\x00", b"P2\n0 1\n1\n0\n",
            b"P2\n1 1\n0\n0\n", b"P2\n1 1\n1\n2\n",
            b"P2\n1 1\n1\n0 1\n", b"P5\n2 1\n255\n\x00",
            b"P5\n1 1\n255\n\x00\x01",
        ]
        for value in invalid:
            with self.subTest(value=value[:16]):
                with self.assertRaises(ValueError):
                    loads_pgm(value)

    def test_checkpoint_to_pgm_inference_path(self):
        model = TinyConvNet(seed=7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.vtnn"
            image = root / "image.pgm"
            save_model(model, checkpoint)
            image.write_bytes(b"P2\n4 3\n1\n0 0 1 0 0 0 1 0 0 0 1 0\n")
            prediction, shape = predict_file(checkpoint, image)
            self.assertIn(prediction, range(3))
            self.assertEqual(shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
