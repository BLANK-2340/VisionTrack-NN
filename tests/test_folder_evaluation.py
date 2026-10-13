import json
import tempfile
import unittest
from pathlib import Path

from vision_baseline import TinyConvNet, save_model
from vision_baseline.evaluate_folder import evaluate_folder, main


PGM = b"P2\n3 3\n1\n0 1 0 0 1 0 0 1 0\n"


class FolderEvaluationTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        model = TinyConvNet(channels=2, classes=3, seed=3)
        for parameter in model.parameters().values():
            parameter.fill(0.0)
        model.bias[:] = [2.0, 1.0, 0.0]
        checkpoint = root / "model.vtnn"
        save_model(model, checkpoint)
        dataset = root / "dataset"
        for label, filename in [
                ("vertical", "b.pgm"), ("vertical", "a.pgm"),
                ("horizontal", "sample.pgm"), ("diagonal", "sample.pgm")]:
            directory = dataset / label
            directory.mkdir(parents=True, exist_ok=True)
            (directory / filename).write_bytes(PGM)
        return checkpoint, dataset

    def test_confusion_matrix_metrics_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, dataset = self.make_fixture(Path(directory))
            report = evaluate_folder(checkpoint, dataset)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["samples"], 4)
        self.assertEqual(report["metrics"]["confusion_matrix"],
                         [[2, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(report["metrics"]["accuracy"], 0.5)
        self.assertAlmostEqual(report["metrics"]["macro_f1"], 2.0 / 9.0)
        self.assertEqual([row["path"] for row in report["predictions"]], [
            "vertical/a.pgm", "vertical/b.pgm", "horizontal/sample.pgm",
            "diagonal/sample.pgm",
        ])
        self.assertTrue(all(not Path(row["path"]).is_absolute()
                            for row in report["predictions"]))

    def test_report_and_cli_artifacts_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, dataset = self.make_fixture(root)
            self.assertEqual(json.dumps(evaluate_folder(checkpoint, dataset), sort_keys=True),
                             json.dumps(evaluate_folder(checkpoint, dataset), sort_keys=True))
            first, second = root / "first.json", root / "second.json"
            self.assertEqual(main([str(checkpoint), str(dataset), "--output", str(first)]), 0)
            self.assertEqual(main([str(checkpoint), str(dataset), "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_invalid_layout_labels_and_temperature_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, dataset = self.make_fixture(root)
            (dataset / "unexpected").mkdir()
            with self.assertRaises(ValueError):
                evaluate_folder(checkpoint, dataset)
            (dataset / "unexpected").rmdir()
            with self.assertRaises(ValueError):
                evaluate_folder(checkpoint, dataset, ["same", "same", "other"])
            with self.assertRaises(ValueError):
                evaluate_folder(checkpoint, dataset, temperature=0.0)

    def test_non_pgm_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, dataset = self.make_fixture(Path(directory))
            (dataset / "vertical" / "notes.txt").write_text("not an image")
            with self.assertRaisesRegex(ValueError, "unexpected dataset entry"):
                evaluate_folder(checkpoint, dataset)


if __name__ == "__main__":
    unittest.main()
