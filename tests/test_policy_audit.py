import json
import tempfile
import unittest
from pathlib import Path

from vision_baseline import TinyConvNet, save_model
from vision_baseline.abstention import AbstentionPolicy, policy_record
from vision_baseline.audit_policy import audit_folder_policy, main


PGM = b"P2\n3 3\n1\n0 1 0 0 1 0 0 1 0\n"


class PolicyAuditTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, AbstentionPolicy]:
        model = TinyConvNet(channels=2, classes=3, seed=3)
        for parameter in model.parameters().values():
            parameter.fill(0.0)
        model.bias[:] = [2.0, 1.0, 0.0]
        checkpoint = root / "model.vtnn"
        save_model(model, checkpoint)
        dataset = root / "dataset"
        for label, names in {
                "vertical": ("b.pgm", "a.pgm"),
                "horizontal": ("sample.pgm",),
                "diagonal": ("sample.pgm",),
        }.items():
            folder = dataset / label
            folder.mkdir(parents=True)
            for name in names:
                (folder / name).write_bytes(PGM)
        return checkpoint, dataset, AbstentionPolicy(.2, .5, .5, 0.0, 2, 4)

    def test_per_class_and_aggregate_selective_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, dataset, policy = self.make_fixture(Path(directory))
            report = audit_folder_policy(checkpoint, dataset, policy)
        self.assertEqual(report["aggregate"], {
            "accepted": 4, "coverage": 1.0, "errors": 2, "risk": .5, "samples": 4,
        })
        self.assertEqual([(row["label"], row["coverage"], row["risk"])
                          for row in report["per_class"]], [
            ("vertical", 1.0, 0.0),
            ("horizontal", 1.0, 1.0),
            ("diagonal", 1.0, 1.0),
        ])
        self.assertEqual([row["path"] for row in report["predictions"]], [
            "vertical/a.pgm", "vertical/b.pgm", "horizontal/sample.pgm",
            "diagonal/sample.pgm",
        ])

    def test_zero_acceptance_is_explicit_for_every_class(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, dataset, policy = self.make_fixture(Path(directory))
            report = audit_folder_policy(checkpoint, dataset,
                                         policy._replace(min_margin=.9))
        self.assertEqual(report["aggregate"]["coverage"], 0.0)
        self.assertIsNone(report["aggregate"]["risk"])
        for row in report["per_class"]:
            self.assertEqual(row["accepted"], 0)
            self.assertEqual(row["errors"], 0)
            self.assertIsNone(row["risk"])

    def test_cli_report_is_byte_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, dataset, policy = self.make_fixture(root)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy_record(policy)), encoding="utf-8")
            first, second = root / "first.json", root / "second.json"
            self.assertEqual(main([str(checkpoint), str(dataset), str(policy_path),
                                   "--output", str(first)]), 0)
            self.assertEqual(main([str(checkpoint), str(dataset), str(policy_path),
                                   "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_invalid_policy_is_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, dataset, policy = self.make_fixture(Path(directory))
            with self.assertRaises(ValueError):
                audit_folder_policy(checkpoint, dataset,
                                    policy._replace(validation_retained=3))


if __name__ == "__main__":
    unittest.main()
