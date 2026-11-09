import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vision_baseline import TinyConvNet, save_model
from vision_baseline.abstention import AbstentionPolicy, policy_record
from vision_baseline.audit_policy import audit_folder_policy, main
from vision_baseline.validate_policy_audit import ValidationError, validate_policy_audit


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

    def test_report_binds_policy_and_labeled_input_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, dataset, policy = self.make_fixture(Path(directory))
            report = audit_folder_policy(checkpoint, dataset, policy)
        input_sha256 = hashlib.sha256(PGM).hexdigest()
        manifest = [
            {"path": "vertical/a.pgm", "sha256": input_sha256, "target_index": 0},
            {"path": "vertical/b.pgm", "sha256": input_sha256, "target_index": 0},
            {"path": "horizontal/sample.pgm", "sha256": input_sha256,
             "target_index": 1},
            {"path": "diagonal/sample.pgm", "sha256": input_sha256,
             "target_index": 2},
        ]
        canonical_policy = json.dumps(
            policy_record(policy), allow_nan=False, separators=(",", ":"),
            sort_keys=True).encode("utf-8")
        canonical_manifest = json.dumps(
            manifest, allow_nan=False, separators=(",", ":"),
            sort_keys=True).encode("utf-8")
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["policy_sha256"],
                         hashlib.sha256(canonical_policy).hexdigest())
        self.assertEqual(report["input_manifest_sha256"],
                         hashlib.sha256(canonical_manifest).hexdigest())
        self.assertTrue(all(row["input_sha256"] == input_sha256
                            for row in report["predictions"]))

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

    def test_strict_validator_accepts_exact_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, dataset, policy = self.make_fixture(root)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy_record(policy)), encoding="utf-8")
            report_path = root / "audit.json"
            report = audit_folder_policy(checkpoint, dataset, policy)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
            self.assertEqual(validate_policy_audit(
                report_path, checkpoint, dataset, policy_path),
                {"accepted": 4, "samples": 4})

    def test_strict_validator_rejects_report_and_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, dataset, policy = self.make_fixture(root)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy_record(policy)), encoding="utf-8")
            report_path = root / "audit.json"
            report = audit_folder_policy(checkpoint, dataset, policy)

            corrupted = copy.deepcopy(report)
            corrupted["checkpoint_sha256"] = "0" * 64
            report_path.write_text(json.dumps(corrupted, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
            with self.assertRaises(ValidationError):
                validate_policy_audit(report_path, checkpoint, dataset, policy_path)

            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
            (dataset / "vertical" / "a.pgm").write_bytes(PGM + b"\n")
            with self.assertRaises(ValidationError):
                validate_policy_audit(report_path, checkpoint, dataset, policy_path)

            (dataset / "vertical" / "a.pgm").write_bytes(PGM)
            changed_policy = policy._replace(min_margin=0.3)
            policy_path.write_text(json.dumps(policy_record(changed_policy)), encoding="utf-8")
            with self.assertRaises(ValidationError):
                validate_policy_audit(report_path, checkpoint, dataset, policy_path)

    def test_strict_validator_rejects_duplicate_and_noncanonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, dataset, policy = self.make_fixture(root)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy_record(policy)), encoding="utf-8")
            report = audit_folder_policy(checkpoint, dataset, policy)
            report_path = root / "audit.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValidationError):
                validate_policy_audit(report_path, checkpoint, dataset, policy_path)
            canonical = json.dumps(report, indent=2, sort_keys=True) + "\n"
            report_path.write_text('{"schema_version":2,' + canonical.lstrip()[1:],
                                   encoding="utf-8")
            with self.assertRaises(ValidationError):
                validate_policy_audit(report_path, checkpoint, dataset, policy_path)


if __name__ == "__main__":
    unittest.main()
