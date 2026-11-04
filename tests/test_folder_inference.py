import json
import tempfile
import unittest
from pathlib import Path

from vision_baseline import TinyConvNet, save_model
from vision_baseline.abstention import AbstentionPolicy, policy_record
from vision_baseline.infer_folder import infer_folder, load_policy, main


PGM = b"P2\n3 3\n1\n0 1 0 0 1 0 0 1 0\n"


class FolderInferenceTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, AbstentionPolicy]:
        model = TinyConvNet(channels=2, classes=3, seed=3)
        for parameter in model.parameters().values():
            parameter.fill(0.0)
        model.bias[:] = [2.0, 1.0, 0.0]
        checkpoint = root / "model.vtnn"
        save_model(model, checkpoint)
        folder = root / "images"
        folder.mkdir()
        (folder / "b.pgm").write_bytes(PGM)
        (folder / "a.pgm").write_bytes(PGM)
        policy = AbstentionPolicy(.2, .5, .5, 0.0, 2, 4)
        return checkpoint, folder, policy

    def test_report_records_sorted_acceptance_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, folder, policy = self.make_fixture(Path(directory))
            accepted = infer_folder(checkpoint, folder, policy)
            abstained = infer_folder(checkpoint, folder, policy._replace(min_margin=.9))
        self.assertEqual([row["path"] for row in accepted["predictions"]],
                         ["a.pgm", "b.pgm"])
        self.assertEqual((accepted["accepted"], accepted["abstained"]), (2, 0))
        self.assertTrue(all(row["accepted"] for row in accepted["predictions"]))
        self.assertEqual((abstained["accepted"], abstained["abstained"]), (0, 2))
        self.assertEqual(abstained["coverage"], 0.0)

    def test_policy_loader_accepts_record_and_evaluation_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, policy = self.make_fixture(root)
            direct = root / "direct.json"
            wrapped = root / "wrapped.json"
            direct.write_text(json.dumps(policy_record(policy)), encoding="utf-8")
            wrapped.write_text(json.dumps({"abstention_policy": policy_record(policy)}),
                               encoding="utf-8")
            self.assertEqual(load_policy(direct), policy)
            self.assertEqual(load_policy(wrapped), policy)

    def test_cli_output_is_byte_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, folder, policy = self.make_fixture(root)
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy_record(policy)), encoding="utf-8")
            first, second = root / "first.json", root / "second.json"
            self.assertEqual(main([str(checkpoint), str(folder), str(policy_path),
                                   "--output", str(first)]), 0)
            self.assertEqual(main([str(checkpoint), str(folder), str(policy_path),
                                   "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_invalid_policy_and_folder_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, folder, policy = self.make_fixture(root)
            invalid = root / "invalid.json"
            invalid.write_text('{"min_margin": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(invalid)
            invalid.write_text('{"min_margin": 0.1, "min_margin": 0.2}',
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_policy(invalid)
            with self.assertRaises(ValueError):
                infer_folder(checkpoint, folder,
                             policy._replace(validation_coverage=.75))
            (folder / "notes.txt").write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected image-folder entries"):
                infer_folder(checkpoint, folder, policy)


if __name__ == "__main__":
    unittest.main()
