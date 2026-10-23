import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_baseline.validate_results import ValidationError, validate_results


SOURCE_RESULTS = Path(__file__).resolve().parents[1] / "results"


class ResultArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.results = Path(self.temporary.name) / "results"
        shutil.copytree(SOURCE_RESULTS, self.results)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def mutate(self, name: str, update) -> None:
        path = self.results / name
        value = json.loads(path.read_text(encoding="utf-8"))
        update(value)
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    def test_committed_artifacts_and_cli_validate(self) -> None:
        summary = validate_results(SOURCE_RESULTS)
        self.assertEqual(summary["artifact_count"], 4)
        self.assertEqual(summary["multiseed_runs"], 3)
        self.assertEqual(summary["pooling_runs"], 6)
        completed = subprocess.run(
            [sys.executable, "-m", "vision_baseline.validate_results", str(SOURCE_RESULTS)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("validated_artifacts=4", completed.stdout)
        self.assertIn(summary["manifest_sha256"], completed.stdout)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        path = self.results / "fashion_mnist_tinyconvnet_summary.json"
        original = path.read_text(encoding="utf-8")
        path.write_text('{"schema_version":1,"schema_version":1,' + original.lstrip()[1:], encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "duplicate JSON key"):
            validate_results(self.results)

    def test_manifest_and_aggregate_tampering_are_rejected(self) -> None:
        self.mutate(
            "fashion_mnist_spatial_comparison.json",
            lambda value: value["dataset"].__setitem__("manifest_sha256", "0" * 64),
        )
        with self.assertRaisesRegex(ValidationError, "different dataset manifest"):
            validate_results(self.results)

        shutil.rmtree(self.results)
        shutil.copytree(SOURCE_RESULTS, self.results)
        self.mutate(
            "fashion_mnist_spatial_multiseed.json",
            lambda value: value["aggregate"]["metrics"]["accuracy"].__setitem__("mean", 0.99),
        )
        with self.assertRaisesRegex(ValidationError, "multiseed.accuracy.mean mismatch"):
            validate_results(self.results)

    def test_confusion_total_and_pair_delta_tampering_are_rejected(self) -> None:
        self.mutate(
            "fashion_mnist_tinyconvnet_summary.json",
            lambda value: value["test"]["confusion_matrix_true_rows"][0].__setitem__(0, 999),
        )
        with self.assertRaisesRegex(ValidationError, "tiny.confusion total"):
            validate_results(self.results)

        shutil.rmtree(self.results)
        shutil.copytree(SOURCE_RESULTS, self.results)
        self.mutate(
            "fashion_mnist_pooling_budget_comparison.json",
            lambda value: value["paired_grid_minus_global"]["per_seed"][0].__setitem__(
                "accuracy_percentage_points", -37.2
            ),
        )
        with self.assertRaisesRegex(ValidationError, r"paired\[0\].accuracy_percentage_points mismatch"):
            validate_results(self.results)


if __name__ == "__main__":
    unittest.main()
