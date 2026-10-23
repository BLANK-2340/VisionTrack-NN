import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_baseline.environment_report import (
    _structured_numpy_build,
    build_environment_report,
    canonical_json,
)


class EnvironmentReportTests(unittest.TestCase):
    def test_report_is_canonical_and_omits_volatile_identity(self) -> None:
        report = build_environment_report()
        self.assertEqual(report["schema_version"], 1)
        self.assertIn("version", report["python"])
        self.assertIn("version", report["numpy"])
        self.assertEqual(canonical_json(report), canonical_json(build_environment_report()))
        rendered = canonical_json(report)
        for forbidden in ("hostname", "username", "executable", "timestamp", "cwd"):
            self.assertNotIn(forbidden, rendered.lower())
        self.assertEqual(json.loads(rendered), report)

    def test_structured_config_keeps_identity_but_drops_paths_and_commands(self) -> None:
        config = {
            "Build Dependencies": {
                "blas": {
                    "name": "example-blas",
                    "found": True,
                    "version": "1.2.3",
                    "detection method": "pkgconfig",
                    "include directory": "/private/build/include",
                    "lib directory": "/private/build/lib",
                    "openblas configuration": "OpenBLAS 1.2.3 DYNAMIC_ARCH",
                }
            },
            "Compilers": {
                "c": {
                    "name": "gcc",
                    "version": "14.2",
                    "linker": "ld",
                    "commands": "/private/toolchain/cc",
                }
            },
            "Python Information": {"path": "/private/build/python"},
            "SIMD Extensions": {"baseline": ["SSE2"], "found": ["AVX2"]},
        }
        rendered = canonical_json(_structured_numpy_build(config))
        self.assertIn("example-blas", rendered)
        self.assertIn("OpenBLAS 1.2.3 DYNAMIC_ARCH", rendered)
        self.assertNotIn("/private", rendered)
        self.assertNotIn("commands", rendered)
        self.assertNotIn("Python Information", rendered)

    def test_cli_writes_byte_repeatable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            for output in (first, second):
                subprocess.run(
                    [sys.executable, "-m", "vision_baseline.environment_report", "--output", str(output)],
                    check=True,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
