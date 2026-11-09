"""Validate a policy-audit report against its exact local sources."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .audit_policy import audit_folder_policy
from .infer_folder import load_policy


MAX_REPORT_BYTES = 8 * 1024 * 1024


class ValidationError(ValueError):
    """Raised when an audit report is malformed or differs from its sources."""


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-standard numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_finite(value: Any, label: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite number at {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, f"{label}[{index}]")


def _load_report(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValidationError("audit report must be a regular non-symlink file")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_REPORT_BYTES:
        raise ValidationError(f"audit report must be 1..{MAX_REPORT_BYTES} bytes")
    try:
        report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError("audit report is not valid UTF-8 JSON") from error
    if not isinstance(report, dict):
        raise ValidationError("audit report root must be an object")
    _require_finite(report)
    canonical = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if raw != canonical:
        raise ValidationError("audit report is not canonical JSON")
    return report, raw


def validate_policy_audit(report_path: str | Path, checkpoint: str | Path,
                          dataset: str | Path, policy_path: str | Path,
                          labels: Sequence[str] | None = None,
                          temperature: float = 1.0) -> dict[str, int]:
    """Require *report_path* to exactly match a fresh audit of its sources."""
    report, _ = _load_report(Path(report_path))
    if report.get("schema_version") != 2:
        raise ValidationError("audit report schema_version must be 2")
    try:
        expected = audit_folder_policy(checkpoint, dataset, load_policy(policy_path),
                                       labels, temperature)
    except (OSError, ValueError) as error:
        raise ValidationError(f"cannot regenerate audit report: {error}") from error
    if report != expected:
        raise ValidationError("audit report differs from its checkpoint, policy or inputs")
    aggregate = expected["aggregate"]
    return {"accepted": aggregate["accepted"], "samples": aggregate["samples"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--labels", nargs="+", help="class directory names in model order")
    parser.add_argument("--temperature", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    try:
        summary = validate_policy_audit(
            arguments.report, arguments.checkpoint, arguments.dataset,
            arguments.policy, arguments.labels, arguments.temperature,
        )
    except ValidationError as error:
        parser.error(str(error))
    print(f"validated={arguments.report} samples={summary['samples']} "
          f"accepted={summary['accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
