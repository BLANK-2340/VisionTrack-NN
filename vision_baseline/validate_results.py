"""Validate committed Fashion-MNIST reports and their cross-file invariants."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


ARTIFACT_NAMES = (
    "fashion_mnist_tinyconvnet_summary.json",
    "fashion_mnist_spatial_comparison.json",
    "fashion_mnist_spatial_multiseed.json",
    "fashion_mnist_pooling_budget_comparison.json",
)
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
EXPECTED_SPLITS = {"train": 1000, "validation": 500, "test": 500}


class ValidationError(ValueError):
    """Raised when a result artifact is malformed or internally inconsistent."""


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-standard numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_finite_numbers(value: Any, label: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite number at {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{label}[{index}]")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"artifact must be a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_ARTIFACT_BYTES:
        raise ValidationError(f"artifact size outside 1..{MAX_ARTIFACT_BYTES}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"artifact root must be an object: {path}")
    _require_finite_numbers(value)
    return value


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValidationError(f"{label} keys differ; missing={missing}, extra={extra}")


def _require_close(actual: float, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise ValidationError(f"{label} must be numeric")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValidationError(f"{label} mismatch: {actual!r} != {expected!r}")


def _dataset_core(dataset: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        manifest = dataset["manifest_sha256"]
        splits = dataset["split_sizes"]
    except (KeyError, TypeError) as error:
        raise ValidationError(f"{label} lacks dataset identity") from error
    if not isinstance(manifest, str) or len(manifest) != 64:
        raise ValidationError(f"{label} has an invalid manifest SHA-256")
    if splits != EXPECTED_SPLITS:
        raise ValidationError(f"{label} split sizes differ: {splits!r}")
    return {"manifest_sha256": manifest, "split_sizes": splits}


def _confusion(matrix: Any, samples: int, label: str) -> None:
    if not isinstance(matrix, list) or len(matrix) != 10:
        raise ValidationError(f"{label} must contain ten rows")
    if any(not isinstance(row, list) or len(row) != 10 for row in matrix):
        raise ValidationError(f"{label} must be 10x10")
    cells = [cell for row in matrix for cell in row]
    if any(not isinstance(cell, int) or isinstance(cell, bool) or cell < 0 for cell in cells):
        raise ValidationError(f"{label} cells must be non-negative integers")
    if sum(cells) != samples:
        raise ValidationError(f"{label} total {sum(cells)} != {samples}")


def _stats(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValidationError("at least two runs are required for sample statistics")
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values),
    }


def _check_stats(actual: dict[str, Any], values: list[float], label: str) -> None:
    _require_keys(actual, {"min", "max", "mean", "sample_stddev"}, label)
    for key, expected in _stats(values).items():
        _require_close(actual[key], expected, f"{label}.{key}")


def _run_metric(run: dict[str, Any], name: str) -> float:
    test = run["test"]
    mapping: dict[str, Callable[[], float]] = {
        "accuracy": lambda: test["calibrated"]["accuracy"],
        "macro_f1": lambda: test["macro_f1"],
        "calibrated_ece": lambda: test["calibrated"]["ece"],
        "calibrated_nll": lambda: test["calibrated"]["nll"],
        "uncalibrated_ece": lambda: test["uncalibrated"]["ece"],
        "uncalibrated_nll": lambda: test["uncalibrated"]["nll"],
        "ece_change": lambda: test["calibrated"]["ece"] - test["uncalibrated"]["ece"],
        "nll_change": lambda: test["calibrated"]["nll"] - test["uncalibrated"]["nll"],
    }
    try:
        value = mapping[name]()
    except (KeyError, TypeError) as error:
        raise ValidationError(f"run lacks metric {name}") from error
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValidationError(f"run metric {name} is not finite numeric data")
    return float(value)


def _validate_suite(suite: dict[str, Any], core: dict[str, Any], label: str) -> None:
    _require_keys(suite, {"schema_version", "dataset", "evaluation", "runs", "aggregate"}, label)
    if suite["schema_version"] != 1:
        raise ValidationError(f"{label} schema_version must be 1")
    if _dataset_core(suite["dataset"], f"{label}.dataset") != core:
        raise ValidationError(f"{label} uses a different dataset")
    if suite["dataset"].get("official_source_verified") is not True:
        raise ValidationError(f"{label} does not record official source verification")
    runs = suite["runs"]
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValidationError(f"{label}.runs must contain at least two runs")
    seeds: list[int] = []
    for index, run in enumerate(runs):
        if run.get("schema_version") != 1:
            raise ValidationError(f"{label}.runs[{index}] schema_version must be 1")
        if _dataset_core(run.get("dataset", {}), f"{label}.runs[{index}]") != core:
            raise ValidationError(f"{label}.runs[{index}] uses a different dataset")
        protocol = run.get("protocol", {})
        seed = protocol.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValidationError(f"{label}.runs[{index}] has an invalid seed")
        seeds.append(seed)
        test = run.get("test", {})
        if test.get("samples") != EXPECTED_SPLITS["test"]:
            raise ValidationError(f"{label}.runs[{index}] has an invalid test count")
        _confusion(test.get("confusion_matrix"), test["samples"], f"{label}.runs[{index}].confusion")
        correct = sum(test["confusion_matrix"][i][i] for i in range(10))
        _require_close(test["calibrated"]["accuracy"], correct / test["samples"],
                       f"{label}.runs[{index}].accuracy")

    aggregate = suite["aggregate"]
    if aggregate.get("run_count") != len(runs) or aggregate.get("seeds") != seeds:
        raise ValidationError(f"{label} aggregate run identity differs from runs")
    metrics = aggregate.get("metrics", {})
    expected_metrics = {
        "accuracy", "macro_f1", "calibrated_ece", "calibrated_nll",
        "uncalibrated_ece", "uncalibrated_nll", "ece_change", "nll_change",
    }
    if set(metrics) != expected_metrics:
        raise ValidationError(f"{label} aggregate metric set differs")
    for name in sorted(expected_metrics):
        _check_stats(metrics[name], [_run_metric(run, name) for run in runs], f"{label}.{name}")
    regressions = aggregate.get("calibration_regressions", {})
    expected_regressions = {
        "ece_increased": sum(_run_metric(run, "ece_change") > 0.0 for run in runs),
        "nll_increased": sum(_run_metric(run, "nll_change") > 0.0 for run in runs),
    }
    if regressions != expected_regressions:
        raise ValidationError(f"{label} calibration regression counts differ")


def validate_results(directory: str | Path) -> dict[str, Any]:
    """Validate all committed reports in *directory* and return a small summary."""
    root = Path(directory)
    artifacts = {name: _load(root / name) for name in ARTIFACT_NAMES}
    tiny, spatial, multiseed, pooling = (artifacts[name] for name in ARTIFACT_NAMES)

    _require_keys(tiny, {"schema_version", "dataset", "protocol", "training", "validation", "test"}, "tiny")
    _require_keys(spatial, {"schema_version", "dataset", "global_average_pool_control", "spatial_grid_model", "spatial_minus_control"}, "spatial")
    _require_keys(pooling, {"schema_version", "scope", "dataset", "arms", "paired_grid_minus_global", "limitations"}, "pooling")
    if any(artifact.get("schema_version") != 1 for artifact in artifacts.values()):
        raise ValidationError("every artifact schema_version must be 1")

    core = _dataset_core(tiny["dataset"], "tiny.dataset")
    for label, artifact in (("spatial", spatial), ("multiseed", multiseed), ("pooling", pooling)):
        if _dataset_core(artifact["dataset"], f"{label}.dataset") != core:
            raise ValidationError(f"{label} uses a different dataset manifest or split")
    if tiny["dataset"].get("official_source_verified") is not True:
        raise ValidationError("tiny artifact does not record official source verification")

    _confusion(tiny["test"].get("confusion_matrix_true_rows"), 500, "tiny.confusion")
    control = spatial["global_average_pool_control"]
    for source, target, field in (
        (tiny["test"], control, "accuracy"),
        (tiny["test"], control, "macro_f1"),
        (tiny["protocol"], control, "parameter_count"),
    ):
        _require_close(target[field], source[field], f"spatial.control.{field}")

    _validate_suite(multiseed, core, "multiseed")
    first = multiseed["runs"][0]
    spatial_model = spatial["spatial_grid_model"]
    for actual, expected, label in (
        (spatial_model["accuracy"], first["test"]["calibrated"]["accuracy"], "spatial.accuracy"),
        (spatial_model["macro_f1"], first["test"]["macro_f1"], "spatial.macro_f1"),
        (spatial_model["parameter_count"], first["protocol"]["parameter_count"], "spatial.parameter_count"),
    ):
        _require_close(actual, expected, label)
    if spatial_model["confusion_matrix_true_rows"] != first["test"]["confusion_matrix"]:
        raise ValidationError("spatial comparison differs from the first multi-seed confusion matrix")
    _require_close(spatial["spatial_minus_control"]["accuracy_percentage_points"],
                   100.0 * (spatial_model["accuracy"] - control["accuracy"]), "spatial.accuracy_delta")
    _require_close(spatial["spatial_minus_control"]["macro_f1"],
                   spatial_model["macro_f1"] - control["macro_f1"], "spatial.macro_f1_delta")

    arms = pooling.get("arms", {})
    if set(arms) != {"grid_pool", "global_pool"}:
        raise ValidationError("pooling artifact must contain grid_pool and global_pool")
    if arms["grid_pool"] != multiseed:
        raise ValidationError("pooling grid arm is not the exact committed multi-seed report")
    _validate_suite(arms["global_pool"], core, "global_pool")
    grid_runs = arms["grid_pool"]["runs"]
    global_runs = arms["global_pool"]["runs"]
    paired = pooling["paired_grid_minus_global"]
    per_seed = paired.get("per_seed", [])
    if len(per_seed) != len(grid_runs) or len(global_runs) != len(grid_runs):
        raise ValidationError("pooling arm and paired run counts differ")
    expected_pairs: dict[str, list[float]] = {
        "accuracy_percentage_points": [], "macro_f1": [],
        "calibrated_ece": [], "calibrated_nll": [],
    }
    for index, (grid, global_run, recorded) in enumerate(zip(grid_runs, global_runs, per_seed)):
        grid_protocol, global_protocol = grid["protocol"], global_run["protocol"]
        if grid_protocol["seed"] != global_protocol["seed"] or recorded.get("seed") != grid_protocol["seed"]:
            raise ValidationError(f"paired seed mismatch at index {index}")
        if grid_protocol["parameter_count"] != global_protocol["parameter_count"]:
            raise ValidationError(f"parameter budgets differ at index {index}")
        for protocol in (grid_protocol, global_protocol):
            formula = protocol["channels"] * (10 + 10 * protocol["grid_size"] ** 2) + 10
            if protocol["parameter_count"] != formula:
                raise ValidationError(f"parameter formula mismatch at index {index}")
        expected = {
            "accuracy_percentage_points": 100.0 * (_run_metric(grid, "accuracy") - _run_metric(global_run, "accuracy")),
            "macro_f1": _run_metric(grid, "macro_f1") - _run_metric(global_run, "macro_f1"),
            "calibrated_ece": _run_metric(grid, "calibrated_ece") - _run_metric(global_run, "calibrated_ece"),
            "calibrated_nll": _run_metric(grid, "calibrated_nll") - _run_metric(global_run, "calibrated_nll"),
        }
        for name, value in expected.items():
            _require_close(recorded[name], value, f"paired[{index}].{name}")
            expected_pairs[name].append(value)
    aggregate = paired.get("aggregate", {})
    if set(aggregate) != set(expected_pairs):
        raise ValidationError("paired aggregate metric set differs")
    for name, values in expected_pairs.items():
        _check_stats(aggregate[name], values, f"paired.{name}")

    return {
        "artifact_count": len(artifacts),
        "manifest_sha256": core["manifest_sha256"],
        "multiseed_runs": len(multiseed["runs"]),
        "pooling_runs": len(grid_runs) + len(global_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results", help="artifact directory")
    arguments = parser.parse_args()
    try:
        summary = validate_results(arguments.directory)
    except ValidationError as error:
        parser.exit(1, f"result validation failed: {error}\n")
    print(
        f"validated_artifacts={summary['artifact_count']} "
        f"multiseed_runs={summary['multiseed_runs']} "
        f"pooling_runs={summary['pooling_runs']} "
        f"manifest_sha256={summary['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
