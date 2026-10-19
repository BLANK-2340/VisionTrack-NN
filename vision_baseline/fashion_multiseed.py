"""Fixed-protocol, multi-seed spatial Fashion-MNIST evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .fashion_spatial_experiment import run_experiment

DEFAULT_SEEDS = (20261014, 20261015, 20261016)


def summarize_reports(reports: list[dict]) -> dict:
    """Summarize matched runs; stddev is sample standard deviation, not a CI."""
    if not 2 <= len(reports) <= 10:
        raise ValueError("multi-seed evaluation requires 2..10 runs")
    reference = reports[0]
    seeds = []
    metrics = {name: [] for name in (
        "accuracy", "macro_f1", "calibrated_nll", "uncalibrated_nll",
        "calibrated_ece", "uncalibrated_ece", "ece_change", "nll_change",
    )}
    for report in reports:
        seed = report["protocol"]["seed"]
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError("run seeds must be distinct nonnegative integers")
        seeds.append(seed)
        protocol = {key: value for key, value in report["protocol"].items() if key != "seed"}
        expected = {key: value for key, value in reference["protocol"].items() if key != "seed"}
        if report["dataset"] != reference["dataset"] or protocol != expected:
            raise ValueError("all seeds must use identical dataset and protocol")
        test = report["test"]
        values = {
            "accuracy": test["calibrated"]["accuracy"],
            "macro_f1": test["macro_f1"],
            "calibrated_nll": test["calibrated"]["nll"],
            "uncalibrated_nll": test["uncalibrated"]["nll"],
            "calibrated_ece": test["calibrated"]["ece"],
            "uncalibrated_ece": test["uncalibrated"]["ece"],
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("test metrics must be finite nonnegative numbers")
        if any(values[key] > 1 for key in ("accuracy", "macro_f1", "calibrated_ece",
                                           "uncalibrated_ece")):
            raise ValueError("accuracy, F1 and ECE must be in [0, 1]")
        values["ece_change"] = values["calibrated_ece"] - values["uncalibrated_ece"]
        values["nll_change"] = values["calibrated_nll"] - values["uncalibrated_nll"]
        for key, value in values.items():
            metrics[key].append(value)
    summary = {
        key: {"mean": statistics.mean(values), "sample_stddev": statistics.stdev(values),
              "min": min(values), "max": max(values)}
        for key, values in metrics.items()
    }
    return {"metrics": summary, "run_count": len(reports), "seeds": seeds,
            "calibration_regressions": {
                "ece_increased": sum(value > 0 for value in metrics["ece_change"]),
                "nll_increased": sum(value > 0 for value in metrics["nll_change"]),
            },
            "scope": "initialization and shuffle variation on one fixed split; not split uncertainty",
            "selection_policy": "all prespecified seeds retained; no best-seed selection"}


def run_suite(manifest_path: str | Path, data_directory: str | Path,
              *, seeds: tuple[int, ...] = DEFAULT_SEEDS, channels: int = 8,
              grid_size: int = 4, epochs: int = 100, learning_rate: float = 0.15,
              batch_size: int = 100, bins: int = 10) -> dict:
    if (not 2 <= len(seeds) <= 10 or any(type(seed) is not int or seed < 0 or
                                       seed >= 2**32 - 1 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("provide 2..10 distinct integer seeds in [0, 2**32-2]")
    manifest_file = Path(manifest_path)
    identity = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    reports = []
    for seed in seeds:
        if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != identity:
            raise ValueError("manifest changed during the seed suite")
        report = run_experiment(
            manifest_file, data_directory, channels=channels, grid_size=grid_size,
            epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
            seed=seed, bins=bins,
        )
        if (report["dataset"]["manifest_sha256"] != identity or
                hashlib.sha256(manifest_file.read_bytes()).hexdigest() != identity):
            raise ValueError("manifest changed during the seed suite")
        reports.append(report)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    records = manifest.get("files", {})
    return {
        "schema_version": 1,
        "dataset": {**reports[0]["dataset"],
                    "official_source_verified": len(records) == 4 and all(
                        record.get("official_md5_match") is True for record in records.values()),
                    "preprocessing": "float64(pixel) / 255.0; no augmentation"},
        "evaluation": {"ece_bins": bins,
                       "calibration": "per-seed temperature fitted on validation NLL only",
                       "test_policy": "one evaluation per seed after training; never used to select seeds"},
        "runs": reports,
        "aggregate": summarize_reports(reports),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fashion-multiseed.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        report = run_suite(args.manifest, args.data_directory, seeds=tuple(args.seeds),
                           channels=args.channels, grid_size=args.grid_size, epochs=args.epochs,
                           learning_rate=args.learning_rate, batch_size=args.batch_size, bins=args.bins)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    accuracy = report["aggregate"]["metrics"]["accuracy"]
    print(f"runs={report['aggregate']['run_count']} accuracy_mean={accuracy['mean']:.6f} "
          f"sample_stddev={accuracy['sample_stddev']:.6f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
