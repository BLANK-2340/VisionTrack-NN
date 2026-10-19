"""Compare grid and global pooling at exactly matched trainable parameter counts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .fashion_multiseed import DEFAULT_SEEDS, run_suite, summarize_reports


def matched_global_channels(channels: int, grid_size: int) -> int:
    """For ten classes: grid has C*(10+10*g*g)+10, global has 20*K+10."""
    if (type(channels) is not int or type(grid_size) is not int or
            channels < 1 or not 2 <= grid_size <= 4):
        raise ValueError("positive integer channels and a grid of size 2..4 required")
    numerator = channels * (1 + grid_size * grid_size)
    if numerator % 2 or numerator // 2 > 128:
        raise ValueError("exact matching requires an integer global width no larger than 128")
    return numerator // 2


def compare_suites(grid: dict, global_pool: dict) -> dict:
    """Validate matched budgets and form paired grid-minus-global seed differences."""
    grid_summary = summarize_reports(grid["runs"])
    global_summary = summarize_reports(global_pool["runs"])
    if (grid["dataset"] != global_pool["dataset"] or
            grid["runs"][0]["dataset"] != global_pool["runs"][0]["dataset"] or
            grid["evaluation"] != global_pool["evaluation"] or
            grid_summary["seeds"] != global_summary["seeds"]):
        raise ValueError("comparison requires identical data, evaluation and ordered seeds")
    differences = []
    for spatial, global_run in zip(grid["runs"], global_pool["runs"]):
        spatial_protocol = spatial["protocol"]
        global_protocol = global_run["protocol"]
        expected_width = matched_global_channels(spatial_protocol["channels"],
                                                 spatial_protocol["grid_size"])
        if global_protocol["grid_size"] != 1 or global_protocol["channels"] != expected_width:
            raise ValueError("global pooling architecture is not parameter matched")
        expected_count = 20 * expected_width + 10
        if (spatial_protocol["parameter_count"] != expected_count or
                global_protocol["parameter_count"] != expected_count):
            raise ValueError("actual trainable parameter counts differ")
        excluded = {"channels", "grid_size"}
        if ({key: value for key, value in spatial_protocol.items() if key not in excluded} !=
                {key: value for key, value in global_protocol.items() if key not in excluded}):
            raise ValueError("training protocol differs between comparison arms")
        differences.append({
            "seed": spatial_protocol["seed"],
            "accuracy_percentage_points": 100 * (spatial["test"]["calibrated"]["accuracy"] -
                                                  global_run["test"]["calibrated"]["accuracy"]),
            "macro_f1": spatial["test"]["macro_f1"] - global_run["test"]["macro_f1"],
            "calibrated_nll": spatial["test"]["calibrated"]["nll"] -
                              global_run["test"]["calibrated"]["nll"],
            "calibrated_ece": spatial["test"]["calibrated"]["ece"] -
                              global_run["test"]["calibrated"]["ece"],
        })
    paired = {}
    for key in differences[0]:
        if key == "seed":
            continue
        values = [row[key] for row in differences]
        paired[key] = {"mean": statistics.mean(values),
                       "sample_stddev": statistics.stdev(values),
                       "min": min(values), "max": max(values)}
    return {
        "schema_version": 1,
        "dataset": grid["dataset"],
        "arms": {"grid_pool": {**grid, "aggregate": grid_summary},
                 "global_pool": {**global_pool, "aggregate": global_summary}},
        "paired_grid_minus_global": {"per_seed": differences, "aggregate": paired},
        "scope": "matched total parameter budget; convolution width and capacity allocation differ",
        "limitations": ["not a pure pooling-only intervention", "not compute matched",
                        "fixed previously evaluated subset; not independent generalization evidence",
                        "same seed labels do not imply identical differently shaped initial weights",
                        "no best-seed selection or test-based tuning"],
    }


def run_comparison(manifest: str | Path, data_directory: str | Path, *,
                   channels: int = 8, grid_size: int = 4,
                   seeds: tuple[int, ...] = DEFAULT_SEEDS, epochs: int = 100,
                   learning_rate: float = 0.15, batch_size: int = 100,
                   bins: int = 10) -> dict:
    global_channels = matched_global_channels(channels, grid_size)
    settings = dict(seeds=seeds, epochs=epochs, learning_rate=learning_rate,
                    batch_size=batch_size, bins=bins)
    grid = run_suite(manifest, data_directory, channels=channels,
                     grid_size=grid_size, **settings)
    global_pool = run_suite(manifest, data_directory, channels=global_channels,
                            grid_size=1, **settings)
    return compare_suites(grid, global_pool)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fashion-pooling-comparison.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        report = run_comparison(args.manifest, args.data_directory,
            channels=args.channels, grid_size=args.grid_size, seeds=tuple(args.seeds),
            epochs=args.epochs, learning_rate=args.learning_rate,
            batch_size=args.batch_size, bins=args.bins)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    delta = report["paired_grid_minus_global"]["aggregate"]["accuracy_percentage_points"]
    print(f"mean_accuracy_grid_minus_global_pp={delta['mean']:.6f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
