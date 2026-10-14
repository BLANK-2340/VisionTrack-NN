"""Evaluate a spatially preserving NumPy CNN on a Fashion-MNIST manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .evaluate import calibration_metrics, fit_temperature, probabilities_from_logits
from .fashion_experiment import classification_metrics
from .fashion_mnist import CLASS_NAMES, load_split
from .spatial_model import SpatialConvNet


def run_experiment(manifest_path: str | Path, data_directory: str | Path,
                   *, channels: int = 8, grid_size: int = 4, epochs: int = 100,
                   learning_rate: float = 0.15, batch_size: int = 100,
                   seed: int = 20261014, bins: int = 10) -> dict[str, object]:
    """Train, calibrate on validation, and evaluate the untouched test split."""
    if (channels < 1 or grid_size < 1 or epochs < 1 or batch_size < 1 or
            not math.isfinite(learning_rate) or learning_rate <= 0.0 or
            bins < 1 or bins > 1000):
        raise ValueError("invalid experiment hyperparameters")
    manifest_file = Path(manifest_path)
    manifest_bytes = manifest_file.read_bytes()
    train_images, train_labels = load_split(manifest_file, data_directory, "train")
    validation_images, validation_labels = load_split(
        manifest_file, data_directory, "validation"
    )
    test_images, test_labels = load_split(manifest_file, data_directory, "test")
    model = SpatialConvNet(channels=channels, classes=len(CLASS_NAMES),
                           grid_size=grid_size, seed=seed)
    initial_loss = model.loss_and_gradients(train_images, train_labels)[0]
    history = model.train(train_images, train_labels, epochs=epochs,
                          learning_rate=learning_rate, batch_size=batch_size,
                          seed=seed + 1)
    validation_logits = model.predict_logits(validation_images)
    temperature = fit_temperature(validation_logits, validation_labels)
    test_logits = model.predict_logits(test_images)
    uncalibrated = probabilities_from_logits(test_logits)
    calibrated = probabilities_from_logits(test_logits, temperature)
    metrics = classification_metrics(calibrated.argmax(axis=1), test_labels,
                                     len(CLASS_NAMES))
    metrics.update({
        "calibrated": calibration_metrics(calibrated, test_labels, bins),
        "samples": int(len(test_labels)),
        "uncalibrated": calibration_metrics(uncalibrated, test_labels, bins),
    })
    validation_before = calibration_metrics(
        probabilities_from_logits(validation_logits), validation_labels, bins
    )
    validation_after = calibration_metrics(
        probabilities_from_logits(validation_logits, temperature), validation_labels, bins
    )
    return {
        "dataset": {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "split_sizes": {"train": int(len(train_labels)),
                            "validation": int(len(validation_labels)),
                            "test": int(len(test_labels))},
        },
        "protocol": {
            "architecture": "conv3x3-relu-grid-average-pool-linear",
            "batch_size": batch_size,
            "channels": channels,
            "epochs": epochs,
            "grid_size": grid_size,
            "learning_rate": learning_rate,
            "parameter_count": sum(int(value.size) for value in model.parameters().values()),
            "seed": seed,
            "test_policy": "evaluated once after training and validation calibration",
        },
        "schema_version": 1,
        "test": metrics,
        "training": {"initial_loss": initial_loss, "final_loss": history[-1],
                     "loss_history": history},
        "validation": {"nll_before": validation_before["nll"],
                       "nll_after": validation_after["nll"],
                       "samples": int(len(validation_labels)),
                       "temperature": temperature},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fashion-spatial.json"))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--bins", type=int, default=10)
    arguments = parser.parse_args(argv)
    try:
        report = run_experiment(
            arguments.manifest, arguments.data_directory,
            channels=arguments.channels, grid_size=arguments.grid_size,
            epochs=arguments.epochs, learning_rate=arguments.learning_rate,
            batch_size=arguments.batch_size, seed=arguments.seed, bins=arguments.bins,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    calibrated = report["test"]["calibrated"]
    print(f"output={arguments.output} samples={report['test']['samples']} "
          f"accuracy={calibrated['accuracy']:.6f} macro_f1={report['test']['macro_f1']:.6f} "
          f"ece={calibrated['ece']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
