"""Reproducible TinyConvNet experiment on a verified Fashion-MNIST manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import save_model
from .evaluate import calibration_metrics, fit_temperature, probabilities_from_logits
from .fashion_mnist import CLASS_NAMES, load_split
from .model import TinyConvNet


def classification_metrics(predictions: np.ndarray, labels: np.ndarray,
                           classes: int) -> dict[str, object]:
    """Return a true-row/predicted-column confusion matrix and class metrics."""
    predicted = np.asarray(predictions)
    targets = np.asarray(labels)
    if (classes < 2 or predicted.shape != targets.shape or predicted.ndim != 1 or
            predicted.size < 1 or
            predicted.dtype.kind not in "iu" or targets.dtype.kind not in "iu" or
            np.any(predicted < 0) or np.any(predicted >= classes) or
            np.any(targets < 0) or np.any(targets >= classes)):
        raise ValueError("predictions and labels must contain valid integer classes")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    rows = []
    for index, name in enumerate(CLASS_NAMES[:classes]):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "f1": f1, "precision": precision,
                     "recall": recall, "support": support})
    return {
        "confusion_matrix": confusion.astype(int).tolist(),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "per_class": rows,
    }


def _manifest_identity(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid manifest file") from error
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def run_experiment(manifest_path: str | Path, data_directory: str | Path,
                   *, channels: int = 8, epochs: int = 30,
                   learning_rate: float = 0.2, batch_size: int = 100,
                   seed: int = 20261014, bins: int = 10) -> tuple[dict[str, object], TinyConvNet]:
    """Train once, calibrate on validation only, and evaluate the untouched test split."""
    if (channels < 1 or epochs < 1 or batch_size < 1 or
            not math.isfinite(learning_rate) or learning_rate <= 0.0 or
            bins < 1 or bins > 1000):
        raise ValueError("invalid experiment hyperparameters")
    manifest_file = Path(manifest_path)
    manifest, manifest_sha256 = _manifest_identity(manifest_file)
    train_images, train_labels = load_split(manifest_file, data_directory, "train")
    validation_images, validation_labels = load_split(
        manifest_file, data_directory, "validation"
    )
    test_images, test_labels = load_split(manifest_file, data_directory, "test")

    model = TinyConvNet(channels=channels, classes=len(CLASS_NAMES), seed=seed)
    initial_loss = model.loss_and_gradients(train_images, train_labels)[0]
    history = model.train(train_images, train_labels, epochs=epochs,
                          learning_rate=learning_rate, batch_size=batch_size,
                          seed=seed + 1)
    validation_logits = model.predict_logits(validation_images)
    temperature = fit_temperature(validation_logits, validation_labels)
    test_logits = model.predict_logits(test_images)
    uncalibrated = probabilities_from_logits(test_logits)
    calibrated = probabilities_from_logits(test_logits, temperature)
    predictions = calibrated.argmax(axis=1)
    metrics = classification_metrics(predictions, test_labels, len(CLASS_NAMES))
    metrics.update({
        "calibrated": calibration_metrics(calibrated, test_labels, bins),
        "samples": int(len(test_labels)),
        "uncalibrated": calibration_metrics(uncalibrated, test_labels, bins),
    })
    parameter_count = sum(int(value.size) for value in model.parameters().values())
    report: dict[str, object] = {
        "dataset": {
            "classes": list(CLASS_NAMES),
            "manifest_seed": manifest.get("seed"),
            "manifest_sha256": manifest_sha256,
            "name": manifest.get("dataset"),
            "official_source_verified": all(
                isinstance(record, dict) and record.get("official_md5_match") is True
                for record in manifest.get("files", {}).values()
            ),
            "split_sizes": {
                "test": int(len(test_labels)),
                "train": int(len(train_labels)),
                "validation": int(len(validation_labels)),
            },
        },
        "protocol": {
            "architecture": "conv3x3-relu-global-average-pool-linear",
            "batch_size": batch_size,
            "calibration": "scalar temperature fitted on validation NLL",
            "channels": channels,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "parameter_count": parameter_count,
            "preprocessing": "float64(pixel) / 255.0; no augmentation",
            "seed": seed,
            "test_policy": "evaluated once after training and validation calibration",
        },
        "schema_version": 1,
        "test": metrics,
        "training": {
            "final_loss": history[-1],
            "initial_loss": initial_loss,
            "loss_history": history,
        },
        "validation": {
            "nll_after": calibration_metrics(
                probabilities_from_logits(validation_logits, temperature),
                validation_labels, bins,
            )["nll"],
            "nll_before": calibration_metrics(
                probabilities_from_logits(validation_logits), validation_labels, bins
            )["nll"],
            "samples": int(len(validation_labels)),
            "temperature": temperature,
        },
    }
    return report, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fashion-experiment.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--bins", type=int, default=10)
    arguments = parser.parse_args(argv)
    try:
        report, model = run_experiment(
            arguments.manifest, arguments.data_directory,
            channels=arguments.channels, epochs=arguments.epochs,
            learning_rate=arguments.learning_rate, batch_size=arguments.batch_size,
            seed=arguments.seed, bins=arguments.bins,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    if arguments.checkpoint is not None:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, arguments.checkpoint)
    calibrated = report["test"]["calibrated"]
    print(f"output={arguments.output} samples={report['test']['samples']} "
          f"accuracy={calibrated['accuracy']:.6f} macro_f1={report['test']['macro_f1']:.6f} "
          f"ece={calibrated['ece']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
