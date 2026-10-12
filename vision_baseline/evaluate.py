"""Deterministic confidence calibration and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .model import TinyConvNet, make_dataset


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert a finite score matrix to stable softmax probabilities."""
    scores = np.asarray(logits, dtype=np.float64)
    if (scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 2 or
            not np.all(np.isfinite(scores))):
        raise ValueError("logits must be a non-empty finite [samples, classes] matrix")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    shifted = scores / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def corrupt_images(images: np.ndarray, seed: int, noise_std: float = 0.4) -> np.ndarray:
    """Apply deterministic clipped Gaussian sensor noise to an image batch."""
    values = np.asarray(images, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] < 1 or not np.all(np.isfinite(values)) or
            np.any(values < 0.0) or np.any(values > 1.0)):
        raise ValueError("images must be a non-empty finite [batch, height, width] array in [0, 1]")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative")
    rng = np.random.default_rng(seed)
    return np.clip(values + rng.normal(0.0, noise_std, values.shape), 0.0, 1.0)


def _validated_labels(labels: np.ndarray, samples: int, classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if (values.shape != (samples,) or values.dtype.kind not in "iu" or
            np.any(values < 0) or np.any(values >= classes)):
        raise ValueError("labels must contain one valid integer class per sample")
    return values.astype(np.int64, copy=False)


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray,
                        bins: int = 10) -> dict[str, object]:
    """Compute accuracy, NLL, multiclass Brier score, ECE, and bin details."""
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2 or
            not np.all(np.isfinite(values)) or np.any(values < 0.0) or
            not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError("probabilities must be a normalized finite matrix")
    if not isinstance(bins, int) or not 1 <= bins <= 1000:
        raise ValueError("bins must be an integer in [1, 1000]")
    targets = _validated_labels(labels, values.shape[0], values.shape[1])
    predictions = values.argmax(axis=1)
    confidence = values[np.arange(len(values)), predictions]
    correct = predictions == targets
    selected = np.clip(values[np.arange(len(values)), targets], np.finfo(np.float64).tiny, 1.0)
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    bin_indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    details = []
    ece = 0.0
    for index in range(bins):
        mask = bin_indices == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else 0.0
        mean_confidence = float(confidence[mask].mean()) if count else 0.0
        gap = abs(accuracy - mean_confidence) if count else 0.0
        ece += count / len(values) * gap
        details.append({
            "accuracy": accuracy,
            "confidence": mean_confidence,
            "count": count,
            "gap": gap,
            "lower": index / bins,
            "upper": (index + 1) / bins,
        })
    return {
        "accuracy": float(correct.mean()),
        "brier": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "nll": float(-np.log(selected).mean()),
        "reliability_bins": details,
    }


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probabilities = probabilities_from_logits(logits, temperature)
    targets = _validated_labels(labels, len(probabilities), probabilities.shape[1])
    selected = np.clip(probabilities[np.arange(len(targets)), targets],
                       np.finfo(np.float64).tiny, 1.0)
    return float(-np.log(selected).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one temperature by deterministic bounded NLL minimization."""
    scores = np.asarray(logits, dtype=np.float64)
    if (scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 2 or
            not np.all(np.isfinite(scores))):
        raise ValueError("logits must be a non-empty finite [samples, classes] matrix")
    targets = _validated_labels(labels, scores.shape[0], scores.shape[1])
    grid = np.linspace(-4.0, 4.0, 257)
    losses = np.asarray([_nll(scores, targets, math.exp(value)) for value in grid])
    best = int(losses.argmin())
    if best in (0, len(grid) - 1):
        candidate = math.exp(float(grid[best]))
    else:
        left, right = float(grid[best - 1]), float(grid[best + 1])
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        c = right - ratio * (right - left)
        d = left + ratio * (right - left)
        fc = _nll(scores, targets, math.exp(c))
        fd = _nll(scores, targets, math.exp(d))
        for _ in range(64):
            if fc <= fd:
                right, d, fd = d, c, fc
                c = right - ratio * (right - left)
                fc = _nll(scores, targets, math.exp(c))
            else:
                left, c, fc = c, d, fd
                d = left + ratio * (right - left)
                fd = _nll(scores, targets, math.exp(d))
        candidate = math.exp((left + right) / 2.0)
    return candidate if _nll(scores, targets, candidate) <= _nll(scores, targets, 1.0) else 1.0


def selective_metrics(probabilities: np.ndarray, labels: np.ndarray,
                      coverages: tuple[float, ...] = (0.5, 0.75, 0.9, 1.0)) -> list[dict[str, float | int]]:
    """Measure error after retaining the most-confident requested fraction."""
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2 or
            not np.all(np.isfinite(values)) or np.any(values < 0.0) or
            not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError("probabilities must be a normalized finite matrix")
    targets = _validated_labels(labels, values.shape[0], values.shape[1])
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in coverages):
        raise ValueError("coverages must be finite values in (0, 1]")
    confidence = values.max(axis=1)
    predictions = values.argmax(axis=1)
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for requested in coverages:
        retained = max(1, math.ceil(requested * len(values)))
        indices = order[:retained]
        rows.append({
            "accuracy": float(np.mean(predictions[indices] == targets[indices])),
            "coverage": retained / len(values),
            "requested_coverage": requested,
            "retained": retained,
            "risk": float(np.mean(predictions[indices] != targets[indices])),
        })
    return rows


def evaluate_model(model: TinyConvNet, calibration_images: np.ndarray,
                   calibration_labels: np.ndarray, test_images: np.ndarray,
                   test_labels: np.ndarray, bins: int = 10) -> dict[str, object]:
    calibration_logits = model.predict_logits(calibration_images)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    test_logits = model.predict_logits(test_images)
    uncalibrated = probabilities_from_logits(test_logits)
    calibrated = probabilities_from_logits(test_logits, temperature)
    return {
        "calibration": {
            "nll_after": _nll(calibration_logits, calibration_labels, temperature),
            "nll_before": _nll(calibration_logits, calibration_labels, 1.0),
            "samples": int(len(calibration_labels)),
            "temperature": temperature,
        },
        "schema_version": 1,
        "test": {
            "calibrated": calibration_metrics(calibrated, test_labels, bins),
            "samples": int(len(test_labels)),
            "selective": selective_metrics(calibrated, test_labels),
            "uncalibrated": calibration_metrics(uncalibrated, test_labels, bins),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--bins", type=int, default=10)
    arguments = parser.parse_args(argv)
    train_images, train_labels = make_dataset(60, seed=11)
    clean_calibration_images, calibration_labels = make_dataset(30, seed=23)
    clean_test_images, test_labels = make_dataset(30, seed=29)
    calibration_images = corrupt_images(clean_calibration_images, seed=101)
    test_images = corrupt_images(clean_test_images, seed=103)
    model = TinyConvNet(seed=7)
    model.train(train_images, train_labels, seed=13)
    report = evaluate_model(model, calibration_images, calibration_labels,
                            test_images, test_labels, arguments.bins)
    report["protocol"] = {
        "calibration_corruption_seed": 101,
        "calibration_data_seed": 23,
        "classes": ["vertical", "horizontal", "diagonal"],
        "dataset": "generated-orientation-lines",
        "noise": "clipped-zero-mean-gaussian",
        "noise_std": 0.4,
        "test_corruption_seed": 103,
        "test_data_seed": 29,
        "train_data_seed": 11,
        "training_seed": 13,
    }
    report["test"]["clean_accuracy"] = model.accuracy(clean_test_images, test_labels)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    calibrated = report["test"]["calibrated"]
    print(f"output={arguments.output} samples={report['test']['samples']} "
          f"temperature={report['calibration']['temperature']:.6f} "
          f"accuracy={calibrated['accuracy']:.6f} ece={calibrated['ece']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
