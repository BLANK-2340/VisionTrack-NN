"""Evaluate a checkpoint against a strict class-directory PGM dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .checkpoint import loads_model
from .pgm import load_pgm

DEFAULT_LABELS = ("vertical", "horizontal", "diagonal")


def _labels_for_classes(classes: int, labels: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(labels) if labels is not None else (
        DEFAULT_LABELS if classes == len(DEFAULT_LABELS)
        else tuple(f"class_{index}" for index in range(classes))
    )
    if (len(values) != classes or any(not value or value in {".", ".."} for value in values)
            or len(set(values)) != len(values)
            or any("/" in value or "\\" in value for value in values)):
        raise ValueError(f"labels must contain {classes} unique directory names")
    return values


def _dataset_files(root: Path, labels: tuple[str, ...]) -> list[tuple[int, Path]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("dataset root must be a non-symlink directory")
    expected = set(labels)
    entries = list(root.iterdir())
    directories = {entry.name for entry in entries
                   if entry.is_dir() and not entry.is_symlink()}
    if directories != expected:
        missing = sorted(expected - directories)
        unexpected = sorted(directories - expected)
        raise ValueError(
            f"dataset class directories mismatch: missing={missing}, unexpected={unexpected}"
        )
    unexpected_entries = sorted(entry.name for entry in entries
                                if entry.name not in expected or entry.is_symlink())
    if unexpected_entries:
        raise ValueError(f"unexpected dataset root entries: {unexpected_entries}")
    samples: list[tuple[int, Path]] = []
    for class_index, label in enumerate(labels):
        directory = root / label
        files = []
        for entry in directory.iterdir():
            if entry.is_symlink() or not entry.is_file() or entry.suffix != ".pgm":
                raise ValueError(f"unexpected dataset entry: {entry.relative_to(root).as_posix()}")
            files.append(entry)
        if not files:
            raise ValueError(f"dataset class has no PGM images: {label}")
        samples.extend((class_index, path) for path in sorted(files, key=lambda item: item.name))
    return samples


def _class_metrics(confusion: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for index, label in enumerate(labels):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "f1": f1,
            "label": label,
            "precision": precision,
            "recall": recall,
            "support": support,
        })
    return rows


def evaluate_folder(checkpoint: str | Path, dataset: str | Path,
                    labels: Sequence[str] | None = None,
                    temperature: float = 1.0) -> dict[str, object]:
    """Return a deterministic report for ``dataset/<label>/*.pgm``."""
    checkpoint_path = Path(checkpoint)
    checkpoint_bytes = checkpoint_path.read_bytes()
    model = loads_model(checkpoint_bytes)
    class_labels = _labels_for_classes(model.bias.size, labels)
    dataset_root = Path(dataset)
    samples = _dataset_files(dataset_root, class_labels)
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    predictions = []
    correct = 0
    for target, path in samples:
        pixels = load_pgm(path)
        probabilities = model.predict_proba(pixels[None, :, :], temperature=temperature)[0]
        predicted = int(probabilities.argmax())
        confusion[target, predicted] += 1
        correct += int(predicted == target)
        predictions.append({
            "confidence": float(probabilities[predicted]),
            "height": int(pixels.shape[0]),
            "path": path.relative_to(dataset_root).as_posix(),
            "predicted_index": predicted,
            "predicted_label": class_labels[predicted],
            "target_index": target,
            "target_label": class_labels[target],
            "width": int(pixels.shape[1]),
        })
    per_class = _class_metrics(confusion, class_labels)
    return {
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "dataset_layout": "class-directory-pgm",
        "labels": list(class_labels),
        "metrics": {
            "accuracy": correct / len(samples),
            "confusion_matrix": confusion.tolist(),
            "macro_f1": float(np.mean([row["f1"] for row in per_class])),
            "macro_precision": float(np.mean([row["precision"] for row in per_class])),
            "macro_recall": float(np.mean([row["recall"] for row in per_class])),
            "per_class": per_class,
        },
        "predictions": predictions,
        "samples": len(samples),
        "schema_version": 1,
        "temperature": float(temperature),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="VTNN checkpoint")
    parser.add_argument("dataset", type=Path, help="root containing one directory per class")
    parser.add_argument("--output", type=Path, default=Path("folder-evaluation.json"))
    parser.add_argument("--labels", nargs="+", help="class directory names in model order")
    parser.add_argument("--temperature", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    try:
        report = evaluate_folder(arguments.checkpoint, arguments.dataset,
                                 arguments.labels, arguments.temperature)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(f"output={arguments.output} samples={report['samples']} "
          f"accuracy={report['metrics']['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
