"""Validated Fashion-MNIST IDX adapter and reproducible split manifests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

DATASET_NAME = "Fashion-MNIST"
DATASET_VERSION = "official-2017"
SOURCE_URL = "https://github.com/zalandoresearch/fashion-mnist"
PAPER_URL = "https://arxiv.org/abs/1708.07747"
LICENSE_URL = "https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE"
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
}
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 60 * 1024 * 1024
MAX_SAMPLES = 70_000


def _file_bytes(path: Path) -> bytes:
    size = path.stat().st_size
    if size < 1 or size > MAX_COMPRESSED_BYTES:
        raise ValueError(f"invalid compressed file size: {path.name}")
    return path.read_bytes()


def _decompress(path: Path) -> tuple[bytes, dict[str, object]]:
    compressed = _file_bytes(path)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_DECOMPRESSED_BYTES + 1)
    except (EOFError, OSError) as error:
        raise ValueError(f"invalid gzip stream: {path.name}") from error
    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise ValueError(f"decompressed file exceeds resource limit: {path.name}")
    return raw, {
        "bytes": len(compressed),
        "md5": hashlib.md5(compressed, usedforsecurity=False).hexdigest(),
        "name": path.name,
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def load_idx_pair(images_path: str | Path,
                  labels_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one validated 28x28 Fashion-MNIST IDX image/label pair."""
    image_raw, _ = _decompress(Path(images_path))
    label_raw, _ = _decompress(Path(labels_path))
    if len(image_raw) < 16 or len(label_raw) < 8:
        raise ValueError("truncated IDX header")
    image_magic, count, rows, columns = struct.unpack(">IIII", image_raw[:16])
    label_magic, label_count = struct.unpack(">II", label_raw[:8])
    if image_magic != 2051 or label_magic != 2049:
        raise ValueError("invalid IDX magic")
    if count < 1 or count > MAX_SAMPLES or label_count != count:
        raise ValueError("invalid or inconsistent IDX sample count")
    if (rows, columns) != (28, 28):
        raise ValueError("Fashion-MNIST images must be 28x28")
    expected_images = 16 + count * rows * columns
    if len(image_raw) != expected_images or len(label_raw) != 8 + count:
        raise ValueError("IDX payload size mismatch")
    images = np.frombuffer(image_raw, dtype=np.uint8, offset=16)
    images = images.reshape(count, rows, columns).copy()
    labels = np.frombuffer(label_raw, dtype=np.uint8, offset=8).copy()
    if np.any(labels >= len(CLASS_NAMES)):
        raise ValueError("Fashion-MNIST labels must be in [0, 9]")
    return images, labels


def _paths(data_directory: Path) -> dict[str, Path]:
    return {key: data_directory / filename for key, (filename, _) in FILES.items()}


def _stratified_partitions(labels: np.ndarray, first_per_class: int,
                           second_per_class: int, seed: int) -> tuple[list[int], list[int]]:
    if first_per_class < 1 or second_per_class < 0:
        raise ValueError("per-class split sizes must be positive (validation may be zero)")
    rng = np.random.default_rng(seed)
    first, second = [], []
    for label in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(labels == label)
        required = first_per_class + second_per_class
        if len(candidates) < required:
            raise ValueError(f"class {label} has {len(candidates)} samples; {required} required")
        chosen = rng.permutation(candidates)[:required]
        first.extend(int(value) for value in chosen[:first_per_class])
        second.extend(int(value) for value in chosen[first_per_class:])
    return sorted(first), sorted(second)


def _counts(labels: np.ndarray, indices: list[int]) -> list[int]:
    return np.bincount(labels[indices], minlength=len(CLASS_NAMES)).astype(int).tolist()


def build_manifest(data_directory: str | Path, train_per_class: int = 500,
                   validation_per_class: int = 100, test_per_class: int = 100,
                   seed: int = 20261014, require_official: bool = True) -> dict[str, object]:
    """Validate source files and return a canonical stratified split manifest."""
    root = Path(data_directory)
    paths = _paths(root)
    train_images, train_labels = load_idx_pair(paths["train_images"], paths["train_labels"])
    test_images, test_labels = load_idx_pair(paths["test_images"], paths["test_labels"])
    del train_images, test_images
    file_records = {}
    for key, path in paths.items():
        _, record = _decompress(path)
        expected_md5 = FILES[key][1]
        record["expected_official_md5"] = expected_md5
        record["official_md5_match"] = record["md5"] == expected_md5
        if require_official and not record["official_md5_match"]:
            raise ValueError(f"official MD5 mismatch: {path.name}")
        file_records[key] = record
    train_indices, validation_indices = _stratified_partitions(
        train_labels, train_per_class, validation_per_class, seed
    )
    test_indices, unused = _stratified_partitions(test_labels, test_per_class, 0, seed + 1)
    assert not unused
    return {
        "classes": list(CLASS_NAMES),
        "dataset": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "files": file_records,
        "license_url": LICENSE_URL,
        "paper_url": PAPER_URL,
        "preprocessing": {
            "input": "uint8 grayscale 28x28",
            "operation": "float64(pixel) / 255.0",
            "output_range": [0.0, 1.0],
            "resize": None,
        },
        "schema_version": 1,
        "seed": seed,
        "source_url": SOURCE_URL,
        "splits": {
            "test": {
                "class_counts": _counts(test_labels, test_indices),
                "indices": test_indices,
                "source": "test",
            },
            "train": {
                "class_counts": _counts(train_labels, train_indices),
                "indices": train_indices,
                "source": "train",
            },
            "validation": {
                "class_counts": _counts(train_labels, validation_indices),
                "indices": validation_indices,
                "source": "train",
            },
        },
    }


def _validated_manifest(manifest: dict[str, object]) -> None:
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") != DATASET_NAME
            or manifest.get("classes") != list(CLASS_NAMES)):
        raise ValueError("unsupported Fashion-MNIST manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("manifest must contain train, validation, and test splits")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ValueError("manifest file table is invalid")
    for key, (filename, _) in FILES.items():
        record = files[key]
        if (not isinstance(record, dict) or record.get("name") != filename
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64):
            raise ValueError(f"invalid manifest file record: {key}")
    seen = {"train": set(), "test": set()}
    for name, expected_source in (("train", "train"), ("validation", "train"), ("test", "test")):
        entry = splits[name]
        if not isinstance(entry, dict) or entry.get("source") != expected_source:
            raise ValueError(f"invalid source for split: {name}")
        indices = entry.get("indices")
        if (not isinstance(indices, list) or not indices
                or any(type(value) is not int or value < 0 or value >= MAX_SAMPLES
                       for value in indices)
                or indices != sorted(indices) or len(indices) != len(set(indices))):
            raise ValueError(f"invalid indices for split: {name}")
        overlap = seen[expected_source].intersection(indices)
        if overlap:
            raise ValueError("manifest splits overlap")
        seen[expected_source].update(indices)


def load_split(manifest_path: str | Path, data_directory: str | Path,
               split: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-verify source identity and materialize one normalized manifest split."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid manifest file") from error
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    _validated_manifest(manifest)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    source = manifest["splits"][split]["source"]
    paths = _paths(Path(data_directory))
    image_key, label_key = f"{source}_images", f"{source}_labels"
    for key in (image_key, label_key):
        compressed = _file_bytes(paths[key])
        expected = manifest.get("files", {}).get(key, {}).get("sha256")
        if not isinstance(expected, str) or hashlib.sha256(compressed).hexdigest() != expected:
            raise ValueError(f"source SHA-256 mismatch: {paths[key].name}")
    images, labels = load_idx_pair(paths[image_key], paths[label_key])
    indices = np.asarray(manifest["splits"][split]["indices"], dtype=np.int64)
    if int(indices[-1]) >= len(labels):
        raise ValueError("manifest index exceeds source sample count")
    selected_labels = labels[indices]
    recorded_counts = manifest["splits"][split].get("class_counts")
    actual_counts = np.bincount(selected_labels, minlength=len(CLASS_NAMES)).astype(int).tolist()
    if recorded_counts != actual_counts:
        raise ValueError("manifest class counts do not match selected indices")
    return images[indices].astype(np.float64) / 255.0, selected_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fashion-mnist-manifest.json"))
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--allow-nonofficial-files", action="store_true",
                        help="intended only for synthetic fixtures and mirrors")
    arguments = parser.parse_args(argv)
    try:
        report = build_manifest(
            arguments.data_directory, arguments.train_per_class,
            arguments.validation_per_class, arguments.test_per_class,
            arguments.seed, not arguments.allow_nonofficial_files,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    sizes = {name: len(value["indices"]) for name, value in report["splits"].items()}
    print(f"output={arguments.output} train={sizes['train']} "
          f"validation={sizes['validation']} test={sizes['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
