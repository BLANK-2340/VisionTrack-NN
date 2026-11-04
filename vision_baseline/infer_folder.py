"""Apply a validation-fixed abstention policy to a folder of PGM images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .abstention import AbstentionPolicy, policy_from_record, policy_record
from .checkpoint import loads_model
from .pgm import load_pgm
from .ranking import rank_probabilities

DEFAULT_LABELS = ("vertical", "horizontal", "diagonal")
MAX_POLICY_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_policy(path: str | Path) -> AbstentionPolicy:
    """Load a policy record or a calibration report containing one."""
    raw = Path(path).read_bytes()
    if not raw or len(raw) > MAX_POLICY_BYTES:
        raise ValueError("policy file must be non-empty and at most 64 KiB")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"non-finite JSON value: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("policy file is not valid UTF-8 JSON") from error
    if isinstance(document, dict) and "abstention_policy" in document:
        document = document["abstention_policy"]
    return policy_from_record(document)


def _labels(classes: int, labels: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(labels) if labels is not None else (
        DEFAULT_LABELS if classes == len(DEFAULT_LABELS)
        else tuple(f"class_{index}" for index in range(classes))
    )
    if (len(values) != classes or len(set(values)) != classes or
            any(not value or value in {".", ".."} or "/" in value or "\\" in value
                for value in values)):
        raise ValueError(f"labels must contain {classes} unique names")
    return values


def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("image folder must be a non-symlink directory")
    entries = list(folder.iterdir())
    if not entries:
        raise ValueError("image folder must not be empty")
    invalid = sorted(entry.name for entry in entries
                     if entry.is_symlink() or not entry.is_file() or entry.suffix != ".pgm")
    if invalid:
        raise ValueError(f"unexpected image-folder entries: {invalid}")
    return sorted(entries, key=lambda item: item.name)


def infer_folder(checkpoint: str | Path, folder: str | Path, policy: AbstentionPolicy,
                 labels: Sequence[str] | None = None,
                 temperature: float = 1.0) -> dict[str, object]:
    """Return deterministic accepted/abstained predictions for a flat folder."""
    policy = policy_from_record(policy_record(policy))
    checkpoint_bytes = Path(checkpoint).read_bytes()
    model = loads_model(checkpoint_bytes)
    class_labels = _labels(model.bias.size, labels)
    root = Path(folder)
    records = []
    accepted = 0
    for path in _image_files(root):
        pixels = load_pgm(path)
        probabilities = model.predict_proba(pixels[None, :, :], temperature=temperature)
        ranking = rank_probabilities(probabilities)
        predicted = int(ranking.indices[0, 0])
        margin = float(ranking.confidence_margin[0])
        decision = margin >= policy.min_margin
        accepted += int(decision)
        records.append({
            "accepted": decision,
            "confidence_margin": margin,
            "height": int(pixels.shape[0]),
            "path": path.relative_to(root).as_posix(),
            "predicted_index": predicted,
            "predicted_label": class_labels[predicted],
            "probability": float(ranking.probabilities[0, 0]),
            "width": int(pixels.shape[1]),
        })
    return {
        "accepted": accepted,
        "abstained": len(records) - accepted,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "coverage": accepted / len(records),
        "labels": list(class_labels),
        "policy": policy_record(policy),
        "predictions": records,
        "samples": len(records),
        "schema_version": 1,
        "temperature": float(temperature),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="VTNN checkpoint")
    parser.add_argument("folder", type=Path, help="flat folder of PGM images")
    parser.add_argument("policy", type=Path,
                        help="policy JSON or calibration report containing one")
    parser.add_argument("--output", type=Path, default=Path("folder-inference.json"))
    parser.add_argument("--labels", nargs="+", help="class labels in model order")
    parser.add_argument("--temperature", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    try:
        policy = load_policy(arguments.policy)
        report = infer_folder(arguments.checkpoint, arguments.folder, policy,
                              arguments.labels, arguments.temperature)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(f"output={arguments.output} samples={report['samples']} "
          f"accepted={report['accepted']} abstained={report['abstained']} "
          f"coverage={report['coverage']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
