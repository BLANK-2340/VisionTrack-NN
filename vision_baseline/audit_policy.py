"""Audit a frozen abstention policy on a labeled class-directory dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .abstention import AbstentionPolicy, policy_from_record, policy_record
from .checkpoint import loads_model
from .evaluate_folder import _dataset_files, _labels_for_classes
from .infer_folder import load_policy
from .pgm import load_pgm
from .ranking import rank_probabilities


def audit_folder_policy(checkpoint: str | Path, dataset: str | Path,
                        policy: AbstentionPolicy,
                        labels: Sequence[str] | None = None,
                        temperature: float = 1.0) -> dict[str, object]:
    """Report aggregate and per-class coverage/risk without refitting policy."""
    policy = policy_from_record(policy_record(policy))
    checkpoint_bytes = Path(checkpoint).read_bytes()
    model = loads_model(checkpoint_bytes)
    class_labels = _labels_for_classes(model.bias.size, labels)
    root = Path(dataset)
    samples = _dataset_files(root, class_labels)
    rows = []
    class_counts = [dict(samples=0, accepted=0, errors=0) for _ in class_labels]
    total_accepted = 0
    total_errors = 0
    for target, path in samples:
        pixels = load_pgm(path)
        probabilities = model.predict_proba(pixels[None, :, :], temperature=temperature)
        ranking = rank_probabilities(probabilities)
        predicted = int(ranking.indices[0, 0])
        margin = float(ranking.confidence_margin[0])
        accepted = margin >= policy.min_margin
        error = accepted and predicted != target
        class_counts[target]["samples"] += 1
        class_counts[target]["accepted"] += int(accepted)
        class_counts[target]["errors"] += int(error)
        total_accepted += int(accepted)
        total_errors += int(error)
        rows.append({
            "accepted": accepted,
            "confidence_margin": margin,
            "path": path.relative_to(root).as_posix(),
            "predicted_index": predicted,
            "predicted_label": class_labels[predicted],
            "probability": float(ranking.probabilities[0, 0]),
            "target_index": target,
            "target_label": class_labels[target],
        })
    per_class = []
    for index, (label, counts) in enumerate(zip(class_labels, class_counts)):
        accepted = counts["accepted"]
        per_class.append({
            "accepted": accepted,
            "coverage": accepted / counts["samples"],
            "errors": counts["errors"],
            "index": index,
            "label": label,
            "risk": counts["errors"] / accepted if accepted else None,
            "samples": counts["samples"],
        })
    return {
        "aggregate": {
            "accepted": total_accepted,
            "coverage": total_accepted / len(samples),
            "errors": total_errors,
            "risk": total_errors / total_accepted if total_accepted else None,
            "samples": len(samples),
        },
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "dataset_layout": "class-directory-pgm",
        "labels": list(class_labels),
        "per_class": per_class,
        "policy": policy_record(policy),
        "predictions": rows,
        "schema_version": 1,
        "temperature": float(temperature),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="VTNN checkpoint")
    parser.add_argument("dataset", type=Path, help="root containing one directory per class")
    parser.add_argument("policy", type=Path,
                        help="policy JSON or calibration report containing one")
    parser.add_argument("--output", type=Path, default=Path("policy-audit.json"))
    parser.add_argument("--labels", nargs="+", help="class directory names in model order")
    parser.add_argument("--temperature", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    try:
        policy = load_policy(arguments.policy)
        report = audit_folder_policy(arguments.checkpoint, arguments.dataset, policy,
                                     arguments.labels, arguments.temperature)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    aggregate = report["aggregate"]
    risk = "null" if aggregate["risk"] is None else f"{aggregate['risk']:.6f}"
    print(f"output={arguments.output} samples={aggregate['samples']} "
          f"accepted={aggregate['accepted']} coverage={aggregate['coverage']:.6f} "
          f"risk={risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
