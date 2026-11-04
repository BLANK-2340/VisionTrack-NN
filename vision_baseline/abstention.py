"""Validation-fixed abstention policies based on top-two probability margin."""

from __future__ import annotations

from typing import NamedTuple
import math

import numpy as np

from .ranking import rank_probabilities


class AbstentionPolicy(NamedTuple):
    """A frozen margin threshold and the validation evidence used to fit it."""

    min_margin: float
    requested_coverage: float
    validation_coverage: float
    validation_risk: float
    validation_retained: int
    validation_samples: int


def _validated_labels(labels: np.ndarray, samples: int, classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if (values.shape != (samples,) or values.dtype.kind not in "iu" or
            np.any(values < 0) or np.any(values >= classes)):
        raise ValueError("labels must contain one valid integer class per sample")
    return values.astype(np.int64, copy=False)


def apply_abstention(probabilities: np.ndarray, policy: AbstentionPolicy) -> np.ndarray:
    """Return the samples accepted by a previously fitted policy."""
    if not isinstance(policy, AbstentionPolicy):
        raise TypeError("policy must be an AbstentionPolicy")
    ranking = rank_probabilities(probabilities)
    return ranking.confidence_margin >= policy.min_margin


def abstention_metrics(probabilities: np.ndarray, labels: np.ndarray,
                       policy: AbstentionPolicy) -> dict[str, float | int | None]:
    """Measure coverage and selective risk without refitting the policy."""
    ranking = rank_probabilities(probabilities)
    targets = _validated_labels(labels, len(ranking.indices),
                                np.asarray(probabilities).shape[1])
    accepted = ranking.confidence_margin >= policy.min_margin
    retained = int(accepted.sum())
    errors = ranking.indices[:, 0] != targets
    return {
        "coverage": retained / len(accepted),
        "risk": float(errors[accepted].mean()) if retained else None,
        "retained": retained,
        "samples": len(accepted),
    }


def fit_abstention_policy(probabilities: np.ndarray, labels: np.ndarray,
                          target_coverage: float = 0.9) -> AbstentionPolicy:
    """Fit a deterministic margin threshold using validation data only.

    The threshold is the margin at the requested coverage rank. All exact ties
    at that boundary are retained, so achieved coverage can exceed the request.
    """
    if (isinstance(target_coverage, (bool, np.bool_)) or
            not isinstance(target_coverage, (int, float, np.integer, np.floating)) or
            not math.isfinite(float(target_coverage)) or
            not 0.0 < float(target_coverage) <= 1.0):
        raise ValueError("target_coverage must be finite and in (0, 1]")
    ranking = rank_probabilities(probabilities)
    targets = _validated_labels(labels, len(ranking.indices),
                                np.asarray(probabilities).shape[1])
    required = max(1, math.ceil(float(target_coverage) * len(targets)))
    order = np.argsort(-ranking.confidence_margin, kind="stable")
    threshold = float(ranking.confidence_margin[order[required - 1]])
    accepted = ranking.confidence_margin >= threshold
    retained = int(accepted.sum())
    risk = float(np.mean(ranking.indices[accepted, 0] != targets[accepted]))
    return AbstentionPolicy(
        min_margin=threshold,
        requested_coverage=float(target_coverage),
        validation_coverage=retained / len(targets),
        validation_risk=risk,
        validation_retained=retained,
        validation_samples=len(targets),
    )


def policy_record(policy: AbstentionPolicy) -> dict[str, float | int]:
    """Return a stable JSON-compatible policy record."""
    return dict(policy._asdict())


def policy_from_record(record: object) -> AbstentionPolicy:
    """Validate and restore a policy from its JSON-compatible record."""
    expected = tuple(AbstentionPolicy._fields)
    if not isinstance(record, dict) or tuple(sorted(record)) != tuple(sorted(expected)):
        raise ValueError("policy record fields do not match the supported schema")
    numeric = (int, float, np.integer, np.floating)
    for name in expected[:4]:
        value = record[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric):
            raise ValueError(f"policy field {name} must be numeric")
    for name in expected[4:]:
        value = record[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"policy field {name} must be an integer")
    policy = AbstentionPolicy(
        min_margin=float(record["min_margin"]),
        requested_coverage=float(record["requested_coverage"]),
        validation_coverage=float(record["validation_coverage"]),
        validation_risk=float(record["validation_risk"]),
        validation_retained=int(record["validation_retained"]),
        validation_samples=int(record["validation_samples"]),
    )
    if (not all(math.isfinite(value) for value in policy[:4]) or
            not 0.0 <= policy.min_margin <= 1.0 or
            not 0.0 < policy.requested_coverage <= 1.0 or
            not policy.requested_coverage <= policy.validation_coverage <= 1.0 or
            not 0.0 <= policy.validation_risk <= 1.0 or
            policy.validation_samples < 1 or
            not 1 <= policy.validation_retained <= policy.validation_samples or
            not math.isclose(policy.validation_coverage,
                             policy.validation_retained / policy.validation_samples,
                             rel_tol=0.0, abs_tol=1.0e-12)):
        raise ValueError("policy record contains inconsistent values")
    return policy
