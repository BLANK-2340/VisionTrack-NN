"""Deterministic top-k ranking for validated class probabilities."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .model import TinyConvNet


class Ranking(NamedTuple):
    """Ranked class indices, probabilities, and top-two confidence margins."""

    indices: np.ndarray
    probabilities: np.ndarray
    confidence_margin: np.ndarray


def rank_probabilities(probabilities: np.ndarray, k: int = 1) -> Ranking:
    """Rank each row by probability, breaking exact ties by class index.

    Inputs must be a finite two-dimensional probability matrix with at least
    two classes. The confidence margin is always top-1 minus top-2 and is
    therefore independent of ``k``.
    """
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch>=1, classes>=2]")
    if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite floating-point values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("each probability row must sum to one")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if k < 1 or k > values.shape[1]:
        raise ValueError("k must be between one and the class count")

    class_indices = np.arange(values.shape[1])
    order = np.empty((values.shape[0], values.shape[1]), dtype=np.int64)
    for row_index, row in enumerate(values):
        order[row_index] = np.lexsort((class_indices, -row))
    selected = order[:, :k]
    selected_probabilities = np.take_along_axis(values, selected, axis=1)
    margins = (np.take_along_axis(values, order[:, :1], axis=1)[:, 0] -
               np.take_along_axis(values, order[:, 1:2], axis=1)[:, 0])
    return Ranking(selected, selected_probabilities, margins)


def predict_top_k(model: TinyConvNet, images: np.ndarray, k: int = 1,
                  temperature: float = 1.0) -> Ranking:
    """Return deterministic top-k predictions from a ``TinyConvNet``."""
    return rank_probabilities(model.predict_proba(images, temperature=temperature), k=k)
