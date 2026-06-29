"""Evaluation helpers for CC-DQML experiments."""

from __future__ import annotations

import numpy as np


def accuracy_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    return float(np.mean(predictions == labels))


def mean_squared_error(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2))
