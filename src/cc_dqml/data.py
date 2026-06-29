"""Synthetic dataset generation from the paper appendix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cc_dqml.config import DataSettings


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    max_abs_pearson: float


def _sample_ball(rng: np.random.Generator, count: int, dim: int, radius: float) -> np.ndarray:
    directions = rng.normal(size=(count, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.random(count) ** (1.0 / dim)
    return directions * radii[:, None]


def generate_synthetic_dataset(settings: DataSettings, seed: int) -> DatasetSplit:
    """Generate clustered 8D binary data similar to Appendix B."""

    if settings.n_features != 8:
        raise ValueError("Sprint 1 expects n_features=8.")
    if settings.n_samples % settings.n_clusters != 0:
        raise ValueError("n_samples must be divisible by n_clusters.")
    if settings.train_size + settings.validation_size != settings.n_samples:
        raise ValueError("train_size + validation_size must equal n_samples.")

    rng = np.random.default_rng(seed)
    samples_per_cluster = settings.n_samples // settings.n_clusters

    centers = rng.choice(
        [-settings.sphere_radius, settings.sphere_radius],
        size=(settings.n_clusters, settings.n_features),
    )
    labels = np.array([-1.0, 1.0] * (settings.n_clusters // 2), dtype=float)
    if len(labels) != settings.n_clusters:
        raise ValueError("n_clusters must be even for balanced binary labels.")
    rng.shuffle(labels)

    xs = []
    ys = []
    for center, label in zip(centers, labels, strict=True):
        cluster = _sample_ball(
            rng,
            samples_per_cluster,
            settings.n_features,
            settings.sphere_radius,
        )
        xs.append(cluster + center)
        ys.append(np.full(samples_per_cluster, label, dtype=float))

    x = np.vstack(xs)
    y = np.concatenate(ys)
    order = rng.permutation(settings.n_samples)
    x = x[order]
    y = y[order]

    correlations = [
        np.corrcoef(x[:, feature], y)[0, 1] for feature in range(settings.n_features)
    ]
    max_abs_pearson = float(np.nanmax(np.abs(correlations)))

    return DatasetSplit(
        x_train=x[: settings.train_size],
        y_train=y[: settings.train_size],
        x_val=x[settings.train_size :],
        y_val=y[settings.train_size :],
        max_abs_pearson=max_abs_pearson,
    )
