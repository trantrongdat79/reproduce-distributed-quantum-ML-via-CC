"""Configuration loading for local CC-DQML runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    model: str
    output_dir: Path
    dataset_seed: int
    init_seed: int


@dataclass(frozen=True)
class DataSettings:
    n_features: int
    n_samples: int
    n_clusters: int
    train_size: int
    validation_size: int
    sphere_radius: float


@dataclass(frozen=True)
class ModelSettings:
    qpus: int
    qubits_per_qpu: int
    convolutional_sub_layers: int
    communication: str
    interpret_weights_initial: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrainingSettings:
    optimizer: str
    learning_rate: float
    batch_size: int
    iterations: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSettings
    data: DataSettings
    model: ModelSettings
    training: TrainingSettings


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config into typed settings."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    experiment = _required(raw, "experiment")
    data = _required(raw, "data")
    model = _required(raw, "model")
    training = _required(raw, "training")

    return ExperimentConfig(
        experiment=ExperimentSettings(
            model=str(_required(experiment, "model")),
            output_dir=Path(_required(experiment, "output_dir")),
            dataset_seed=int(_required(experiment, "dataset_seed")),
            init_seed=int(_required(experiment, "init_seed")),
        ),
        data=DataSettings(
            n_features=int(_required(data, "n_features")),
            n_samples=int(_required(data, "n_samples")),
            n_clusters=int(_required(data, "n_clusters")),
            train_size=int(_required(data, "train_size")),
            validation_size=int(_required(data, "validation_size")),
            sphere_radius=float(_required(data, "sphere_radius")),
        ),
        model=ModelSettings(
            qpus=int(_required(model, "qpus")),
            qubits_per_qpu=int(_required(model, "qubits_per_qpu")),
            convolutional_sub_layers=int(_required(model, "convolutional_sub_layers")),
            communication=str(_required(model, "communication")),
            interpret_weights_initial=tuple(
                float(value) for value in _required(model, "interpret_weights_initial")
            ),
        ),
        training=TrainingSettings(
            optimizer=str(_required(training, "optimizer")),
            learning_rate=float(_required(training, "learning_rate")),
            batch_size=int(_required(training, "batch_size")),
            iterations=int(_required(training, "iterations")),
        ),
    )
