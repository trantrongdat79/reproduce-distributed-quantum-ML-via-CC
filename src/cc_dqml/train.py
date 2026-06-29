"""Training loop for the Sprint 1 CC-DQML prototype."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from cc_dqml.circuits import (
    initialize_parameters,
    interpret_probabilities,
    make_cc_dqml_qnode,
    parameters_to_dict,
)
from cc_dqml.config import ExperimentConfig
from cc_dqml.data import generate_synthetic_dataset
from cc_dqml.evaluate import accuracy_from_scores, mean_squared_error


def _batch_indices(
    rng: np.random.Generator,
    n_items: int,
    batch_size: int,
) -> np.ndarray:
    size = min(batch_size, n_items)
    return rng.choice(n_items, size=size, replace=False)


def _scores(
    circuit: qml.QNode,
    params: tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray],
    x_values: np.ndarray,
) -> pnp.ndarray:
    named = parameters_to_dict(params)
    values = []
    for x in x_values:
        probs = circuit(
            pnp.array(x, requires_grad=False),
            named["conv"],
            named["pool"],
            named["feedforward"],
        )
        values.append(interpret_probabilities(probs, named["interpret"]))
    return pnp.stack(values)


def _loss(
    params: tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray],
    circuit: qml.QNode,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
) -> pnp.ndarray:
    predictions = _scores(circuit, params, x_batch)
    labels = pnp.array(y_batch, requires_grad=False)
    return pnp.mean((predictions - labels) ** 2)


def _evaluate(
    circuit: qml.QNode,
    params: tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray],
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, float]:
    scores = np.asarray(_scores(circuit, params, x_values), dtype=float)
    return {
        "loss": mean_squared_error(scores, y_values),
        "accuracy": accuracy_from_scores(scores, y_values),
    }


def _write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: ExperimentConfig) -> dict[str, float | str]:
    """Run the Sprint 1 CC-DQML training experiment."""

    if config.experiment.model != "cc_dqml":
        raise ValueError("Sprint 1 only supports experiment.model=cc_dqml.")
    if config.model.communication != "classical":
        raise ValueError("Sprint 1 expects model.communication=classical.")

    output_dir = config.experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_dataset(config.data, config.experiment.dataset_seed)
    params = initialize_parameters(
        config.model.convolutional_sub_layers,
        config.model.interpret_weights_initial,
        config.experiment.init_seed,
    )
    circuit = make_cc_dqml_qnode()
    optimizer = qml.AdamOptimizer(stepsize=config.training.learning_rate)
    rng = np.random.default_rng(config.experiment.init_seed)

    metrics: list[dict[str, float]] = []
    for iteration in range(1, config.training.iterations + 1):
        indices = _batch_indices(
            rng,
            len(dataset.x_train),
            config.training.batch_size,
        )
        x_batch = dataset.x_train[indices]
        y_batch = dataset.y_train[indices]
        updated, train_loss = optimizer.step_and_cost(
            lambda conv, pool, feedforward, interpret: _loss(
                (conv, pool, feedforward, interpret),
                circuit,
                x_batch,
                y_batch,
            ),
            *params,
        )
        params = tuple(updated)

        if iteration == 1 or iteration == config.training.iterations or iteration % 5 == 0:
            val_metrics = _evaluate(circuit, params, dataset.x_val, dataset.y_val)
            metrics.append(
                {
                    "iteration": float(iteration),
                    "train_loss": float(train_loss),
                    "validation_loss": val_metrics["loss"],
                    "validation_accuracy": val_metrics["accuracy"],
                }
            )
            print(
                f"iteration={iteration} "
                f"train_loss={float(train_loss):.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

    final_train = _evaluate(circuit, params, dataset.x_train, dataset.y_train)
    final_val = _evaluate(circuit, params, dataset.x_val, dataset.y_val)
    summary: dict[str, float | str] = {
        "model": config.experiment.model,
        "communication": config.model.communication,
        "convolutional_sub_layers": float(config.model.convolutional_sub_layers),
        "dataset_seed": float(config.experiment.dataset_seed),
        "init_seed": float(config.experiment.init_seed),
        "max_abs_pearson": dataset.max_abs_pearson,
        "train_loss": final_train["loss"],
        "train_accuracy": final_train["accuracy"],
        "validation_loss": final_val["loss"],
        "validation_accuracy": final_val["accuracy"],
    }

    _write_metrics(output_dir / "metrics.csv", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)

    return summary
