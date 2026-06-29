"""PennyLane CC-DQML circuit prototype."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp


N_QUBITS = 8
READOUT_WIRES = [3, 7]
PARAMETER_NAMES = ("conv", "pool", "feedforward", "interpret")


def parameter_shapes(convolutional_sub_layers: int) -> dict[str, tuple[int, ...]]:
    """Return trainable parameter shapes for the Sprint 1 CC-DQML circuit."""

    return {
        "conv": (convolutional_sub_layers, 2, 3, 3),
        "pool": (2, 2),
        "feedforward": (2, 2),
        "interpret": (4,),
    }


def initialize_parameters(
    convolutional_sub_layers: int,
    interpret_initial: tuple[float, float, float, float],
    seed: int,
) -> tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray]:
    """Initialize trainable circuit and interpret-function parameters."""

    rng = pnp.random.default_rng(seed)
    shapes = parameter_shapes(convolutional_sub_layers)
    return (
        pnp.array(rng.uniform(0.0, 2.0 * pnp.pi, size=shapes["conv"]), requires_grad=True),
        pnp.array(rng.uniform(0.0, 2.0 * pnp.pi, size=shapes["pool"]), requires_grad=True),
        pnp.array(
            rng.uniform(0.0, 2.0 * pnp.pi, size=shapes["feedforward"]),
            requires_grad=True,
        ),
        pnp.array(interpret_initial, requires_grad=True),
    )


def parameters_to_dict(
    params: tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray],
) -> dict[str, pnp.ndarray]:
    """Return a named view of the trainable parameter tuple."""

    return dict(zip(PARAMETER_NAMES, params, strict=True))


def _embed_features(x: pnp.ndarray) -> None:
    for wire in range(N_QUBITS):
        qml.RY(x[wire], wires=wire)
        qml.RZ(0.5 * x[wire], wires=wire)


def _two_qubit_block(params: pnp.ndarray, wires: tuple[int, int]) -> None:
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(params[2], wires=wires[1])
    qml.CNOT(wires=(wires[1], wires[0]))


def _convolution_layer(params: pnp.ndarray) -> None:
    qpu_pairs = (
        ((0, 1), (2, 3), (1, 2)),
        ((4, 5), (6, 7), (5, 6)),
    )
    for qpu_index, pairs in enumerate(qpu_pairs):
        for pair_index, wires in enumerate(pairs):
            _two_qubit_block(params[qpu_index, pair_index], wires)


def _pool_and_classically_communicate(
    pool_params: pnp.ndarray,
    feedforward_params: pnp.ndarray,
) -> None:
    # Statevector simulation uses the circuit-equivalent controlled operations
    # for mid-circuit measurement plus classical feedforward.
    qml.CRY(pool_params[0, 0], wires=(0, 1))
    qml.CRY(pool_params[0, 1], wires=(1, 3))
    qml.CRY(pool_params[1, 0], wires=(4, 5))
    qml.CRY(pool_params[1, 1], wires=(5, 7))

    qml.CRY(feedforward_params[0, 0], wires=(1, 7))
    qml.CRZ(feedforward_params[0, 1], wires=(1, 7))
    qml.CRY(feedforward_params[1, 0], wires=(5, 3))
    qml.CRZ(feedforward_params[1, 1], wires=(5, 3))


def make_cc_dqml_qnode() -> qml.QNode:
    """Create the CC-DQML probability circuit."""

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd")
    def circuit(
        x: pnp.ndarray,
        conv: pnp.ndarray,
        pool: pnp.ndarray,
        feedforward: pnp.ndarray,
    ) -> pnp.ndarray:
        _embed_features(x)
        for layer_params in conv:
            _convolution_layer(layer_params)
        _pool_and_classically_communicate(pool, feedforward)
        return qml.probs(wires=READOUT_WIRES)

    return circuit


def interpret_probabilities(probabilities: pnp.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
    """Map two readout-qubit probabilities to a scalar prediction score."""

    return pnp.dot(probabilities, weights)
