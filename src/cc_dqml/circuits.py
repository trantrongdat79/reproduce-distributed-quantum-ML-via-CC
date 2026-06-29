"""PennyLane CC-DQML circuit blocks based on Fig. 4b."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp


N_QUBITS = 8
READOUT_WIRES = [3, 7]
PARAMETER_NAMES = ("conv", "pool", "feedforward", "interpret")


def parameter_shapes(convolutional_sub_layers: int) -> dict[str, tuple[int, ...]]:
    """Return trainable parameter shapes for the Fig. 4b-style CC-DQML circuit."""

    return {
        "conv": (convolutional_sub_layers, 2, 4),
        "pool": (2, 3, 4),
        "feedforward": (4, 4),
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
    for offset in (0, 4):
        qpu_wires = tuple(range(offset, offset + 4))
        qpu_features = x[offset : offset + 4]

        for wire, feature in zip(qpu_wires, qpu_features, strict=True):
            qml.Hadamard(wires=wire)
            qml.RZ(feature, wires=wire)

        ring_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
        for left, right in ring_pairs:
            qml.IsingZZ(
                qpu_features[left] * qpu_features[right],
                wires=(qpu_wires[left], qpu_wires[right]),
            )


def _convolution_block(params: pnp.ndarray, wires: tuple[int, int], local_pair: tuple[int, int]) -> None:
    for wire in wires:
        qml.Hadamard(wires=wire)
    qml.CZ(wires=wires)
    qml.RX(params[local_pair[0]], wires=wires[0])
    qml.RX(params[local_pair[1]], wires=wires[1])


def _convolution_layer(params: pnp.ndarray) -> None:
    local_pairs = ((0, 1), (2, 3), (1, 2))
    for qpu_index, offset in enumerate((0, 4)):
        for local_pair in local_pairs:
            wires = (offset + local_pair[0], offset + local_pair[1])
            _convolution_block(params[qpu_index], wires, local_pair)


def _controlled_pool_block(
    params: pnp.ndarray,
    control_wire: int,
    target_wire: int,
) -> None:
    """Equivalent of measuring control and applying Z/X gates by outcome.

    params are ordered as (z_if_0, x_if_0, z_if_1, x_if_1), matching the four
    outcome-dependent parameters described for each Fig. 4b pooling block.
    """

    qml.PauliX(wires=control_wire)
    qml.CRZ(params[0], wires=(control_wire, target_wire))
    qml.CRX(params[1], wires=(control_wire, target_wire))
    qml.PauliX(wires=control_wire)

    qml.CRZ(params[2], wires=(control_wire, target_wire))
    qml.CRX(params[3], wires=(control_wire, target_wire))


def _pool_and_classically_communicate(
    pool_params: pnp.ndarray,
    feedforward_params: pnp.ndarray,
) -> None:
    # Local Fig. 4b pooling: 4 qubits -> 2 qubits -> 1 readout qubit per QPU.
    local_pool_blocks = (
        (0, 1, 0),
        (2, 3, 1),
        (1, 3, 2),
        (4, 5, 0),
        (6, 7, 1),
        (5, 7, 2),
    )
    for control, target, block_index in local_pool_blocks:
        qpu_index = 0 if control < 4 else 1
        _controlled_pool_block(pool_params[qpu_index, block_index], control, target)

    # CC-DQML cross-QPU feedforward. These are the statevector equivalents of
    # measured classical outcomes controlling Z/X gates on the other QPU.
    cross_blocks = (
        (0, 7),
        (1, 7),
        (4, 3),
        (5, 3),
    )
    for block_index, (control, target) in enumerate(cross_blocks):
        _controlled_pool_block(feedforward_params[block_index], control, target)


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
