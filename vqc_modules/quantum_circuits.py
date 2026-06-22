"""quantum_circuits.py — Circuit construction for the VQC pipeline.

Builds Qiskit circuits for feature maps and ansätze using the Qiskit ≥ 2.1
function API (``zz_feature_map``, ``efficient_su2``, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger("vqc")

# -------------------------------------------------------------------------
# Feature maps
# -------------------------------------------------------------------------

def build_feature_map(n_qubits: int, config: dict[str, Any]) -> "QuantumCircuit":
    """
    Build and return a Qiskit feature-map circuit.

    Parameters
    ----------
    n_qubits : int
    config   : pipeline config (reads ``fm_type``, ``fm_reps``)

    Returns
    -------
    QuantumCircuit
    """
    from qiskit.circuit.library import (  # noqa: PLC0415
        ZZFeatureMap,
        ZFeatureMap,
        PauliFeatureMap,
    )

    fm_type = config.get("fm_type", "ZZFeatureMap")
    reps = config.get("fm_reps", 1)

    if fm_type == "ZZFeatureMap":
        fm = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    elif fm_type == "ZFeatureMap":
        fm = ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    elif fm_type == "PauliFeatureMap":
        fm = PauliFeatureMap(feature_dimension=n_qubits, reps=reps)
    else:
        raise ValueError(f"Unknown feature map type: {fm_type!r}")

    logger.debug("Feature map: %s (reps=%d, depth=%d)", fm_type, reps, fm.depth())
    return fm


def build_ansatz(n_qubits: int, config: dict[str, Any]) -> "QuantumCircuit":
    """
    Build and return a Qiskit ansatz circuit.

    Parameters
    ----------
    n_qubits : int
    config   : pipeline config (reads ``ansatz_type``, ``ansatz_reps``,
               ``ansatz_entanglement``, ``ansatz_rotation_blocks``)

    Returns
    -------
    QuantumCircuit
    """
    from qiskit.circuit.library import (  # noqa: PLC0415
        RealAmplitudes,
        EfficientSU2,
        TwoLocal,
    )

    ansatz_type = config.get("ansatz_type", "EfficientSU2")
    reps = config.get("ansatz_reps", 1)
    entanglement = config.get("ansatz_entanglement", "circular")
    rotation_blocks = config.get("ansatz_rotation_blocks", ["ry", "rz"])

    if ansatz_type == "RealAmplitudes":
        ans = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    elif ansatz_type == "EfficientSU2":
        ans = EfficientSU2(
            num_qubits=n_qubits,
            reps=reps,
            entanglement=entanglement,
            su2_gates=rotation_blocks,
        )
    elif ansatz_type == "TwoLocal":
        ans = TwoLocal(
            num_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks="cx",
            reps=reps,
            entanglement=entanglement,
        )
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type!r}")

    logger.debug(
        "Ansatz: %s (reps=%d, entanglement=%s, depth=%d)",
        ansatz_type,
        reps,
        entanglement,
        ans.depth(),
    )
    return ans


def build_vqc_circuit(n_qubits: int, config: dict[str, Any]) -> "QuantumCircuit":
    """
    Compose feature map + ansatz into a full VQC circuit.

    Returns the composed QuantumCircuit.
    """
    fm = build_feature_map(n_qubits, config)
    ans = build_ansatz(n_qubits, config)
    circuit = fm.compose(ans)
    logger.info(
        "VQC circuit: %d qubits, depth=%d, parameters=%d",
        n_qubits,
        circuit.depth(),
        circuit.num_parameters,
    )
    return circuit


def count_parameters(circuit: "QuantumCircuit") -> int:
    """Return the number of trainable parameters in *circuit*."""
    return circuit.num_parameters
