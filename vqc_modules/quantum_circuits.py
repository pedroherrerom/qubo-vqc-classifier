"""Quantum circuit parameterization and structural construction interfaces."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    TwoLocal,
    efficient_su2,
    pauli_feature_map,
    real_amplitudes,
    z_feature_map,
    zz_feature_map,
)

# Standardized Qiskit functional API routing map. These methods return explicitly 
# decomposed primitive-gate circuits compatible directly with Polypus and Aer.
FEATURE_MAP_REGISTRY = {
    "ZZFeatureMap": zz_feature_map,
    "ZFeatureMap": z_feature_map,
    "PauliFeatureMap": pauli_feature_map,
}

# Ansatz circuit mapping. TwoLocal leverages the object-oriented API and requires 
# specific decomposition logic applied dynamically inside build_ansatz.
ANSATZ_REGISTRY = {
    "RealAmplitudes": real_amplitudes,
    "EfficientSU2": efficient_su2,
    "TwoLocal": TwoLocal,          
}

# Recognized base logic operators for gate decomposition validation.
PRIMITIVE_GATES = {
    "barrier", "measure", "reset", "id", "x", "y", "z", "h", "s", "sdg",
    "sx", "sxdg", "t", "tdg", "rx", "ry", "rz", "r", "u", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "swap", "ecr", "cp", "crx", "cry", "crz",
    "ccx", "cswap", "p",
}


def build_feature_map(
    fm_type: str,
    num_qubits: int,
    reps: int = 2,
    entanglement: str = "full",
    paulis=None,
    ):
    """Return a primitive-gate feature-map QuantumCircuit via the function API."""
    fn = FEATURE_MAP_REGISTRY[fm_type]
    kwargs = {"feature_dimension": num_qubits, "reps": reps, "entanglement": entanglement}
    if fn is pauli_feature_map and paulis:
        kwargs["paulis"] = paulis
    return fn(**kwargs)


def build_ansatz(
    ansatz_type: str,
    num_qubits: int,
    reps: int = 2,
    entanglement: str = "full",
    rotation_blocks="ry",
    entanglement_blocks="cx",
    ):
    """Construct a parameter-bound primitive-gate ansatz circuit."""
    if ansatz_type == "RealAmplitudes":
        return real_amplitudes(
            num_qubits=num_qubits, reps=reps, entanglement=entanglement,
        )
    if ansatz_type == "EfficientSU2":
        return efficient_su2(
            num_qubits=num_qubits, reps=reps, su2_gates=rotation_blocks, entanglement=entanglement,
        )
        
    # Execute explicit decomposition for object-oriented class instances
    return TwoLocal(
        num_qubits=num_qubits,
        reps=reps,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
    ).decompose(reps=10)


def build_vqc_circuit(
    num_qubits: int,
    fm_type: str,
    fm_reps: int,
    fm_entanglement: str,
    fm_paulis,
    ansatz_type: str,
    ansatz_reps: int,
    ansatz_entanglement: str,
    rotation_blocks,
    entanglement_blocks,
    ):
    """Compile the combined VQC circuit ready for evaluation bindings.

    Returns:
        tuple: (circuit, feature_params, ansatz_params)
    """
    feature_map = build_feature_map(
        fm_type, num_qubits, reps=fm_reps, entanglement=fm_entanglement, paulis=fm_paulis,
    )
    ansatz = build_ansatz(
        ansatz_type, num_qubits, reps=ansatz_reps, entanglement=ansatz_entanglement,
        rotation_blocks=rotation_blocks, entanglement_blocks=entanglement_blocks,
    )
    fm_params = list(feature_map.parameters)
    an_params = list(ansatz.parameters)
    
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    
    return circuit, fm_params, an_params


def build_vqc_components(
    num_qubits: int,
    fm_type: str,
    fm_reps: int,
    fm_entanglement: str,
    fm_paulis,
    ansatz_type: str,
    ansatz_reps: int,
    ansatz_entanglement: str,
    rotation_blocks,
    entanglement_blocks,
    ):
    """Return isolated pre-compiled feature-map and ansatz logic blocks.

    Returns:
        tuple: (feature_map, ansatz)
    """
    feature_map = build_feature_map(
        fm_type, num_qubits, reps=fm_reps, entanglement=fm_entanglement, paulis=fm_paulis,
    )
    ansatz = build_ansatz(
        ansatz_type, num_qubits, reps=ansatz_reps, entanglement=ansatz_entanglement,
        rotation_blocks=rotation_blocks, entanglement_blocks=entanglement_blocks,
    )
    return feature_map, ansatz


def remaining_non_primitive_gates(circuit) -> set:
    """Evaluate structural integrity against standard hardware gate primitives."""
    return {gate.operation.name for gate in circuit.data} - PRIMITIVE_GATES


def build_bound_circuits(circuit, feature_params, ansatz_params, x_batch, ansatz_param_values) -> list:
    """Generate executable bounds attaching active feature limits and final measurements."""
    circuits = []
    for x_sample in x_batch:
        bindings = {param: float(value) for param, value in zip(feature_params, x_sample)}
        bindings.update({param: float(value) for param, value in zip(ansatz_params, ansatz_param_values)})
        
        bound = circuit.assign_parameters(bindings)
        bound.measure_all()
        circuits.append(bound)
        
    return circuits