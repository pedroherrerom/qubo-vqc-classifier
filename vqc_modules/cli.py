"""Command-line interface and configuration parser for the QML pipeline."""

import argparse
import json

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and integrate optional JSON configuration files."""
    parser = argparse.ArgumentParser(description="Unified entry point for the core QML/VQC execution pipeline.")

    # --- I/O Operations ---
    io = parser.add_argument_group("I/O Configuration")
    io.add_argument("--data", default="datasets/Student Depression Dataset.csv", help="Target path to the input CSV dataset.")
    io.add_argument("--train-path", default=None, help="Optional explicit path to an external training dataset split.")
    io.add_argument("--test-path", default=None, help="Optional explicit path to an external test dataset split.")
    io.add_argument("--target", default="DIAGNOSIS", help="Target classification column.")
    io.add_argument("--outdir", default="results", help="Directory for output generation and logging.")
    io.add_argument("--config", default=None, help="Path to a JSON configuration file overlay.")
    io.add_argument("--stages", nargs="+", choices=["annealing", "quantum"], default=["annealing", "quantum"], help="Pipeline execution stages.")
    io.add_argument("--pca", action="store_true", default=False, help="Enable PCA instead of QUBO feature selection (n_components defined by --k).")    
    io.add_argument("--id-cols", nargs="+", default=["SMILES"], help="Columns targeted for immediate dropping during preprocessing.")
    io.add_argument("--k", type=int, default=5, help="Target feature selection bound via QUBO-QFS or PCA.")

    # --- Experiment Control ---
    exp = parser.add_argument_group("Experiment Control")
    exp.add_argument("--seed", type=int, default=42, help="Global random seed ensuring reproducibility.")
    exp.add_argument("--num-runs", type=int, default=5, help="Independent statistical execution cycles.")
    exp.add_argument("--smoke-test", action="store_true", default=False, help="Injects a trivially separable dataset to validate optimizer infrastructure mechanics.")

    # --- Distributed & Job Array Architecture ---
    dist = parser.add_argument_group("Distributed Computing Context")
    dist.add_argument("--exp-dir", default=None, help="Triggers job-array task mode using a pre-allocated experiment directory.")
    dist.add_argument("--run-offset", type=int, default=0, help="Global index offset pointer for current task processing block.")
    dist.add_argument("--run-count", type=int, default=None, help="Processing load bound for the current task block.")
    dist.add_argument("--aggregate", action="store_true", default=False, help="Triggers node-level aggregation routines for distributed artifacts.")

    # --- Data Splitting Strategies ---
    split = parser.add_argument_group("Data Partitions")
    split.add_argument("--test-size", type=float, default=0.20, help="Test dataset reserve ratio.")
    split.add_argument("--max-samples", type=int, default=None, help="Global sub-sampling limit ceiling.") 
    
    # --- Simulated Annealing (QUBO-QFS) ---
    sa = parser.add_argument_group("Simulated Annealing Configuration")
    sa.add_argument("--sa-num-reads", type=int, default=300, help="Annealing sampler read sweep depth.")
    sa.add_argument("--sa-bins", type=int, default=20, help="Continuous data discretization bin count.")
    sa.add_argument("--sa-epsilon", type=float, default=1e-8, help="Zero-division mitigation constant.")
    sa.add_argument("--sa-alpha", type=float, default=1.0, help="Objective function relevance weight.")
    sa.add_argument("--sa-beta", type=float, default=0.6, help="Objective function redundancy penalty weight.")

    # --- VQC Optimization Engine ---
    qopt = parser.add_argument_group("Quantum Optimization Subsystem")
    qopt.add_argument("--optimizer", default="PSO", choices=["PSO", "COBYLA"], help="Global optimizer architecture selection.")    
    qopt.add_argument("--opt-maxiter", type=int, default=100, help="Optimizer convergence generation limits.")
    qopt.add_argument("--opt-tol", type=float, default=1e-4, help="Convergence delta tolerance.")
    qopt.add_argument("--opt-population-size", type=int, default=10, help="Swarm/Population bounds.")
    qopt.add_argument("--checkpoint-every", type=int, default=None, help="History training (fitness/mean_best) to csv every N generations, instead at the end of training. None or 0 to deactivate")

    # --- Quantum Backend Connectivity ---
    qbe = parser.add_argument_group("Quantum Processing Interfaces")
    qbe.add_argument("--vqc-num-shots", type=int, default=128, help="Circuit measurement sampling depth.")
    qbe.add_argument("--vqc-n-workers", type=int, default=1, help="Hardware concurrent thread/QPU allocation matrix.")
    qbe.add_argument("--vqc-train-infrastructure", default="local", choices=["local", "cunqa"], help="Polypus internal training backend router.")
    qbe.add_argument("--vqc-readout", default="single_qubit", choices=["single_qubit", "parity"], help="Observable topology mapping policy.")
    qbe.add_argument("--vqc-test-infrastructure", default="cunqa", choices=["cunqa", "local"], help="Test dataset loss evaluation backend router.")
    qbe.add_argument("--vqc-eval-backend", default=None, choices=["cunqa", "aer", None], help="Deprecated evaluation router overriding.")

    # Legacy configuration properties
    qbe.add_argument("--vqc-num-nodes", type=int, default=1, help="(Legacy) Node matrix distribution scale.")
    qbe.add_argument("--vqc-cores-per-qpu", type=int, default=2, help="(Legacy) CPU threading boundary per virtual QPU.")
    qbe.add_argument("--GPU", action="store_true", default=False, help="Hardware acceleration context flag for local execution.")
    qbe.add_argument("--parallelize", action="store_true", default=True, help="Implicit multiprocessing context enabler.")
    qbe.add_argument("--no-parallelize", dest="parallelize", action="store_false", help="Forces sequential CPU execution models.")

    # --- Quantum Circuit Architecture (Feature Map) ---
    qfm = parser.add_argument_group("Feature Encoding Map")
    qfm.add_argument("--fm-type", choices=["ZZFeatureMap", "ZFeatureMap", "PauliFeatureMap"], default="ZZFeatureMap", help="Encoding circuit blueprint selection.")
    qfm.add_argument("--fm-reps", type=int, default=2, help="Circuit layer repetition depth.")
    qfm.add_argument("--fm-entanglement", default="linear", help="Qubit correlation strategy map.")
    qfm.add_argument("--fm-paulis", nargs="+", default=["Z", "ZZ"], help="Pauli projection operators targeting feature spaces.")

    # --- Quantum Circuit Architecture (Ansatz) ---
    qan = parser.add_argument_group("Variational Ansatz")
    qan.add_argument("--ansatz-type", choices=["RealAmplitudes", "TwoLocal", "EfficientSU2"], default="RealAmplitudes", help="Parameterized circuit blueprint selection.")
    qan.add_argument("--ansatz-reps", type=int, default=3, help="Trainable layer depth.")
    qan.add_argument("--ansatz-entanglement", default="linear", help="Ansatz internal entanglement strategy.")
    qan.add_argument("--ansatz-rotation-blocks", nargs="+", default=["ry"], help="Parametrized rotation block mapping.")
    qan.add_argument("--ansatz-entanglement-blocks", nargs="+", default=["cx"], help="Parametrized entangling gate arrays.")

    args = parser.parse_args()

    # --- Configuration Overrides ---
    if args.config:
        with open(args.config, encoding="utf-8") as config_file:
            config = json.load(config_file)
        vars(args).update({key.replace("-", "_"): value for key, value in config.items()})

    if args.vqc_eval_backend is not None:
        args.vqc_test_infrastructure = "local" if args.vqc_eval_backend == "aer" else args.vqc_eval_backend

    return args