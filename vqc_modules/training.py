"""Variational Quantum Classifier execution and optimization models.

Optimizer Configuration:
  PSO / DE ->  Managed via polypus.qml.train() with target execution bounds 
               derived from vqc_train_infrastructure parameters:
               - "local": Standard unmanaged Aer instances (Fast mapping, minimal allocation).
               - "cunqa": Self-managed dynamic vQPU architecture cluster (via qraise).

Evaluation Analytics (Final Loss vs Benchmark Set):
  Targets derived via vqc_test_infrastructure maps:
  - "cunqa": CUNQA network logic (defaults to local Aer execution if vQPU allocations fail).
  - "local": Execution managed directly via device threading logic bounds controlled by:
             --GPU (Triggers device mapping state mapping logic bounds).
             --parallelize (Regulates threading density logic mapped to internal processors).

Threshold Calibration:
  After training, the optimal decision threshold is calibrated via Youden's J
  statistic on the training set (or fold-train set during CV).  The calibrated
  threshold is then applied to the test / validation set without re-fitting, 
  preventing any information leakage from the held-out data.
"""

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polypus

from .backends import (
    get_cunqa_qpus,
    resolve_qpus,
    run_batch,
    run_batch_local,
)
from .metrics import (
    compute_loss_from_counts,
    evaluate_counts,
    measurement_prob,
)
from .quantum_circuits import (
    build_bound_circuits,
    build_vqc_circuit,
    build_vqc_components,
    remaining_non_primitive_gates,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Core Architectural Bindings
# ---------------------------------------------------------------------------

def _log_slurm_resources(logger: logging.Logger) -> None:
    """Document allocated standard cluster processor footprints."""
    keys = (
        "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE", "SLURM_NNODES", "SLURM_JOB_NODELIST",
    )
    info = " | ".join(f"{k.replace('SLURM_', '')}={os.environ.get(k, '-')}" for k in keys)
    logger.info("SLURM allocation | %s", info)
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info("Process peak RSS: %.1f MB", rss_mb)
    except Exception:
        pass


def _make_loss_evaluator( circuit, feature_params, ansatz_params, x_train, y_train, executor, readout: str = "single_qubit", ):
    """Closure encapsulation: yields a callable that evaluates cross-entropy loss."""
    def eval_loss(params) -> float:
        circuits = build_bound_circuits(circuit, feature_params, ansatz_params, x_train, params)
        return compute_loss_from_counts(executor(circuits), y_train)
    return eval_loss


def _make_expectation(y_train, readout: str = "single_qubit"):
    """Evaluate training bounds mappings inside main Polypus structures."""
    y_arr = np.asarray(y_train)
    n = len(y_arr)
    counter = [0]

    def expectation_fn(array_counts) -> list[float]:
        batch_size = len(array_counts)
        start_idx = counter[0] % n
        expectations = []
        for i, counts in enumerate(array_counts):
            y_true = int(y_arr[(start_idx + i) % n])
            total = sum(counts.values())
            prob_1 = measurement_prob(counts, total, readout=readout)
            p_correct = prob_1 if y_true == 1 else 1.0 - prob_1
            expectations.append(float(np.log(p_correct + 1e-10)))
        counter[0] += batch_size
        return expectations

    return expectation_fn


def _make_synthetic_dataset(num_qubits: int, num_samples: int = 100):
    """Trivially separable dataset instantiation for baseline mapping validity testing."""
    synth_x = np.zeros((num_samples, num_qubits))
    synth_y = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        if i % 2 == 0:
            synth_x[i, :] = np.pi
            synth_y[i] = 1
    return synth_x, synth_y


def _append_final_loss_csv(csv_path: Path, run_idx: int, num_generations: int, final_loss: float):
    """Append the final cross-entropy loss row.

    The per-generation fitness/mean_best rows are written incrementally by
    _PolypusStdoutInterceptor during training, so only the closing final_loss
    row is added here (Epoch = num_generations + 1 to stay consistent).
    """
    record = {"Run": run_idx + 1, "Epoch": num_generations + 1, "Metric": "final_loss", "Value": final_loss}
    pd.DataFrame([record]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


# ---------------------------------------------------------------------------
# Core Training Subsystems
# ---------------------------------------------------------------------------

def _run_cobyla_qml(
    circuit,
    feature_params,
    ansatz_params,
    x_train,
    y_train_arr,
    max_iter: int,
    tol: float,
    num_shots: int,
    n_params: int,
    readout: str,
    use_gpu: bool,
    parallelize: bool,
    history_csv_path,
    run_idx: int,
    checkpoint_every,
    seed: int,
    logger: logging.Logger,
    ) -> tuple[list, list[float], list[float]]:
    """Gradient-free COBYLA training as a drop-in alternative to polypus PSO.

    Training runs on local Aer in-process: like the PSO path, CUNQA QPUs are
    unreachable for the per-iteration objective evaluations. The objective is the
    SAME cross-entropy (compute_loss_from_counts) used elsewhere, so the trained
    model is directly comparable to PSO at evaluation time (identical
    evaluate_counts path on the held-out test set).
    """
    from scipy.optimize import minimize

    def _executor(circuits):
        return run_batch_local(
            circuits, num_shots, use_gpu=use_gpu, parallelize=parallelize, logger=logger,
        )

    eval_loss = _make_loss_evaluator(
        circuit, feature_params, ansatz_params, x_train, y_train_arr,
        _executor, readout=readout,
    )

    loss_history: list[float] = []
    flushed = [0]

    def _write_rows():
        """Append new 'fitness' rows to the history CSV (same schema as PSO)."""
        if history_csv_path is None:
            return
        import csv as _csv
        start, end = flushed[0], len(loss_history)
        if end <= start:
            return
        write_header = not Path(history_csv_path).exists()
        with open(history_csv_path, "a", newline="") as f:
            w = _csv.writer(f)
            if write_header:
                w.writerow(["Run", "Epoch", "Metric", "Value"])
            for ep in range(start, end):
                w.writerow([run_idx + 1, ep + 1, "fitness", loss_history[ep]])
        flushed[0] = end

    def _objective(params) -> float:
        loss = eval_loss(np.asarray(params, dtype=float))
        loss_history.append(float(loss))
        gen = len(loss_history)
        if gen == 1 or gen % 10 == 0:
            logger.info("COBYLA | eval %d/%d | loss=%.6f", gen, max_iter, loss)
        if checkpoint_every and gen % checkpoint_every == 0:
            try:
                _write_rows()
            except Exception:
                pass
        return loss

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=n_params)

    logger.info(
        "Solver: COBYLA | maxiter=%d | params=%d | readout=%s | local-Aer (in-process)",
        max_iter, n_params, readout,
    )
    _log_slurm_resources(logger)

    t_train = time.time()
    res = minimize(
        _objective,
        x0,
        method="COBYLA",
        tol=tol,
        options={"maxiter": max_iter, "rhobeg": np.pi / 2, "disp": False},
    )
    train_elapsed = time.time() - t_train

    # Final flush — guarantees the history CSV exists even when checkpoint_every is None
    # (mirrors the interceptor's __exit__ behaviour on the PSO path).
    try:
        _write_rows()
    except Exception:
        pass

    best_params = list(np.asarray(res.x, dtype=float))
    # Single-trajectory method: no swarm. Returned only for dict parity; not written to CSV,
    # so plot_pso_trajectory will simply draw the 'fitness' line and skip 'mean_best'.
    mean_best_history = list(loss_history)

    if loss_history:
        logger.info(
            "COBYLA done | %d evals | loss %.6f -> %.6f (Δ=%.6f) | %.2f s",
            len(loss_history), loss_history[0], loss_history[-1],
            loss_history[-1] - loss_history[0], train_elapsed,
        )
    else:
        logger.warning("COBYLA produced no objective evaluations.")

    return best_params, loss_history, mean_best_history

def _run_polypus_qml(
    feature_map,
    ansatz,
    x_train,
    y_train_arr,
    optimizer: str,
    max_iter: int,
    population_size: int,
    tol: float,
    num_shots: int,
    actual_workers: int,
    n_params: int,
    num_nodes: int,
    cores_per_worker: int,
    run_id: str,
    eval_loss,
    train_infrastructure: str,
    readout: str,
    history_csv_path,      
    run_idx: int,
    checkpoint_every,
    logger: logging.Logger) -> tuple[list, list[float], list[float]]:
    """Initiate and supervise the native Rust QML solver integration."""
    expectation_fn = _make_expectation(y_train_arr, readout=readout)

    if optimizer == "PSO":
        method_obj = polypus.PSO(
            generations=max_iter, population_size=population_size,
            bounds=(-np.pi, np.pi), tolerance=tol,
        )
    elif optimizer == "DE":
        method_obj = polypus.DE(
            generations=max_iter, population_size=population_size, tolerance=tol,
        )
    else:
        logger.warning(
            "Unrecognized optimizer '%s'. Falling back to PSO.", optimizer,
        )
        method_obj = polypus.PSO(
            generations=max_iter, population_size=population_size,
            bounds=(-np.pi, np.pi), tolerance=tol,
        )

    n_circuits_per_gen = population_size * len(x_train)
    logger.info(
        "Solver: %s | iters=%d | pop=%d | infra=%s | workers=%d | "
        "~%d circuits/gen (~%d total)",
        optimizer, max_iter, population_size, train_infrastructure, actual_workers,
        n_circuits_per_gen, n_circuits_per_gen * max_iter,
    )
    if train_infrastructure == "cunqa":
        logger.info("Polypus will raise/destroy its own vQPU family internally (qraise).")
    _log_slurm_resources(logger)

    if checkpoint_every and history_csv_path is not None:
        os.environ["POLYPUS_CHECKPOINT_EVERY"] = str(checkpoint_every)
        os.environ["POLYPUS_CHECKPOINT_DIR"] = str(Path(history_csv_path).parent)
    else:
        os.environ.pop("POLYPUS_CHECKPOINT_EVERY", None)
        os.environ.pop("POLYPUS_CHECKPOINT_DIR", None)

    t_train = time.time()
    from .serialization import _PolypusStdoutInterceptor
    try:
        with _PolypusStdoutInterceptor(csv_path=history_csv_path, run_idx=run_idx,checkpoint_every=checkpoint_every,) as interceptor:
            result_params = polypus.qml.train(
                feature_map,
                ansatz,
                x_train,
                method_obj,
                shots=num_shots,
                dimensions=n_params,
                expectation_function=expectation_fn,
                infrastructure=train_infrastructure,
                nodes=num_nodes,
                id=run_id,
                n_qpus=actual_workers,
                cores_per_qpu=cores_per_worker,
            )
    except Exception:
        logger.exception(
            "polypus.qml.train raised an exception (infra=%s, workers=%d)",
            train_infrastructure, actual_workers,
        )
        raise

    train_elapsed = time.time() - t_train
    best_params = list(result_params)
    loss_history = interceptor.fitness_history
    mean_best_history = interceptor.mean_best_history

    if not loss_history:
        logger.warning(
            "No fitness history captured. Check Rust output for 'BestFitness' / 'MeanBest'.",
        )
    else:
        logger.info(
            "Fitness | gen 1: %.6f -> gen %d: %.6f (Δ=%.6f) | %.2f s/gen avg",
            loss_history[0], len(loss_history), loss_history[-1],
            loss_history[-1] - loss_history[0],
            train_elapsed / max(len(loss_history), 1),
        )

    if eval_loss is not None:
        t_eval = time.time()
        final_loss = eval_loss(np.asarray(best_params))
        logger.info(
            "Optimization complete | final loss=%.6f (eval %.1f s) | train %.1f s (%.2f min)",
            final_loss, time.time() - t_eval, train_elapsed, train_elapsed / 60,
        )
    else:
        logger.info(
            "Optimization complete | loss deferred | train %.1f s (%.2f min)",
            train_elapsed, train_elapsed / 60,
        )
    _log_slurm_resources(logger)

    return best_params, loss_history, mean_best_history


# ---------------------------------------------------------------------------
# Execution Controller Interface
# ---------------------------------------------------------------------------

def run_vqc_polypus(
    X_train,
    X_test,
    y_train,
    y_test,
    num_qubits: int,
    fm_type: str = "ZZFeatureMap",
    fm_reps: int = 2,
    fm_entanglement: str = "full",
    fm_paulis=None,
    ansatz_type: str = "RealAmplitudes",
    ansatz_reps: int = 2,
    ansatz_entanglement: str = "linear",
    rotation_blocks="ry",
    entanglement_blocks="cx",
    optimizer: str = "PSO",
    max_iter: int = 100,
    num_shots: int = 512,
    n_workers: int = 10,
    train_infrastructure: str = "local",
    vqc_test_infrastructure: str = "cunqa",
    use_gpu: bool = False,
    parallelize: bool = True,
    readout: str = "single_qubit",
    num_nodes: int = 1,
    cores_per_worker: int = 2,
    tol: float = 1e-4,
    population_size: int = 48,
    seed: int = 42,
    run_idx: int = 0,
    outdir=".",
    smoke_test: bool = False,
    checkpoint_every: int | None = None,
    logger: logging.Logger = None) -> dict:

    if logger is None:
        logger = logging.getLogger(__name__)
    np.random.seed(seed)

    if smoke_test:
        logger.warning(
            "SMOKE TEST: replacing dataset with trivially separable synthetic data.",
        )
        X_train, y_train = _make_synthetic_dataset(num_qubits)
        X_test, y_test = X_train.copy(), y_train.copy()

    feature_map, ansatz = build_vqc_components(
        num_qubits, fm_type, fm_reps, fm_entanglement, fm_paulis,
        ansatz_type, ansatz_reps, ansatz_entanglement, rotation_blocks, entanglement_blocks,
    )
    circuit, feature_params, ansatz_params = build_vqc_circuit(
        num_qubits, fm_type, fm_reps, fm_entanglement, fm_paulis,
        ansatz_type, ansatz_reps, ansatz_entanglement, rotation_blocks, entanglement_blocks,
    )

    high_level_gates = remaining_non_primitive_gates(circuit)
    if high_level_gates:
        logger.warning("Non-primitive gates after decomposition: %s", high_level_gates)
    logger.info(
        "Gates: %s", {g.operation.name for g in circuit.data},
    )

    n_params = len(ansatz_params)
    logger.info(
        "Circuit | qubits=%d | fm_params=%d | ansatz_params=%d | depth=%d",
        num_qubits, len(feature_params), n_params, circuit.depth(),
    )

    actual_workers = resolve_qpus(n_workers, logger)
    x_train = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    history_csv_path = Path(outdir) / "quantum_train_historical.csv" if outdir is not None else None
    t0 = time.time()
    if optimizer == "COBYLA":
        result_params, loss_history, mean_best_history = _run_cobyla_qml(
            circuit=circuit,
            feature_params=feature_params,
            ansatz_params=ansatz_params,
            x_train=x_train,
            y_train_arr=y_train_arr,
            max_iter=max_iter,
            tol=tol,
            num_shots=num_shots,
            n_params=n_params,
            readout=readout,
            use_gpu=use_gpu,
            parallelize=parallelize,
            history_csv_path=history_csv_path,
            run_idx=run_idx,
            checkpoint_every=checkpoint_every,
            seed=seed,
            logger=logger,
        )
    else:
        result_params, loss_history, mean_best_history = _run_polypus_qml(
            feature_map=feature_map,
            ansatz=ansatz,
            x_train=x_train,
            y_train_arr=y_train_arr,
            optimizer=optimizer,
            max_iter=max_iter,
            population_size=population_size,
            tol=tol,
            num_shots=num_shots,
            actual_workers=actual_workers,
            n_params=n_params,
            num_nodes=num_nodes,
            cores_per_worker=cores_per_worker,
            run_id=f"{optimizer.lower()}_run{run_idx}",
            eval_loss=None,
            train_infrastructure=train_infrastructure,
            readout=readout,
            history_csv_path=history_csv_path,
            run_idx=run_idx,
            checkpoint_every=checkpoint_every,
            logger=logger,
        )
    result_params = np.asarray(result_params)
    # ------------------------------------------------------------------
    # Backend setup for evaluation
    # ------------------------------------------------------------------
    qpus = get_cunqa_qpus(actual_workers, logger) if vqc_test_infrastructure == "cunqa" else None
    use_cunqa = qpus is not None

    if use_cunqa:
        logger.info(
            "Eval backend: CUNQA (%d QPUs)", len(qpus),
        )
    else:
        mode = "GPU" if use_gpu else ("CPU-parallel" if parallelize else "CPU-sequential")
        logger.info("Eval backend: local Aer (%s)", mode)

    def execute_circuits(circuits):
        if use_cunqa:
            return run_batch(circuits, qpus, num_shots, logger=logger)
        return run_batch_local(
            circuits, num_shots, use_gpu=use_gpu, parallelize=parallelize, logger=logger,
        )

    eval_loss = _make_loss_evaluator(
        circuit, feature_params, ansatz_params, x_train, y_train_arr,
        execute_circuits, readout=readout,
    )

    final_loss = eval_loss(result_params)
    elapsed = time.time() - t0
    logger.info(
        "%s complete | %d gens | loss=%.6f | %.1f s (%.3f min)",
        optimizer, max_iter, final_loss, elapsed, elapsed / 60,
    )

    # ── Write final-loss row (history already checkpointed by interceptor) ──────
    if history_csv_path is not None:
        _append_final_loss_csv(history_csv_path, run_idx, len(loss_history), final_loss)
        logger.info("Recorded final loss in %s (%d generations checkpointed)", history_csv_path.name, len(loss_history))

    # ------------------------------------------------------------------
    # Threshold calibration via Youden's J — fitted on TRAINING SET ONLY
    # ------------------------------------------------------------------
    logger.info(
        "Calibrating decision threshold via Youden's J on training set (%d samples)...",
        len(y_train_arr),
    )
    t_cal = time.time()
    train_circuits = build_bound_circuits(
        circuit, feature_params, ansatz_params, x_train, result_params,
    )
    train_counts = execute_circuits(train_circuits)
    train_metrics = evaluate_counts(
        y_train_arr, train_counts, readout=readout, threshold=None,
    )
    t_opt = train_metrics["threshold_used"]
    logger.info(
        "Youden's J calibration complete | t*=%.4f | train Acc=%.4f F1=%.4f ROC-AUC=%.4f"
        " (%.1f s)",
        t_opt,
        train_metrics["accuracy"],
        train_metrics["macro_f1"],
        train_metrics["roc_auc"],
        time.time() - t_cal,
    )

    # ------------------------------------------------------------------
    # Test evaluation — threshold t* applied, never re-fitted
    # ------------------------------------------------------------------
    logger.info(
        "Evaluating on test set (%d samples, backend=%s, threshold=%.4f)...",
        len(np.asarray(y_test)), "cunqa" if use_cunqa else "aer", t_opt,
    )
    t_test = time.time()
    x_test = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    test_circuits = build_bound_circuits(
        circuit, feature_params, ansatz_params, x_test, result_params,
    )
    metrics = evaluate_counts(
        y_test_arr,
        execute_circuits(test_circuits),
        readout=readout,
        threshold=t_opt,         # pre-calibrated - no leakage
    )
    logger.info("Test evaluation took %.1f s", time.time() - t_test)
    logger.info(
        "[VQC] Acc=%.4f  F1=%.4f  ROC-AUC=%.4f  threshold=%.4f  Gens=%d",
        metrics["accuracy"], metrics["macro_f1"], metrics["roc_auc"],
        metrics["threshold_used"], max_iter,
    )

    # ------------------------------------------------------------------
    # Persist test predictions
    # ------------------------------------------------------------------
    if outdir is not None:
        preds_csv = Path(outdir) / "quantum_aggregated_predictions.csv"
        pd.DataFrame({
            "Run":             run_idx + 1,
            "True_Label":      metrics["y_true"],
            "Predicted_Label": metrics["y_pred"],
            "Prob_1":          metrics["y_prob"],
        }).to_csv(preds_csv, mode="a", header=not preds_csv.exists(), index=False)
        logger.info(
            "Appended %d test predictions -> %s", len(metrics["y_true"]), preds_csv.name,
        )

    roc_auc = metrics["roc_auc"]
    return {
        "label":              "VQC",
        "n_features":         num_qubits,
        "n_test":             int(len(y_test_arr)),
        "accuracy":           round(float(metrics["accuracy"]), 4),
        "macro_f1":           round(float(metrics["macro_f1"]), 4),
        "roc_auc":            round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
        "threshold_youden":   round(float(t_opt), 4),
        "epochs_ran":         max_iter,
        "train_infrastructure": train_infrastructure,
        "eval_backend":       "cunqa" if use_cunqa else "aer",
        "loss_history":       loss_history,
        "mean_best_history":  mean_best_history,
        "report":             metrics["report"],
        "y_true":             metrics["y_true"],
        "y_prob":             metrics["y_prob"],
    }


def run_vqc(args, X_train, X_test, y_train, y_test, num_qubits: int, outdir, logger, run_idx: int = 0, ) -> dict:
    """Adapter: maps CLI/config namespace onto run_vqc_polypus kwargs."""
    n_workers = getattr(args, "vqc_n_workers", getattr(args, "vqc_n_qpus", 10))
    cores_per_worker = getattr(args, "vqc_cores_per_worker", getattr(args, "vqc_cores_per_qpu", 2))

    return run_vqc_polypus(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_qubits=num_qubits,
        fm_type=args.fm_type,
        fm_reps=args.fm_reps,
        fm_entanglement=args.fm_entanglement,
        fm_paulis=args.fm_paulis,
        ansatz_type=args.ansatz_type,
        ansatz_reps=args.ansatz_reps,
        ansatz_entanglement=args.ansatz_entanglement,
        rotation_blocks=args.ansatz_rotation_blocks,
        entanglement_blocks=args.ansatz_entanglement_blocks,
        optimizer=args.optimizer,
        max_iter=args.opt_maxiter,
        num_shots=args.vqc_num_shots,
        n_workers=n_workers,
        train_infrastructure=getattr(args, "vqc_train_infrastructure", "local"),
        vqc_test_infrastructure=getattr(args, "vqc_test_infrastructure", "cunqa"),
        use_gpu=getattr(args, "GPU", False),
        parallelize=getattr(args, "parallelize", True),
        readout=getattr(args, "vqc_readout", "single_qubit"),
        num_nodes=args.vqc_num_nodes,
        cores_per_worker=cores_per_worker,
        tol=args.opt_tol,
        population_size=args.opt_population_size,
        seed=args.seed,
        run_idx=run_idx,
        outdir=outdir,
        smoke_test=getattr(args, "smoke_test", False),
        checkpoint_every=getattr(args, "checkpoint_every", None),
        logger=logger,
    )