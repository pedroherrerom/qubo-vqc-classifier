"""Cross-framework metric analytical pipelines (Classical, QSVC, VQC baselines).

Outputs compiled metrics generating grouped variance bar chart frames evaluating parameter stability.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Standard operational metric sorting layout constraint.
METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    """Build standard macro-averaged analytical framework metrics mapping block."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )
    
    # Ensure inputs are standard integer arrays to prevent sklearn type warnings
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    out = {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC-AUC":   float("nan"),
    }
    
    # Safely compute ROC-AUC only if probability/decision scores are available
    if y_score is not None:
        try:
            out["ROC-AUC"] = float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
        except Exception:
            out["ROC-AUC"] = float("nan")
    return out


def _vqc_run_to_metrics(run: dict) -> dict:
    """Map one polypus VQC run dict onto the canonical metric set."""
    # Safely extract nested dictionaries to avoid KeyErrors
    report = run.get("report", {}) or {}
    macro = report.get("macro avg", {}) or {}
    roc = run.get("roc_auc", None)
    
    return {
        "Accuracy":  float(run.get("accuracy", float("nan"))),
        "Precision": float(macro.get("precision", float("nan"))),
        "Recall":    float(macro.get("recall", float("nan"))),
        "F1":        float(run.get("macro_f1", macro.get("f1-score", float("nan")))),
        "ROC-AUC":   float(roc) if roc is not None else float("nan"),
    }


def _aggregate(model_name: str, metric_dicts: list[dict]) -> list[dict]:
    """Calculate mean boundaries mapping standard statistical errors over metric matrices."""
    rows = []
    for metric in METRIC_ORDER:
        vals = np.array([d.get(metric, np.nan) for d in metric_dicts], dtype=float)
        
        # Filter out missing values before statistical calculation
        vals = vals[~np.isnan(vals)]
        
        mean = float(vals.mean()) if vals.size else float("nan")
        std = float(vals.std()) if vals.size else 0.0
        
        rows.append({"Model": model_name, "Metric": metric, "Mean": mean, "Std": std})
    return rows


def run_qsvc(X_train, y_train, X_test, y_test, *, k, fm_reps, fm_entanglement, num_shots, seed, parallelize, logger) -> dict | None:
    """Process Quantum Support Vector Classification (QSVC) mappings over statevectors."""
    try:
        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    except Exception as exc:
        logger.warning("QSVC execution framework unresolvable (%s).", exc)
        return None

    try: 
        from qiskit_machine_learning.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    except Exception:
        pass

    fm = zz_feature_map(feature_dimension=k, reps=fm_reps, entanglement=fm_entanglement)
    kernel = FidelityStatevectorKernel(
        feature_map=fm,
        shots=num_shots if (num_shots and num_shots > 0) else None,
    )

    X_train = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_test = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_train = np.asarray(y_train).astype(int)

    try:
        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(X_train, y_train)
        y_pred = qsvc.predict(X_test)
        
        try:
            y_score = qsvc.decision_function(X_test)
        except Exception:
            y_score = y_pred
            
        return binary_metrics(y_test, y_pred, y_score)
    except Exception as exc:
        logger.warning("QSVC mapping/fit routine aborted (%s). Bypassing run matrix step.", exc)
        return None


def _runs_rows(model_name: str, metric_dicts: list[dict]) -> list[dict]:
    """Parse linear matrices structure logs out from dimensional evaluation mappings."""
    rows = []
    for i, d in enumerate(metric_dicts):
        for metric in METRIC_ORDER:
            v = d.get(metric, np.nan)
            if v == v:  
                rows.append({"Model": model_name, "Run": i + 1, "Metric": metric, "Value": float(v)})
    return rows


def build_model_comparison(*, args, X_train_sel, X_test_sel, y_train, y_test, vqc_runs_metrics, classical_metrics, base_seed, logger):
    """Aggregate model metrics returning evaluated tables."""
    agg, runs = [], []

    for name, m in (classical_metrics or {}).items():
        agg += _aggregate(name, [m]); runs += _runs_rows(name, [m])

    if getattr(args, "run_qsvc", True):
        n_runs = int(getattr(args, "qsvc_num_runs", args.num_runs))
        k = X_train_sel.shape[1]
        logger.info("----------- Model Analysis Component: QSVC Evaluation (%d cycles) -----------", n_runs)
        qsvc_metrics = []
        for r in range(n_runs):
            m = run_qsvc(
                X_train_sel, y_train, X_test_sel, y_test, k=k,
                fm_reps=getattr(args, "fm_reps", 2),
                fm_entanglement=getattr(args, "fm_entanglement", "linear"),
                num_shots=getattr(args, "vqc_num_shots", 1024),
                seed=base_seed + r, parallelize=getattr(args, "parallelize", True), logger=logger,
            )
            if m is not None:
                logger.info("[QSVC Evaluation Step %d/%d] Acc=%.4f F1=%.4f ROC-AUC=%.4f",
                            r + 1, n_runs, m["Accuracy"], m["F1"], m["ROC-AUC"])
                qsvc_metrics.append(m)
        if qsvc_metrics:
            agg += _aggregate("QSVC", qsvc_metrics); runs += _runs_rows("QSVC", qsvc_metrics)

    vqc_metrics = [_vqc_run_to_metrics(r) for r in (vqc_runs_metrics or [])]
    if vqc_metrics:
        agg += _aggregate("VQC", vqc_metrics)
        runs += _runs_rows("VQC", vqc_metrics)

    return (pd.DataFrame(agg, columns=["Model", "Metric", "Mean", "Std"]), pd.DataFrame(runs, columns=["Model", "Run", "Metric", "Value"]))