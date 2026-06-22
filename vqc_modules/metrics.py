"""Evaluation structures and state-loss measurement functions."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def measurement_prob(counts: dict, total_shots: int, readout: str = "single_qubit") -> float:
    """Evaluate target P(label=1) probability bounds from an active state configuration.

    readout modes:
    - "parity": Evaluates probability subset P(odd number of 1s in string).
      Exhibits global maximally nonlinear behaviors.
    - "single_qubit": Evaluates local configuration subset P(qubit 0 = |1⟩).
      Exhibits linear smooth topological gradients.
    """
    if not counts or total_shots <= 0:
        return 0.0

    if readout == "parity":
        hits = sum(
            cnt for bitstring, cnt in counts.items()
            if bitstring.replace(" ", "").count("1") % 2 == 1
        )
    elif readout == "single_qubit":
        hits = sum(
            cnt for bitstring, cnt in counts.items()
            if bitstring.replace(" ", "")[-1] == "1"
        )
    else:
        raise ValueError(
            f"Operational mode unrecognized: {readout!r} "
            "(Requires 'parity' or 'single_qubit')."
        )

    return float(hits) / float(total_shots)


def compute_loss_from_counts(counts_list, y_batch, readout: str = "single_qubit") -> float:
    """Evaluate topological cross-entropy loss against standard binary evaluation targets."""
    total_loss = 0.0
    for counts, y_true in zip(counts_list, y_batch):
        total = sum(counts.values())
        prob_1 = measurement_prob(counts, total, readout=readout)
        p_correct = prob_1 if int(y_true) == 1 else 1.0 - prob_1
        total_loss -= np.log(p_correct + 1e-10)
    return total_loss / len(y_batch)


def calibrate_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the optimal decision threshold via Youden's J statistic.

    Youden's J is defined as: J(t) = Sensitivity(t) + Specificity(t) - 1 = TPR(t) - FPR(t)

    The threshold that maximises J gives the best trade-off between
    sensitivity and specificity without assuming equal class costs. It is
    robust to class imbalance.

    This function must be called exclusively on training data and the returned threshold applied to the held-out set, 
    so as not to leak test information into the calibration step.

    Args:
        y_true: Ground-truth binary labels (0/1) for the calibration set.
        y_prob: Predicted probabilities of the positive class, in [0, 1].

    Returns:
        Optimal threshold t* in [0, 1] that maximises J on the calibration set.
        Falls back to 0.5 if the ROC curve cannot be computed (e.g. single
        class present in y_true).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return float(thresholds[best_idx])
    except Exception:
        print("Failed to optimize decision threshold via Youden's J-index. Reverting to default fallback threshold (0.5)")
        return 0.5


def evaluate_counts(y_true, counts_list, readout: str = "single_qubit", threshold: float | None = None) -> dict:
    """Evaluate quantum circuit outputs against ground-truth labels.

    Decision threshold behaviour
    ----------------------------
    - If threshold is None (default): the threshold is calibrated on
      the provided y_true / counts_list via Youden's J statistic.
      Use this mode when calling on training or fold-train data to obtain
      the optimal threshold for subsequent application to held-out data.
    - If threshold is a float: that value is applied directly without
      any re-calibration. Use this mode when calling on validation or test
      data to avoid information leakage.

    Args:
        y_true: Ground-truth binary labels.
        counts_list: List of measurement count dicts, one per sample.
        readout: Readout mode passed to measurement_prob.
        threshold: Pre-calibrated decision threshold, or None to calibrate here.

    Returns:
        dict with keys: accuracy, macro_f1, roc_auc, report, threshold_used,
        threshold_source, y_true (list), y_pred (list), y_prob (list).
    """
    y_true = np.asarray(y_true)
    y_prob = np.array([
        measurement_prob(counts, sum(counts.values()), readout=readout)
        for counts in counts_list
    ])

    # ------------------------------------------------------------------
    # Threshold determination
    # ------------------------------------------------------------------
    if threshold is None:
        t_opt = calibrate_threshold_youden(y_true, y_prob)
        threshold_source = "youden_calibrated"
    else:
        t_opt = float(threshold)
        threshold_source = "pre_calibrated"

    y_pred = (y_prob >= t_opt).astype(int)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc,
        "report": report,
        "threshold_used": t_opt,
        "threshold_source": threshold_source,
        "y_true": np.asarray(y_true).astype(int).tolist(),
        "y_pred": np.asarray(y_pred).astype(int).tolist(),
        "y_prob": y_prob.tolist(),
    }