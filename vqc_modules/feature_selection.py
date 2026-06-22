"""QUBO mathematical feature selection utilities leveraging simulated annealing."""

import argparse
import logging

import dimod
import neal
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def _discretize_feature(series, B=20, logger=None):
    """Translate continuous feature vectors into strict discrete domain clusters."""
    try:
        discretized = pd.qcut(series, q=B, labels=False, duplicates="drop")
        if discretized.isna().any():
            discretized = discretized.fillna(-1)
        return discretized
    except ValueError:
        try:
            discretized = pd.cut(series, bins=B, labels=False)
            if logger:
                logger.warning("Feature '%s' threshold routing failed. Degrading algorithm to standard cut.", series.name)
            return discretized.fillna(-1)
        except Exception:
            if logger:
                logger.info("Feature '%s' could not be discretized. Assigning all to one bin.", series.name)
            return pd.Series(0, index=series.index)


def _calculate_mi_matrices(X, y, B=20, logger=None):
    """Derive foundational importance and spatial redundancy topologies via mutual information."""
    features = X.columns.tolist()
    n_features = len(features)
    X_disc = X.apply(lambda column: _discretize_feature(column, B=B, logger=logger))

    importance = pd.Series(index=features, dtype=float)
    for feature in features:
        importance[feature] = mutual_info_score(X_disc[feature], y)

    redundancy = pd.DataFrame(0.0, index=features, columns=features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = mutual_info_score(X_disc[features[i]], X_disc[features[j]])
            redundancy.loc[features[i], features[j]] = mi
            redundancy.loc[features[j], features[i]] = mi

    return importance, redundancy


def select_features_qfs(X_train_s, y_train, k, num_reads=1000, logger=None, B=20, epsilon=1e-8, seed=None):
    """Execute QUBO Feature Selection bounds mapping, extracting optimal subset.

    Executes binary search across alpha domain mappings according to the framework 
    established in Muecke et al. "Feature selection on quantum computers", 2023.

    Args:
        X_train_s (pd.DataFrame): Training set feature topology.
        y_train (np.ndarray): Target classifications array.
        k (int): Exact target subset parameter cap.
        num_reads (int): Annealing configuration sweeps.
        logger (logging.Logger): Operational framework logging target.
        B (int): Matrix feature discretization bin size constraint.
        epsilon (float): Structural convergence zero-shift tolerance.
        seed (int | None): Sampler strict replication baseline configuration.
    """
    logger = logger or logging.getLogger(__name__)
    importance, redundancy = _calculate_mi_matrices(X_train_s, y_train, B=B, logger=logger)
    features = X_train_s.columns.tolist()
    n_features = len(features)

    lower_alpha = 0.0
    upper_alpha = 1.0
    alpha = 0.5
    best_subset = []
    sampler = neal.SimulatedAnnealingSampler()

    sample_kwargs = {"num_reads": num_reads}
    if seed is not None:
        sample_kwargs["seed"] = int(seed)

    logger.info(
        "Starting QFS Binary Search for k=%d features out of %d (seed=%s)",
        k, n_features, "None" if seed is None else int(seed),
    )

    while True:
        qubo = {}
        raw_qubo = {}
        max_val = -np.inf

        for i in range(n_features):
            feature_i = features[i]
            raw_qubo[(feature_i, feature_i)] = -alpha * importance[feature_i]
            max_val = max(max_val, raw_qubo[(feature_i, feature_i)])

            for j in range(i + 1, n_features):
                feature_j = features[j]
                value = 2 * (1 - alpha) * redundancy.loc[feature_i, feature_j]
                raw_qubo[(feature_i, feature_j)] = value
                max_val = max(max_val, value)

        mu = max_val if max_val > 0 else 1.0

        for i in range(n_features):
            feature_i = features[i]
            qubo[(feature_i, feature_i)] = (
                mu if alpha * importance[feature_i] < epsilon else raw_qubo[(feature_i, feature_i)]
            )
            for j in range(i + 1, n_features):
                feature_j = features[j]
                qubo[(feature_i, feature_j)] = raw_qubo[(feature_i, feature_j)]

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        sampleset = sampler.sample(bqm, **sample_kwargs)
        selected = [feature for feature, value in sampleset.first.sample.items() if value == 1]
        k_prime = len(selected)

        logger.info("Alpha evaluation frame: %.5f -> Subset convergence mapping resolved: %d items.", alpha, k_prime)

        if k_prime == k:
            best_subset = selected
            break
        if k_prime > k:
            upper_alpha = alpha
        else:
            lower_alpha = alpha

        if (upper_alpha - lower_alpha) < 1e-6:
            logger.warning("Alpha parameter collapsed internally. Defaulting to operational nearest mapping subset (%d metrics).", k_prime)
            best_subset = selected
            break

        alpha = (lower_alpha + upper_alpha) / 2.0

    return best_subset