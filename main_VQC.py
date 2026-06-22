#!/usr/bin/env python3
"""Unified entry point for the full QML/VQC pipeline.

This module acts as the single public face of the project:
- Exposes preprocessing and feature-selection helpers.
- Exposes VQC training entry points.
- Runs the full orchestration when executed as a script.
"""

from vqc_modules.backends import extract_counts, get_cunqa_qpus, resolve_qpus, run_batch
from vqc_modules.cli import parse_args
from vqc_modules.data_processing import load_and_preprocess
from vqc_modules.experiment import append_history_records, build_experiment_id, make_experiment_dir, setup_logging
from vqc_modules.feature_selection import select_features_qfs
from vqc_modules.metrics import compute_loss_from_counts, evaluate_counts
from vqc_modules.pipeline import run_pipeline
from vqc_modules.quantum_circuits import (
    build_ansatz,
    build_bound_circuits,
    build_feature_map,
    build_vqc_circuit,
    remaining_non_primitive_gates,
)
from vqc_modules.serialization import NpEncoder
from vqc_modules.training import run_vqc, run_vqc_polypus

__all__ = [
    "NpEncoder",
    "append_history_records",
    "build_ansatz",
    "build_bound_circuits",
    "build_experiment_id",
    "build_feature_map",
    "build_vqc_circuit",
    "compute_loss_from_counts",
    "evaluate_counts",
    "extract_counts",
    "get_cunqa_qpus",
    "load_and_preprocess",
    "main",
    "make_experiment_dir",
    "parse_args",
    "resolve_qpus",
    "run_batch",
    "run_pipeline",
    "run_vqc",
    "run_vqc_polypus",
    "select_features_qfs",
    "setup_logging",
]


def main() -> None:
    """Parse CLI arguments and run the complete pipeline."""
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()