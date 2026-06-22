"""Experiment tracking, configuration persistence, and logging interfaces."""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

from .serialization import NpEncoder


def _slug(text: str) -> str:
    """Format string to filesystem-safe slug component."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def build_experiment_id(args: argparse.Namespace) -> str:
    """Construct a unique filesystem tracking tag from structural parameters."""
    feature_mode = "qubo" if "annealing" in args.stages else "allfeat"
    common = [f"test{args.test_size:g}"]
    
    if getattr(args, "max_samples", None) is not None:
        common.append(f"maxs{args.max_samples}")
    
    common.append(f"fs{feature_mode}")
    if "annealing" in args.stages or args.pca:
        common.append(f"k{args.k}")
    
    common.append(f"runs{args.num_runs}")
    
    # Extract parallelization metric scaling mapping bounds
    n_workers = getattr(args, "vqc_n_workers", getattr(args, "vqc_n_qpus", 1))
    
    model_parts = [
        args.optimizer,
        f"iter{args.opt_maxiter}",
        f"workers{n_workers}", 
    ]
    
    exp_id = _slug("_".join(common + model_parts))
    
    # Hash truncation for strict OS directory path bounds
    if len(exp_id) > 96:
        digest = hashlib.sha1(exp_id.encode()).hexdigest()[:10]
        exp_id = f"{exp_id[:85].rstrip('-_')}_{digest}"
        
    return exp_id


def make_experiment_dir(base: Path, args: argparse.Namespace) -> Path:
    """Generate and configure a safe output directory hierarchy."""
    base.mkdir(parents=True, exist_ok=True)
    max_prefix = 0
    for p in base.iterdir():
        if p.is_dir():
            m = re.match(r"(\d+)_", p.name)
            if m:
                max_prefix = max(max_prefix, int(m.group(1)))
                
    exp_name = f"{max_prefix + 1}_{build_experiment_id(args)}"
    exp_dir = base / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=4, cls=NpEncoder)
        
    return exp_dir


def setup_logging(qml_outdir=None) -> logging.Logger:
    """Initialize standard console and artifact file logger streams."""
    logger = logging.getLogger("qml")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if qml_outdir:
        file_handler = logging.FileHandler(qml_outdir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def append_history_records(records: list, run_idx: int, history: dict, metrics: list) -> tuple:
    """Append structural training histories via long-table schema layouts."""
    train_records, val_records = [], []
    for key, values in history.items():
        is_val = key.startswith("val_")
        base = key[4:] if is_val else key
        
        if base not in metrics:
            continue
            
        for epoch_idx, val in enumerate(values):
            record = {
                "Run": run_idx + 1,
                "Epoch": epoch_idx + 1,
                "Metric": base,
                "Value": float(val),
            }
            if is_val:
                val_records.append(record)
            else:
                train_records.append(record)
                
    records.extend(train_records)
    records.extend(val_records)
    
    return train_records, val_records