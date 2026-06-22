"""Post-processing visualization and analytical statistics generation module."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from .visualizations import (
    plot_confusion_matrix_academic,
    plot_final_loss_distribution,
    plot_model_comparison_bars,
    plot_pso_trajectory,
    plot_quantum_roc_curves,
)


# --- Support Structures ---

def _report(run: dict) -> dict:
    """Safely extract nested report structures."""
    return run.get("report_dict") or run.get("report") or {}


def _setup_logging(outdir: Path) -> logging.Logger:
    """Initialize discrete processing logic handler."""
    logger = logging.getLogger("metrics_processor")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    fmt = logging.Formatter("%(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(outdir / "post_processing.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger


# --- Core Processing Logic ---

def aggregate_classification_reports(
    json_path: Path,
    model_name: str,
    logger: logging.Logger) -> dict:
    """Read serialized arrays, compute variance mappings, and log formatted tables."""
    if not json_path.exists():
        logger.warning("File not found: %s. Skipping %s summary.", json_path.name, model_name)
        return {}

    with open(json_path) as f:
        runs = json.load(f)

    if not runs or not _report(runs[0]):
        logger.warning("Invalid format in %s.", json_path.name)
        return {}

    agg = {}
    for key in _report(runs[0]):
        if key == "accuracy":
            vals = [_report(r)[key] for r in runs]
            agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}
        else:
            agg[key] = {}
            for metric in ("precision", "recall", "f1-score", "support"):
                vals = [_report(r)[key][metric] for r in runs]
                agg[key][metric] = {"mean": np.mean(vals), "std": np.std(vals)}

    lines = [
        f"\n--- Aggregated Classification Analytics Matrix: {model_name} ({len(runs)} Cycles) ---",
        f"{'Classification':<15} {'Precision Target':<18} {'Recall Curve':<18} {'F1 Validation':<18} {'Metric Support'}",
        "-" * 80,
    ]
    for key, m in agg.items():
        if key == "accuracy":
            continue
        lines.append(
            f"{key:<15} "
            f"{m['precision']['mean']:.3f}±{m['precision']['std']:.3f}  "
            f"{m['recall']['mean']:.3f}±{m['recall']['std']:.3f}  "
            f"{m['f1-score']['mean']:.3f}±{m['f1-score']['std']:.3f}  "
            f"{m['support']['mean']:.1f}"
        )
    lines += ["-" * 80,
              f"{'accuracy threshold':<15} {'':18} {'':18} "
              f"{agg['accuracy']['mean']:.3f}±{agg['accuracy']['std']:.3f}"]
    
    logger.info("\n".join(lines))
    return agg


def discover_experiment_dirs(data_dir: Path) -> list[Path]:
    """Scan and isolate directories containing valid tracking frameworks."""
    discovered = []
    if (data_dir / "quantum_raw_metrics.json").exists():
        discovered.append(data_dir)
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "quantum_raw_metrics.json").exists():
            discovered.append(p)
    return discovered


def process_experiment_dir(exp_dir: Path, plot_metrics: list, logger: logging.Logger) -> dict:
    """Aggregate metrics and generate each plot as a separate image."""
    logger.info("\n--- Processing experiment: %s ---", exp_dir.name)
    plots_dir = exp_dir / "plots"
    stats = {}

    q = aggregate_classification_reports(exp_dir / "quantum_raw_metrics.json", "Quantum VQC Engine", logger)
    if q:
        stats["quantum_vqc"] = q

    train_csv = exp_dir / "quantum_train_historical.csv"
    plot_pso_trajectory(train_csv, plots_dir, logger)
    plot_final_loss_distribution(train_csv, plots_dir, logger)

    preds_csv = exp_dir / "quantum_aggregated_predictions.csv"
    if preds_csv.exists():
        plot_confusion_matrix_academic(preds_csv, plots_dir, logger)
        plot_quantum_roc_curves(preds_csv, plots_dir, logger)
    else:
        logger.warning("Classification tracking map %s absent; bypassing confusion matrix visualization.", preds_csv.name)

    comp_csv = exp_dir / "model_comparison.csv"
    if comp_csv.exists():
        runs_csv = exp_dir / "model_comparison_runs.csv"
        plot_model_comparison_bars(comp_csv, plots_dir, logger, runs=runs_csv if runs_csv.exists() else None)
    else:
        logger.info("No model_comparison.csv; skipping comparison bars (run regen_comparison.py to create it).")

    if stats:
        out = exp_dir / "summary_statistics.json"
        with open(out, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info("Structural evaluation complete. Saved Summary -> %s", out)
        
    return stats


def run_post_processing(args) -> None:
    """Execution interface handling terminal routing commands."""
    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"I/O Exception: Target workspace '{data_dir}' inaccessible.", file=sys.stderr)
        sys.exit(1)

    logger = _setup_logging(data_dir)
    logger.info("=" * 44)
    logger.info("Post-Processing Routine Activated: %s", data_dir.resolve())
    logger.info("=" * 44)

    experiments = discover_experiment_dirs(data_dir)
    if not experiments:
        logger.warning("No QML experiment directories found in %s.", data_dir.name)
        return

    combined = {}
    for exp_dir in experiments:
        s = process_experiment_dir(exp_dir, args.plot_metrics, logger)
        if s:
            combined[f"QML/{exp_dir.name}"] = s

    if combined and len(experiments) > 1:
        out = data_dir / "summary_statistics_combined.json"
        with open(out, "w") as f:
            json.dump(combined, f, indent=4)
        logger.info("Saved combined summary: %s", out)

    logger.info("=" * 44)
    logger.info("Post-processing visualization stack finalized.")