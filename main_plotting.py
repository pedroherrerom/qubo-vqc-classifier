#!/usr/bin/env python3
"""Independent CLI module for QML metrics post-processing and visualization."""

import argparse
import sys
from pathlib import Path

from vqc_modules.process_metrics import run_post_processing


def parse_args():
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generation of plots and summaries for QML (VQC) experiments."
    )
    
    # Results directory (Required)
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing the results (e.g., folder with quantum_train_historical.csv)"
    )
    
    # Metrics to plot (List with default values)
    parser.add_argument(
        "--plot_metrics",
        nargs="+",
        default=["loss"],
        help="List of metrics to plot in learning curves (e.g., --plot_metrics loss accuracy). Default: loss"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate that the directory exists before starting execution
    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Error: The specified directory does not exist: {target_dir.resolve()}", file=sys.stderr)
        sys.exit(1)
        
    print("Starting independent visualization module...")
    print(f"Target directory: {target_dir}")
    print(f"Metrics to plot: {args.plot_metrics}")
    print("-" * 50)
    
    # Execute the core post-processing routine
    run_post_processing(args)