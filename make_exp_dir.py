#!/usr/bin/env python3
"""Login-node helper: Creates the numbered experiment directory and prints its path.

This script is intentionally kept separate from main_VQC.py. While main_VQC 
imports vqc_modules.training (which loads the `polypus` Rust extension requiring 
specific cluster modules), this helper only targets 'cli' and 'experiment'. 
This makes it lightweight and safe to execute directly on the login node.
"""

from pathlib import Path

from vqc_modules.cli import parse_args
from vqc_modules.experiment import make_experiment_dir


def main() -> None:
    """Generate the experimental path directory safely."""
    args = parse_args()
    base = Path(args.outdir)
    base.mkdir(parents=True, exist_ok=True)
    exp_dir = make_experiment_dir(base, args)
    print(exp_dir.resolve())


if __name__ == "__main__":
    main()