"""Data persistence mappings and low-level subsystem interception controllers."""

import json
import os
import re
import sys
import threading
import csv

from pathlib import Path

import numpy as np


class NpEncoder(json.JSONEncoder):
    """Bridge encoder adapting intrinsic NumPy configurations to standard JSON parameters."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class _PolypusStdoutInterceptor:
    """Intercept OS-level fd 1 to capture output from C/Rust extensions.

    Optionally checkpoints the per-generation fitness/mean_best history to a CSV
    every `checkpoint_every` generations (from the drain thread), so a job that
    dies mid-run still leaves a partial trajectory on disk.
    """

    _RE = re.compile(r"Generation\s+(\d+):.*BestFitness:\s+([-\d.eE+]+),\s*MeanBest:\s+([-\d.eE+]+)")

    def __init__(self, csv_path=None, run_idx=0, checkpoint_every=None):
        self.fitness_history: list[float] = []
        self.mean_best_history: list[float] = []
        self._csv_path = str(csv_path) if csv_path is not None else None
        self._run_idx = run_idx
        self._checkpoint_every = checkpoint_every if (checkpoint_every and checkpoint_every > 0) else None
        self._flushed_count = 0
        self._original_fd = sys.stdout.fileno()
        self._saved_fd: int | None = None
        self._pipe_r: int | None = None
        self._pipe_w: int | None = None
        self._thread: threading.Thread | None = None

    def _flush_checkpoint(self) -> int:
        """Append newly-parsed generations to the history CSV.

        Best-effort: writes only the rows not yet flushed (Run, Epoch, fitness/
        mean_best). The final_loss row is added later by the caller. Safe to call
        from the drain thread (writes to a real file, not fd 1).
        """
        if self._csv_path is None:
            return 0
        start = self._flushed_count
        end = min(len(self.fitness_history), len(self.mean_best_history))
        if end <= start:
            return 0
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Run", "Epoch", "Metric", "Value"])
            for epoch in range(start, end):
                writer.writerow([self._run_idx + 1, epoch + 1, "fitness", self.fitness_history[epoch]])
                writer.writerow([self._run_idx + 1, epoch + 1, "mean_best", self.mean_best_history[epoch]])
        self._flushed_count = end
        return end - start

    def __enter__(self):
        self._pipe_r, self._pipe_w = os.pipe()
        self._saved_fd = os.dup(self._original_fd)
        sys.stdout.flush()
        os.dup2(self._pipe_w, self._original_fd)
        os.close(self._pipe_w)
        self._pipe_w = None

        saved_fd = self._saved_fd

        def _drain():
            with os.fdopen(self._pipe_r, "r", errors="replace") as reader, \
                 os.fdopen(os.dup(saved_fd), "w", errors="replace") as terminal:
                for line in reader:
                    terminal.write(line)
                    terminal.flush()
                    m = self._RE.search(line)
                    if m:
                        self.fitness_history.append(float(m.group(2)))
                        self.mean_best_history.append(float(m.group(3)))
                        if self._checkpoint_every and len(self.fitness_history) % self._checkpoint_every == 0:
                            try:
                                self._flush_checkpoint()
                            except Exception:
                                # Never let a checkpoint write kill the drain thread;
                                # the __exit__ flush will retry the remainder.
                                pass

        self._thread = threading.Thread(target=_drain, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        sys.stdout.flush()
        os.dup2(self._saved_fd, self._original_fd)
        os.close(self._saved_fd)
        self._saved_fd = None
        self._thread.join()
        # Final flush: persist any generations not yet checkpointed (and, when
        # checkpoint_every is None, this writes the whole history at once —
        # identical to the previous behaviour).
        try:
            self._flush_checkpoint()
        except Exception:
            pass

def load_param_checkpoint(outdir: Path, run_id: str) -> np.ndarray | None:
    """Recover the last checkpointed best-parameter vector for a crashed run."""
    path = Path(outdir) / f"{run_id}_params_checkpoint.csv"
    if not path.exists():
        return None
    return np.loadtxt(path, delimiter=",")