# QML-VQC Pipeline

A hybrid quantum–classical machine learning pipeline for binary classification. It combines **QUBO-based feature selection** (via simulated annealing) with a **Variational Quantum Classifier (VQC)**, designed to run on real quantum processing units through the [CUNQA](https://cunqa.readthedocs.io) framework and [polypus](https://polypus.readthedocs.io), targeting the [CESGA FT3](https://www.cesga.es) HPC cluster.

> **Status:** Active research — TFM (Master's Thesis) project.

- Fix local to cunqa (import source code from polypus github) _POLYPUS_QML_INFRA in training

---

## Table of contents

- [Overview](#overview)
- [Pipeline stages](#pipeline-stages)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Local run](#local-run)
  - [HPC submission (CESGA FT3)](#hpc-submission-cesga-ft3)
- [Optimizers](#optimizers)
- [Outputs](#outputs)
- [Known issues](#known-issues)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

The pipeline addresses the practical constraints of near-term quantum devices (NISQ era): limited qubit counts and noisy gates. It tackles both constraints explicitly:

1. **Feature selection via QUBO** — reduces the number of input features to exactly *k*, so the VQC circuit fits on available qubits without depth blowup.
2. **Flexible optimizers** — PSO and DE via the polypus built-in QML branch, with a HYBRID strategy planned.
3. **Hardware-aware execution** — circuits are dispatched in batches across multiple QPUs via CUNQA, with automatic fallback to polypus serial mode when QPUs are unavailable.

```plain
CSV dataset
    │
    ▼
Data preprocessing  ──►  QUBO feature selection (k features)
                                  │
                                  ▼
                        VQC circuit (k qubits)
                                  │
                          ┌───────┴───────┐
                      CUNQA batch     polypus fallback
                          └───────┬───────┘
                                  │
                              Optimizer loop
                               (PSO / DE)
                                  │
                                  ▼
                        Evaluation & export
                    (accuracy, F1, ROC-AUC, CSV, JSON)
```

---

## Pipeline stages

### Stage 1 — Data preprocessing (`data_processing.py`)

- Drops irrelevant columns, zero-variance columns, and high-cardinality object columns (> 30 unique values).
- Label-encodes all remaining categorical variables.
- Optional stratified subsampling (`max_samples`) to limit dataset size for faster iteration.
- Stratified train/test split → median imputation → `StandardScaler` normalization.

### Stage 2 — QUBO feature selection (`feature_selection.py`)

Implements the method from [Muecke et al. (2023)](#references):

- Builds importance and pairwise redundancy matrices using **mutual information**.
- Formulates a QUBO that balances relevance vs. redundancy, parameterized by `α`.
- Solves with `neal.SimulatedAnnealingSampler`.
- Binary search over `α` until exactly `k` features are selected.

This stage can be skipped (`stages: ["quantum"]`), in which case all features are passed to the VQC.

### Stage 3 — VQC training (`training.py`, `quantum_circuits.py`)

The number of selected features becomes the qubit count. A fully parametrized circuit is built by composing:

- **Feature map** — encodes classical data into quantum states (`ZZFeatureMap`, `ZFeatureMap`, or `PauliFeatureMap`).
- **Ansatz** — trainable unitary (`RealAmplitudes`, `EfficientSU2`, or `TwoLocal`).

The circuit is fully decomposed to primitive gates before execution. The feature map and ansatz are passed separately to `polypus.qml.train`, which handles data encoding and parameter binding internally. Training uses the polypus QML branch (PSO or DE); evaluation runs on CUNQA QPUs directly.

**Classification rule:** the *odd-parity rule* — `P(label=1)` equals the fraction of shots with an odd number of 1s in the measurement bitstring. A global parity-flip correction is applied at evaluation time if the inverted threshold yields better accuracy.

---

## Project structure

```plain
.
├── main_VQC.py               # Entry point — exposes public API and runs pipeline
├── main_plotting.py          # Main plotting script
├── configVQC.json            # Default experiment configuration
├── submit.sh                 # HPC submission script (CESGA FT3)
├── qraise_job.sh             # SLURM job: provision CUNQA QPUs
├── vqc_job.sh                # SLURM job: run the pipeline (depends on qraise)
└── vqc_modules/
    ├── __init__.py
    ├── cli.py                # Argument parsing (CLI + JSON config overlay)
    ├── pipeline.py           # Top-level orchestration
    ├── data_processing.py    # Preprocessing utilities
    ├── feature_selection.py  # QUBO-QFS (simulated annealing)
    ├── quantum_circuits.py   # Circuit construction and binding
    ├── training.py           # Optimizer routing (PSO, DE via polypus.qml.train)
    ├── backends.py           # CUNQA / polypus execution backends
    ├── metrics.py            # Loss, parity probability, evaluation metrics
    ├── process_metrics.py    # Post-processing: aggregation and plot generation
    ├── visualizations.py     # Learning curves and confusion matrix plots
    ├── experiment.py         # Experiment IDs, directories, logging, history export
    └── serialization.py      # NumPy-safe JSON encoder
```

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| [Qiskit](https://qiskit.org/) | Circuit construction and decomposition |
| [dimod](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/) | QUBO / BQM formulation |
| [neal](https://docs.ocean.dwavesys.com/projects/neal/) | Simulated annealing sampler |
| [polypus](https://polypus.readthedocs.io) *(QML branch)* | VQC training and fallback executor |
| [cunqa](https://cunqa.readthedocs.io) | Native QPU batch execution (HPC only) |
| scikit-learn | Preprocessing and classical metrics |
| NumPy, pandas, matplotlib, seaborn | Data handling and visualization |

> `cunqa` is only required for HPC execution. Local runs fall back to `polypus` automatically.

## Configuration

All parameters are controlled through `configVQC.json`. CLI flags override JSON values.

```jsonc
{
  // Data
  "data": "datasets/Student Depression Dataset.csv",
  "target": "DIAGNOSIS",
  "outdir": "results",
  "stages": ["annealing", "quantum"],   // drop "annealing" to skip feature selection
  "irrelevant_cols": ["id", "City"],
  "max_samples": 500,                   // null = use full dataset

  // Feature selection
  "sa_k": 5,                            // number of features to select
  "sa_num_reads": 500,                  // SA iterations per QUBO solve

  // VQC optimizer
  "optimizer": "PSO",                   // PSO | DE
  "opt_maxiter": 25,

  // Circuit
  "fm_type": "ZZFeatureMap",
  "fm_reps": 2,
  "ansatz_type": "EfficientSU2",
  "ansatz_reps": 2,

  // Backend
  "vqc_num_shots": 1024,
  "vqc_n_qpus": 4,
  "vqc_infrastructure": "cunqa"         // "cunqa" | "aer"
}
```

See `cli.py` for the full list of parameters and their defaults.

---

## Usage

### Local run

```bash
python main_VQC.py --config configVQC.json
```

Override individual parameters on the command line:

```bash
python main_VQC.py \
  --config configVQC.json \
  --optimizer PSO \
  --opt-maxiter 50 \
  --sa-k 3 \
  --vqc-num-shots 512
```

Run only the quantum stage (skip feature selection):

```bash
python main_VQC.py --config configVQC.json --stages quantum
```

Post-process results and generate plots:

```bash
python main_VQC.py process --dir results/ --plot-metrics loss
```

### HPC submission (CESGA FT3)

`submit.sh` handles the two-job dependency chain — it first provisions QPUs via `qraise`, then launches the pipeline once they are ready.

```bash
bash submit.sh --config configVQC.json --vqc-time 01:00:00 --vqc-cpus 10 --vqc-mem 8G
```

Minimum CPU rule of thumb:

```bash
--vqc-cpus ≥ 2 + (vqc_n_qpus × cores_per_qpu)
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `configVQC.json` | Experiment config file |
| `--n-qpus` | read from config | Number of QPUs to provision |
| `--cores-per-qpu` | `2` | CPU cores per QPU process |
| `--vqc-time` | `08:00:00` | Wall-clock limit for the VQC job |
| `--vqc-cpus` | `4` | CPU cores for the VQC job |
| `--vqc-mem` | `8G` | Memory for the VQC job |
| `--qraise-margin` | `15` | Extra minutes added to QPU lifetime beyond VQC wall time |

Monitor jobs:

```bash
watch -n 10 squeue --me
tail -f logs/vqc_pipeline-<JID>.out
tail -f logs/qraise_vqc-<JID>.out
```

---

## Optimizers

PSO and DE are provided by `polypus.qml.train` (QML branch). The feature map and ansatz are passed separately; polypus handles data encoding and parameter binding internally.

| Name | Strategy | Best for |
|---|---|---|
| `PSO` | Particle Swarm Optimization | Noisy landscapes, global search |
| `DE` | Differential Evolution | Population-based global search |

---

## Outputs

Each experiment writes to `<outdir>/<experiment_id>/`:

```bash
<experiment_id>/
├── experiment_config.json        # Full config snapshot
├── experiment.log                # Timestamped run log
├── quantum_train_historical.csv  # Long-format per-generation loss history across all runs
├── quantum_raw_metrics.json      # Per-run accuracy, F1, ROC-AUC, classification report
└── plots/
    ├── quantum_loss_learning_curve.png   # Mean ±1 std across runs
    └── quantum_ConfusionMatrix.png       # Aggregated over all runs (to be implemented)
```

The experiment ID is deterministically derived from the config (test size, optimizer, QPU count, etc.) and truncated with a SHA-1 suffix if too long, ensuring reproducibility and avoiding directory collisions.

---

## Known issues

- **CUNQA QPU timeout:** `run_batch` enforces a per-chunk timeout (`chunk_timeout_s=120`). Long circuits or heavy load may hit this limit; increase via the keyword argument or restructure into smaller chunks.

---

## Roadmap

- [ ] **HYBRID optimizer** — COBYLA local phases alternating with PSO/DE escape cycles for better convergence on barren plateaus.
- [ ] **Confusion matrix plot** — aggregate predictions across runs and generate a heatmap automatically after training.
- [ ] **QNG optimizer** — Quantum Natural Gradient once supported by the polypus QML branch.
- [ ] **Real quantum hardware** — submit circuits to IBM Quantum or IonQ via Qiskit Runtime.
- [ ] **Quantum feature selection** — replace the classical SA sampler with a D-Wave quantum annealer or a QAOA-based QUBO solver.
- [ ] **Multi-class classification** — extend the parity rule and loss function beyond binary labels.
- [ ] **Noise-aware training** — incorporate device noise models (depolarizing, readout error) into the training loss.
- [ ] **Ansatz search** — automate circuit architecture selection (reps, entanglement, gate set) via hyperparameter optimization.
- [ ] **Benchmarking suite** — systematic comparison against classical baselines (SVM, XGBoost, MLP) with statistical significance testing.
- [ ] **`requirements.txt` / `pyproject.toml`** — pinned dependency file for reproducible environments.
- [ ] **Unit tests** — circuit construction, parity metric, QUBO formulation, and backend dispatch.
- [ ] **Experiment tracking** — optional MLflow or Weights & Biases logging alongside CSV/JSON export.

---

## References

- Muecke, S., Heese, R., Müller, S., Wolter, M., & Piatkowski, N. (2023). *Feature selection on quantum computers*. Quantum Machine Intelligence, 5(1), 11. [doi:10.1007/s42484-023-00099-z](https://doi.org/10.1007/s42484-023-00099-z)
- Cerezo, M., et al. (2021). *Variational quantum algorithms*. Nature Reviews Physics, 3(9), 625–644. [doi:10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
- Qiskit contributors (2023). *Qiskit: An Open-source Framework for Quantum Computing*. [doi:10.5281/zenodo.2573505](https://doi.org/10.5281/zenodo.2573505)

---

## License

This project is part of a Master's Thesis (TFM). License to be added upon publication.
