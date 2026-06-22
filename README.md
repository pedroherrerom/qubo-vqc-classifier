# qubo-vqc-classifier

A hybrid quantum–classical pipeline for binary molecular classification (Master's Thesis, University of Vigo). Combines **QUBO-based feature selection** via simulated annealing with a **Variational Quantum Classifier (VQC)**, benchmarked against classical baselines (Logistic Regression, SVM-RBF) and QSVC. Designed for HPC execution on CESGA Finisterrae III using [Qiskit](https://qiskit.org/), [polypus](https://github.com/polypus) and [CUNQA](https://cunqa.readthedocs.io).

> **Status:** Active research — TFM (Master's Thesis) project.

---

## Table of contents

- [Overview](#overview)
- [Pipeline stages](#pipeline-stages)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Local / single-node run](#local--single-node-run)
  - [HPC submission — single job (CESGA FT3)](#hpc-submission--single-job-cesga-ft3)
  - [HPC submission — job array (CESGA FT3)](#hpc-submission--job-array-cesga-ft3)
  - [Post-processing and plots](#post-processing-and-plots)
- [Outputs](#outputs)
- [Roadmap](#roadmap)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Overview

The pipeline targets the practical constraints of near-term quantum devices (NISQ era): limited qubit counts and noisy gates. It tackles both explicitly:

1. **Feature selection via QUBO** — reduces input features to exactly *k*, so the VQC circuit fits on available qubits without depth blowup. Solved with simulated annealing (`neal`), following Muecke et al. (2023).
2. **VQC training via polypus** — a Rust-core QML library (`polypus.qml.train`) drives PSO optimization over the variational parameters. The Rust extension handles data encoding and parameter binding internally.
3. **Hardware-aware execution** — training runs on local Aer; evaluation can be dispatched to CUNQA QPUs with automatic fallback to local Aer.
4. **Distributed job-array architecture** — independent runs are spread across SLURM array tasks on separate nodes, with a gated aggregation job that merges results and generates plots.

```
CSV dataset
    │
    ▼
Data preprocessing  ──►  QUBO feature selection (k features, fixed seed)
                                  │
                                  ▼
                        VQC circuit (k qubits)
                        Feature Map + Trainable Ansatz
                                  │
                          polypus.qml.train (PSO)
                                  │
                                  ▼
                    Youden's J threshold calibration
                         (on training set only)
                                  │
                                  ▼
                    Test evaluation + metrics export
               (ROC-AUC, F1, accuracy, CSV, JSON, plots)
```

---

## Pipeline stages

### Stage 1 — Data preprocessing (`data_processing.py`)

Supports two input modes:

- **Dual-CSV mode** (recommended): separate `train_path` and `test_path` files preserving the official dataset split.
- **Legacy single-CSV mode**: stratified random split via `test_size`.

Steps applied (all fitted on train, applied to test):

- Drop ID/irrelevant columns (`id_cols`), zero-variance columns, and high-cardinality object columns (> 30 unique values).
- Label-encode remaining categorical variables.
- Median imputation.
- MinMax scaling to **[0, π]** — mandatory for angle encoding; `StandardScaler` produces unbounded angles that break the feature map.

### Stage 2 — QUBO feature selection (`feature_selection.py`)

Implements Muecke et al. (2023):

- Builds importance and pairwise redundancy matrices via **mutual information**.
- Formulates a QUBO balancing relevance vs. redundancy, parameterized by `α`.
- Solves with `neal.SimulatedAnnealingSampler`.
- Binary search over `α` until exactly `k` features are selected.
- A **fixed seed** is enforced in array mode to guarantee identical feature subsets across all distributed tasks.

This stage can be skipped (`"stages": ["quantum"]`), passing all features directly to the VQC.

### Stage 3 — VQC training (`training.py`, `quantum_circuits.py`)

The number of selected features equals the qubit count. A fully decomposed primitive-gate circuit is built by composing:

- **Feature map** — encodes classical data into quantum states (`ZZFeatureMap`, `ZFeatureMap`, or `PauliFeatureMap`).
- **Ansatz** — trainable unitary (`RealAmplitudes`, `EfficientSU2`, or `TwoLocal`).

The feature map and ansatz are passed separately to `polypus.qml.train`. Per-generation fitness and mean-best values are captured from Rust stdout via `_PolypusStdoutInterceptor` (OS fd-level pipe) and checkpointed to CSV every N generations.

**Readout:** `single_qubit` mode — `P(qubit_0 = |1⟩)` — is the default and outperforms global parity, which is maximally nonlinear for odd qubit counts.

**Threshold calibration:** after training, Youden's J statistic is computed on the training set to find the optimal decision threshold `t*`. This threshold is then applied to the test set without re-fitting, preventing information leakage.

---

## Project structure

```
.
├── main_VQC.py               # Entry point — public API and pipeline runner
├── main_plotting.py          # Standalone post-processing and plot generation
├── make_exp_dir.py           # Login-node helper: creates numbered experiment dir
├── configVQC.json            # Default experiment configuration
├── submit.sh                 # HPC single-job submission (CESGA FT3)
├── qraise_job.sh             # SLURM job: provision CUNQA QPUs
├── vqc_job.sh                # SLURM job: run the pipeline (depends on qraise)
├── submit_array.sh           # HPC job-array submission (CESGA FT3)
├── vqc_array_job.sh          # SLURM array task: one chunk of independent runs
├── aggregate_job.sh          # SLURM aggregation job: merges all task outputs
└── vqc_modules/
    ├── __init__.py
    ├── cli.py                # Argument parsing (CLI + JSON config overlay)
    ├── pipeline.py           # Orchestration (single-node, task, aggregate modes)
    ├── data_processing.py    # Preprocessing utilities
    ├── feature_selection.py  # QUBO-QFS via simulated annealing
    ├── quantum_circuits.py   # Circuit construction and parameter binding
    ├── training.py           # polypus.qml.train adapter (PSO optimizer)
    ├── backends.py           # CUNQA / local Aer execution backends
    ├── metrics.py            # Loss, readout probability, Youden calibration
    ├── model_comparison.py   # Classical baselines (LogReg, SVM-RBF) and QSVC
    ├── process_metrics.py    # Post-processing: aggregation and plot dispatch
    ├── visualizations.py     # ROC curves, confusion matrix, PSO trajectory, bars
    ├── experiment.py         # Experiment IDs, directories, logging
    └── serialization.py      # NumPy-safe JSON encoder + stdout interceptor
```

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| [Qiskit](https://qiskit.org/) ≥ 2.1 | Circuit construction (function API: `zz_feature_map`, `efficient_su2`) |
| [Qiskit Aer](https://qiskit.github.io/qiskit-aer/) | Local circuit simulation |
| [dimod](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/) | QUBO / BQM formulation |
| [neal](https://docs.ocean.dwavesys.com/projects/neal/) | Simulated annealing sampler |
| [polypus](https://github.com/polypus) *(QML branch)* | VQC training via Rust PSO |
| [CUNQA](https://cunqa.readthedocs.io) ≥ 2.4.0 | QPU batch execution (HPC only) |
| scikit-learn | Preprocessing, classical baselines, metrics |
| RDKit | Molecular descriptors (dataset-specific) |
| NumPy, pandas, matplotlib, seaborn | Data handling and visualization |

> `cunqa` and `polypus` require specific cluster modules on CESGA (`gcc/14.3.0`, `openmpi/5.0.9`, `rust/1.88.0`). Local runs use Aer only. Installation guide to be added.

---

## Configuration

All parameters are controlled via `configVQC.json`. CLI flags override JSON values.

**Example**:

```jsonc
{
  // Data
  "train_path": "datasets/DIA_trainingset_RDKit_descriptors.csv",
  "test_path":  "datasets/DIA_testset_RDKit_descriptors.csv",
  "target": "Label",
  "outdir": "results",
  "id_cols": ["SMILES", "fr_para_hydroxylation"],
  "stages": ["annealing", "quantum"],   // drop "annealing" to skip QUBO selection

  // Experiment
  "seed": 42,
  "num_runs": 5,
  "k": 6,                               // features to select / qubits

  // QUBO feature selection
  "sa_num_reads": 500,
  "sa_bins": 10,

  // VQC optimizer
  "optimizer": "PSO",
  "opt_maxiter": 70,
  "opt_population_size": 128,
  "checkpoint_every": 10,              // checkpoint fitness to CSV every N generations

  // Circuit architecture
  "fm_type": "ZZFeatureMap",
  "fm_reps": 1,
  "ansatz_type": "EfficientSU2",
  "ansatz_reps": 1,
  "ansatz_entanglement": "circular",
  "ansatz_rotation_blocks": ["ry", "rz"],

  // Backend
  "vqc_num_shots": 1024,
  "vqc_readout": "single_qubit",
  "vqc_train_infrastructure": "local",
  "vqc_test_infrastructure": "local",
  "vqc_n_workers": 32,
  "vqc_cores_per_worker": 2
}
```

See `cli.py` for the full parameter reference.

---

## Usage

### Local / single-node run

```bash
python main_VQC.py --config configVQC.json
```

Override individual parameters on the command line:

```bash
python main_VQC.py \
  --config configVQC.json \
  --opt-maxiter 70 \
  --fm-reps 1 \
  --ansatz-reps 1 \
  --vqc-num-shots 1024
```

### HPC submission — single job (CESGA FT3)

Handles the two-job SLURM chain: provisions CUNQA QPUs via `qraise`, then launches the pipeline once they are ready (skipped automatically when `vqc_test_infrastructure=local`).

```bash
bash submit.sh --config configVQC.json --vqc-time 08:00:00 --vqc-mem 8G
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `configVQC.json` | Experiment config file |
| `--vqc-n-workers` | read from config | PSO parallel workers |
| `--vqc-cores-per-worker` | read from config | CPU cores per worker |
| `--vqc-time` | `08:00:00` | Wall-clock limit |
| `--vqc-mem` | `8G` | Memory per job |
| `--qraise-time` | VQC time + 1 h | CUNQA QPU lifetime |

### HPC submission — job array (CESGA FT3)

Spreads independent runs across SLURM array tasks (one task per run by default), then merges all outputs with a gated aggregation job. Requires `vqc_test_infrastructure=local`.

```bash
bash submit_array.sh \
  --config configVQC.json \
  --num-runs 10 \
  --runs-per-task 1 \
  --vqc-time 04:00:00 \
  --vqc-mem 8G
```

| Flag | Default | Description |
|---|---|---|
| `--num-runs` | read from config | Total independent runs |
| `--runs-per-task` | `1` | Runs per array task |
| `--max-concurrent` | `0` (no cap) | Max simultaneous tasks |
| `--agg-time` | `01:00:00` | Aggregation job wall-clock |

Monitor jobs:

```bash
watch -n 10 squeue --me
tail -f logs/vqc_array-<AID>_<TID>.out
tail -f logs/vqc_aggregate-<JID>.out
```

### Post-processing and plots

```bash
python main_plotting.py --dir results/<experiment_id>/
```

Regenerates all plots from existing CSVs without re-running training.

---

## Outputs

Each experiment writes to `<outdir>/<N>_<experiment_id>/`:

```
<experiment_id>/
├── experiment_config.json              # Full config snapshot
├── experiment.log                      # Timestamped run log
├── quantum_train_historical.csv        # Per-generation fitness + mean_best (all runs)
├── quantum_aggregated_predictions.csv  # Per-sample y_true, y_pred, y_prob (all runs)
├── quantum_raw_metrics.json            # Per-run accuracy, F1, ROC-AUC, report
├── model_comparison.csv                # Aggregated mean ± std across all models
├── model_comparison_runs.csv           # Per-run values for scatter overlay
└── plots/
    ├── quantum_pso_trajectory.png         # Global best + swarm mean across generations
    ├── quantum_final_loss_distribution.png
    ├── quantum_confusion_matrix.png       # Mean ± std across runs
    ├── quantum_roc_curves.png             # Per-run + interpolated mean ± std band
    └── model_comparison_bars.png          # Grouped bars with per-run scatter
```

In job-array mode, each task writes its own outputs under `tasks/task_NNNN/` before the aggregation job merges them into the experiment root.

---

## Roadmap

- [ ] Multiclass classification (architecture changes identified, deferred)
- [ ] Real quantum hardware submission via Qiskit Runtime
- [ ] Quantum feature selection (D-Wave annealer or QAOA-based QUBO solver)
- [ ] `requirements.txt` / `pyproject.toml` with pinned dependencies
- [ ] Unit tests (circuit construction, readout, QUBO formulation, backend dispatch)

---

## References

- Muecke, S., Heese, R., Müller, S., Wolter, M., & Piatkowski, N. (2023). *Feature selection on quantum computers*. Quantum Machine Intelligence, 5(1), 11. [doi:10.1007/s42484-023-00099-z](https://doi.org/10.1007/s42484-023-00099-z)
- Cerezo, M., et al. (2021). *Variational quantum algorithms*. Nature Reviews Physics, 3(9), 625–644. [doi:10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
- Qiskit contributors (2023). *Qiskit: An Open-source Framework for Quantum Computing*. [doi:10.5281/zenodo.2573505](https://doi.org/10.5281/zenodo.2573505)

---

## License

This project is part of a Master's Thesis (TFM) developed at the Universidade da Vigo, supervised by Eduardo Mosqueira (Universidade da Coruña) and Sergio Figueiras (Bahía Software SLU). License to be added upon publication.

---

## Acknowledgements

This research project was made possible through the access granted by the Galician Supercomputing Center (CESGA) to its supercomputing infrastructure. The supercomputer FinisTerrae III and its permanent data storage system have been funded by the NextGeneration EU 2021 Recovery, Transformation and Resilience Plan, ICT2021-006904, and also from the Pluriregional Operational Programme of Spain 2014-2020 of the European Regional Development Fund (ERDF), ICTS-2019-02-CESGA-3, and from the State Programme for the Promotion of Scientific and Technical Research of Excellence of the State Plan for Scientific and Technical Research and Innovation 2013-2016 State subprogramme for scientific and technical infrastructures and equipment of ERDF, CESG15-DE-3114.
