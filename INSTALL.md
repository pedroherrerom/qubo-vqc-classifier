# Installation Guide

This guide covers two installation paths:

- **[Local development (WSL/Linux)](#local-development-wsllinux)** — for code development and smoke tests. CUNQA is not available; training and evaluation run on local Aer.
- **[HPC production (CESGA FinisTerrae III)](#hpc-production-cesga-finisterrae-iii)** — full pipeline with CUNQA QPU orchestration and SLURM job arrays.

---

## Local Development (WSL/Linux)

### Prerequisites

| Tool | Required version | Notes |
|---|---|---|
| Python | ≥ 3.10 | `python3 --version` |
| Rust + Cargo | ≥ 1.78 | Install via `rustup` |
| git | any | `git --version` |

**Install Rust** (if not present):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**Install pip and venv** (Ubuntu/WSL):
```bash
sudo apt update && sudo apt install python3-pip python3-venv -y
```

---

### 1. Create the virtual environment

From the project root:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

---

### 2. Install Python dependencies

```bash
pip install qiskit qiskit-aer qiskit-machine-learning \
            scikit-learn numpy pandas matplotlib seaborn \
            dimod dwave-neal maturin
```

Verify:
```bash
python3 -c "
import qiskit, qiskit_aer, qiskit_machine_learning
import sklearn, numpy, pandas, matplotlib, seaborn, dimod, neal
print('All dependencies OK.')
"
```

---

### 3. Install Polypus

The public `main` branch of polypus has diverged from the version used on CESGA. The `polypus.qml.train` API (which accepts a Qiskit feature map, ansatz, and training data) and the critical fix to `pso.rs` (passing measurement dicts instead of raw bitstrings to the expectation function) are only present in the CESGA fork. You must clone from the public repo, pin to the correct commit, and then patch `pso.rs` from CESGA.

#### 3.1 Clone and pin to the compatible commit

```bash
cd ~/projects   # or wherever you keep your code
git clone https://github.com/Bahia-Software/polypus.git polypus_qml
cd polypus_qml
git checkout d249fd7   # "New polypus.qml module" — last commit with the qml.train API
```

#### 3.2 Patch `pso.rs`

The public commit at `d249fd7` still contains the original `pso.rs`, which routes measurement results through `polypus_python.expectation_values`. That function iterates over **unique bitstrings** rather than whole samples, so the sample-to-label alignment in `_make_expectation` immediately drifts and the optimizer scores samples against random labels. This causes an `AttributeError: 'str' object has no attribute 'values'` panic at runtime.

The fix is in `evaluate_qml_candidate` inside `src/algorithms/vqc/pso.rs`. Replace the inner loop body that calls `polypus_python`:

**Before** (original `d249fd7`):
```rust
        let running_result = runner.run(&batch_args);
        let batch_expectations: Vec<f64> = Python::with_gil(|py| {
            PyModule::import(py, "polypus_python")
                .expect("Failed to import polypus_python")
                .call_method("expectation_values", (running_result, expectation_function), None)
                .expect("Error computing expectation values")
                .extract::<Vec<f64>>()
                .expect("Failed to extract expectation values")
        });
        all_expectations.extend(batch_expectations);
```

**After** (patched version):
```rust
        let running_result = runner.run(&batch_args);
        let batch_expectations: Vec<f64> = Python::with_gil(|py| {
            expectation_function.call1(py, (running_result,))
                .expect("Error calling QML expectation function")
                .extract::<Vec<f64>>(py)
                .expect("Failed to extract Vec<f64> from QML expectation function")
        });
        all_expectations.extend(batch_expectations);
```

The patched version calls the Python `expectation_function` directly with the raw list of measurement dicts, bypassing `polypus_python.expectation_values` entirely. This lets `_make_expectation` in `training.py` receive the counts dicts it expects and index into `y_train` deterministically.

Apply the patch manually with the diff above, or copy the patched file directly from CESGA:

```bash
scp <user>@ft3.cesga.es:<cesga_polypus_qml_path>/src/algorithms/vqc/pso.rs \
    polypus_qml/src/algorithms/vqc/pso.rs
```

#### 3.3 Compile and install

From the repo root (where `Cargo.toml` lives):

```bash
source ~/projects/qubo-vqc-classifier/env/bin/activate
maturin develop --release
pip install packages/polypus_python/
```

> **Troubleshooting:** if `maturin develop` reports `Can't find Cargo.toml`, you are too deep in the directory tree. The workspace `Cargo.toml` lives at `polypus_qml/`, not inside `packages/polypus_python/`.

#### 3.4 Verify

```bash
python3 -c "import polypus; print('Polypus loaded successfully.')"
```

---

### 4. Smoke test

Run a minimal end-to-end check with synthetic data (no dataset required):

```bash
cd ~/projects/qubo-vqc-classifier
source env/bin/activate
python3 main_VQC.py --config configVQC.json --smoke-test
```

A successful run will log PSO generation progress and report `VQC` metrics. CUNQA-related warnings are expected and harmless in local mode.

---

## HPC Production (CESGA FinisTerrae III)

### Prerequisites

The following CESGA modules must be loaded for compilation and all SLURM jobs:

```bash
module purge
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30
```

Rust is additionally required for compiling polypus:
```bash
module load rust/1.88.0
```

> **Why `rust/1.88.0`?** The system Cargo (Gentoo, `1.69.0`) cannot parse this repo's `Cargo.lock` (lock file v4 requires Cargo ≥ 1.78). Always load the CESGA module before any `maturin` or `cargo` call.
>
> Verify: `cargo --version` should report `1.88.0`, and `which cargo` should resolve under the module tree, not `/software/2025/gentoo/...`.

---

### 1. Create the virtual environment

```bash
export STORE="/mnt/lustre/scratch/nlsas/home/uvi/et/phm"
export VENV="${STORE}/polypus/env"

python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Install CUNQA

CUNQA must be compiled from source on a compute node to link correctly against the cluster's MPI and OpenBLAS libraries. Compiling on the login node is not supported.

#### 2.1 Prepare the configuration script (`configure_cunqa.sh`)

This script forces CMake to use the venv Python and links OpenBLAS. Make it executable:
```bash
chmod +x configure_cunqa.sh
```

#### 2.2 Submit the installation job

```bash
sbatch install_cunqa.sh
```

This script reinstalls `pybind11` and uses `mpicc`/`mpicxx` wrappers to ensure MPI compatibility. Wait for the job to complete before proceeding.

#### 2.3 Verify

```bash
python3 -c "import cunqa; print('CUNQA loaded successfully.')"
```

---

### 3. Install Polypus

#### 3.1 Clone the repository

```bash
cd "${STORE}"
git clone https://github.com/Bahia-Software/polypus.git polypus_qml
cd polypus_qml
git checkout d249fd7   # pin to the qml.train API commit
```

#### 3.2 Apply the pso.rs patch

The `pso.rs` fix described in [section 3.2 above](#32-patch-psors-from-cesga) must also be applied here if starting from scratch. If your CESGA repo already has the patched file, skip this step.

#### 3.3 Compile and install

Ensure all modules and the venv are active:
```bash
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30 rust/1.88.0
source "${VENV}/bin/activate"

cd "${STORE}/polypus_qml"
maturin develop --release
pip install packages/polypus_python/ --force-reinstall
```

#### 3.4 Verify

```bash
python3 -c "import polypus; print('Polypus loaded successfully.')"
```

---

### 4. SLURM environment override (required for CUNQA)

CESGA's `/etc/bashrc` forces the legacy `cesga/2020` module stack into all SLURM jobs, which breaks `srun --mpi=pmix_v2` (the MPI backend used by CUNQA's QPU launcher). Add the following block to `~/.bashrc` to override it safely — the guard ensures this only fires inside SLURM jobs, not on the login node:

```bash
# Override CESGA legacy environment for SLURM jobs (required for CUNQA/polypus)
if [ -n "$SLURM_JOB_ID" ]; then
    module purge > /dev/null 2>&1
    module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30 > /dev/null 2>&1
fi
```

> **Why this is needed:** without this override, QPU launcher jobs submitted by `qraise` inherit `cesga/2020` and fail with `mpi/pmix_v2: init: can not load PMIx library`. The QPU server processes never start, and the pipeline silently falls back to local Aer for evaluation.

---

### 5. Recompiling after a Rust source change

If you modify any `.rs` file in `polypus_qml` (e.g. `src/algorithms/vqc/pso.rs`), recompile before running:

```bash
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30 rust/1.88.0
source "${VENV}/bin/activate"
cd "${STORE}/polypus_qml"   # repo root, not packages/polypus_python/
maturin develop --release
```

A successful import does not confirm your change took effect — run a smoke test before submitting a long SLURM job:

```bash
python3 main_VQC.py --config configVQC.json --smoke-test
```

---

## Dependency Summary

| Dependency | Local (WSL) | CESGA HPC |
|---|---|---|
| Python ≥ 3.10 | ✅ | ✅ |
| Qiskit ≥ 2.1 | ✅ | ✅ |
| Qiskit Aer | ✅ | ✅ |
| qiskit-machine-learning | ✅ | ✅ |
| scikit-learn, numpy, pandas | ✅ | ✅ |
| dimod, dwave-neal | ✅ | ✅ |
| polypus (pinned + patched) | ✅ | ✅ |
| CUNQA | ❌ not available | ✅ |
| SLURM / MPI | ❌ not available | ✅ |