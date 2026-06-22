#!/bin/bash
# =============================================================================
# vqc_job.sh  —  SLURM job that runs the full QML/VQC Python pipeline.
#
# WHAT THIS DOES:
#   1. Sets POLYPUS_CUNQA_FAMILY so the Python code has a fallback reference.
#   2. Runs main_VQC.py with the JSON config.
#   3. Polypus itself will spawn, manage, and destroy the QPU processes 
#      internally during its qml.train routine if train_infrastructure=cunqa.
#
# REQUIRED ENV VARS (passed automatically via --export=ALL from submit.sh):
#   FAMILY_NAME   Unique identifier for this specific run
#   CONFIG_FILE   path to configVQC.json (or similar)
#   PIPELINE      path to the Python pipeline directory
#   VENV          path to the Python virtual environment
# =============================================================================
#SBATCH -J vqc_pipeline
#SBATCH -p medium
#SBATCH --mail-user=pedroherreromaldonado@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Clear and Establish Cluster Execution Environment ------------------------
module purge
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30
source "${VENV}/bin/activate"

export POLYPUS_CUNQA_FAMILY="${FAMILY_NAME:-default_cunqa}"

echo "=== VQC Execution Node Routine Dispatched: $(date) ==="
echo "  FAMILY_NAME = ${POLYPUS_CUNQA_FAMILY}"
echo "  CONFIG_FILE = ${CONFIG_FILE}"
echo "=== Git tracking: branch=$(git -C "${PIPELINE}" branch --show-current) commit=$(git -C "${PIPELINE}" rev-parse --short HEAD) ==="

# --- Thread Contention Management (Rust Parallelism vs Qiskit Engines) -------

# Empower Rayon (Rust/Polypus) to consume the full SLURM CPU profile bounds
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# Prevent thread trashing and oversubscription by locking BLAS/MKL backends down
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Disable Qiskit Python multiprocessing layers to avoid structural deadlock
export QISKIT_PARALLEL=FALSE

# --- Run Pipeline Execution --------------------------------------------------
python3 "${PIPELINE}/main_VQC.py" --config "${CONFIG_FILE}"
EXIT_CODE=$?

echo "=== VQC Routine Finalized (exit status: ${EXIT_CODE}) · $(date) ==="
exit "${EXIT_CODE}"