#!/bin/bash
# =============================================================================
# aggregate_job.sh  —  Runs ONCE after the whole array (afterok dependency).
#
# Merges every ${EXP_DIR}/tasks/task_*/ into the canonical experiment-level
# CSVs/JSON, recomputes the classical + QSVC baselines once, writes the
# model-comparison tables, and finally generates all plots.
#
# REQUIRED ENV (exported by submit_array.sh via --export=ALL):
#   CONFIG_FILE, EXP_DIR, PIPELINE, VENV
# =============================================================================
#SBATCH -J vqc_aggregate
#SBATCH -p medium
#SBATCH --mail-user=pedro2002phm@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Load Analytical Execution Framework Context -----------------------------
module purge
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30
source "${VENV}/bin/activate"
cd "${PIPELINE}"

# Allocate execution bounds to maximize standard BLAS processing capacity
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== Post-Processing Evaluation Step Started: $(date) ==="
echo "  EXP_DIR = ${EXP_DIR}"

# --- Step 1: Merge Array Datasets & Evaluate Performance Tables --------------
python3 "${PIPELINE}/main_VQC.py" \
    --config "${CONFIG_FILE}" \
    --exp-dir "${EXP_DIR}" \
    --aggregate
AGG_EXIT=$?
echo "  Dataset consolidation final exit monitoring state: ${AGG_EXIT}"

# --- Step 2: Execute Thesis Optimization Visualizations ----------------------
# Errors handled gracefully here to protect output storage vectors from dropping
python3 "${PIPELINE}/main_plotting.py" --dir "${EXP_DIR}" || \
    echo "WARNING: Plot generation framework failed; compiled metrics remain intact within: ${EXP_DIR}."

echo "=== Post-Processing Aggregation Complete (exit status: ${AGG_EXIT}) · $(date) ==="
exit "${AGG_EXIT}"