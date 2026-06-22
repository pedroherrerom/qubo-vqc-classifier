#!/bin/bash
# =============================================================================
# vqc_array_job.sh  —  One SLURM ARRAY TASK of the QML/VQC pipeline.
#
# Each task owns a contiguous chunk of runs:
#   RUN_OFFSET = SLURM_ARRAY_TASK_ID * RUNS_PER_TASK
#   RUN_COUNT  = min(RUNS_PER_TASK, NUM_RUNS - RUN_OFFSET)
# and writes ONLY its own outputs under ${EXP_DIR}/tasks/task_<offset>/.
#
# THREAD MANAGEMENT:
#   Thread/parallelism env (RAYON_NUM_THREADS, OMP/MKL/OPENBLAS=1,
#   QISKIT_PARALLEL) is centralized in vqc_modules.training.setup_environment(),
#   called at the start of run_vqc_polypus — identical config to the single-node
#   path. Not set here.
#
# REQUIRED ENV (exported by submit_array.sh via --export=ALL):
#   CONFIG_FILE, EXP_DIR, NUM_RUNS, RUNS_PER_TASK, PIPELINE, VENV
# =============================================================================
#SBATCH -J vqc_array
#SBATCH -p medium
#SBATCH --mail-user=pedroherreromaldonado@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Cluster Environment Instantiation ---------------------------------------
module purge
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30
source "${VENV}/bin/activate"
cd "${PIPELINE}"

# --- Evaluate Running Offset Tasks -------------------------------------------
TID="${SLURM_ARRAY_TASK_ID:-0}"
RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
RUN_OFFSET=$(( TID * RUNS_PER_TASK ))
REMAINING=$(( NUM_RUNS - RUN_OFFSET ))

if (( REMAINING <= 0 )); then
    echo "Array worker task allocation bounds exceeded (offset ${RUN_OFFSET} >= bounds ${NUM_RUNS}). Exiting."
    exit 0
fi
RUN_COUNT=$(( RUNS_PER_TASK < REMAINING ? RUNS_PER_TASK : REMAINING ))

echo "=== VQC Array Chunk Job Task ${TID} Dispatched: $(date) ==="
echo "  EXP_DIR    = ${EXP_DIR}"
echo "  RUN_OFFSET = ${RUN_OFFSET}"
echo "  RUN_COUNT  = ${RUN_COUNT} (Target ceiling: NUM_RUNS=${NUM_RUNS})"
echo "  CPUS/TASK  = ${SLURM_CPUS_PER_TASK:-?}"
echo "=== Git tracking: branch=$(git -C "${PIPELINE}" branch --show-current) commit=$(git -C "${PIPELINE}" rev-parse --short HEAD) ==="

# Note: Environment limits are managed natively by setup_environment inline hooks
python3 "${PIPELINE}/main_VQC.py" \
    --config "${CONFIG_FILE}" \
    --exp-dir "${EXP_DIR}" \
    --run-offset "${RUN_OFFSET}" \
    --run-count "${RUN_COUNT}"
EXIT_CODE=$?

echo "=== VQC Array Chunk Job Task ${TID} Finalized (exit status: ${EXIT_CODE}) · $(date) ==="
exit "${EXIT_CODE}"