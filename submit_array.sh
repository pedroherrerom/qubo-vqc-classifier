#!/bin/bash
# =============================================================================
# submit_array.sh  —  Multi-node launcher for the QML/VQC pipeline on CESGA FT3.
#
# WHAT THIS DOES (runs on the LOGIN node):
#   1. Creates the numbered experiment dir ONCE (make_exp_dir.py, no polypus).
#   2. Submits a SLURM JOB ARRAY: one task per chunk of runs. SLURM spreads the
#      tasks across nodes, so the independent runs execute in parallel.
#   3. Submits an AGGREGATION job with afterok-dependency on the whole array; it
#      merges every per-task output into the canonical experiment-level files
#      and builds the model-comparison tables + plots.
#
#   Each array task is hermetic: local-Aer training AND local-Aer evaluation.
#   No CUNQA here — virtual QPUs are only reachable inside their own allocation,
#   so a shared eval family cannot span nodes, and a per-task qraise would blow
#   past the QPU budget. Use the original submit.sh for the CUNQA eval path.
#
# USAGE:
#   bash submit_array.sh [options]
#
# OPTIONS (sensible defaults / read from config):
#   --config               PATH     JSON config            (default: configVQC.json)
#   --num-runs             N        Total independent runs (default: config num_runs)
#   --runs-per-task        N        Runs per array task    (default: 1)
#   --max-concurrent       N        Cap simultaneous tasks (default: 0 = no cap)
#   --vqc-n-workers        N        PSO workers per task   (default: config)
#   --vqc-cores-per-worker N        Cores per worker       (default: config)
#   --vqc-time             HH:MM:SS Wall-clock per task    (default: 04:00:00)
#   --vqc-mem              MEM      RAM per task            (default: 8G)
#   --agg-time             HH:MM:SS Wall-clock aggregation (default: 01:00:00)
#   --agg-mem              MEM      RAM aggregation         (default: 8G)
# =============================================================================
set -euo pipefail

# --- Fixed Cluster Paths -----------------------------------------------------
export PIPELINE="/mnt/lustre/scratch/nlsas/home/uvi/et/phm/Pipeline_VQC_definitive"
export STORE="/mnt/lustre/scratch/nlsas/home/uvi/et/phm"
export VENV="${STORE}/polypus/TFM_env"
cd "${PIPELINE}"
source "${VENV}/bin/activate"

# --- Default Configurations --------------------------------------------------
CONFIG_FILE="${PIPELINE}/configVQC.json"
NUM_RUNS="";              NUM_RUNS_SET=0
RUNS_PER_TASK=1
MAX_CONCURRENT=0
VQC_N_WORKERS="";         VQC_N_WORKERS_SET=0
VQC_CORES_PER_WORKER="";  VQC_CORES_PER_WORKER_SET=0
VQC_TIME="04:00:00"
VQC_MEM="8G"
AGG_TIME="01:00:00"
AGG_MEM="8G"

# --- Parse Command Line Arguments --------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)                 CONFIG_FILE="$2"; shift 2 ;;
        --num-runs)               NUM_RUNS="$2"; NUM_RUNS_SET=1; shift 2 ;;
        --runs-per-task)          RUNS_PER_TASK="$2"; shift 2 ;;
        --max-concurrent)         MAX_CONCURRENT="$2"; shift 2 ;;
        --vqc-n-workers|--vqc-n-qpus)
                                  VQC_N_WORKERS="$2"; VQC_N_WORKERS_SET=1; shift 2 ;;
        --vqc-cores-per-worker|--vqc-cores-per-qpu)
                                  VQC_CORES_PER_WORKER="$2"; VQC_CORES_PER_WORKER_SET=1; shift 2 ;;
        --vqc-time)               VQC_TIME="$2"; shift 2 ;;
        --vqc-mem)                VQC_MEM="$2"; shift 2 ;;
        --agg-time)               AGG_TIME="$2"; shift 2 ;;
        --agg-mem)                AGG_MEM="$2"; shift 2 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: Configuration file not found: ${CONFIG_FILE}" >&2; exit 1; }

# --- Internal Configuration Parser Helper ------------------------------------
cfg_get() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get(sys.argv[2], sys.argv[3]))" "${CONFIG_FILE}" "$1" "$2"; }

if [[ "${VQC_N_WORKERS_SET}" -eq 0 ]]; then
    VQC_N_WORKERS=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('vqc_n_workers',d.get('vqc_n_qpus',10)))" "${CONFIG_FILE}")
fi
if [[ "${VQC_CORES_PER_WORKER_SET}" -eq 0 ]]; then
    VQC_CORES_PER_WORKER=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('vqc_cores_per_worker',d.get('vqc_cores_per_qpu',2)))" "${CONFIG_FILE}")
fi
if [[ "${NUM_RUNS_SET}" -eq 0 ]]; then
    NUM_RUNS=$(cfg_get num_runs 5)
fi

# --- Cluster Bounds and Network Constraint Enforcement ----------------------
EVAL_INFRA=$(cfg_get vqc_test_infrastructure cunqa)
if [[ "${EVAL_INFRA}" != "local" ]]; then
    echo "ERROR: vqc_test_infrastructure='${EVAL_INFRA}', but the job-array engine requires 'local' execution." >&2
    echo "       Shared virtual QPUs cannot span cross-node network barriers seamlessly." >&2
    echo "       -> Set \"vqc_test_infrastructure\": \"local\" in ${CONFIG_FILE}, or pivot to standard single submission." >&2
    exit 1
fi

# --- Derived Multi-processing Calculations -----------------------------------
TRAIN_INFRA=$(cfg_get vqc_train_infrastructure local)

# Cunqa limit calculation
TOTAL_SLURM_CPUS=$(( VQC_N_WORKERS * VQC_CORES_PER_WORKER + 2 ))

# Smart self-correction: If the training is local, cap at the node's maximum capacity
if [[ "${TRAIN_INFRA}" == "local" ]]; then
    if (( TOTAL_SLURM_CPUS > 64 )); then
        echo "  [INFO] 100% local execution detected. Capping SLURM request to the node's maximum capacity (64 cores)."
        TOTAL_SLURM_CPUS=64
    fi
fi

N_TASKS=$(( (NUM_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
(( N_TASKS >= 1 )) || { echo "ERROR: Invalid N_TASKS evaluation state=${N_TASKS}." >&2; exit 1; }
ARRAY_SPEC="0-$(( N_TASKS - 1 ))"
if (( MAX_CONCURRENT > 0 )); then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

mkdir -p "${PIPELINE}/logs"

# --- Step 0: Initialize Workspace Context Directories Safely -----------------
EXP_DIR=$(python3 "${PIPELINE}/make_exp_dir.py" --config "${CONFIG_FILE}")
[[ -n "${EXP_DIR}" && -d "${EXP_DIR}" ]] || { echo "ERROR: Root experiment workspace initialization failure." >&2; exit 1; }

# --- Global Environment Transmissions ----------------------------------------
export CONFIG_FILE EXP_DIR NUM_RUNS RUNS_PER_TASK
export VQC_N_WORKERS VQC_CORES_PER_WORKER
export VQC_N_QPUS="${VQC_N_WORKERS}"
export VQC_CORES_PER_QPU="${VQC_CORES_PER_WORKER}"

echo "════════════════════════════════════════"
echo "  QML Pipeline — Job-Array Submission   "
echo "  $(date)"
echo "════════════════════════════════════════"
echo "  Config       : ${CONFIG_FILE}"
echo "  Experiment   : ${EXP_DIR}"
echo "  Total runs   : ${NUM_RUNS} (${RUNS_PER_TASK}/task -> ${N_TASKS} separate tasks)"
echo "  Array spec   : ${ARRAY_SPEC}"
echo "  Per task     : workers=${VQC_N_WORKERS} cores=${VQC_CORES_PER_WORKER} -> ${TOTAL_SLURM_CPUS} CPUs"
echo "  Per task     : time=${VQC_TIME} mem=${VQC_MEM}"
echo "  Eval backend : Local Aer Engine"
echo "  Aggregation  : time=${AGG_TIME} mem=${AGG_MEM}"
echo "════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Step 1: Submit Parallel Decoupled Array Tasks ---------------------------
ARRAY_JID=$(sbatch --parsable \
    --export=ALL \
    --array="${ARRAY_SPEC}" \
    --time="${VQC_TIME}" \
    --cpus-per-task="${TOTAL_SLURM_CPUS}" \
    --mem="${VQC_MEM}" \
    --output="${PIPELINE}/logs/vqc_array-%A_%a.out" \
    "${SCRIPT_DIR}/vqc_array_job.sh")
echo "  [1/2] Job array allocation processing dispatched -> ID: ${ARRAY_JID}"

# --- Step 2: Queue Aggregation Pipelines -------------------------------------
AGG_JID=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:"${ARRAY_JID}" \
    --time="${AGG_TIME}" \
    --cpus-per-task="${TOTAL_SLURM_CPUS}" \
    --mem="${AGG_MEM}" \
    --output="${PIPELINE}/logs/vqc_aggregate-%j.out" \
    "${SCRIPT_DIR}/aggregate_job.sh")
echo "  [2/2] Aggregation reporting framework attached -> ID: ${AGG_JID}"
echo ""
echo "  Monitoring context routes:"
echo "    watch -n 10 squeue --me"
echo "    Results target directory: ${EXP_DIR}"