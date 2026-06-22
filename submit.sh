#!/bin/bash
# =============================================================================
# submit.sh  —  Single entry-point for the QML/VQC pipeline on CESGA FT3.
#
# WHAT THIS SCRIPT DOES (runs on the LOGIN node, not on a compute node):
#   1. Reads your settings (from CLI flags or the JSON config).
#   2. IF the config requests CUNQA for evaluation (vqc_test_infrastructure=cunqa),
#      it first submits qraise_job.sh to provision a *persistent* eval-side QPU
#      family, then submits vqc_job.sh with a dependency so the pipeline only
#      starts once those QPUs are registered.
#   3. IF evaluation is local (vqc_test_infrastructure=local), it skips qraise
#      entirely and just submits the pipeline.
#
# USAGE:
#   bash submit.sh [options]
#
# OPTIONS (all optional — sensible defaults are provided or read from config):
#   --config               PATH        JSON config file           (default: configVQC.json)
#   --vqc-n-workers        N           Number of parallel workers (default: read from config)
#   --vqc-cores-per-worker N           Cores allocated per worker (default: read from config)
#   --vqc-time             HH:MM:SS    Wall-clock for Python job  (default: 08:00:00)
#   --vqc-mem              MEM         RAM for Python job         (default: 8G)
#   --qraise-time          HH:MM:SS    How long eval QPUs stay up (default: VQC_TIME + 1h)
# =============================================================================
# --- Fixed Cluster Paths -----------------------------------------------------
export PIPELINE="/mnt/lustre/scratch/nlsas/home/uvi/et/phm/Pipeline_VQC_definitive"
export STORE="/mnt/lustre/scratch/nlsas/home/uvi/et/phm"
export VENV="${STORE}/polypus/TFM_env"

# --- Default Configurations --------------------------------------------------
CONFIG_FILE="${PIPELINE}/configVQC.json"

VQC_N_WORKERS=10
VQC_N_WORKERS_SET=0          

VQC_CORES_PER_WORKER=2
VQC_CORES_PER_WORKER_SET=0

VQC_TIME="08:00:00"
VQC_MEM="8G"
QRAISE_TIME=""

# Unique environment identifier shared between jobs
FAMILY_NAME="cunqa_$(date +%Y%m%d_%H%M%S)"

# --- Parse Command Line Arguments --------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"; shift 2 ;;
        --vqc-n-workers | --vqc-n-qpus | --vqc-n-cpus) 
            VQC_N_WORKERS="$2"; VQC_N_WORKERS_SET=1; shift 2 ;;
        --vqc-cores-per-worker | --vqc-cores-per-qpu) 
            VQC_CORES_PER_WORKER="$2"; VQC_CORES_PER_WORKER_SET=1; shift 2 ;;
        --vqc-time) 
            VQC_TIME="$2"; shift 2 ;;
        --vqc-mem) 
            VQC_MEM="$2"; shift 2 ;;
        --qraise-time) 
            QRAISE_TIME="$2"; shift 2 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# --- Validate Workspace Configurations ---------------------------------------
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: Configuration file not found: ${CONFIG_FILE}" >&2; exit 1; }

# --- Dynamically Extract Properties from JSON Config -------------------------
if [[ "${VQC_N_WORKERS_SET}" -eq 0 ]]; then
    VQC_N_WORKERS=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('vqc_n_workers', d.get('vqc_n_qpus', d.get('vqc_n_cpus', 10))))
" "${CONFIG_FILE}")
    echo "  VQC_N_WORKERS parsed from config: ${VQC_N_WORKERS}"
fi

if [[ "${VQC_CORES_PER_WORKER_SET}" -eq 0 ]]; then
    VQC_CORES_PER_WORKER=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('vqc_cores_per_worker', d.get('vqc_cores_per_qpu', 2)))
" "${CONFIG_FILE}")
    echo "  VQC_CORES_PER_WORKER parsed from config: ${VQC_CORES_PER_WORKER}"
fi

EVAL_INFRA=$(python3 -c \
    "import json,sys; print(json.load(open(sys.argv[1])).get('vqc_test_infrastructure', 'cunqa'))" \
    "${CONFIG_FILE}")
echo "  Evaluation infrastructure backend: ${EVAL_INFRA}"

# --- Derive Execution Window Bounds ------------------------------------------
if [[ -z "${QRAISE_TIME}" ]]; then
    IFS=':' read -r vh vm vs <<< "${VQC_TIME}"
    QRAISE_TIME=$(printf "%02d:%02d:%02d" $((10#$vh + 1)) $((10#$vm)) $((10#$vs)))
fi

# --- SLURM Hardware Allocation Calculation ----------------------------------
TRAIN_INFRA=$(python3 -c \
    "import json,sys; print(json.load(open(sys.argv[1])).get('vqc_train_infrastructure', 'local'))" \
    "${CONFIG_FILE}")

# Calculation base: (Workers * Cores per Worker) + 2 management buffer cores
TOTAL_SLURM_CPUS=$(( VQC_N_WORKERS * VQC_CORES_PER_WORKER + 2 ))

# Context-aware capping
if [[ "${TRAIN_INFRA}" == "local" ]]; then
    if (( TOTAL_SLURM_CPUS > 64 )); then
        echo "  [INFO] 100% local execution detected. Limiting to 64 physical cores."
        TOTAL_SLURM_CPUS=64
    fi
fi

# --- Export Global Environment Context ---------------------------------------
export CONFIG_FILE FAMILY_NAME VQC_N_WORKERS VQC_CORES_PER_WORKER VQC_TIME TOTAL_SLURM_CPUS VQC_MEM QRAISE_TIME
export POLYPUS_CUNQA_FAMILY="${FAMILY_NAME}"
export VQC_N_QPUS="${VQC_N_WORKERS}"
export VQC_CORES_PER_QPU="${VQC_CORES_PER_WORKER}"

mkdir -p "${PIPELINE}/logs"

echo "════════════════════════════════════════"
echo "  QML Pipeline — CESGA FT3 Submission   "
echo "  $(date)"
echo "════════════════════════════════════════"
echo "  Config        : ${CONFIG_FILE}"
echo "  Family        : ${FAMILY_NAME}"
echo "  Eval backend  : ${EVAL_INFRA}"
echo "  Workers       : ${VQC_N_WORKERS} (${VQC_CORES_PER_WORKER} threads/worker)"
echo "  Total CPUs    : ${TOTAL_SLURM_CPUS} physical cores requested"
echo "  VQC job       : time=${VQC_TIME} mem=${VQC_MEM}"
echo "  QRAISE_TIME   : ${QRAISE_TIME} (CUNQA cluster lifetime)"
echo "════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Step 1: Provision Persistent QPU Backends if needed ---------------------
DEP_ARG=""
if [[ "${EVAL_INFRA}" == "cunqa" ]]; then
    echo "  CUNQA Infrastructure detected -> Submitting allocation dependencies."
    
    QRAISE_CPUS=$(( VQC_N_WORKERS * VQC_CORES_PER_WORKER ))
    QRAISE_MEM=$(( VQC_N_WORKERS * 1))G

    QRAISE_JID=$(sbatch --parsable \
        --export=ALL \
        --time="${QRAISE_TIME}" \
        --cpus-per-task="${QRAISE_CPUS}" \
        --mem="${QRAISE_MEM}" \
        --output="${PIPELINE}/logs/qraise-%j.out" \
        "${SCRIPT_DIR}/qraise_job.sh")
    echo "  [1/2] Qraise environment job submitted -> ID: ${QRAISE_JID}"
    
    DEP_ARG="--dependency=afterok:${QRAISE_JID}"
else
    echo "  Local execution backend -> Skipping infrastructure pre-allocation steps."
fi

# --- Step 2: Submit Core Processing Job --------------------------------------
VQC_JID=$(sbatch --parsable \
    --export=ALL \
    ${DEP_ARG} \
    --time="${VQC_TIME}" \
    --cpus-per-task="${TOTAL_SLURM_CPUS}" \
    --mem="${VQC_MEM}" \
    --output="${PIPELINE}/logs/vqc_pipeline-%j.out" \
    "${SCRIPT_DIR}/vqc_job.sh")

if [[ -n "${DEP_ARG}" ]]; then
    echo "  [2/2] VQC pipeline execution queued -> ID: ${VQC_JID} (Awaiting Qraise allocation ${QRAISE_JID})"
else
    echo "  [1/1] VQC pipeline execution queued -> ID: ${VQC_JID}"
fi
echo ""
echo "  Monitoring command recommendations:"
echo "    watch -n 10 squeue --me"
echo "    tail -f ${PIPELINE}/logs/vqc_pipeline-${VQC_JID}.out"