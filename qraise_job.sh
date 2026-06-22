#!/bin/bash
# =============================================================================
# qraise_job.sh  —  SLURM job that provisions CUNQA QPUs.
#
# WHAT THIS DOES:
#   Calls `qraise`, which registers N quantum simulator processes with the
#   CUNQA daemon and writes their addresses to qpus.json.
#   The script then exits — the QPUs keep running independently for QRAISE_TIME.
#   The SLURM job itself stays alive (holding resources) until its wall time.
#
# SLURM directives (static hardware request):
#   -c 1 and --mem=4G are enough — this job just calls qraise and waits.
#   --time and -o (log file) are injected by submit.sh on the sbatch command
#   line, so they must NOT appear as #SBATCH directives here. SLURM parses
#   #SBATCH lines before the shell runs, so variables like $STORE would not
#   be expanded — they'd be treated as literal strings.
#
# REQUIRED ENV VARS (passed automatically via --export=ALL from submit.sh):
#   FAMILY_NAME            unique tag linking these QPUs to your vqc_job
#   VQC_N_WORKERS          number of virtual QPUs (workers) to provision
#   VQC_CORES_PER_WORKER   CPU cores allocated to each worker process
#   QRAISE_TIME            how long QPUs stay registered (must exceed VQC job wall time)
#   VENV                   path to the Python virtual environment
# =============================================================================
#SBATCH -J qraise_vqc
#SBATCH -p medium

# --- Initialize Target Interprocess Middleware -------------------------------
module purge
module load cesga/2025 gcc/14.3.0 openmpi/5.0.9 openblas/0.3.30
source "${VENV}/bin/activate"

# Fallback parameter evaluations
FAMILY_NAME="${FAMILY_NAME:-default_cunqa}"
VQC_N_WORKERS="${VQC_N_WORKERS:-1}"
VQC_CORES_PER_WORKER="${VQC_CORES_PER_WORKER:-2}"
QRAISE_TIME="${QRAISE_TIME:-02:30:00}"

echo "=== CUNQA Network Node Initialization Dispatched: $(date) ==="
echo "  FAMILY_NAME          = ${FAMILY_NAME}"
echo "  VQC_N_WORKERS        = ${VQC_N_WORKERS}"
echo "  VQC_CORES_PER_WORKER = ${VQC_CORES_PER_WORKER}"
echo "  QRAISE_TIME          = ${QRAISE_TIME}"

# --- Provision and Register Backend Cluster Instantiations -------------------
qraise \
    -n "${VQC_N_WORKERS}" \
    -t "${QRAISE_TIME}" \
    --family_name="${FAMILY_NAME}" \
    --co-located \
    -c "${VQC_CORES_PER_WORKER}" \
    -N 1 \
    --qpuN "${VQC_N_WORKERS}"

echo "=== CUNQA Network Allocation Job Finished (exit status: $?) · $(date) ==="