#!/bin/bash
# Submit GLMsingle beta-version QC decoding as a SINGLE SLURM job that runs
# all subjects internally via `xargs -P` (not a 59-way array job — see below).
# Runs the same standardized whole-brain category decoder on each GLMsingle
# model type (B/C/D; A is skipped, see run_beta_qc_decoding.py), writing one
# tidy CSV per subject. Category decoding is a pipeline-validation probe; use
# it to check whether the denoising/ridge steps improve decodability here.
#
# Usage (from repo root):
#   bash multivariate/submit_beta_qc_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_beta_qc_decoding.sh 01 05 12   # specific subjects
#   NPROC=4 bash multivariate/submit_beta_qc_decoding.sh    # override concurrency
#
# Why one job instead of an array: measured on a compute node via `srun` +
# /usr/bin/time -v (one subject, B+C+D) — 35s wall time, 866 MB peak RSS,
# ~1 CPU core (LinearSVC/liblinear doesn't thread). This is not memory-bound
# (unlike GLMsingle itself) and far too short-lived to be worth a separate
# array-task scheduling/queueing overhead per subject. A single job with
# NPROC-way internal parallelism (same pattern as run_local.sh's beta_qc
# pipeline, just dispatched through Slurm instead of over SSH) finishes all
# subjects in a few minutes instead of 59 separate job slots.
# run_beta_qc_decoding.py skips subjects already done, so this is resumable
# if the job is killed partway through.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_qc"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="/home/hfluhr/data/learninghabits/participants_mvpa.tsv"

# Concurrent subjects within the single job allocation. Cluster's `standard`
# partition nodes have >=8 cores; 8 is a safe default that schedules easily
# and keeps every worker single-threaded (no BLAS thread contention).
NPROC="${NPROC:-8}"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
SUBJECTS_FILE="${LOG_DIR}/subjects.txt"

if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@" > "$SUBJECTS_FILE"
else
    if [ ! -f "$PARTICIPANTS_TSV" ]; then
        echo "ERROR: participants file not found: ${PARTICIPANTS_TSV}" >&2
        exit 1
    fi
    cp "$PARTICIPANTS_TSV" "$SUBJECTS_FILE"
fi

N=$(wc -l < "$SUBJECTS_FILE")
if [ "$N" -eq 0 ]; then
    echo "ERROR: subject list is empty" >&2
    exit 1
fi

echo "Submitting 1 job for ${N} subjects (NPROC=${NPROC} concurrent):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=beta_qc
#SBATCH --output=${LOG_DIR}/beta_qc_%j.out
#SBATCH --error=${LOG_DIR}/beta_qc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --partition=standard

set -eo pipefail

# Pin every worker to 1 thread — with NPROC subjects running concurrently,
# letting each spawn its own BLAS threads would oversubscribe the cpus.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_one() {
    local s="\$1"
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_beta_qc_decoding.py" \\
        --subject "\$s" \\
        --bids-dir "${BIDS_DIR}" \\
        --glmsingle-dir "${GLMSINGLE_DIR}" \\
        --output-dir "${OUTPUT_DIR}"
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All beta_qc subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
