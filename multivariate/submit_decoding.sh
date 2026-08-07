#!/bin/bash
# Submit stimulus category decoding as a SINGLE SLURM job that runs all
# subjects internally via `xargs -P` (not a per-subject array job — see below).
#
# Usage (from repo root):
#   bash multivariate/submit_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_decoding.sh 01 05 12   # specific subjects
#   NPROC=4 bash multivariate/submit_decoding.sh    # override concurrency
#   OVERWRITE=1 bash multivariate/submit_decoding.sh # force rerun of existing subjects
#
# Why one job instead of an array: measured on a compute node (3-subject
# smoke test, job 4388817) — ~20-26s wall time, ~1.2GB peak RSS per subject,
# single core (LinearSVC/liblinear doesn't thread). Same reasoning as
# submit_beta_qc_decoding.sh: this is far too light and short-lived to be
# worth a separate array-task scheduling/queueing overhead per subject. A
# single job with NPROC-way internal parallelism finishes all subjects in a
# couple of minutes instead of 59 separate job slots.
# run_decoding.py skips subjects already done, so this is resumable if the
# job is killed partway through.
#
# Prerequisites: visual_cortex_mask.nii.gz must exist in OUTPUT_DIR.
# Build it locally with:
#   python multivariate/build_visual_cortex_mask.py --output-dir <local_path>
# then scp to OUTPUT_DIR on the cluster.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding"
VIS_MASK="${OUTPUT_DIR}/visual_cortex_mask.nii.gz"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"

# Concurrent subjects within the single job allocation. Cluster's `standard`
# partition nodes have >=8 cores; 8 is a safe default that schedules easily
# and keeps every worker single-threaded (no BLAS thread contention).
NPROC="${NPROC:-8}"

# Set OVERWRITE=1 to force a rerun of subjects that already have a decoding
# CSV (e.g. after a change to run_decoding.py's output schema).
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"

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

if [ ! -f "$VIS_MASK" ]; then
    echo "ERROR: visual cortex mask not found: ${VIS_MASK}" >&2
    echo "       Build locally: python multivariate/build_visual_cortex_mask.py --output-dir <path>" >&2
    echo "       Then scp to: ${VIS_MASK}" >&2
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
#SBATCH --job-name=decoding
#SBATCH --output=${LOG_DIR}/decoding_%j.out
#SBATCH --error=${LOG_DIR}/decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=8G
#SBATCH --time=10:00
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
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_decoding.py" \\
        --subject "\$s" \\
        --base-dir "${BASE_DIR}" \\
        --bids-dir "${BIDS_DIR}" \\
        --glmsingle-dir "${GLMSINGLE_DIR}" \\
        --output-dir "${OUTPUT_DIR}" \\
        --visual-cortex-mask "${VIS_MASK}" \\
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All decoding subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
