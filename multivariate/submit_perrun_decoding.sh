#!/bin/bash
# Submit per-run vs combined-run stimulus category decoding as a SINGLE SLURM
# job that runs all subjects internally via `xargs -P` (not a per-subject
# array job — see below).
#
# Usage (from repo root):
#   bash multivariate/submit_perrun_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_perrun_decoding.sh 01 05 12   # specific subjects
#   NPROC=4 bash multivariate/submit_perrun_decoding.sh    # override concurrency
#   OVERWRITE=1 bash multivariate/submit_perrun_decoding.sh # force rerun of existing subjects
#
# Compares the production between-run LeaveOneGroupOut CV (`combined_logo`,
# reproduced unchanged) against 3 within-run repeated-StratifiedKFold schemes
# (`run-learning1`, `run-learning2`, `run-test`), on the same category-decoding
# probe (type-D betas, LinearSVC, standardize=True), whole-brain + visual-cortex
# ROI. See run_perrun_decoding.py's docstring.
#
# Why one job instead of an array: same reasoning as submit_decoding.sh (light,
# short-lived per-subject work not worth array-task overhead), but budget more
# wall time here — the 3 within-run schemes add ~25x more LinearSVC fits per
# subject (5 folds x 10 repeats x 3 runs = 150 extra fits vs LOGO's 3),
# partly offset by much smaller per-run training sets. Estimated ~90-100s/subject
# vs run_decoding.py's ~20-26s; recalibrate with a smoke test (--qos=debug,
# 1-2 subjects) before trusting this budget for the full 59-subject run.
# run_perrun_decoding.py skips subjects already done, so this is resumable if
# the job is killed partway through.
#
# Prerequisites: visual_cortex_mask.nii.gz must already exist under the
# production decoding output dir (built once for run_decoding.py).

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/perrun_decoding"
# Reuse the existing production visual-cortex mask directly (read-only) —
# no need to build/copy a separate one for this experiment.
VIS_MASK="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"

# Concurrent subjects within the single job allocation. Cluster's `standard`
# partition nodes have >=8 cores; 8 is a safe default that schedules easily
# and keeps every worker single-threaded (no BLAS thread contention).
NPROC="${NPROC:-8}"

# Set OVERWRITE=1 to force a rerun of subjects that already have a decoding
# CSV (e.g. after a change to run_perrun_decoding.py's output schema).
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
#SBATCH --job-name=perrun_decoding
#SBATCH --output=${LOG_DIR}/perrun_decoding_%j.out
#SBATCH --error=${LOG_DIR}/perrun_decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=8G
#SBATCH --time=45:00
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
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_perrun_decoding.py" \\
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

echo "All perrun_decoding subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
