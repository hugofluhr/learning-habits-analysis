#!/bin/bash
# Submit CV-scheme comparison decoding as a SINGLE SLURM job that runs all
# subjects internally via `xargs -P` (not a 59-way array job — see below).
#
# Compares the production between-run LeaveOneGroupOut CV (3 runs) against a
# within-run StratifiedKFold(3) that ignores run boundaries entirely, on the
# *same* category-decoding validation probe (type-D betas, LinearSVC,
# standardize=True), whole-brain + visual-cortex ROI. Purpose: check whether
# the between-run design is empirically protecting against real temporal
# leakage, or whether the two schemes give statistically indistinguishable
# accuracy for this probe. See run_cv_comparison_decoding.py's docstring.
#
# Usage (from repo root):
#   bash multivariate/submit_cv_comparison_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_cv_comparison_decoding.sh 01 05 12   # specific subjects
#   NPROC=4 bash multivariate/submit_cv_comparison_decoding.sh    # override concurrency
#   OVERWRITE=1 bash multivariate/submit_cv_comparison_decoding.sh # force rerun of existing subjects
#
# Same per-subject cost profile as submit_beta_qc_decoding.sh (light, CPU-bound,
# ~35-45s, <1GB) — one job with NPROC-way internal parallelism, not an array job.
# run_cv_comparison_decoding.py skips subjects already done, so this is
# resumable if the job is killed partway through.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/cv_comparison"
VIS_MASK="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="/home/hfluhr/data/learninghabits/participants_mvpa.tsv"

# Concurrent subjects within the single job allocation.
NPROC="${NPROC:-8}"

# Set OVERWRITE=1 to force a rerun of subjects that already have output.
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
    echo "ERROR: visual cortex mask not found: $VIS_MASK" >&2
    echo "       Build it: python multivariate/build_visual_cortex_mask.py --output-dir <decoding-dir>" >&2
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
#SBATCH --job-name=cv_comparison
#SBATCH --output=${LOG_DIR}/cv_comparison_%j.out
#SBATCH --error=${LOG_DIR}/cv_comparison_%j.err
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
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_cv_comparison_decoding.py" \\
        --subject "\$s" \\
        --bids-dir "${BIDS_DIR}" \\
        --glmsingle-dir "${GLMSINGLE_DIR}" \\
        --output-dir "${OUTPUT_DIR}" \\
        --visual-cortex-mask "${VIS_MASK}" \\
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All cv_comparison subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
