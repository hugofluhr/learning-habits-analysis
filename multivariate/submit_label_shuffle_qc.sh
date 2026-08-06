#!/bin/bash
# Submit the label-shuffle decoding robustness check as a SINGLE SLURM job that
# runs all subjects internally via `xargs -P` (same pattern as
# submit_beta_qc_decoding.sh, not a 59-way array job).
#
# Negative-control / permutation-test sanity check: decode stimulus category
# from type-D betas with the true labels, then N times more with `stim_cat`
# shuffled, same masks (whole-brain + visual-cortex ROI) and CV as
# run_decoding.py. See run_label_shuffle_qc.py for details.
#
# Usage (from repo root):
#   bash multivariate/submit_label_shuffle_qc.sh              # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_label_shuffle_qc.sh 01 05 12      # specific subjects
#   NPROC=4 bash multivariate/submit_label_shuffle_qc.sh       # override concurrency
#   N_PERMUTATIONS=50 bash multivariate/submit_label_shuffle_qc.sh
#   OVERWRITE=1 bash multivariate/submit_label_shuffle_qc.sh   # force rerun of existing subjects
#
# Cost: run_beta_qc_decoding.py measured ~6s/fit (3 beta types x 2 masks in
# 35s). This script does 2 masks x (N_PERMUTATIONS+1) fits per subject with X
# transformed once per mask and reused — at N_PERMUTATIONS=100 that's roughly
# 2 x 101 x 6s ~= 20 min/subject. Same "single job, NPROC-way internal
# parallelism" reasoning as submit_beta_qc_decoding.sh: too short-lived and
# light (no memory pressure) to be worth per-task array scheduling overhead.
# run_label_shuffle_qc.py skips subjects already done, so this is resumable.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_qc"
VIS_MASK="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="/home/hfluhr/data/learninghabits/participants_mvpa.tsv"

# Concurrent subjects within the single job allocation.
NPROC="${NPROC:-8}"

# Label-shuffle iterations per subject/mask (see cost note above).
N_PERMUTATIONS="${N_PERMUTATIONS:-100}"

# Set OVERWRITE=1 to force a rerun of subjects that already have a QC CSV.
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
SUBJECTS_FILE="${LOG_DIR}/subjects_label_shuffle.txt"

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

echo "Submitting 1 job for ${N} subjects (NPROC=${NPROC} concurrent, N_PERMUTATIONS=${N_PERMUTATIONS}):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=label_shuffle_qc
#SBATCH --output=${LOG_DIR}/label_shuffle_qc_%j.out
#SBATCH --error=${LOG_DIR}/label_shuffle_qc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=16G
#SBATCH --time=04:00:00
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
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_label_shuffle_qc.py" \\
        --subject "\$s" \\
        --bids-dir "${BIDS_DIR}" \\
        --glmsingle-dir "${GLMSINGLE_DIR}" \\
        --output-dir "${OUTPUT_DIR}" \\
        --visual-cortex-mask "${VIS_MASK}" \\
        --n-permutations "${N_PERMUTATIONS}" \\
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All label_shuffle_qc subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
