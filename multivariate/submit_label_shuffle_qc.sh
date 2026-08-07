#!/bin/bash
# Submit the label-shuffle decoding robustness check as a SLURM array job —
# one task per subject (like submit_decoding.sh), throttled with `%THROTTLE`
# so it doesn't hog the standard partition.
#
# Negative-control / permutation-test sanity check: decode stimulus category
# from type-D betas with the true labels, then N times more with `stim_cat`
# shuffled, same masks (whole-brain + visual-cortex ROI) and CV as
# run_decoding.py. See run_label_shuffle_qc.py for details.
#
# Usage (from repo root):
#   bash multivariate/submit_label_shuffle_qc.sh              # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_label_shuffle_qc.sh 01 05 12      # specific subjects
#   THROTTLE=10 bash multivariate/submit_label_shuffle_qc.sh   # override max concurrent tasks
#   N_PERMUTATIONS=50 bash multivariate/submit_label_shuffle_qc.sh
#   OVERWRITE=1 bash multivariate/submit_label_shuffle_qc.sh   # force rerun of existing subjects
#
# Cost, measured on the cluster (sub-01, standardize=True, 100 permutations,
# both masks): 13m47s wall, single-threaded (LinearSVC/liblinear doesn't
# thread), well under 1GB RSS (see run_beta_qc_decoding.py's measurement for
# the same decoder). Originally this was a single job looping subjects via
# `xargs -P` (same shape as submit_beta_qc_decoding.sh) — wrong choice here:
# that pattern only pays off when per-subject cost is light enough that
# array-task scheduling overhead matters (beta_qc: ~35-45s/subject); at
# ~14min/subject the array job's genuine per-subject concurrency dominates.
# `%THROTTLE` (default 20) caps concurrent tasks so this doesn't monopolize
# shared partition capacity from other users' jobs.
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

# Max concurrent array tasks — a courtesy cap, not a hard cluster limit (no
# per-user MaxJobs/MaxSubmit QOS limit is set for hfluhr as of writing).
THROTTLE="${THROTTLE:-20}"

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

echo "Submitting ${N} subjects (array 1-${N}%${THROTTLE}, N_PERMUTATIONS=${N_PERMUTATIONS}):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one array task per subject
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=label_shuffle_qc
#SBATCH --output=${LOG_DIR}/label_shuffle_qc_%A_%a.out
#SBATCH --error=${LOG_DIR}/label_shuffle_qc_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30:00
#SBATCH --partition=standard
#SBATCH --array=1-${N}%${THROTTLE}

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SUBJECT=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS_FILE}")
echo "=== sub-\${SUBJECT}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_label_shuffle_qc.py" \\
    --subject "\$SUBJECT" \\
    --bids-dir "${BIDS_DIR}" \\
    --glmsingle-dir "${GLMSINGLE_DIR}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --visual-cortex-mask "${VIS_MASK}" \\
    --n-permutations "${N_PERMUTATIONS}" \\
    ${OVERWRITE_FLAG}
EOF
