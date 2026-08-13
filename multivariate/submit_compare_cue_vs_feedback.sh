#!/bin/bash
# Submit the cue-vs-feedback beta comparison as a SLURM array job — one task per subject.
#
# Usage (from repo root):
#   bash multivariate/submit_compare_cue_vs_feedback.sh                # all subjects
#   bash multivariate/submit_compare_cue_vs_feedback.sh 01 02 03 05 06 # a first read
#
# Array rather than a single looping job (unlike submit_beta_qc_decoding.sh): each task
# decompresses ~185 MB of cue betas plus ~110 MB of feedback betas, so per-subject cost
# is I/O-dominated at a couple of minutes — enough that 59 of them serially is worth
# avoiding, and light enough in RAM to fan out freely.
#
# OVERWRITE=1 bash multivariate/submit_compare_cue_vs_feedback.sh   # force a rerun

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit for cluster
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
CUE_DIR="${CUE_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle}"
FEEDBACK_DIR="${FEEDBACK_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_feedback}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/cue_vs_feedback}"
VIS_MASK="${VIS_MASK:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"

EXTRA_ARGS=""
[ "${OVERWRITE:-0}" = "1" ] && EXTRA_ARGS="--overwrite"

# ---------------------------------------------------------------------------
# Prerequisite checks (fail early)
# ---------------------------------------------------------------------------
[ -d "$CUE_DIR" ]      || { echo "ERROR: cue betas dir not found: $CUE_DIR" >&2; exit 1; }
[ -d "$FEEDBACK_DIR" ] || { echo "ERROR: feedback betas dir not found: $FEEDBACK_DIR (run submit_glmsingle_feedback.sh first)" >&2; exit 1; }
if [ ! -f "$VIS_MASK" ]; then
    echo "ERROR: visual cortex mask not found: $VIS_MASK" >&2
    echo "       Build it: python multivariate/build_visual_cortex_mask.py --output-dir \$(dirname $VIS_MASK)" >&2
    exit 1
fi

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

N=$(grep -c . "$SUBJECTS_FILE" || true)
if [ "$N" -eq 0 ]; then
    echo "ERROR: subject list is empty" >&2
    exit 1
fi
echo "Submitting ${N} subjects from ${SUBJECTS_FILE} (array 1-${N}):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=cue_vs_fb
#SBATCH --output=${LOG_DIR}/cue_vs_feedback_%A_%a.out
#SBATCH --error=${LOG_DIR}/cue_vs_feedback_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --array=1-${N}

set -eo pipefail

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

SUBJECT=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS_FILE}")

echo "=== sub-\${SUBJECT}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/compare_cue_vs_feedback_betas.py" \\
    --subject "\$SUBJECT" \\
    --base-dir "${BASE_DIR}" \\
    --bids-dir "${BIDS_DIR}" \\
    --cue-dir "${CUE_DIR}" \\
    --feedback-dir "${FEEDBACK_DIR}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --roi-mask visualcortex "${VIS_MASK}" \\
    --save-maps ${EXTRA_ARGS}
EOF
