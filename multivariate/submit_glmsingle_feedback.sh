#!/bin/bash
# Submit feedback-locked GLMsingle estimation as a SLURM array job — one job per subject.
#
# Usage (from repo root):
#   bash multivariate/submit_glmsingle_feedback.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_glmsingle_feedback.sh 01 05 12   # specific subjects
#
# Same shape as submit_glmsingle.sh; separate OUTPUT_DIR so the feedback-locked betas
# never overwrite the cue-locked ones. Walltime is kept at 1h even though only 2 of the
# 3 runs are fitted — GLMsingle's cost is dominated by the per-voxel HRF/fracridge
# sweeps, and the learning runs are the short ones anyway (~426 vols vs the test run's
# ~593), so the saving is modest and not worth cutting the margin for.
#
# Adjust BASE_DIR / BIDS_DIR / OUTPUT_DIR for the cluster file system.
# The subject list is written to a temp file so the array index maps cleanly
# even when subject IDs have gaps.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit for cluster
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_feedback}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

# Single source of truth for the analysis sample (excludes motion/QC failures)
PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

SUBJECTS_FILE="${LOG_DIR}/subjects.txt"

if [ "$#" -gt 0 ]; then
    # Subjects passed on the command line
    printf "%s\n" "$@" > "$SUBJECTS_FILE"
else
    # Use the canonical participants list
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
echo "Submitting ${N} subjects from ${SUBJECTS_FILE} (array 1-${N}):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=glmsingle_fb
#SBATCH --output=${LOG_DIR}/glmsingle_feedback_%A_%a.out
#SBATCH --error=${LOG_DIR}/glmsingle_feedback_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=1-${N}

set -eo pipefail

# Cap threaded libs to the allocated CPUs
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

SUBJECT=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS_FILE}")

echo "=== sub-\${SUBJECT}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_glmsingle_feedback.py" \\
    --subject "\$SUBJECT" \\
    --base-dir "${BASE_DIR}" \\
    --bids-dir "${BIDS_DIR}" \\
    --output-dir "${OUTPUT_DIR}"
EOF
