#!/bin/bash
# Submit GLMsingle estimation as a SLURM array job — one job per subject.
#
# Usage (from repo root):
#   bash multivariate/submit_glmsingle.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_glmsingle.sh 01 05 12   # specific subjects
#
# Adjust BASE_DIR / BIDS_DIR / OUTPUT_DIR for the cluster file system.
# The subject list is written to a temp file so the array index maps cleanly
# even when subject IDs have gaps.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit for cluster
# ---------------------------------------------------------------------------
BASE_DIR="/mnt/data/learning-habits"
BIDS_DIR="${BASE_DIR}/bids_dataset/derivatives/fmriprep-24.0.1-noSDC"
OUTPUT_DIR="${BASE_DIR}/bids_dataset/derivatives/glmsingle"
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
#!/bin/bash
#SBATCH --job-name=glmsingle
#SBATCH --output=${LOG_DIR}/glmsingle_%A_%a.out
#SBATCH --error=${LOG_DIR}/glmsingle_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --array=1-${N}

# Cap threaded libs to the allocated CPUs
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MALLOC_ARENA_MAX=2

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate neuroim   # adjust to your cluster env name

SUBJECT=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS_FILE}")

echo "=== sub-\${SUBJECT}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

python "${REPO}/multivariate/run_glmsingle.py" \\
    --subject "\$SUBJECT" \\
    --base-dir "${BASE_DIR}" \\
    --bids-dir "${BIDS_DIR}" \\
    --output-dir "${OUTPUT_DIR}"
EOF
