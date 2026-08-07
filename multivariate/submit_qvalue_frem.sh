#!/bin/bash
# Submit reward-value (Q-value) FREM regression as a SLURM array job — one job per
# subject. Regression counterpart of run_frem.py: fits a whole-brain FREMRegressor
# to a continuous target (default first_stim_value, the objective reward level)
# from the GLMsingle CUES betas, target sourced from the BBT (same table used for
# the SPM first-levels).
#
# Usage (from repo root):
#   bash multivariate/submit_qvalue_frem.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_qvalue_frem.sh 01 05 12   # specific subjects
#   BBT=/path/to/bbt.csv bash multivariate/submit_qvalue_frem.sh
#   TARGET_COL=first_stim_value_rl bash multivariate/submit_qvalue_frem.sh   # model RL Q instead

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/frem"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="/home/hfluhr/data/learninghabits/participants_mvpa.tsv"

# BBT target table (same one the SPM first-levels use). Override with BBT=...
BBT="${BBT:-/home/hfluhr/data/learninghabits/bbt.csv}"
TARGET_COL="${TARGET_COL:-first_stim_value}"

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

if [ ! -f "$BBT" ]; then
    echo "ERROR: BBT target table not found: ${BBT} (set BBT=/path/to/bbt.csv)" >&2
    exit 1
fi

echo "Submitting ${N} subjects (array 1-${N}), target='${TARGET_COL}', bbt='${BBT}':"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=qval_frem
#SBATCH --output=${LOG_DIR}/qvalue_frem_%A_%a.out
#SBATCH --error=${LOG_DIR}/qvalue_frem_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --array=1-${N}

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SUBJECT=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS_FILE}")
echo "=== sub-\${SUBJECT}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_qvalue_frem.py" \\
    --subject "\$SUBJECT" \\
    --bids-dir "${BIDS_DIR}" \\
    --glmsingle-dir "${GLMSINGLE_DIR}" \\
    --bbt "${BBT}" \\
    --target-col "${TARGET_COL}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --n-jobs \$SLURM_CPUS_PER_TASK
EOF
