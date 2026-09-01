#!/bin/bash
# Submit RSA searchlight as a SLURM array job — one job per subject.
# Each subject gets a full 5-term RSA regression at every voxel (6mm radius),
# producing 5 beta maps (category, value, frequency, second_stim_value,
# choice_rate). Same sphere radius and parallelism as the frequency searchlight.
#
# Usage (from repo root):
#   bash multivariate/submit_rsa_searchlight.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_rsa_searchlight.sh 01 05 12   # specific subjects
#   BBT=/path/to/bbt.csv bash multivariate/submit_rsa_searchlight.sh
#
# run_rsa_searchlight.py skips subjects already done, so this is resumable.
# sub-46 has no BBT entry and will fail that one array task; expect n=58/59.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/rsa_searchlight}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"
BBT="${BBT:-${BASE_DIR}/bbt.csv}"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
SUBJECTS_FILE="${LOG_DIR}/subjects_rsa_searchlight.txt"

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

echo "Submitting ${N} subjects (array 1-${N}), bbt='${BBT}':"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
# RSA searchlight is heavier than decoding searchlight: crossnobis + regression
# per sphere instead of SVM.fit + cross_val_score.  Budget 45 min (decoding was
# 30 min); memory 16G should suffice (same data footprint, no SVM overhead).
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=rsa_sl
#SBATCH --output=${LOG_DIR}/rsa_searchlight_%A_%a.out
#SBATCH --error=${LOG_DIR}/rsa_searchlight_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:45:00
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

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_rsa_searchlight.py" \\
    --subject "\$SUBJECT" \\
    --base-dir "${BASE_DIR}" \\
    --bids-dir "${BIDS_DIR}" \\
    --glmsingle-dir "${GLMSINGLE_DIR}" \\
    --bbt "${BBT}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --n-jobs \$SLURM_CPUS_PER_TASK
EOF
