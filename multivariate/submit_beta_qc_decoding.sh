#!/bin/bash
# Submit GLMsingle beta-version QC decoding as a SLURM array job — one job per
# subject. Runs the same standardized whole-brain decoder on each GLMsingle model
# type (A/B/C/D) for both targets (category probe + reward), writing one tidy CSV
# per subject. Use it to check whether the denoising/ridge steps improve
# decodability on this dataset.
#
# Usage (from repo root):
#   bash multivariate/submit_beta_qc_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_beta_qc_decoding.sh 01 05 12   # specific subjects
#   BBT=/path/to/bbt.csv TARGETS=reward bash multivariate/submit_beta_qc_decoding.sh
#
# Higher --mem than the other decoders: each of TYPE{A,B,C,D}.npy is a full-brain
# betas array loaded one at a time.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_qc"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="/home/hfluhr/data/learninghabits/participants_mvpa.tsv"

BBT="${BBT:-/home/hfluhr/data/learninghabits/bbt.csv}"
TARGET_COL="${TARGET_COL:-first_stim_value}"
TARGETS="${TARGETS:-category,reward}"

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

if [[ "$TARGETS" == *reward* ]] && [ ! -f "$BBT" ]; then
    echo "ERROR: BBT target table not found: ${BBT} (set BBT=/path/to/bbt.csv)" >&2
    exit 1
fi

echo "Submitting ${N} subjects (array 1-${N}), targets='${TARGETS}', bbt='${BBT}':"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=beta_qc
#SBATCH --output=${LOG_DIR}/beta_qc_%A_%a.out
#SBATCH --error=${LOG_DIR}/beta_qc_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --array=1-${N}

set -eo pipefail

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

SUBJECT=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SUBJECTS_FILE}")
echo "=== sub-\${SUBJECT}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_beta_qc_decoding.py" \\
    --subject "\$SUBJECT" \\
    --bids-dir "${BIDS_DIR}" \\
    --glmsingle-dir "${GLMSINGLE_DIR}" \\
    --bbt "${BBT}" \\
    --target-col "${TARGET_COL}" \\
    --targets "${TARGETS}" \\
    --output-dir "${OUTPUT_DIR}"
EOF
