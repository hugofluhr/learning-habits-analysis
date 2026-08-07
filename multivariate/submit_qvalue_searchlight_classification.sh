#!/bin/bash
# Submit reward high/low classification searchlight as a SLURM array job — one job per
# subject. Searchlight counterpart of submit_qvalue_classification.sh (not of
# submit_qvalue_searchlight.sh, which drives the untested regression draft): per-subject
# array job, not a single xargs -P job, because SearchLight.fit() over the whole brain is
# the expensive pipeline here (like submit_searchlight.sh), unlike the whole-ROI decoders'
# ~seconds-per-subject cost.
#
# Usage (from repo root):
#   bash multivariate/submit_qvalue_searchlight_classification.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_qvalue_searchlight_classification.sh 01 05 12   # specific subjects
#   BBT=/path/to/bbt.csv bash multivariate/submit_qvalue_searchlight_classification.sh
#   TARGET_COL=first_stim_value_rl bash multivariate/submit_qvalue_searchlight_classification.sh
#   LOW_MAX=1 HIGH_MIN=5 bash multivariate/submit_qvalue_searchlight_classification.sh  # stricter extremes-only split
#
# run_qvalue_searchlight_classification.py skips subjects already done, so this is
# resumable if the job is killed partway through. sub-46 has no BBT entry (known,
# unresolved gap) and will fail that one array task; expect n=58/59 on completion.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/searchlight"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"

# BBT target table (same one the SPM first-levels use). Override with BBT=...
BBT="${BBT:-${BASE_DIR}/bbt.csv}"
TARGET_COL="${TARGET_COL:-first_stim_value}"
LOW_MAX="${LOW_MAX:-2}"
HIGH_MIN="${HIGH_MIN:-4}"

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

echo "Submitting ${N} subjects (array 1-${N}), target='${TARGET_COL}' (low<=${LOW_MAX}, high>=${HIGH_MIN}), bbt='${BBT}':"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=qval_sl_clf
#SBATCH --output=${LOG_DIR}/qvalue_searchlight_classification_%A_%a.out
#SBATCH --error=${LOG_DIR}/qvalue_searchlight_classification_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
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

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_qvalue_searchlight_classification.py" \\
    --subject "\$SUBJECT" \\
    --bids-dir "${BIDS_DIR}" \\
    --glmsingle-dir "${GLMSINGLE_DIR}" \\
    --bbt "${BBT}" \\
    --target-col "${TARGET_COL}" \\
    --low-max "${LOW_MAX}" \\
    --high-min "${HIGH_MIN}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --n-jobs \$SLURM_CPUS_PER_TASK
EOF
