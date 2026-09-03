#!/bin/bash
# Submit GLMsingle cue-beta cross-run reliability as a SINGLE SLURM job that runs
# all subjects internally via `xargs -P` (same pattern as submit_beta_qc_decoding.sh).
# Per stimulus, per mask (wholebrain/visualcortex/fusiform), correlates the mean beta
# pattern between each pair of runs (learning1/learning2/test) — finishes
# glmsingle_qc.ipynb §8, which was written but never run to completion / persisted.
#
# Usage (from repo root):
#   bash multivariate/submit_beta_crossrun_reliability.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_beta_crossrun_reliability.sh 01 05 12   # specific subjects
#   NPROC=8 bash multivariate/submit_beta_crossrun_reliability.sh    # override concurrency
#   OVERWRITE=1 bash multivariate/submit_beta_crossrun_reliability.sh # force rerun

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_qc"
DECODING_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding"
MASK_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/masks"
VIS_MASK="${VIS_MASK:-${DECODING_DIR}/visual_cortex_mask.nii.gz}"
FUSIFORM_MASK="${FUSIFORM_MASK:-${MASK_DIR}/fusiform_mask_MNI152NLin2009cAsym.nii}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="/home/hfluhr/data/learninghabits/participants_mvpa.tsv"

# Concurrent subjects within the single job allocation.
NPROC="${NPROC:-8}"

OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
SUBJECTS_FILE="${LOG_DIR}/subjects_crossrun.txt"

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

for mask_var in VIS_MASK FUSIFORM_MASK; do
    path="${!mask_var}"
    if [ ! -f "$path" ]; then
        echo "ERROR: ${mask_var} not found: $path" >&2
        exit 1
    fi
done

echo "Submitting 1 job for ${N} subjects (NPROC=${NPROC} concurrent):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=crossrun_rel
#SBATCH --output=${LOG_DIR}/crossrun_rel_%j.out
#SBATCH --error=${LOG_DIR}/crossrun_rel_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --partition=standard

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_one() {
    local s="\$1"
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_beta_crossrun_reliability.py" \\
        --subject "\$s" \\
        --bids-dir "${BIDS_DIR}" \\
        --glmsingle-dir "${GLMSINGLE_DIR}" \\
        --output-dir "${OUTPUT_DIR}" \\
        --visual-cortex-mask "${VIS_MASK}" \\
        --fusiform-mask "${FUSIFORM_MASK}" \\
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All crossrun_reliability subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
