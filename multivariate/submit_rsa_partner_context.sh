#!/bin/bash
# Submit the RSA partner-context pollution test as a SINGLE SLURM job that runs all
# subjects internally via `xargs -P` (same shape as submit_rsa_roi.sh/
# submit_beta_crossrun_reliability.sh -- per-subject cost is dominated by masking the
# 4D betas once; the per-stimulus 2-condition crossnobis calls are trivial).
#
# Usage (from repo root):
#   bash multivariate/submit_rsa_partner_context.sh                     # all subjects, scope=learning
#   SCOPE=test bash multivariate/submit_rsa_partner_context.sh          # all subjects, scope=test
#   SCOPE=test bash multivariate/submit_rsa_partner_context.sh 01       # single-subject pilot
#   NPROC=8 bash multivariate/submit_rsa_partner_context.sh             # override concurrency
#   OVERWRITE=1 bash multivariate/submit_rsa_partner_context.sh         # force rerun
#
# run_rsa_partner_context.py skips subjects that already have a results CSV (per
# scope), so this is resumable.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
DECODING_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/rsa_partner_context}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"
BBT="${BBT:-${BASE_DIR}/bbt.csv}"

MASK_DIR="${BASE_DIR}/masks/MNI152NLin2009cAsym"
VIS_MASK="${VIS_MASK:-${DECODING_DIR}/visual_cortex_mask.nii.gz}"
FUSIFORM_MASK="${FUSIFORM_MASK:-${MASK_DIR}/fusiform_mask_MNI152NLin2009cAsym.nii}"

SCOPE="${SCOPE:-learning}"
NPROC="${NPROC:-8}"

OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
SUBJECTS_FILE="${LOG_DIR}/subjects_partner_context_${SCOPE}.txt"

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

echo "Submitting 1 job for ${N} subjects, scope=${SCOPE} (NPROC=${NPROC} concurrent):"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=rsa_partner_${SCOPE}
#SBATCH --output=${LOG_DIR}/rsa_partner_${SCOPE}_%j.out
#SBATCH --error=${LOG_DIR}/rsa_partner_${SCOPE}_%j.err
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
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_rsa_partner_context.py" \\
        --subject "\$s" \\
        --scope "${SCOPE}" \\
        --base-dir "${BASE_DIR}" \\
        --bids-dir "${BIDS_DIR}" \\
        --glmsingle-dir "${GLMSINGLE_DIR}" \\
        --bbt "${BBT}" \\
        --output-dir "${OUTPUT_DIR}" \\
        --roi-mask visualcortex "${VIS_MASK}" \\
        --roi-mask fusiform "${FUSIFORM_MASK}" \\
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All rsa_partner_context subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
