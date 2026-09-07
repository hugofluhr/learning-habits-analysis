#!/bin/bash
# Submit second-stimulus category decoding as a SINGLE SLURM job that runs all
# subjects internally via `xargs -P` (same pattern as submit_frequency_decoding.sh).
#
# Uses the same 8 ROI masks as the RSA pipeline (submit_rsa_roi.sh) for direct comparison.
#
# Usage (from repo root):
#   bash multivariate/submit_stim2_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_stim2_decoding.sh 01 05 12    # specific subjects
#   NPROC=4 bash multivariate/submit_stim2_decoding.sh    # override concurrency
#   OVERWRITE=1 bash multivariate/submit_stim2_decoding.sh
#
# run_stim2_decoding.py skips subjects already done, so this is resumable.
# sub-46 has no BBT entry and will fail; expect n=58/59 on completion.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/stim2_decoding"
DECODING_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding"
MASK_DIR="${BASE_DIR}/masks/MNI152NLin2009cAsym"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"
BBT="${BBT:-${BASE_DIR}/bbt.csv}"

# ROI masks — same 8 as submit_rsa_roi.sh / submit_frequency_decoding.sh; whole-brain
# is added automatically
VIS_MASK="${VIS_MASK:-${DECODING_DIR}/visual_cortex_mask.nii.gz}"
FUSIFORM_MASK="${FUSIFORM_MASK:-${MASK_DIR}/fusiform_mask_MNI152NLin2009cAsym.nii}"
VMPFC_MASK="${VMPFC_MASK:-${MASK_DIR}/vmpfc_bartra2013_MNI152NLin2009cAsym.nii}"
STRIATUM_MASK="${STRIATUM_MASK:-${MASK_DIR}/striatum_bartra2013_MNI152NLin2009cAsym.nii}"
HABIT_MASK="${HABIT_MASK:-${MASK_DIR}/habit_Guida2022_MNI152NLin2009cAsym.nii}"
PUTAMEN_MASK="${PUTAMEN_MASK:-${MASK_DIR}/putamen_AAL_MNI152NLin2009cAsym.nii}"
PREMOTOR_MASK="${PREMOTOR_MASK:-${MASK_DIR}/premotor_HMAT_MNI152NLin2009cAsym.nii}"
PARIETAL_MASK="${PARIETAL_MASK:-${MASK_DIR}/parietal_AAL_MNI152NLin2009cAsym.nii}"

# Concurrent subjects within the single job allocation. A clean 8-subject real-batch
# test at NPROC=8 (job 5518511) completed successfully in 34.5 min -- no stall, just
# genuinely ~4.3 min/subject net (9 masks x 2 variants x 4-class LOGO-CV); the earlier
# 0/59-in-15min stall (job 5495879) was not reproduced and was most likely a one-off
# node-contention/scheduling artifact, not a systematic property of this NPROC. At
# NPROC=8, all 59 subjects would take ~4.3h -- over the 4h walltime -- so bumped
# moderately to 16 (not back to the earlier untested NPROC=24 guess) to fit comfortably.
NPROC="${NPROC:-16}"

# Set OVERWRITE=1 to force a rerun of subjects that already have a decoding CSV.
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"

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
for mask_var in VIS_MASK FUSIFORM_MASK VMPFC_MASK STRIATUM_MASK HABIT_MASK PUTAMEN_MASK PREMOTOR_MASK PARIETAL_MASK; do
    mask_path="${!mask_var}"
    if [ ! -f "$mask_path" ]; then
        echo "ERROR: ${mask_var} not found: ${mask_path}" >&2
        exit 1
    fi
done

echo "Submitting 1 job for ${N} subjects (NPROC=${NPROC} concurrent), bbt='${BBT}':"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=stim2_decode
#SBATCH --output=${LOG_DIR}/stim2_decoding_%j.out
#SBATCH --error=${LOG_DIR}/stim2_decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=36G
#SBATCH --time=4:00:00
#SBATCH --partition=standard

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_one() {
    local s="\$1"
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_stim2_decoding.py" \
        --subject "\$s" \
        --base-dir "${BASE_DIR}" \
        --bids-dir "${BIDS_DIR}" \
        --glmsingle-dir "${GLMSINGLE_DIR}" \
        --bbt "${BBT}" \
        --output-dir "${OUTPUT_DIR}" \
        --roi-mask visualcortex "${VIS_MASK}" \
        --roi-mask fusiform "${FUSIFORM_MASK}" \
        --roi-mask vmpfc "${VMPFC_MASK}" \
        --roi-mask striatum "${STRIATUM_MASK}" \
        --roi-mask habit "${HABIT_MASK}" \
        --roi-mask putamen "${PUTAMEN_MASK}" \
        --roi-mask premotor "${PREMOTOR_MASK}" \
        --roi-mask parietal "${PARIETAL_MASK}" \
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All stim2_decoding subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
