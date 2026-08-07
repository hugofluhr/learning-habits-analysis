#!/bin/bash
# Submit reward-value (Q-value) whole-brain + ROI decoding as a SINGLE SLURM job that
# runs all subjects internally via `xargs -P` (not a per-subject array job — see
# submit_decoding.sh, whose pattern this mirrors). Measured via a sub-01 srun smoke test
# (job 4390771, 4 cores, all 4 masks incl. wholebrain's 65,617 voxels): ~5s wall, well
# under a single CPU core (RidgeCV's internal alpha selection uses sklearn's efficient
# built-in generalized-CV shortcut, not brute-force refitting per alpha) — same
# performance class as submit_decoding.sh's category decoder. --mem/--time below still
# carry a safety margin over that single-subject measurement for the full-cohort run.
#
# Usage (from repo root):
#   bash multivariate/submit_qvalue_decoding.sh            # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_qvalue_decoding.sh 01 05 12   # specific subjects
#   NPROC=4 bash multivariate/submit_qvalue_decoding.sh    # override concurrency
#   OVERWRITE=1 bash multivariate/submit_qvalue_decoding.sh # force rerun of existing subjects
#   BBT=/path/to/bbt.csv TARGET_COL=first_stim_value_rl bash multivariate/submit_qvalue_decoding.sh
#
# Prerequisites: visual_cortex_mask.nii.gz must exist in OUTPUT_DIR (built already on
# the cluster by build_visual_cortex_mask.py, shared with the category decoder); the
# vmPFC/striatum Bartra masks must exist under the shared masks/ directory.
#
# run_qvalue_decoding.py skips subjects already done, so this is resumable if the job
# is killed partway through.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
OUTPUT_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"

# BBT target table (same one the SPM first-levels use). Override with BBT=...
BBT="${BBT:-${BASE_DIR}/bbt.csv}"
TARGET_COL="${TARGET_COL:-first_stim_value}"

# ROI masks (whole-brain is added automatically by run_qvalue_decoding.py from the
# subject's own functional brain mask). Override any of these with e.g. VMPFC_MASK=...
VIS_MASK="${VIS_MASK:-${OUTPUT_DIR}/visual_cortex_mask.nii.gz}"
VMPFC_MASK="${VMPFC_MASK:-${BASE_DIR}/masks/MNI152NLin2009cAsym/vmpfc_bartra2013_MNI152NLin2009cAsym.nii}"
STRIATUM_MASK="${STRIATUM_MASK:-${BASE_DIR}/masks/MNI152NLin2009cAsym/striatum_bartra2013_MNI152NLin2009cAsym.nii}"

# Concurrent subjects within the single job allocation.
NPROC="${NPROC:-8}"

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
for mask_var in VIS_MASK VMPFC_MASK STRIATUM_MASK; do
    mask_path="${!mask_var}"
    if [ ! -f "$mask_path" ]; then
        echo "ERROR: ${mask_var} not found: ${mask_path}" >&2
        exit 1
    fi
done

echo "Submitting 1 job for ${N} subjects (NPROC=${NPROC} concurrent), target='${TARGET_COL}', bbt='${BBT}':"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=qval_decoding
#SBATCH --output=${LOG_DIR}/qvalue_decoding_%j.out
#SBATCH --error=${LOG_DIR}/qvalue_decoding_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=12G
#SBATCH --time=20:00
#SBATCH --partition=standard

set -eo pipefail

# Pin every worker to 1 thread — with NPROC subjects running concurrently, letting
# each spawn its own BLAS threads would oversubscribe the cpus.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_one() {
    local s="\$1"
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_qvalue_decoding.py" \
        --subject "\$s" \
        --base-dir "${BASE_DIR}" \
        --bids-dir "${BIDS_DIR}" \
        --glmsingle-dir "${GLMSINGLE_DIR}" \
        --bbt "${BBT}" \
        --target-col "${TARGET_COL}" \
        --output-dir "${OUTPUT_DIR}" \
        --roi-mask visualcortex "${VIS_MASK}" \
        --roi-mask vmpfc "${VMPFC_MASK}" \
        --roi-mask striatum "${STRIATUM_MASK}" \
        ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All qvalue_decoding subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
