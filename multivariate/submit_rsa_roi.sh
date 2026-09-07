#!/bin/bash
# Submit ROI crossnobis RSA as a SINGLE SLURM job that runs all subjects internally
# via `xargs -P` (same shape as submit_qvalue_classification.sh — the per-subject cost
# is dominated by masking the 4D betas once, the RDM algebra itself is trivial: 8
# conditions, a handful of masks).
#
# Usage (from repo root):
#   bash multivariate/submit_rsa_roi.sh                 # all subjects in PARTICIPANTS_TSV
#   bash multivariate/submit_rsa_roi.sh 01 05 12        # specific subjects
#   NPROC=4 bash multivariate/submit_rsa_roi.sh         # override concurrency
#   OVERWRITE=1 bash multivariate/submit_rsa_roi.sh     # force rerun of existing subjects
#   SPLIT=blocked bash multivariate/submit_rsa_roi.sh   # blocked within-run folds
#
# Negative control (writes to a separate tree so it can never be mistaken for the
# real thing — run_rsa_roi.py permutes stimulus labels within run):
#   SHUFFLE_SEED=1 OUTPUT_DIR=.../derivatives/rsa_shuffled bash multivariate/submit_rsa_roi.sh
#
# Symmetric stim-2 model (adds s2_category/s2_frequency/s2_identity — see
# run_rsa_roi.py --symmetric docstring). Separate output tree, same as SHUFFLE/REMOVE_MEAN:
#   SYMMETRIC=1 OUTPUT_DIR=.../derivatives/rsa_symmetric bash multivariate/submit_rsa_roi.sh
#
# Prerequisites: visual_cortex_mask.nii.gz in the decoding output dir (shared with the
# category decoder); the Bartra vmPFC/striatum masks and the fusiform mask under the
# shared masks/ directory; rsatoolbox in the conda env only if you pass
# --validate-against-rsatoolbox (this submitter does not).
#
# run_rsa_roi.py skips subjects that already have a results CSV, so this is resumable.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
GLMSINGLE_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle"
DECODING_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/rsa}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

PARTICIPANTS_TSV="${BASE_DIR}/participants_mvpa.tsv"
BBT="${BBT:-${BASE_DIR}/bbt.csv}"

MASK_DIR="${BASE_DIR}/masks/MNI152NLin2009cAsym"
VIS_MASK="${VIS_MASK:-${DECODING_DIR}/visual_cortex_mask.nii.gz}"
VMPFC_MASK="${VMPFC_MASK:-${MASK_DIR}/vmpfc_bartra2013_MNI152NLin2009cAsym.nii}"
STRIATUM_MASK="${STRIATUM_MASK:-${MASK_DIR}/striatum_bartra2013_MNI152NLin2009cAsym.nii}"
FUSIFORM_MASK="${FUSIFORM_MASK:-${MASK_DIR}/fusiform_mask_MNI152NLin2009cAsym.nii}"
HABIT_MASK="${HABIT_MASK:-${MASK_DIR}/habit_Guida2022_MNI152NLin2009cAsym.nii}"
PUTAMEN_MASK="${PUTAMEN_MASK:-${MASK_DIR}/putamen_AAL_MNI152NLin2009cAsym.nii}"
PREMOTOR_MASK="${PREMOTOR_MASK:-${MASK_DIR}/premotor_HMAT_MNI152NLin2009cAsym.nii}"
PARIETAL_MASK="${PARIETAL_MASK:-${MASK_DIR}/parietal_AAL_MNI152NLin2009cAsym.nii}"

NPROC="${NPROC:-8}"
SPLIT="${SPLIT:-interleaved}"

OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = "1" ] && OVERWRITE_FLAG="--overwrite"

SHUFFLE_FLAG=""
if [ -n "${SHUFFLE_SEED:-}" ]; then
    SHUFFLE_FLAG="--shuffle-seed ${SHUFFLE_SEED}"
    case "$OUTPUT_DIR" in
        *shuffl*) ;;
        *) echo "ERROR: SHUFFLE_SEED set but OUTPUT_DIR ('${OUTPUT_DIR}') does not look" >&2
           echo "       like a shuffled-control tree. Refusing to overwrite real results." >&2
           exit 1 ;;
    esac
fi

# REMOVE_MEAN=1: amplitude-confound control (see run_rsa_roi.py --remove-mean).
# Same guardrail idea as the shuffle: never write a variant into the main rsa tree.
REMOVE_MEAN_FLAG=""
if [ "${REMOVE_MEAN:-0}" = "1" ]; then
    REMOVE_MEAN_FLAG="--remove-mean"
    case "$OUTPUT_DIR" in
        *remove_mean*|*removemean*) ;;
        *) echo "ERROR: REMOVE_MEAN=1 but OUTPUT_DIR ('${OUTPUT_DIR}') does not look" >&2
           echo "       like a remove-mean tree. Refusing to overwrite real results." >&2
           exit 1 ;;
    esac
fi

# SYMMETRIC=1: adds stim-2 predictors (see run_rsa_roi.py --symmetric). Same
# guardrail idea: never write a variant into the main rsa tree.
SYMMETRIC_FLAG=""
if [ "${SYMMETRIC:-0}" = "1" ]; then
    SYMMETRIC_FLAG="--symmetric"
    case "$OUTPUT_DIR" in
        *symmetric*) ;;
        *) echo "ERROR: SYMMETRIC=1 but OUTPUT_DIR ('${OUTPUT_DIR}') does not look" >&2
           echo "       like a symmetric-model tree. Refusing to overwrite real results." >&2
           exit 1 ;;
    esac
fi

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
    echo "ERROR: BBT not found: ${BBT} (set BBT=/path/to/bbt.csv)" >&2
    exit 1
fi
for mask_var in VIS_MASK VMPFC_MASK STRIATUM_MASK FUSIFORM_MASK HABIT_MASK PUTAMEN_MASK PREMOTOR_MASK PARIETAL_MASK; do
    mask_path="${!mask_var}"
    if [ ! -f "$mask_path" ]; then
        echo "ERROR: ${mask_var} not found: ${mask_path}" >&2
        exit 1
    fi
done

echo "Submitting 1 job for ${N} subjects (NPROC=${NPROC} concurrent)"
echo "  output    : ${OUTPUT_DIR}"
echo "  bbt       : ${BBT}"
echo "  split     : ${SPLIT}${SHUFFLE_FLAG:+   [SHUFFLE CONTROL: seed ${SHUFFLE_SEED}]}${REMOVE_MEAN_FLAG:+   [REMOVE-MEAN control]}${SYMMETRIC_FLAG:+   [SYMMETRIC stim-2 model]}"
cat "$SUBJECTS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission — one job, NPROC-way internal parallelism via xargs
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=rsa_roi
#SBATCH --output=${LOG_DIR}/rsa_roi_%j.out
#SBATCH --error=${LOG_DIR}/rsa_roi_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NPROC}
#SBATCH --mem=24G
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
    /home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_rsa_roi.py" \
        --subject "\$s" \
        --base-dir "${BASE_DIR}" \
        --bids-dir "${BIDS_DIR}" \
        --glmsingle-dir "${GLMSINGLE_DIR}" \
        --bbt "${BBT}" \
        --output-dir "${OUTPUT_DIR}" \
        --within-run-split "${SPLIT}" \
        --roi-mask visualcortex "${VIS_MASK}" \
        --roi-mask fusiform "${FUSIFORM_MASK}" \
        --roi-mask vmpfc "${VMPFC_MASK}" \
        --roi-mask striatum "${STRIATUM_MASK}" \
        --roi-mask habit "${HABIT_MASK}" \
        --roi-mask putamen "${PUTAMEN_MASK}" \
        --roi-mask premotor "${PREMOTOR_MASK}" \
        --roi-mask parietal "${PARIETAL_MASK}" \
        ${SHUFFLE_FLAG} ${REMOVE_MEAN_FLAG} ${SYMMETRIC_FLAG} ${OVERWRITE_FLAG}
}
export -f run_one

xargs -a "${SUBJECTS_FILE}" -P ${NPROC} -I{} bash -c 'run_one "\$@"' _ {}

echo "All rsa_roi subjects finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
EOF
