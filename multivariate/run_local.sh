#!/bin/bash
# Local parallel driver for the multivariate pipelines on a single machine
# (replaces the SLURM submit_*.sh array jobs when no scheduler is available,
#  e.g. running on a standalone uzh.vm).
#
# Usage (from repo root):
#   bash multivariate/run_local.sh glmsingle                 # all subjects
#   bash multivariate/run_local.sh searchlight 01 05 12      # specific subjects
#   NPROC=8 THREADS=3 bash multivariate/run_local.sh glmsingle   # override defaults
#
# Pipelines:  glmsingle | searchlight | decoding | frem
#
# Why the per-pipeline defaults differ (this is the whole point of the file):
#   glmsingle    -> MEMORY-bound. Loads 3 BOLD runs as float32 + GLMdenoise/ridge
#                   copies (cluster allocated 32 GB/subject). Wants a few BLAS
#                   threads. -> few subjects at once, several threads each.
#   searchlight  -> CPU-bound, light RAM (~2 GB). Whole-brain SVC x5 fits.
#                   -> many subjects at once, single-threaded (--n-jobs 1).
#   decoding     -> light + fast. -> many subjects at once, single-threaded.
#
# The Python runners (run_*.py) are single-subject and skip work that's already
# done, so this driver is safe to re-run / resume after an interruption.

set -euo pipefail

PIPELINE="${1:-}"
if [ -z "$PIPELINE" ]; then
    echo "Usage: bash multivariate/run_local.sh {glmsingle|searchlight|decoding|frem} [SUBJECT ...]" >&2
    exit 1
fi
shift

# ---------------------------------------------------------------------------
# Paths — EDIT these to match the VM's filesystem layout.
# Defaults follow the /mnt/data convention documented in run_glmsingle.py.
# Any of these can be overridden from the environment.
# ---------------------------------------------------------------------------
BASE_DIR="${BASE_DIR:-/mnt/data/learning-habits}"
BIDS_DIR="${BIDS_DIR:-${BASE_DIR}/bids_dataset/derivatives/fmriprep-24.0.1-noSDC}"
GLMSINGLE_DIR="${GLMSINGLE_DIR:-${BASE_DIR}/bids_dataset/derivatives/glmsingle}"
SEARCHLIGHT_DIR="${SEARCHLIGHT_DIR:-${BASE_DIR}/bids_dataset/derivatives/searchlight}"
DECODING_DIR="${DECODING_DIR:-${BASE_DIR}/bids_dataset/derivatives/decoding}"
FREM_DIR="${FREM_DIR:-${BASE_DIR}/bids_dataset/derivatives/frem}"
VIS_MASK="${VIS_MASK:-${DECODING_DIR}/visual_cortex_mask.nii.gz}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"   # override with the env's python, e.g. PY=~/miniforge3/envs/neuroim/bin/python
PARTICIPANTS_TSV="${PARTICIPANTS_TSV:-${BASE_DIR}/participants_mvpa.tsv}"

TOTAL_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 30)"

# ---------------------------------------------------------------------------
# Per-pipeline defaults: (NPROC = concurrent subjects, THREADS = BLAS threads
# per subject). Chosen so NPROC*THREADS <= cores, and NPROC respects the memory
# ceiling for the memory-bound step.  Override via env: NPROC=.. THREADS=..
# ---------------------------------------------------------------------------
case "$PIPELINE" in
    glmsingle)
        OUTPUT_DIR="$GLMSINGLE_DIR"
        DEF_NPROC=6 ; DEF_THREADS=4     # ~20 GB/subject budget under 120 GB
        ;;
    searchlight)
        OUTPUT_DIR="$SEARCHLIGHT_DIR"
        DEF_NPROC=28 ; DEF_THREADS=1
        ;;
    decoding)
        OUTPUT_DIR="$DECODING_DIR"
        DEF_NPROC=28 ; DEF_THREADS=1
        ;;
    frem)
        OUTPUT_DIR="$FREM_DIR"
        DEF_NPROC=28 ; DEF_THREADS=1
        ;;
    *)
        echo "ERROR: unknown pipeline '$PIPELINE' (expected glmsingle|searchlight|decoding|frem)" >&2
        exit 1
        ;;
esac

NPROC="${NPROC:-$DEF_NPROC}"
THREADS="${THREADS:-$DEF_THREADS}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Build subject list (same logic as the submit_*.sh scripts)
# ---------------------------------------------------------------------------
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
N=$(grep -c . "$SUBJECTS_FILE" || true)
[ "$N" -gt 0 ] || { echo "ERROR: subject list is empty" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisite checks (fail early, not 3 hours in)
# ---------------------------------------------------------------------------
case "$PIPELINE" in
    glmsingle)
        [ -d "$BIDS_DIR" ] || { echo "ERROR: fMRIPrep dir not found: $BIDS_DIR" >&2; exit 1; } ;;
    searchlight|decoding|frem)
        [ -d "$GLMSINGLE_DIR" ] || { echo "ERROR: GLMsingle betas dir not found: $GLMSINGLE_DIR (run 'glmsingle' first)" >&2; exit 1; } ;;
esac
if [ "$PIPELINE" = "decoding" ] && [ ! -f "$VIS_MASK" ]; then
    echo "ERROR: visual cortex mask not found: $VIS_MASK" >&2
    echo "       Build it: $PY multivariate/build_visual_cortex_mask.py --output-dir $DECODING_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Thread pinning — inherited by every xargs child. Critical: without this,
# NPROC processes each spawn ~cores BLAS threads and thrash the machine.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo " pipeline      : $PIPELINE"
echo " subjects      : $N   (from ${SUBJECTS_FILE})"
echo " concurrency   : $NPROC subjects x $THREADS threads = $((NPROC*THREADS)) (machine has $TOTAL_CORES cores)"
echo " output dir    : $OUTPUT_DIR"
echo " python        : $PY"
echo "=========================================================="
if [ "$((NPROC*THREADS))" -gt "$TOTAL_CORES" ]; then
    echo "WARNING: NPROC*THREADS ($((NPROC*THREADS))) exceeds cores ($TOTAL_CORES) — expect contention." >&2
fi
echo

# ---------------------------------------------------------------------------
# Per-subject command + parallel launch
# ---------------------------------------------------------------------------
run_one() {
    local s="$1"
    case "$PIPELINE" in
        glmsingle)
            "$PY" -u "${REPO}/multivariate/run_glmsingle.py" \
                --subject "$s" --base-dir "$BASE_DIR" \
                --bids-dir "$BIDS_DIR" --output-dir "$GLMSINGLE_DIR" ;;
        searchlight)
            "$PY" -u "${REPO}/multivariate/run_searchlight.py" \
                --subject "$s" --bids-dir "$BIDS_DIR" \
                --glmsingle-dir "$GLMSINGLE_DIR" --output-dir "$SEARCHLIGHT_DIR" \
                --n-jobs 1 ;;
        decoding)
            "$PY" -u "${REPO}/multivariate/run_decoding.py" \
                --subject "$s" --base-dir "$BASE_DIR" --bids-dir "$BIDS_DIR" \
                --glmsingle-dir "$GLMSINGLE_DIR" --output-dir "$DECODING_DIR" \
                --visual-cortex-mask "$VIS_MASK" ;;
        frem)
            "$PY" -u "${REPO}/multivariate/run_frem.py" \
                --subject "$s" --bids-dir "$BIDS_DIR" \
                --glmsingle-dir "$GLMSINGLE_DIR" --output-dir "$FREM_DIR" \
                --n-jobs 1 ;;
    esac
}
export -f run_one
export PIPELINE REPO PY BASE_DIR BIDS_DIR GLMSINGLE_DIR SEARCHLIGHT_DIR DECODING_DIR FREM_DIR VIS_MASK

# xargs -P runs NPROC subjects at a time and picks up the next as each finishes
# (natural load-balancing across the 59-subject list). Exit non-zero if any fail.
xargs -a "$SUBJECTS_FILE" -P "$NPROC" -I{} \
    bash -c 'run_one "$@"' _ {}

echo
echo "All $PIPELINE jobs finished. Per-subject logs under: ${OUTPUT_DIR}/sub-*/"
