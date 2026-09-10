#!/bin/bash
# Prep BIDS data for SPM on the cluster: unzip BOLD/mask into a flat
# sub-XX/func/ layout, write basic motion/events files, then add the
# scrubbing-aware *_motion_with_dummies.txt confound file GLM scripts use.
#
# Usage (from repo root):
#   bash scripts/submit_spm_prep.sh            # all subjects in participants_mvpa.tsv
#   bash scripts/submit_spm_prep.sh 01          # smoke test on sub-01 only
#   bash scripts/submit_spm_prep.sh 01 05 12    # specific subjects
#
# Prerequisite: run this before spm_smooth_data.m / glm2_all_runs_diff_timing.m,
# which read the flat sub-XX/func/ layout this produces.

set -euo pipefail

BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
OUTPUT_DIR="${BASE_DIR}/spm_format_noSDC"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

SUBJECTS_ARGS=""
if [ "$#" -gt 0 ]; then
    SUBJECTS_ARGS="--subjects $*"
fi

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=spm_prep
#SBATCH --output=${LOG_DIR}/spm_prep_%j.out
#SBATCH --error=${LOG_DIR}/spm_prep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=standard

set -eo pipefail
module load miniforge3
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate learning-habits
export PYTHONUNBUFFERED=1

cd "${REPO}"

python -u scripts/prepare_bids_spm.py \
    --base-dir "${BASE_DIR}" \
    --bids-dir "${BIDS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    ${SUBJECTS_ARGS}

python -u scripts/prepare_bids_spm_add_dummies.py \
    --base-dir "${BASE_DIR}" \
    --bids-dir "${BIDS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    ${SUBJECTS_ARGS}
EOF
