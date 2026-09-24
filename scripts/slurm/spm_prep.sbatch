#!/bin/bash -l
# Prepare fMRIPrep output for SPM for a list of subjects.
# Submitted by scripts/submit_spm_prep.sh, which sets these environment variables:
#   SUBJECTS (bare IDs like "01 15"), OUTPUT_DIR, REPO
#SBATCH --job-name=spm_prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=standard

set -eo pipefail

BASE_DIR=/home/hfluhr/data/learninghabits
BIDS_DIR=/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate learning-habits
export PYTHONUNBUFFERED=1
cd "$REPO"

python -u scripts/prepare_bids_spm.py \
    --base-dir "$BASE_DIR" --bids-dir "$BIDS_DIR" --output-dir "$OUTPUT_DIR" --subjects $SUBJECTS

python -u scripts/prepare_bids_spm_add_dummies.py \
    --base-dir "$BASE_DIR" --bids-dir "$BIDS_DIR" --output-dir "$OUTPUT_DIR" --subjects $SUBJECTS
