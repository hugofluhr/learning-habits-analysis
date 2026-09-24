#!/bin/bash -l
# One task of a first-level SLURM array: runs one GLM script for one subject.
# Submitted by scripts/submit_first_lvl.sh, which sets these environment variables:
#   SCRIPT, SUBJECTS_FILE, SPM_PATH, DATA_DIR, BBT_PATH, CURRENT_DATE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=standard
# MATLAB's Apptainer container fails on these GPU nodes ("Failed to create user namespace")
#SBATCH --exclude=u24-cva0ls0-[509-516]

set -eo pipefail

# Task N handles line N+1 of the subject list
SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
echo "Task $SLURM_ARRAY_TASK_ID: $SUB"

module load matlab
matlab -batch "spmpath = '$SPM_PATH'; data_dir = '$DATA_DIR'; analysis_dir = '$DATA_DIR'; bbt_path = '$BBT_PATH'; current_date = '$CURRENT_DATE'; subjects_override = {'$SUB'}; run('$SCRIPT');"
