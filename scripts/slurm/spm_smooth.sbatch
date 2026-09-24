#!/bin/bash -l
# Smooth (5 mm FWHM) the BOLD in spm_format/sub-*/func/; already-smoothed files are skipped.
# Submitted by scripts/submit_spm_smooth.sh, which sets REPO.
#SBATCH --job-name=spm_smooth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=standard
# MATLAB's Apptainer container fails on these GPU nodes ("Failed to create user namespace")
#SBATCH --exclude=u24-cva0ls0-[509-516]

set -eo pipefail
module load matlab
matlab -batch "spmpath = '/home/hfluhr/repos/spm12'; base_dir = '/home/hfluhr/data/learninghabits/spm_format'; run('$REPO/matlab/spm_smooth_data.m');"
