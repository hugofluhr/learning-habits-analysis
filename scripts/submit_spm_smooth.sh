#!/bin/bash
# Smooth (5mm FWHM) the flat SPM-format BOLD produced by submit_spm_prep.sh,
# on the cluster. Loops over every sub-* dir found under base_dir, so run
# submit_spm_prep.sh first (optionally for a subset of subjects, for a
# smoke test) to control scope.
#
# Usage (from repo root):
#   bash scripts/submit_spm_smooth.sh

set -euo pipefail

SPM_PATH="/home/hfluhr/repos/spm12"
BASE_DIR="/home/hfluhr/data/learninghabits/spm_format"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$LOG_DIR"
# MATLAB runs in an Apptainer container, which fails on the L4 GPU nodes u24-cva0ls0-[509-516]
# ("Failed to create user namespace: Permission denied", seen 2026-09-23). Exclude them;
# override with EXCLUDE="" or another node list.
EXCLUDE="${EXCLUDE-u24-cva0ls0-[509-516]}"

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=spm_smooth
#SBATCH --output=${LOG_DIR}/spm_smooth_%j.out
#SBATCH --error=${LOG_DIR}/spm_smooth_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=standard
${EXCLUDE:+#SBATCH --exclude=${EXCLUDE}}

set -eo pipefail
module load matlab

matlab -batch "spmpath = '${SPM_PATH}'; base_dir = '${BASE_DIR}'; run('${REPO}/matlab/spm_smooth_data.m');"
EOF
