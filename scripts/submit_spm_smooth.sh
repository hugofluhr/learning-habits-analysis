#!/bin/bash
# Smooth (5 mm FWHM) the BOLD in spm_format/sub-*/func/, in one job for all subjects.
# Files that are already smoothed are skipped, so only new subjects get processed.
#
# Usage: bash scripts/submit_spm_smooth.sh
#
# Environment (optional): EXCLUDE overrides the excluded nodes set below.

set -euo pipefail

SPM_PATH="/home/hfluhr/repos/spm12"
BASE_DIR="/home/hfluhr/data/learninghabits/spm_format"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$LOG_DIR"
# MATLAB's Apptainer container fails on these GPU nodes ("Failed to create user namespace")
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
