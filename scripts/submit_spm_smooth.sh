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
BASE_DIR="/home/hfluhr/data/learninghabits/spm_format_noSDC"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=spm_smooth
#SBATCH --output=${LOG_DIR}/spm_smooth_%j.out
#SBATCH --error=${LOG_DIR}/spm_smooth_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=standard

set -eo pipefail
module load matlab

matlab -nodisplay -nosplash -nodesktop -r "\
spmpath = '${SPM_PATH}'; \
base_dir = '${BASE_DIR}'; \
run('${REPO}/matlab/spm_smooth_data.m'); \
exit;"
EOF
