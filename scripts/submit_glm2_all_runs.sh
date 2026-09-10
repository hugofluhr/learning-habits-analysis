#!/bin/bash
# Run glm2_all_runs.m on the cluster.
#
# Primary use right now: verification reference. This script already has a
# saved VM run (outputs/glm2_all_runs_scrubbed_2025-12-11-12-44) to diff
# against, unlike the brand-new glm2_all_runs_diff_timing.m — so running it
# on cluster-prepped data and comparing design matrices/contrasts against
# that VM output validates the ported data-prep pipeline itself (smoothing,
# confounds), independent of anything about the new script.
#
# Usage (from repo root):
#   bash scripts/submit_glm2_all_runs.sh sub-01      # smoke test on sub-01 only
#   bash scripts/submit_glm2_all_runs.sh              # all subjects in bbt.csv
#
# Prerequisite: submit_spm_prep.sh + submit_spm_smooth.sh must already have
# produced smoothed_*_bold.nii and *_motion_with_dummies.txt for the target
# subject(s) under DATA_DIR.

set -euo pipefail

SPM_PATH="/home/hfluhr/repos/spm12"
DATA_DIR="/home/hfluhr/data/learninghabits/spm_format_noSDC"
BBT_PATH="/home/hfluhr/data/learninghabits/bbt.csv"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${DATA_DIR}/logs"
mkdir -p "$LOG_DIR"

SUBJECTS_OVERRIDE_MATLAB="{}"
if [ "$#" -gt 0 ]; then
    QUOTED=$(printf "'%s'," "$@")
    SUBJECTS_OVERRIDE_MATLAB="{${QUOTED%,}}"
fi

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=glm2_all_runs
#SBATCH --output=${LOG_DIR}/glm2_all_runs_%j.out
#SBATCH --error=${LOG_DIR}/glm2_all_runs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=standard

set -eo pipefail
module load matlab

matlab -nodisplay -nosplash -nodesktop -r "\
spmpath = '${SPM_PATH}'; \
data_dir = '${DATA_DIR}'; \
analysis_dir = '${DATA_DIR}'; \
bbt_path = '${BBT_PATH}'; \
subjects_override = ${SUBJECTS_OVERRIDE_MATLAB}; \
run('${REPO}/matlab/first_lvl/glm2_all_runs.m'); \
exit;"
EOF
