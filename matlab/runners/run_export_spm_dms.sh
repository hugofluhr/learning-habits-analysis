#!/bin/bash
# Export SPM design matrices (design_matrix.csv + column_names.txt) for VIF inspection.

set -euo pipefail

MATLAB=/opt/apps/containers/matlab/r2023a/usr/local/MATLAB/R2023a/bin/matlab

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_DIR="$REPO/matlab"

declare -a GLM_DIRS=(
    "/mnt/data/learning-habits/spm_format/outputs/glm2_chosen_Qval_2026-05-26-02-43"
)

for glm_dir in "${GLM_DIRS[@]}"; do
    echo "===== Exporting DMs: $(basename "$glm_dir") ====="
    $MATLAB -nodisplay -r \
        "addpath('$MATLAB_DIR'); export_spm_dms('$glm_dir'); exit" \
        2>&1
    echo "Done."
    echo ""
done
