#!/bin/bash
# Step 1: Append per-session contrasts to each GLM's SPM.mat files.
# Run this before step 2 or 3.

set -euo pipefail

module load matlab/r2023a

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_SCRIPT="$REPO/matlab/first_lvl/add_session_contrasts_glm2.m"

GLM_ROOTS=(
    "/mnt/data/learning-habits/spm_format/outputs/glm2_chosen_Qval_2026-05-26-02-43"
)

for glm_root in "${GLM_ROOTS[@]}"; do
    log_dir="$glm_root/logs"
    mkdir -p "$log_dir"
    log_file="$log_dir/add_session_contrasts_$(date +%Y%m%d_%H%M%S).log"

    echo "===== Step 1: $(basename "$glm_root") ====="
    echo "Log: $log_file"

    matlab -nodisplay -r \
        "glm_root = '$glm_root'; run('$MATLAB_SCRIPT'); exit" \
        2>&1 | tee "$log_file"
    matlab_exit=${PIPESTATUS[0]}

    if [ "$matlab_exit" -ne 0 ]; then
        echo "ERROR: MATLAB exited with code $matlab_exit for $glm_root" >&2
        exit "$matlab_exit"
    fi

    echo ""
done

echo "Step 1 complete for all GLMs."
