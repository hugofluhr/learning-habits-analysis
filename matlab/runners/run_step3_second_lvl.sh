#!/bin/bash
# Step 3: Run second-level (group) analysis for all GLMs.
# Run after step 2 has completed for all GLMs.

set -euo pipefail

module load matlab/r2023a

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_SCRIPT="$REPO/matlab/second_lvl/second_lvl_all_runs.m"

EXPORT_ROOTS=(
    "/mnt/data/learning-habits/spm_outputs/glm2_mf_chosenval_2026-06-05-09-17"
)

for export_root in "${EXPORT_ROOTS[@]}"; do
    log_dir="$export_root/logs"
    mkdir -p "$log_dir"
    log_file="$log_dir/second_lvl_$(date +%Y%m%d_%H%M%S).log"

    echo "===== Step 3: $(basename "$export_root") ====="
    echo "Log: $log_file"

    matlab -nodisplay -r \
        "export_root = '$export_root'; run('$MATLAB_SCRIPT'); exit" \
        2>&1 | tee "$log_file"
    matlab_exit=${PIPESTATUS[0]}

    if [ "$matlab_exit" -ne 0 ]; then
        echo "ERROR: MATLAB exited with code $matlab_exit for $export_root" >&2
        exit "$matlab_exit"
    fi

    echo ""
done

echo "Step 3 complete for all GLMs."
