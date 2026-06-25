#!/bin/bash
set -euo pipefail

source /etc/profile.d/z00-lmod.sh
module load matlab/r2023a

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_DIR="$REPO/matlab"
SPM_PATH="/home/ubuntu/repos/spm12"

declare -a GLM_ROOTS=(
    "/mnt/data/learning-habits/spm_format/outputs/glm2_all_runs_run_zscore_scrubbed_2026-06-23-10-00"
    "/mnt/data/learning-habits/spm_format/outputs/glm2_chosen_all_runs_run_zscore_scrubbed_2026-06-23-11-35"
)

declare -a EXPORT_ROOTS=(
    "/mnt/data/learning-habits/spm_outputs/glm2_all_runs_run_zscore_scrubbed_2026-06-23-10-00"
    "/mnt/data/learning-habits/spm_outputs/glm2_chosen_all_runs_run_zscore_scrubbed_2026-06-23-11-35"
)

for i in "${!GLM_ROOTS[@]}"; do
    glm_root="${GLM_ROOTS[$i]}"
    export_root="${EXPORT_ROOTS[$i]}"

    log_dir="$export_root/logs"
    mkdir -p "$log_dir"
    log_file="$log_dir/export_contrasts_$(date +%Y%m%d_%H%M%S).log"

    echo "===== Step 2: $(basename "$glm_root") ====="
    echo "  -> $export_root"
    echo "Log: $log_file"

    matlab -nodisplay -r         "addpath('$MATLAB_DIR'); addpath('$SPM_PATH'); spm('Defaults','fMRI'); spm_jobman('initcfg'); export_first_lvl_contrasts_with_sessions('$glm_root', '$export_root'); exit"         2>&1 | tee "$log_file"
    matlab_exit=${PIPESTATUS[0]}

    if [ "$matlab_exit" -ne 0 ]; then
        echo "ERROR: MATLAB exited with code $matlab_exit for $glm_root" >&2
        exit "$matlab_exit"
    fi

    echo ""
done

echo "Step 2 complete for all GLMs."
