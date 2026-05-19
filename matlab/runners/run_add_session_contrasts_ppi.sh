#!/bin/bash
# Append per-session PPI contrasts to each subject's PPI_putamen/SPM.mat.

set -euo pipefail

module load matlab/r2023a

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_SCRIPT="$REPO/matlab/connectivity/add_session_contrasts_ppi.m"

GPPI_ROOT="/mnt/data/learning-habits/spm_format/outputs/PPI/gppi_putamen_Hvalchosen_deconv_2026-03-18-07-39-25"

log_dir="$GPPI_ROOT/logs"
mkdir -p "$log_dir"
log_file="$log_dir/add_session_contrasts_$(date +%Y%m%d_%H%M%S).log"

echo "===== Adding per-session PPI contrasts ====="
echo "Root: $GPPI_ROOT"
echo "Log:  $log_file"

matlab -nodisplay -r \
    "gppi_root = '$GPPI_ROOT'; run('$MATLAB_SCRIPT'); exit" \
    2>&1 | tee "$log_file"
matlab_exit=${PIPESTATUS[0]}

if [ "$matlab_exit" -ne 0 ]; then
    echo "ERROR: MATLAB exited with code $matlab_exit" >&2
    exit "$matlab_exit"
fi

echo "Done."
