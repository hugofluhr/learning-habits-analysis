#!/bin/bash
# Second-level (one-sample t-tests) for glm2_chosen_reduced_052026.
# Run after run_step2_export_contrasts_reduced_052026.sh.

set -euo pipefail

MATLAB=/opt/apps/containers/matlab/r2023a/usr/local/MATLAB/R2023a/bin/matlab
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_SCRIPT="$REPO/matlab/second_lvl/second_lvl_all_runs.m"

EXPORT_ROOT=/mnt/data/learning-habits/spm_outputs/glm2_chosen_all_runs_reduced_Q5_Hpretest_scrubbed_2026-05-20-09-58

mkdir -p "$EXPORT_ROOT/logs"
LOG="$EXPORT_ROOT/logs/second_lvl_$(date +%Y%m%d_%H%M%S).log"

echo "===== Second level: $(basename $EXPORT_ROOT) ====="
echo "Log: $LOG"

$MATLAB -nodisplay -nosplash -nodesktop -r \
  "export_root = '$EXPORT_ROOT'; run('$MATLAB_SCRIPT'); exit" \
  2>&1 | tee "$LOG"

echo "Second level done."
