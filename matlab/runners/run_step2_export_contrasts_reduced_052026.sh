#!/bin/bash
# Export all-runs contrast images for glm2_chosen_reduced_052026 (no session contrasts).

set -euo pipefail

MATLAB=/opt/apps/containers/matlab/r2023a/usr/local/MATLAB/R2023a/bin/matlab
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_DIR="$REPO/matlab"

GLM_ROOT=/mnt/data/learning-habits/spm_format/outputs/glm2_chosen_all_runs_reduced_Q5_Hpretest_scrubbed_2026-05-20-09-58
OUTDIR=/mnt/data/learning-habits/spm_outputs/glm2_chosen_all_runs_reduced_Q5_Hpretest_scrubbed_2026-05-20-09-58

mkdir -p "$OUTDIR"
echo "===== Export: $(basename $GLM_ROOT) ====="
echo "  -> $OUTDIR"

$MATLAB -nodisplay -nosplash -nodesktop -r \
  "addpath('/home/ubuntu/repos/spm12'); addpath('$MATLAB_DIR'); export_first_lvl_contrasts('$GLM_ROOT', '$OUTDIR', 'copy', true); exit" \
  2>&1

echo "Export done."
