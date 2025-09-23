#!/bin/bash

# Simple script to run export_first_lvl_contrasts
# Edit the paths below as needed

FIRSTLVL_ROOT="/home/ubuntu/data/learning-habits/spm_format_20250603/outputs/glm2_combined_long_pmod_2025-09-22-05-42"
OUTDIR="/home/ubuntu/data/learning-habits/spm_outputs/glm2_combined_long_pmod_2025-09-22-05-42"  # Leave empty for in-place aliasing, or set to output directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_DIR="$(dirname "$SCRIPT_DIR")/matlab"

echo "Exporting first-level contrasts..."
echo "Input: $FIRSTLVL_ROOT"
if [[ -n "$OUTDIR" ]]; then
    echo "Output: $OUTDIR"
else
    echo "Mode: In-place aliasing"
fi

module load matlab
matlab -nodisplay -nosplash -nodesktop -r "addpath('/home/ubuntu/repos/spm12'); addpath('$MATLAB_DIR'); export_first_lvl_contrasts('$FIRSTLVL_ROOT', '$OUTDIR'); exit"
