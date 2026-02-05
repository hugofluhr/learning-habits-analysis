#!/bin/bash

# Simple script to run export_first_lvl_contrasts
# Edit the paths below as needed

FIRSTLVL_ROOT="/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs/glm2_all_runs_sustained_2026-02-05-03-03"
OUTDIR="/home/ubuntu/data/learning-habits/spm_outputs_noSDC/glm2_all_runs_sustained_2026-02-05-03-03"
# Leave empty for in-place aliasing, or set to output directory

# Default behavior: create symlinks (do not copy).
COPY_MODE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_DIR="$(dirname "$SCRIPT_DIR")/matlab"

echo "Exporting first-level contrasts..."
echo "Input: $FIRSTLVL_ROOT"
if [[ -n "$OUTDIR" ]]; then
    echo "Output: $OUTDIR"
else
    echo "Mode: In-place aliasing"
fi
if [[ "$COPY_MODE" == true ]]; then
    echo "Copy mode: enabled (will pass 'copy', true to MATLAB)"
else
    echo "Copy mode: disabled (will create symlinks when possible)"
fi

module load matlab
# Build optional MATLAB argument for copy mode (pass in as additional args)
MATLAB_COPY_ARG=""
if [[ "$COPY_MODE" == true ]]; then
    MATLAB_COPY_ARG=", 'copy', true"
fi
matlab -nodisplay -nosplash -nodesktop -r "addpath('/home/ubuntu/repos/spm12'); addpath('$MATLAB_DIR'); export_first_lvl_contrasts('$FIRSTLVL_ROOT', '$OUTDIR'$MATLAB_COPY_ARG); exit"