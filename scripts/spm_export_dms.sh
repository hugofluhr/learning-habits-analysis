#!/bin/bash

# Simple script to run export_spm_dms
# Edit the paths below as needed

BASE_DIR="/home/ubuntu/data/learning-habits/spm_format_20250603/outputs/glm21_2025-09-23-02-55"
# Add any extra options as needed, e.g. OVERWRITE="true"
OVERWRITE="false"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_DIR="$(dirname "$SCRIPT_DIR")/matlab"

echo "Exporting SPM design matrices..."
echo "Input: $BASE_DIR"

module load matlab
matlab -nodisplay -nosplash -nodesktop -r "addpath('/home/ubuntu/repos/spm12'); addpath('$MATLAB_DIR'); export_spm_dms('$BASE_DIR', 'Overwrite', $OVERWRITE); exit"