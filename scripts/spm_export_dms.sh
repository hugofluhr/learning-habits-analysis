#!/bin/bash

# Simple script to run export_spm_dms
# Edit the paths below as needed

ALLRUNS_DEMEAN="/home/ubuntu/data/learning-habits/spm_format/outputs/glm2_all_runs_scrubbed_demeaned_2026-05-11-08-26"
CHOSEN_DEMEAN="/home/ubuntu/data/learning-habits/spm_format/outputs/glm2_chosen_all_runs_scrubbed_demeaned_2026-05-11-08-21"
OVERWRITE="false"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATLAB_DIR="$(dirname "$SCRIPT_DIR")/matlab"

echo "Exporting SPM design matrices..."

module load matlab/r2023a
matlab -nodisplay -nosplash -nodesktop -r "addpath('/home/ubuntu/repos/spm12'); addpath('$MATLAB_DIR'); \
  export_spm_dms('$ALLRUNS_DEMEAN', 'Overwrite', $OVERWRITE); \
  export_spm_dms('$CHOSEN_DEMEAN',  'Overwrite', $OVERWRITE); \
  exit"