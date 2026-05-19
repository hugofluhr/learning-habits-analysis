#!/bin/bash
set -euo pipefail
module load matlab/r2023a
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/mnt/data/learning-habits/spm_format/outputs/PPI/gppi_putamen_Hvalchosen_deconv_2026-03-18-07-39-25/second-lvl/session-02-03"
mkdir -p "$LOG_DIR"
matlab -nodisplay -nosplash -nodesktop -r \
    "run('$REPO/matlab/connectivity/second_lvl_ppi_sn23.m'); exit;" \
    2>&1 | tee "$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
