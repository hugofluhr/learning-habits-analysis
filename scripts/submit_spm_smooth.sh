#!/bin/bash
# Smooth (5 mm FWHM) the BOLD in spm_format/sub-*/func/, in one job for all subjects.
# Files that are already smoothed are skipped, so only new subjects get processed.
#
# Usage: bash scripts/submit_spm_smooth.sh
# The job itself is scripts/slurm/spm_smooth.sbatch.

set -euo pipefail

export REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR=/home/hfluhr/data/learninghabits/spm_format/logs
mkdir -p "$LOG_DIR"
sbatch --output="$LOG_DIR/spm_smooth_%j.out" \
    --error="$LOG_DIR/spm_smooth_%j.err" \
    "$REPO/scripts/slurm/spm_smooth.sbatch"
