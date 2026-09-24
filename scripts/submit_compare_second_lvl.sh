#!/bin/bash
# Compare re-run second levels with the old runs the manuscript and vault results come from.
#
# Usage: bash scripts/submit_compare_second_lvl.sh <date tag>
#   <date tag>  the re-run's output folder date, e.g. 2026-09-24-14-56
#
# Writes spm_outputs/compare_second_lvl_<date tag>/report.md, plus svc/<run>.csv (one peak table per run).
# The old runs are in spm_format/reference/second_lvl/ (copied from the VM-era outputs); the pairs are listed below.
# The concat model has no old run: it is compared with the re-run glm2_all_runs and checked against glm2_all_runs' claims.
# The job itself is scripts/slurm/compare_second_lvl.sbatch.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    sed -n '2,10p' "$0" >&2
    exit 1
fi
TAG="$1"

export REPO="$(cd "$(dirname "$0")/.." && pwd)"
export SPM_PATH=/home/hfluhr/repos/spm12
export OLD_ROOT=/home/hfluhr/data/learninghabits/spm_format/reference/second_lvl
export NEW_ROOT=/home/hfluhr/data/learninghabits/spm_outputs
export OUT_DIR="$NEW_ROOT/compare_second_lvl_$TAG"
export PAIRS="
glm2_all_runs glm2_all_runs_scrubbed_2025-12-11-12-44 glm2_all_runs_scrubbed_$TAG
glm2_chosen_all_runs glm2_chosen_all_runs_scrubbed_2025-12-11-11-22 glm2_chosen_all_runs_scrubbed_$TAG
glm3_chosen_choice_var glm3_chosen_choice_var_scrubbed_2026-04-01-01-29 glm3_chosen_choice_var_scrubbed_$TAG
glm2_mf_val glm2_mf_chosenval_2026-06-05-09-17 glm2_mf_chosenval_$TAG
glm2_mf_frequ glm2_mf_chosenfrequ_2026-06-04-12-12 glm2_mf_chosenfrequ_$TAG
glm2_all_runs_concat=glm2_all_runs glm2_all_runs_scrubbed_$TAG glm2_all_runs_concat_scrubbed_$TAG
"

mkdir -p "$OUT_DIR"
sbatch --output="$OUT_DIR/compare_%j.out" --error="$OUT_DIR/compare_%j.err" "$REPO/scripts/slurm/compare_second_lvl.sbatch"
