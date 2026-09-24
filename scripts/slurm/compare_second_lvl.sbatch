#!/bin/bash -l
# Peak/SVC tables for old and re-run second levels, then the comparison report.
# Submitted by scripts/submit_compare_second_lvl.sh, which sets these environment variables:
#   REPO, SPM_PATH, OLD_ROOT, NEW_ROOT, OUT_DIR, PAIRS (lines of "model old_run new_run")
#SBATCH --job-name=compare_2nd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=standard
# MATLAB's Apptainer container fails on these GPU nodes ("Failed to create user namespace")
#SBATCH --exclude=u24-cva0ls0-[509-516]

set -eo pipefail
module load matlab
module load miniforge3
mkdir -p "$OUT_DIR/svc"

pair_args=()
while read -r model old new; do
    [ -z "$model" ] && continue
    pair_args+=(--pair "$model" "$old" "$new")
    for run in "$old" "$new"; do
        # An old run is in reference/second_lvl/; a run can also be a re-run (the concat model is compared with glm2_all_runs)
        if [ -d "$OLD_ROOT/$run" ]; then root="$OLD_ROOT/$run"; else root="$NEW_ROOT/$run/second-lvl"; fi
        if [ ! -f "$OUT_DIR/svc/$run.csv" ]; then
            echo "===== svc_report $run"
            matlab -batch "addpath('$REPO/matlab/second_lvl'); svc_report('$root', '$OUT_DIR/svc/$run.csv', '', '$SPM_PATH');"
        fi
    done
done <<< "$PAIRS"

echo "===== compare"
/home/hfluhr/data/conda/envs/learning-habits/bin/python "$REPO/scripts/compare_second_lvl.py" \
    --svc-dir "$OUT_DIR/svc" --old-root "$OLD_ROOT" --new-root "$NEW_ROOT" "${pair_args[@]}" \
    --claims "$REPO/scripts/second_lvl_claims.csv" --out "$OUT_DIR/report.md"
