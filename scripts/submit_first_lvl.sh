#!/bin/bash
# Submit a first-level GLM script (matlab/first_lvl/) as a SLURM array, one subject per task.
#
# Usage: bash scripts/submit_first_lvl.sh <script.m> [sub-XX ...]
#   bash scripts/submit_first_lvl.sh glm2_chosen_all_runs.m                 # every subject in the bbt
#   bash scripts/submit_first_lvl.sh glm2_chosen_all_runs.m sub-01 sub-15   # specific subjects
#
# All tasks share one CURRENT_DATE, so they write into the same output folder.
# To add subjects to an existing folder, set CURRENT_DATE to that folder's date tag.
# The GLM scripts skip sub-04 and sub-45 themselves.
# The job itself is scripts/slurm/first_lvl.sbatch.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    sed -n '2,11p' "$0" >&2
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$1" .m)"
shift

export SCRIPT="$REPO/matlab/first_lvl/$NAME.m"
export SPM_PATH=/home/hfluhr/repos/spm12
export DATA_DIR=/home/hfluhr/data/learninghabits/spm_format
export BBT_PATH=/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv
export CURRENT_DATE="${CURRENT_DATE:-$(date +%Y-%m-%d-%H-%M)}"

if [ ! -f "$SCRIPT" ]; then
    echo "No such script: $SCRIPT" >&2
    exit 1
fi

LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

# Subject list: the arguments, or every sub_id in the bbt
export SUBJECTS_FILE="$LOG_DIR/${NAME}_${CURRENT_DATE}_subjects.txt"
if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@" > "$SUBJECTS_FILE"
else
    awk -F, 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "sub_id") col = i; next} {print $col}' "$BBT_PATH" \
        | sort -u > "$SUBJECTS_FILE"
fi
N=$(wc -l < "$SUBJECTS_FILE")

echo "$NAME: $N subjects, output folder date tag $CURRENT_DATE"
sbatch --job-name="$NAME" \
    --array="0-$((N - 1))%20" \
    --output="$LOG_DIR/${NAME}_%A_%a.out" \
    --error="$LOG_DIR/${NAME}_%A_%a.err" \
    "$REPO/scripts/slurm/first_lvl.sbatch"
