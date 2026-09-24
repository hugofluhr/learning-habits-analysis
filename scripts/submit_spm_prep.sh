#!/bin/bash
# Prepare fMRIPrep output for SPM, in one job for all subjects (about 16 s each):
#   - unzip BOLD and brain masks into spm_format/sub-XX/func/
#   - write *_motion_with_dummies.txt (the confounds the GLMs use), *_motion.txt, *_events.mat
#
# Usage: bash scripts/submit_spm_prep.sh [XX ...]
#   bash scripts/submit_spm_prep.sh          # every subject in the bbt
#   bash scripts/submit_spm_prep.sh 01 15    # specific subjects ("15" or "sub-15")
#
# The default is the bbt, not participants_mvpa.tsv: sub-46 is in the MVPA list but has no bbt row.
# Run before submit_spm_smooth.sh. The job itself is scripts/slurm/spm_prep.sbatch.

set -euo pipefail

export REPO="$(cd "$(dirname "$0")/.." && pwd)"
export OUTPUT_DIR=/home/hfluhr/data/learninghabits/spm_format
BBT_PATH=/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv

# Bare IDs ("01"): the prep scripts build the "sub-" folder names themselves
if [ "$#" -gt 0 ]; then
    SUBJECTS=$(printf "%s\n" "$@" | sed 's/^sub-//' | tr '\n' ' ')
else
    SUBJECTS=$(awk -F, 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "sub_id") col = i; next} {print $col}' "$BBT_PATH" \
        | sort -u | sed 's/^sub-//' | tr '\n' ' ')
fi
export SUBJECTS

mkdir -p "$OUTPUT_DIR/logs"
echo "Subjects ($(echo $SUBJECTS | wc -w)): $SUBJECTS"
sbatch --output="$OUTPUT_DIR/logs/spm_prep_%j.out" \
    --error="$OUTPUT_DIR/logs/spm_prep_%j.err" \
    "$REPO/scripts/slurm/spm_prep.sbatch"
