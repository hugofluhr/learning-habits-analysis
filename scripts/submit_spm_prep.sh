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
# Run before submit_spm_smooth.sh.
#
# Environment (all optional):
#   DRY_RUN=1              print the job and run sbatch --test-only
#   BBT_PATH, OUTPUT_DIR   override the defaults set below

set -euo pipefail

BASE_DIR="/home/hfluhr/data/learninghabits"
BIDS_DIR="/home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/spm_format}"
BBT_PATH="${BBT_PATH:-${BASE_DIR}/bbt_062026_mf_cols.csv}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Subject list as bare IDs ("01"): the prep scripts build the "sub-" folder names themselves
if [ "$#" -gt 0 ]; then
    SUBJECTS=$(printf "%s\n" "$@" | sed 's/^sub-//' | tr '\n' ' ')
else
    SUBJECTS=$(awk -F, 'NR==1 {for (i = 1; i <= NF; i++) if ($i == "sub_id") c = i; next} {print $c}' "$BBT_PATH" \
        | sort -u | sed 's/^sub-//' | tr '\n' ' ')
fi
N=$(echo "$SUBJECTS" | wc -w)
if [ "$N" -eq 0 ]; then
    echo "ERROR: empty subject list" >&2
    exit 1
fi
echo "Subjects: ${N}"
echo "Output:   ${OUTPUT_DIR}"

SBATCH_ARGS=()
[ "${DRY_RUN:-0}" = "1" ] && SBATCH_ARGS+=(--test-only)

JOB=$(cat <<EOF
#!/bin/bash -l
#SBATCH --job-name=spm_prep
#SBATCH --output=${LOG_DIR}/spm_prep_%j.out
#SBATCH --error=${LOG_DIR}/spm_prep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=standard

set -eo pipefail
module load miniforge3
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate learning-habits
export PYTHONUNBUFFERED=1
cd "${REPO}"

python -u scripts/prepare_bids_spm.py \\
    --base-dir "${BASE_DIR}" --bids-dir "${BIDS_DIR}" --output-dir "${OUTPUT_DIR}" \\
    --subjects ${SUBJECTS}

python -u scripts/prepare_bids_spm_add_dummies.py \\
    --base-dir "${BASE_DIR}" --bids-dir "${BIDS_DIR}" --output-dir "${OUTPUT_DIR}" \\
    --subjects ${SUBJECTS}
EOF
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "----- job script -----"
    echo "$JOB"
    echo "----------------------"
fi
echo "$JOB" | sbatch ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}
