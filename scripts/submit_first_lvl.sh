#!/bin/bash
# Submit any first-level GLM script in matlab/first_lvl/ as a SLURM array, one subject per task.
#
# Usage (from the repo root, on the cluster):
#   bash scripts/submit_first_lvl.sh glm2_chosen_all_runs.m                  # every subject in the bbt
#   bash scripts/submit_first_lvl.sh glm2_chosen_all_runs.m sub-01 sub-15    # specific subjects
#   DRY_RUN=1 bash scripts/submit_first_lvl.sh glm2_all_runs.m sub-01       # print the job, sbatch --test-only
#
# Overridable via environment: BBT_PATH, DATA_DIR, SPM_PATH, CURRENT_DATE, THROTTLE, TIME, MEM.
#
# All tasks get the same injected current_date, so they write into one output folder
# (<DATA_DIR>/outputs/<script's own prefix><CURRENT_DATE>/sub-XX). The GLM scripts skip
# sub-04 and sub-45 themselves, so those tasks finish immediately.
#
# Prerequisite: submit_spm_prep.sh + submit_spm_smooth.sh have produced smoothed BOLD and
# *_motion_with_dummies.txt for the target subjects under DATA_DIR.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/submit_first_lvl.sh <script.m> [sub-XX ...]" >&2
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_NAME="$(basename "$1" .m)"
SCRIPT="${REPO}/matlab/first_lvl/${SCRIPT_NAME}.m"
shift
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no such script: ${SCRIPT}" >&2
    exit 1
fi

SPM_PATH="${SPM_PATH:-/home/hfluhr/repos/spm12}"
DATA_DIR="${DATA_DIR:-/home/hfluhr/data/learninghabits/spm_format}"
BBT_PATH="${BBT_PATH:-/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv}"
CURRENT_DATE="${CURRENT_DATE:-$(date +%Y-%m-%d-%H-%M)}"
THROTTLE="${THROTTLE:-20}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-16G}"
# MATLAB runs in an Apptainer container, which fails on the L4 GPU nodes u24-cva0ls0-[509-516]
# ("Failed to create user namespace: Permission denied", seen 2026-09-23). Exclude them;
# override with EXCLUDE="" or another node list.
EXCLUDE="${EXCLUDE-u24-cva0ls0-[509-516]}"
LOG_DIR="${DATA_DIR}/logs"
mkdir -p "$LOG_DIR"

# Subject list: arguments, or every sub_id in the bbt
SUBJECTS_FILE="${LOG_DIR}/${SCRIPT_NAME}_${CURRENT_DATE}_subjects.txt"
if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@" > "$SUBJECTS_FILE"
else
    awk -F, 'NR==1 {for (i = 1; i <= NF; i++) if ($i == "sub_id") c = i; next} {print $c}' "$BBT_PATH" \
        | sort -u > "$SUBJECTS_FILE"
fi
N=$(wc -l < "$SUBJECTS_FILE")
if [ "$N" -eq 0 ]; then
    echo "ERROR: empty subject list" >&2
    exit 1
fi

echo "Script:   ${SCRIPT}"
echo "Subjects: ${N} (${SUBJECTS_FILE})"
echo "bbt:      ${BBT_PATH}"
echo "Output:   ${DATA_DIR}/outputs/<${SCRIPT_NAME} prefix>${CURRENT_DATE}"

SBATCH_ARGS=()
[ "${DRY_RUN:-0}" = "1" ] && SBATCH_ARGS+=(--test-only)

JOB=$(cat <<EOF
#!/bin/bash -l
#SBATCH --job-name=${SCRIPT_NAME}
#SBATCH --output=${LOG_DIR}/${SCRIPT_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${SCRIPT_NAME}_%A_%a.err
#SBATCH --array=0-$((N - 1))%${THROTTLE}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --partition=standard
${EXCLUDE:+#SBATCH --exclude=${EXCLUDE}}

set -eo pipefail
SUB=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${SUBJECTS_FILE}")
echo "Task \${SLURM_ARRAY_TASK_ID}: \${SUB}"
module load matlab

matlab -batch "spmpath = '${SPM_PATH}'; data_dir = '${DATA_DIR}'; analysis_dir = '${DATA_DIR}'; bbt_path = '${BBT_PATH}'; current_date = '${CURRENT_DATE}'; subjects_override = {'\${SUB}'}; run('${SCRIPT}');"
EOF
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "----- job script -----"
    echo "$JOB"
    echo "----------------------"
fi
echo "$JOB" | sbatch ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}
