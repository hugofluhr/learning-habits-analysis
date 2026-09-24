#!/bin/bash
# Submit the steps that follow a first level, one SLURM job per step.
#
# Usage: bash scripts/submit_downstream.sh <step> <glm>
#   <glm>   first-level folder name under $OUTPUTS_DIR, or a full path
#   <step>  contrasts | export | second | sn23 | all
#     contrasts  per-session t-contrasts appended to each SPM.mat
#     export     contrast images by session, into $EXPORTS_DIR/<glm>
#     second     one-sample t-tests for allruns/ and session-0X/
#     sn23       average session-02 and session-03, then one-sample t-tests
#     all        all four; second and sn23 both wait for export
#
# Environment: CONNAMES (optional MATLAB cell, e.g. "{'first_stim'}"; by default the conditions come
# from the model's own contrasts), DEPENDENCY, DRY_RUN=1, SPM_PATH, OUTPUTS_DIR, EXPORTS_DIR, EXCLUDE.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    sed -n '2,14p' "$0" >&2
    exit 1
fi
STEP="$1"
GLM="$2"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SPM_PATH="${SPM_PATH:-/home/hfluhr/repos/spm12}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/home/hfluhr/data/learninghabits/spm_format/outputs}"
EXPORTS_DIR="${EXPORTS_DIR:-/home/hfluhr/data/learninghabits/spm_outputs}"
# MATLAB's Apptainer container fails on these GPU nodes ("Failed to create user namespace")
EXCLUDE="${EXCLUDE-u24-cva0ls0-[509-516]}"

if [[ "$GLM" == */* ]]; then GLM_ROOT="$GLM"; else GLM_ROOT="${OUTPUTS_DIR}/${GLM}"; fi
GLM_NAME="$(basename "$GLM_ROOT")"
EXPORT_ROOT="${EXPORTS_DIR}/${GLM_NAME}"
if [ ! -d "$GLM_ROOT" ]; then
    echo "ERROR: first-level folder not found: ${GLM_ROOT}" >&2
    exit 1
fi
mkdir -p "${GLM_ROOT}/logs" "${EXPORT_ROOT}/logs"

submit_job() {
    local name="$1" time="$2" log_dir="$3" mcmd="$4" post="${5:-}"
    local dep_args=() test_args=()
    [ -n "${DEPENDENCY:-}" ] && dep_args+=(--dependency="afterok:${DEPENDENCY}")
    [ "${DRY_RUN:-0}" = "1" ] && test_args+=(--test-only)
    local job
    job=$(cat <<EOF
#!/bin/bash -l
#SBATCH --job-name=${name}
#SBATCH --output=${log_dir}/${name}_%j.out
#SBATCH --error=${log_dir}/${name}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=${time}
#SBATCH --partition=standard
${EXCLUDE:+#SBATCH --exclude=${EXCLUDE}}

set -eo pipefail
EXPORT_ROOT="${EXPORT_ROOT}"
module load matlab
matlab -batch "${mcmd}"
${post}
EOF
)
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "----- ${name} -----" >&2
        echo "$job" >&2
    fi
    echo "$job" | sbatch --parsable ${dep_args[@]+"${dep_args[@]}"} ${test_args[@]+"${test_args[@]}"}
}

# Fallback for links the export's system() call didn't create (it failed on the VM)
read -r -d '' SYMLINKS <<'EOF' || true
n_made=0
if [ -f "$EXPORT_ROOT/allruns/contrasts_manifest.tsv" ]; then
    while IFS=$'\t' read -r _tok _idx _name dst src; do
        [ -z "$dst" ] && continue
        [ -e "$dst" ] || { ln -s "$src" "$dst" && n_made=$((n_made+1)); }
    done < <(tail -n +2 "$EXPORT_ROOT/allruns/contrasts_manifest.tsv")
fi
if [ -f "$EXPORT_ROOT/contrasts_manifest_sessions.tsv" ]; then
    while IFS=$'\t' read -r _tok _sess _idx _name dst src; do
        [ -z "$dst" ] && continue
        [ -e "$dst" ] || { ln -s "$src" "$dst" && n_made=$((n_made+1)); }
    done < <(tail -n +2 "$EXPORT_ROOT/contrasts_manifest_sessions.tsv")
fi
echo "Symlinks created from manifests: $n_made"
EOF

CONNAMES_ARG=""
[ -n "${CONNAMES:-}" ] && CONNAMES_ARG="connames = ${CONNAMES}; "

run_step() {
    case "$1" in
        contrasts)
            submit_job "session_contrasts" "04:00:00" "${GLM_ROOT}/logs" \
                "spmpath = '${SPM_PATH}'; glm_root = '${GLM_ROOT}'; ${CONNAMES_ARG}run('${REPO}/matlab/first_lvl/add_session_contrasts_glm2.m');" ;;
        export)
            submit_job "export_contrasts" "01:00:00" "${EXPORT_ROOT}/logs" \
                "addpath('${REPO}/matlab'); addpath('${SPM_PATH}'); spm('Defaults','fMRI'); spm_jobman('initcfg'); export_first_lvl_contrasts_with_sessions('${GLM_ROOT}', '${EXPORT_ROOT}', 'copy', false);" \
                "$SYMLINKS" ;;
        second)
            submit_job "second_lvl" "04:00:00" "${EXPORT_ROOT}/logs" \
                "spmpath = '${SPM_PATH}'; export_root = '${EXPORT_ROOT}'; run('${REPO}/matlab/second_lvl/second_lvl_all_runs.m');" ;;
        sn23)
            # Separate MATLAB sessions: neither script clears its workspace
            submit_job "second_lvl_sn23" "02:00:00" "${EXPORT_ROOT}/logs" \
                "spmpath = '${SPM_PATH}'; root_dir = '${EXPORT_ROOT}'; run('${REPO}/matlab/average_sn2_sn3_contrasts.m');" \
                "matlab -batch \"spmpath = '${SPM_PATH}'; export_root = '${EXPORT_ROOT}'; run('${REPO}/matlab/second_lvl_sn2_sn3.m');\"" ;;
        *)
            echo "ERROR: unknown step '$1'" >&2; exit 1 ;;
    esac
}

echo "First level: ${GLM_ROOT}" >&2
echo "Export:      ${EXPORT_ROOT}" >&2
if [ "$STEP" = "all" ]; then
    for s in contrasts export; do
        id=$(run_step "$s")
        echo "${s}: job ${id}" >&2
        [ "${DRY_RUN:-0}" = "1" ] || DEPENDENCY="${id%%;*}"
    done
    for s in second sn23; do
        id=$(run_step "$s")
        echo "${s}: job ${id}" >&2
    done
else
    id=$(run_step "$STEP")
    echo "${STEP}: job ${id}" >&2
    echo "$id"
fi
