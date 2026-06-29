#!/bin/bash
set -euo pipefail

source /etc/profile.d/z00-lmod.sh
module load matlab/r2023a

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_DIR="$REPO/matlab"
SPM_PATH="/home/ubuntu/repos/spm12"

declare -a GLM_ROOTS=(
    "/mnt/data/learning-habits/spm_format/outputs/glm2_all_runs_run_zscore_scrubbed_2026-06-23-10-00"
    "/mnt/data/learning-habits/spm_format/outputs/glm2_chosen_all_runs_run_zscore_scrubbed_2026-06-23-11-35"
)

declare -a EXPORT_ROOTS=(
    "/mnt/data/learning-habits/spm_outputs/glm2_all_runs_run_zscore_scrubbed_2026-06-23-10-00"
    "/mnt/data/learning-habits/spm_outputs/glm2_chosen_all_runs_run_zscore_scrubbed_2026-06-23-11-35"
)

for i in "${!GLM_ROOTS[@]}"; do
    glm_root="${GLM_ROOTS[$i]}"
    export_root="${EXPORT_ROOTS[$i]}"

    log_dir="$export_root/logs"
    mkdir -p "$log_dir"
    log_file="$log_dir/export_contrasts_$(date +%Y%m%d_%H%M%S).log"

    echo "===== Step 2: $(basename "$glm_root") ====="
    echo "  -> $export_root"
    echo "Log: $log_file"

    matlab -nodisplay -r         "addpath('$MATLAB_DIR'); addpath('$SPM_PATH'); spm('Defaults','fMRI'); spm_jobman('initcfg'); export_first_lvl_contrasts_with_sessions('$glm_root', '$export_root', 'copy', false); exit"         2>&1 | tee "$log_file"
    matlab_exit=${PIPESTATUS[0]}

    if [ "$matlab_exit" -ne 0 ]; then
        echo "ERROR: MATLAB exited with code $matlab_exit for $glm_root" >&2
        exit "$matlab_exit"
    fi

    # Create the symlinks in the shell from the manifests written by the export.
    # MATLAB's system() is unreliable in this environment (returns 127, no shell),
    # so 'copy', false leaves the link targets recorded in the manifests but the
    # links uncreated. ln -s works fine here, so we make them ourselves.
    n_made=0
    # all-runs manifest: dst=col4, src=col5
    if [ -f "$export_root/allruns/contrasts_manifest.tsv" ]; then
        while IFS=$'\t' read -r _tok _idx _name dst src; do
            [ -z "$dst" ] && continue
            [ -e "$dst" ] || { ln -s "$src" "$dst" && n_made=$((n_made+1)); }
        done < <(tail -n +2 "$export_root/allruns/contrasts_manifest.tsv")
    fi
    # per-session manifest: dst=col5, src=col6
    if [ -f "$export_root/contrasts_manifest_sessions.tsv" ]; then
        while IFS=$'\t' read -r _tok _sess _idx _name dst src; do
            [ -z "$dst" ] && continue
            [ -e "$dst" ] || { ln -s "$src" "$dst" && n_made=$((n_made+1)); }
        done < <(tail -n +2 "$export_root/contrasts_manifest_sessions.tsv")
    fi
    echo "  Symlinks created from manifests: $n_made"

    echo ""
done

echo "Step 2 complete for all GLMs."
