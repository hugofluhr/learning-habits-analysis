#!/bin/bash -l
# The steps that follow a first level, run in order in one job; stops at the first failure.
# Submitted by scripts/submit_downstream.sh, which sets these environment variables:
#   STEPS, GLM_ROOT, EXPORT_ROOT, REPO, SPM_PATH, and optionally CONNAMES
#SBATCH --job-name=downstream
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=standard
# MATLAB's Apptainer container fails on these GPU nodes ("Failed to create user namespace")
#SBATCH --exclude=u24-cva0ls0-[509-516]

set -eo pipefail
module load matlab

# The export writes a manifest of the symlinks it should create; make any that are missing.
# (On the VM, MATLAB's system() call that creates them failed; on the cluster it works.)
make_missing_symlinks() {
    local made=0 dst src
    if [ -f "$EXPORT_ROOT/allruns/contrasts_manifest.tsv" ]; then
        while IFS=$'\t' read -r _ _ _ dst src; do
            if [ -n "$dst" ] && [ ! -e "$dst" ]; then ln -s "$src" "$dst"; made=$((made + 1)); fi
        done < <(tail -n +2 "$EXPORT_ROOT/allruns/contrasts_manifest.tsv")
    fi
    if [ -f "$EXPORT_ROOT/contrasts_manifest_sessions.tsv" ]; then
        while IFS=$'\t' read -r _ _ _ _ dst src; do
            if [ -n "$dst" ] && [ ! -e "$dst" ]; then ln -s "$src" "$dst"; made=$((made + 1)); fi
        done < <(tail -n +2 "$EXPORT_ROOT/contrasts_manifest_sessions.tsv")
    fi
    echo "Symlinks created from manifests: $made"
}

connames_arg=""
if [ -n "${CONNAMES:-}" ]; then
    connames_arg="connames = $CONNAMES;"
fi

for step in $STEPS; do
    echo "===== $step"
    case $step in
        contrasts)
            matlab -batch "spmpath = '$SPM_PATH'; glm_root = '$GLM_ROOT'; $connames_arg run('$REPO/matlab/first_lvl/add_session_contrasts_glm2.m');"
            ;;
        export)
            matlab -batch "addpath('$REPO/matlab'); addpath('$SPM_PATH'); spm('Defaults','fMRI'); spm_jobman('initcfg'); export_first_lvl_contrasts_with_sessions('$GLM_ROOT', '$EXPORT_ROOT', 'copy', false);"
            make_missing_symlinks
            ;;
        second)
            matlab -batch "spmpath = '$SPM_PATH'; export_root = '$EXPORT_ROOT'; run('$REPO/matlab/second_lvl/second_lvl_all_runs.m');"
            ;;
        sn23)
            # Two MATLAB sessions, because neither script clears its workspace
            matlab -batch "spmpath = '$SPM_PATH'; root_dir = '$EXPORT_ROOT'; run('$REPO/matlab/average_sn2_sn3_contrasts.m');"
            matlab -batch "spmpath = '$SPM_PATH'; export_root = '$EXPORT_ROOT'; run('$REPO/matlab/second_lvl_sn2_sn3.m');"
            ;;
    esac
done
