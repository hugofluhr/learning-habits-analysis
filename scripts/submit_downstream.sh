#!/bin/bash
# Submit the steps that follow a first level, run in order in one SLURM job.
#
# Usage: bash scripts/submit_downstream.sh <glm> [step ...]
#   <glm>   first-level folder name under spm_format/outputs, or a full path
#   step    any of the steps below, in this order (default: all four)
#     contrasts  per-session t-contrasts appended to each SPM.mat
#     export     contrast images by session, into spm_outputs/<glm>
#     second     one-sample t-tests for allruns/ and session-0X/
#     sn23       average session-02 and session-03, then one-sample t-tests
#
# CONNAMES="{'first_stim'}" overrides the conditions used for the session contrasts.
# By default they come from the model's own contrasts.
# The job itself is scripts/slurm/downstream.sbatch.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    sed -n '2,14p' "$0" >&2
    exit 1
fi

GLM="$1"
shift

export REPO="$(cd "$(dirname "$0")/.." && pwd)"
export SPM_PATH=/home/hfluhr/repos/spm12
export CONNAMES="${CONNAMES:-}"
export STEPS="${*:-contrasts export second sn23}"

if [[ "$GLM" == */* ]]; then
    export GLM_ROOT="$GLM"
else
    export GLM_ROOT="/home/hfluhr/data/learninghabits/spm_format/outputs/$GLM"
fi
export EXPORT_ROOT="/home/hfluhr/data/learninghabits/spm_outputs/$(basename "$GLM_ROOT")"

if [ ! -d "$GLM_ROOT" ]; then
    echo "First-level folder not found: $GLM_ROOT" >&2
    exit 1
fi
for step in $STEPS; do
    case $step in
        contrasts|export|second|sn23) ;;
        *) echo "Unknown step: $step" >&2; exit 1 ;;
    esac
done

mkdir -p "$EXPORT_ROOT/logs"
echo "Steps: $STEPS"
echo "First level: $GLM_ROOT"
echo "Export:      $EXPORT_ROOT"
sbatch --output="$EXPORT_ROOT/logs/downstream_%j.out" \
    --error="$EXPORT_ROOT/logs/downstream_%j.err" \
    "$REPO/scripts/slurm/downstream.sbatch"
