#!/bin/bash
# Submit RSA searchlight group-level cluster-FWE inference as a SLURM array job —
# one job per TERM (not per subject: this is a single group-level test over
# all subjects' already-computed searchlight maps, run_rsa_group_stats.py).
#
# Local smoke test (2026-09-03, n=58, interaction_value_freq): n_perm=200 took
# ~10s on 8 cores. n_perm=10000 (the script's default) extrapolates to
# ~10 min/term without --tfce, ~50 min/term with --tfce. This job is cheap —
# the 1h walltime below is generous headroom, not a real expectation.
#
# Usage (from repo root):
#   bash multivariate/submit_rsa_group_stats.sh                 # all 6 terms
#   bash multivariate/submit_rsa_group_stats.sh value frequency interaction_value_freq
#   TFCE=1 bash multivariate/submit_rsa_group_stats.sh interaction_value_freq
#   N_PERM=2000 bash multivariate/submit_rsa_group_stats.sh value   # quicker rerun
#
# run_rsa_group_stats.py skips terms already done (checks for <term>_params.json),
# so this is resumable and safe to re-submit.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
INPUT_DIR="${INPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/rsa_searchlight}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hfluhr/shares-hare/ds-learning-habits/derivatives/rsa_searchlight_group_stats}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

N_PERM="${N_PERM:-10000}"
THRESHOLD="${THRESHOLD:-0.001}"
TFCE_FLAG=""
if [ "${TFCE:-0}" = "1" ]; then
    TFCE_FLAG="--tfce"
fi

ALL_TERMS=(category value frequency second_stim_value choice_rate interaction_value_freq)

# ---------------------------------------------------------------------------
# Build term list
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
TERMS_FILE="${LOG_DIR}/terms_rsa_group_stats.txt"

if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@" > "$TERMS_FILE"
else
    printf "%s\n" "${ALL_TERMS[@]}" > "$TERMS_FILE"
fi

N=$(wc -l < "$TERMS_FILE")
if [ "$N" -eq 0 ]; then
    echo "ERROR: term list is empty" >&2
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: searchlight input directory not found: ${INPUT_DIR}" >&2
    exit 1
fi

echo "Submitting ${N} term(s) (array 1-${N}), n_perm=${N_PERM}, threshold=${THRESHOLD}, tfce=${TFCE:-0}:"
cat "$TERMS_FILE"
echo

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------
sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=rsa_grpstats
#SBATCH --output=${LOG_DIR}/rsa_group_stats_%A_%a.out
#SBATCH --error=${LOG_DIR}/rsa_group_stats_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --array=1-${N}

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TERM=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${TERMS_FILE}")
echo "=== term=\${TERM}  (task \${SLURM_ARRAY_TASK_ID}/\${SLURM_ARRAY_TASK_COUNT}) ==="

/home/hfluhr/data/conda/envs/learning-habits/bin/python -u "${REPO}/multivariate/run_rsa_group_stats.py" \\
    --input-dir "${INPUT_DIR}" \\
    --term "\$TERM" \\
    --output-dir "${OUTPUT_DIR}" \\
    --n-perm ${N_PERM} \\
    --threshold ${THRESHOLD} \\
    ${TFCE_FLAG} \\
    --n-jobs \$SLURM_CPUS_PER_TASK
EOF
