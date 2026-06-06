#!/bin/bash
# Build the learning-habits conda env on a compute node.
# Run from the repo root: bash multivariate/build_env.sh

set -eo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PATH="$HOME/data/conda/envs/learning-habits"

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=build_learning-habits
#SBATCH --partition=lowprio
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=$HOME/logs/build_learning-habits_%j.txt

set -eo pipefail

module load miniforge3
source "\$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="\$HOME/data/conda/pkgs"
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# env may already exist (partial build) — update rather than recreate
if [ -d "${ENV_PATH}" ]; then
    conda env update -p "${ENV_PATH}" -f "${REPO}/environment.yml" --prune
else
    conda env create -p "${ENV_PATH}" -f "${REPO}/environment.yml"
fi
echo "Done. Env at: ${ENV_PATH}"
EOF
