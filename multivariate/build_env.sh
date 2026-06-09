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

# Strip pip dependencies so conda doesn't invoke pip itself
# Removes "  - pip:" header and any indented pip packages below it
awk '!/^  - pip:/ && !/glmsingle/' "${REPO}/environment.yml" > /tmp/env_nopip.yml

if [ -d "${ENV_PATH}" ]; then
    conda env update -p "${ENV_PATH}" -f /tmp/env_nopip.yml --prune
else
    conda env create -p "${ENV_PATH}" -f /tmp/env_nopip.yml
fi

# Install pip packages separately with explicit SSL cert
"${ENV_PATH}/bin/pip" install \
    --cert /etc/ssl/certs/ca-certificates.crt \
    glmsingle==1.2

echo "Done. Env at: ${ENV_PATH}"
EOF
