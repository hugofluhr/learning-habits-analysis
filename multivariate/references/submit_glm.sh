#!/bin/bash
#SBATCH --job-name=fit_glmsingle_ses-both_smoothed
#SBATCH --output=/home/aldavy/logs/fit_glmsingle_both_smoothed_%A-%a.txt
#SBATCH --error=/home/aldavy/logs/fit_glmsingle_both_smoothed_%A-%a.err
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=3:30:00
#SBATCH --mem=64G  # Request more memory
#SBATCH --array=1-3 
# change for test subjects

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate numloss_env

# cap threaded libs to avoid oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MALLOC_ARENA_MAX=2

export PARTICIPANT_LABEL="pil$(printf "%02d" $SLURM_ARRAY_TASK_ID)" # remove pil for test subjects

if [ "$PARTICIPANT_LABEL" = "pil01" ]; then
  N_RUNS=5
else
  N_RUNS=10
fi

python $HOME/numloss/numloss/glm/estimate_single_trials.py \
  "$PARTICIPANT_LABEL" \
  0 \
  --bids_folder /shares/zne.uzh/aldavy/ds-numloss \
  --n_runs $N_RUNS \
  --smoothed