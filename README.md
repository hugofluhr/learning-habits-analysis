# Learning Habits Analysis

fMRI analysis pipeline for a reward-learning habits study. Subjects complete three sessions (`learning1`, `learning2`, `test`). GLM estimation runs in SPM12 (MATLAB); design matrix inspection, QC, and secondary analyses run in Python (nilearn).

---

## Repository structure

| Directory | Contents |
|-----------|----------|
| `matlab/first_lvl/` | First-level GLM scripts (glm1–glm4 variants) |
| `matlab/second_lvl/` | Second-level one-sample t-tests |
| `matlab/connectivity/` | PPI analyses (PPPI toolbox) |
| `matlab/runners/` | Shell scripts that loop the 3-step session-contrast pipeline over all GLMs |
| `multivariate/` | GLMsingle single-trial betas, stimulus category decoding, and searchlight — runs on SLURM cluster |
| `modeling/` | RL/CK agent classes generating trial-level Q- and H-values |
| `utils/` | Python data model (`data.py`) and analysis helpers (`analysis.py`) |
| `scripts/` | One-off Python/R/shell scripts (BIDS conversion, data prep, export) |
| `notebooks/behavior/` | Behavioral analysis and computational modeling |
| `notebooks/data_prep/` | BIDS conversion and participant identification |
| `notebooks/glm/` | SPM GLM inspection (design matrices, VIFs, residuals, model comparisons) |
| `notebooks/nilearn_pipeline/` | Archived nilearn first/second-level pipeline development |
| `notebooks/ppi/` | PPI/connectivity inspection |
| `notebooks/qc/` | fMRI data quality control (SFNR, tSNR, scrubbing, timing) |
| `notebooks/results/` | Results-facing notebooks (figures, second-level summaries) |
| `notebooks/roi/` | ROI masks, PFC signal, coordinate decoding |
| `notebooks/social_risk/` | Social-risk task (separate dataset) |
| `physio/` | Physiological noise modelling (TAPAS PhysIO, SPM batch) |
| `cluster/` | Cluster-side data management scripts |
| `defacing/` | Defacing pipeline scripts |

---

## GLMs

| GLM | Parametric modulator |
|-----|---------------------|
| `glm1` | Q/H-value on first stimulus only |
| `glm2` | Q/H-value on both stimuli|
| `glm2_all_runs` | GLM2 pooling all runs (no session separation) - **preregistered model** |
| `glm2_chosen*` | Variants using chosen-stimulus value instead of first/second |
| `glm3` | choice variable (weighted sum of Q + H-values) |
| `glm4_learning_reward_chosen` | exploratory, using the feedback reward |

---

## Python utilities

- **`utils/data.py`**: `Subject` / `Block` data model; loads behavioural `.mat` files, builds trial/event DataFrames, attaches modeling data.
- **`utils/analysis.py`**: HRF convolution, contrast- and regressor-level VIF (`est_c_vifs`, `est_vifs`), design efficiency.
- **`modeling/classes.py`**: Task environment and RL/CK agents.

---

## Multivariate pipeline

Single-trial betas are estimated with GLMsingle, then used for stimulus category decoding and searchlight. All three steps run per-subject on the SLURM cluster:

```bash
bash multivariate/submit_glmsingle.sh    # GLMsingle type-D betas
bash multivariate/submit_decoding.sh     # whole-brain + visual cortex LinearSVC
bash multivariate/submit_searchlight.sh  # whole-brain searchlight (6 mm radius)
```

Each script wraps the corresponding `run_*.py` and injects cluster paths. QC notebooks are in `multivariate/glmsingle_qc.ipynb` and `multivariate/decoding_results.ipynb`.

---

## Infrastructure

- **VM** (`uzh.vm`): SPM12 at `/home/ubuntu/repos/spm12`; data at `/mnt/data/learning-habits/spm_format/outputs/` (raw GLM outputs) and `/mnt/data/learning-habits/spm_outputs/` (exported contrasts)
- **SLURM cluster** (`uzh.cluster`): multivariate pipeline; outputs at `shares-hare/derivatives/glmsingle` and `shares-hare/derivatives/decoding`
