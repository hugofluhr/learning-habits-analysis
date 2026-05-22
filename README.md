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
| `modeling/` | RL/CK agent classes generating trial-level Q- and H-values |
| `utils/` | Python data model (`data.py`) and analysis helpers (`analysis.py`) |
| `scripts/` | One-off Python/R/shell scripts (BIDS conversion, data prep, export) |
| `notebooks/` | Development and inspection notebooks |
| `results_notebooks/` | Results-facing notebooks (figures, second-level summaries) |

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

## Infrastructure

- SPM12 at `/home/ubuntu/repos/spm12` (remote VM)
- Data at `/mnt/data/learning-habits/spm_format/outputs/` (raw GLM outputs) and `/mnt/data/learning-habits/spm_outputs/` (exported contrasts)
