# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Neuroimaging (fMRI) analysis pipeline for a reward-learning habits study. The experiment has three sessions per subject: `learning1`, `learning2`, and `test`. Analysis uses both SPM12 (MATLAB) for GLM estimation and nilearn (Python) for design matrix inspection and secondary analyses.

## Running MATLAB scripts

SPM12 is at `/home/ubuntu/repos/spm12`. Load it and run a script:

```bash
module load matlab/r2023a
matlab -nodisplay -nosplash -nodesktop -r "run('/home/ubuntu/repos/learning-habits-analysis/matlab/first_lvl/glm2_all_runs.m'); exit;"
```

To inject a variable before running a script (the scripts use `if ~exist('var','var') || isempty(var)` guards to preserve externally-set variables):

```bash
matlab -nodisplay -r "glm_root = '/path/to/glm'; run('script.m'); exit"
```

Always pipe through `tee` to log output:

```bash
matlab -nodisplay -r "..." 2>&1 | tee logfile.log
```

## Session contrasts + export + second-level pipeline

The three-step pipeline is documented in `INSTRUCTIONS_session_contrasts_and_secondlvl.md`. Runner scripts that loop over all three GLMs are in `matlab/runners/`:

```bash
bash matlab/runners/run_step1_add_session_contrasts.sh   # appends per-session contrasts to SPM.mat
bash matlab/runners/run_step2_export_contrasts.sh         # exports contrast images by session
bash matlab/runners/run_step3_second_lvl.sh               # one-sample t-tests per contrast
```

Key scripts:
- `matlab/first_lvl/add_session_contrasts_glm2.m` — safe to re-run (skips subjects already processed)
- `matlab/export_first_lvl_contrasts_with_sessions.m` — MATLAB function, use `copy=true` for self-contained output
- `matlab/second_lvl/second_lvl_all_runs.m` — loops over `allruns/`, `session-01/`, `session-02/`, `session-03/` automatically

Excluded subjects at second level: `sub-44, sub-48, sub-68, sub-17, sub-31`.

## SPM export (non-session variant)

```bash
# Edit paths inside the script, then:
bash scripts/spm_export_first_lvl.sh
```

## Data paths

| Location | Path |
|----------|------|
| Raw GLM outputs | `/mnt/data/learning-habits/spm_format/outputs/` |
| Session contrast exports | `/mnt/data/learning-habits/spm_outputs/session_contrasts_exports/` |
| Local data (alternative) | `/home/ubuntu/data/learning-habits/` |

GLM output directories are timestamped, e.g. `glm2_all_runs_scrubbed_2025-12-11-12-44`.

## Architecture

### GLM hierarchy (matlab/first_lvl/)

| GLM | Description |
|-----|-------------|
| `glm1` | Q/H-value modulation on first stimulus only |
| `glm2` | Q/H-value modulation on both stimuli — main workhorse |
| `glm2_all_runs` | GLM2 pooling all runs (no session separation in design matrix) |
| `glm2_chosen*` | Variants using the chosen-stimulus value instead of H-value |
| `glm3` | Prediction error modulation |
| `glm4_learning_reward_chosen` | Learning/reward/chosen combined |

### Python data model (utils/data.py)

- `Subject(base_dir, subject_id, ...)` — loads behavioral `.mat` files; exposes `.learning1`, `.learning2`, `.test` (each a `Block`).
- `Block` — wraps a single fMRI run; `.trials` DataFrame, `.events` DataFrame, `.extended_trials` (after `add_modeling_data()`).
- `StimuliInfo` — stimulus assignment, reward values, presentation frequencies.
- `create_dummy_regressors(sample_mask, n_scans)` — builds scrubbing regressors for excluded volumes.

Key utility: `Subject.load_confounds(run)` calls nilearn's `load_confounds` with motion + WM/CSF + scrubbing strategy and trims cosine columns to 5.

### Python analysis functions (utils/analysis.py)

- `compute_parametric_modulator()` — convolves a trial-level modulator with HRF; supports `center` or `zscore` normalization.
- `est_c_vifs(desmat, contrasts)` — contrast-level VIF (Mumford method).
- `est_vifs(desmat, regressors)` — traditional regressor-level VIF.
- `est_efficiency(desmat, contrasts)` — design efficiency (1/variance of contrast estimate).

### Computational models (modeling/classes.py)

Implements the task environment and RL/CK (Rescorla-Wagner / Choice Kernel) agents used to generate trial-level Q- and H-values that feed into the GLM parametric modulators.

### Connectivity (matlab/connectivity/)

PPI (psychophysiological interaction) analyses using PPPI toolbox. `PPPI_wrapper.m` is the entry point; `extract_voi.m` extracts the seed region timeseries.

## MATLAB scripting conventions

- Scripts use `diary(log_path)` for logging when run non-interactively.
- Always wrap hardcoded path assignments in `if ~exist('var','var') || isempty(var)` so runner scripts can inject values via `-r "var='...'; run('script.m')"`.
- Never use `clear;` at the top of scripts that may receive injected variables.
- Use `delete=0` with `spm_contrasts` to append (not overwrite) contrasts.
- Ghost contrasts (defined in `SPM.xCon` but never estimated) appear in some GLMs — the export script handles these with a `[SKIP]` warning rather than an error.
