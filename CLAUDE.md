# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session checkpoints

This repo keeps `session-notes/` — a git-tracked, per-session log where
findings are backed by rerunnable code, not just prose (see the `checkpoint`
skill). Numbers and claims computed in scratch code and only ever stated in
prose are exactly what gets lost between sessions; don't let that happen here.

Self-trigger the `checkpoint` skill (don't wait to be asked) in either of these
two situations:

1. **Right after stating a numeric/statistical conclusion or decision as
   settled** — e.g. writing/updating a findings-summary section, or answering
   "what did we conclude" — if it isn't yet backed by a saved notebook
   cell/script.
2. **Right before a topic or focus shift** in a long session (moving from one
   analysis thread to a different one), if the thread being left behind
   produced findings that aren't checkpointed yet.

Use `session-notes/2026-08-13_feedback-glmsingle-and-cue-redundancy.md` as the
template for section structure — not for length. Keep new findings to 2-4
lines each pointing at the notebook that backs them; push full derivations
into the notebook's markdown cells instead of restating them in the note.

## Jupyter notebooks

Always use `NotebookEdit` to create or edit `.ipynb` files — never `Write`.

## Python environment

For anything involving brain images (NIfTI files, fMRI data, nilearn, nibabel, etc.) use the `neuroim` conda environment:

```bash
conda run -n neuroim python script.py
# or activate it first:
conda activate neuroim
```

## Project overview

Neuroimaging (fMRI) analysis pipeline for a reward-learning habits study. The experiment has three sessions per subject: `learning1`, `learning2`, and `test`. Analysis uses both SPM12 (MATLAB) for GLM estimation and nilearn (Python) for design matrix inspection and secondary analyses.

## Compute environments

**All remote compute runs on the SLURM cluster (UZH sciencecluster). The analysis VM
(`uzh.vm`) is off-limits — do not ssh to it, run jobs on it, or write paths for it.**
See the `sciencecluster` skill for SLURM specifics.

| Where | What it's for |
|-------|---------------|
| Cluster (`hfluhr@cluster.s3it.uzh.ch`) | Everything that touches the full dataset — GLMsingle, decoding, searchlight, RSA |
| Local (this machine) | Editing code, notebooks, aggregating results, smoke tests against `dev_sample` |
| ~~VM (`uzh.vm`)~~ | **Off-limits.** Legacy paths below (`/mnt/data/…`, `/home/ubuntu/…`) refer to it |

Cluster access — two aliases, and they are not interchangeable:

```bash
ssh uzh.cluster.cmd "squeue -u hfluhr"   # inline commands, rsync, scp — always this one
ssh uzh.cluster                          # interactive shell only (has RemoteCommand=zsh)
```

**Never run compute on the login node**, not even a quick smoke test — wrap it in
`srun` or submit it with `sbatch`.

Get code onto the cluster by pushing and pulling, never by copying into its working tree:

```bash
git add … && git commit -m "…" && git push                                  # local
ssh uzh.cluster.cmd "cd ~/repos/learning-habits-analysis && git pull"       # cluster
```

Cluster paths (these are what the `multivariate/submit_*.sh` scripts inject):

| Location | Path |
|----------|------|
| Repo | `/home/hfluhr/repos/learning-habits-analysis` |
| Data root (`--base-dir`) | `/home/hfluhr/data/learninghabits` |
| Derivatives | `/home/hfluhr/shares-hare/ds-learning-habits/derivatives/{fmriprep-24.0.1-noSDC,glmsingle,decoding,searchlight,frem,rsa}` |
| Conda env | `/home/hfluhr/data/conda/envs/learning-habits` (build with `multivariate/build_env.sh`) |

For local smoke tests use `/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/`
(has `bbt.csv` and masks, but **no** GLMsingle betas).

## Running MATLAB scripts

> These paths are on the decommissioned VM and have no cluster equivalent recorded yet —
> ask before running the MATLAB/SPM pipeline.

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

## Subject lists

Always use `participants_mvpa.tsv` as the canonical subject list for multivariate analyses — never construct subject ranges (e.g. `seq 01 73`) since not all IDs exist. All `submit_*.sh` scripts default to this file when called with no arguments:

```bash
bash multivariate/submit_searchlight.sh        # correct — uses participants_mvpa.tsv
bash multivariate/submit_searchlight.sh 01 05  # correct — specific subjects only when intentional
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

For cluster paths — the ones you almost always want — see **Compute environments** above.
The SPM/MATLAB paths below are **VM paths, and the VM is off-limits**; they are kept only
as a record of where that pipeline last ran.

| Location | Path |
|----------|------|
| Raw GLM outputs (VM) | `/mnt/data/learning-habits/spm_format/outputs/` |
| Session contrast exports (VM) | `/mnt/data/learning-habits/spm_outputs/session_contrasts_exports/` |
| Local data, VM-side alternative | `/home/ubuntu/data/learning-habits/` |

GLM output directories are timestamped, e.g. `glm2_all_runs_scrubbed_2025-12-11-12-44`.

## Architecture

### GLM hierarchy (matlab/first_lvl/)

| GLM | Description |
|-----|-------------|
| `glm1` | Q/H-value modulation on first stimulus only |
| `glm2` | Q/H-value modulation on both stimuli — main workhorse |
| `glm2_all_runs` | GLM2 pooling all runs (no session separation in design matrix) |
| `glm2_chosen*` | Variants using the chosen-stimulus value instead of H-value |
| `glm3` | Choice variable (weighted sum of Q + H-values) |
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
