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
| SPM | `/home/hfluhr/repos/spm12` — despite the folder name this is an **SPM25** development checkout, not SPM12 r7771 as on the VM |
| SPM-ready data | `/home/hfluhr/data/learninghabits/spm_format` (flat `sub-XX/func/`; see its `README.md`) |
| bbt for GLMs | `/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv` (baseline modelling values; used by every first level) |
| First-level outputs | `/home/hfluhr/data/learninghabits/spm_format/outputs/<model>_<date>/` |
| Contrast exports + second levels | `/home/hfluhr/data/learninghabits/spm_outputs/<model>_<date>/` |

For local smoke tests use `/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/`
(has `bbt.csv` and masks, but **no** GLMsingle betas).

## Running MATLAB scripts

The SPM pipeline runs on the cluster through sbatch wrappers in `scripts/` (run them from the
repo root on the cluster, never on the login node itself). Each supports `DRY_RUN=1`, which prints
the job and calls `sbatch --test-only`.

```bash
bash scripts/submit_spm_prep.sh                         # unzip fMRIPrep noSDC + confounds -> spm_format/ (every bbt subject)
bash scripts/submit_spm_smooth.sh                       # 5 mm smoothing, skips already-smoothed files
bash scripts/submit_first_lvl.sh glm2_chosen_all_runs.m # any matlab/first_lvl/*.m, SLURM array, one subject per task
bash scripts/submit_first_lvl.sh glm2_all_runs.m sub-01 sub-15   # specific subjects
bash scripts/submit_downstream.sh all <first-level folder>       # session contrasts -> export -> second levels
```

Only scripts that have been ported can run on the cluster: they need injectable paths with
cluster defaults, no `clear;`, and an injectable `current_date`/`output_dir`. As of 2026-09-23
that means `glm2_all_runs.m` and `glm2_chosen_all_runs.m`; the others still carry VM paths
(see the vault note "LH - Model re-run tracker").

The wrappers inject variables before `run()`. The scripts keep externally-set values through
`if ~exist('var','var') || isempty(var)` guards:

```bash
module load matlab
matlab -batch "glm_root = '/path/to/glm'; run('script.m');"
```

Use `matlab -batch`, not `-r "...; exit"`: with `-r`, an error inside `run()` leaves MATLAB at
its prompt until the job's walltime runs out, whereas `-batch` exits non-zero and SLURM marks the job FAILED.

## Subject lists

Always use `participants_mvpa.tsv` as the canonical subject list for multivariate analyses — never construct subject ranges (e.g. `seq 01 73`) since not all IDs exist. All `submit_*.sh` scripts default to this file when called with no arguments:

```bash
bash multivariate/submit_searchlight.sh        # correct — uses participants_mvpa.tsv
bash multivariate/submit_searchlight.sh 01 05  # correct — specific subjects only when intentional
```

The SPM first-level pipeline is the exception: it runs on **every subject in the bbt** (62;
the GLM scripts skip sub-04 and sub-45), and exclusions happen at second level.
`participants_mvpa.tsv` is not a subset of the bbt (sub-46 has no bbt row), so don't use it there.

## Session contrasts + export + second-level pipeline

The pipeline is documented in `INSTRUCTIONS_session_contrasts_and_secondlvl.md`, which predates the
cluster port and still describes VM paths. On the cluster, one wrapper submits each step for one
first-level folder:

```bash
bash scripts/submit_downstream.sh contrasts <glm>   # appends per-session contrasts to each SPM.mat
bash scripts/submit_downstream.sh export <glm>      # exports contrast images by session, creates symlinks
bash scripts/submit_downstream.sh second <glm>      # one-sample t-tests for allruns/ and session-0X/
bash scripts/submit_downstream.sh sn23 <glm>        # average Sn2+Sn3 contrasts, then second level (drops Sn1)
bash scripts/submit_downstream.sh all <glm>         # all four, chained with afterok
```

`add_session_contrasts_glm2.m` takes its conditions from each model's own contrasts: every existing
t-contrast named after a regressor, so combined contrasts like `Qval_sum` are left out. A name that
matches no regressor is an error. `CONNAMES="{'first_stim', ...}"` overrides the list.

Key scripts:
- `matlab/first_lvl/add_session_contrasts_glm2.m` — safe to re-run (skips subjects already processed)
- `matlab/export_first_lvl_contrasts_with_sessions.m` — MATLAB function, use `copy=true` for self-contained output
- `matlab/second_lvl/second_lvl_all_runs.m` — loops over `allruns/`, `session-01/`, `session-02/`, `session-03/` automatically

Excluded subjects at second level: `sub-44, sub-48, sub-68, sub-17, sub-31`.

## SPM export (non-session variant)

VM-era, not ported to the cluster:

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
- Always wrap hardcoded path assignments in `if ~exist('var','var') || isempty(var)` so runner scripts can inject values via `-batch "var='...'; run('script.m');"` (see "Running MATLAB scripts" for why `-batch` rather than `-r`).
- Never use `clear;` at the top of scripts that may receive injected variables.
- First-level scripts run as SLURM arrays, one subject per task, so `current_date` (or `output_dir`)
  must be injectable: all tasks then write into one output folder. They also need one diary log per
  subject set, so parallel tasks don't overwrite each other.
- Use `delete=0` with `spm_contrasts` to append (not overwrite) contrasts.
- Ghost contrasts (defined in `SPM.xCon` but never estimated) appear in some GLMs — the export script handles these with a `[SKIP]` warning rather than an error.
