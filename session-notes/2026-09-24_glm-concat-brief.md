# Brief — side quest: a concatenated-design first-level model

**Date:** 2026-09-24 · **Branch:** `glm-concat` (from `spm-cluster-port` @ `2216712`)
**Worktrees:** local `~/phd_local/repositories/lh-concat`, cluster `~/repos/lh-concat`.
The main checkouts (`learning-habits-analysis`) belong to the ongoing cluster-port / re-run work: **don't switch or edit them from here.**

This is a hand-off to a fresh session. Read CLAUDE.md (the SPM-on-the-cluster sections) and
`session-notes/2026-09-23_spm-cluster-port-validation.md` first.

## Goal

Build a first-level model that **concatenates the 3 runs into one session** instead of the current per-session design
(`sessrep = 'repl'`, one block of task regressors per run). This is Philippe's suggestion (vault: [[Reply to Philippe]], top line:
"concatenate instead of per session design to get rid of the VIF issue with h-values, don't forget constant per session term").
The target is the Sn(1) Hval problem ([[VIF issue]]). In Sn(1), H-values barely vary, so the Hval pmod is almost perfectly anti-correlated with
`second_stim` (r ≈ −0.94). A single pmod regressor spanning all runs should dilute that. Concatenation was never tried before
(vault daily 2026-09-23, "Investigating why we never went for proper concatenation").

## Starting point

- Copy `matlab/first_lvl/glm2_chosen_all_runs.m` to a new script (e.g. `glm2_chosen_all_runs_concat.m`). It's already ported to the cluster
  (injectable paths, no `clear;`, injectable `current_date`) and has the purple_frame fix. Keep those properties.
- Data: `/home/hfluhr/data/learninghabits/spm_format` (currently sub-01 and sub-15 only; the full cohort comes with the main work's Phase B).
  bbt: `bbt_062026_mf_cols.csv`. SPM: `~/repos/spm12` (SPM25).
- Run with `scripts/submit_first_lvl.sh <script.m> sub-01` from `~/repos/lh-concat` on the cluster. The wrapper finds the repo from
  its own path, so the worktree's scripts are used. Output lands in `spm_format/outputs/<the new script's prefix>_<date>`.

## How SPM does it (checked in `~/repos/spm12/spm_fmri_concatenate.m`)

1. Specify **one session**: all volumes of the 3 runs, with onsets of runs 2 and 3 shifted by the cumulative run length
   (`nscan × TR`; TR = 2.33384; about 426 / 426 / 593 volumes, but read the actual counts from the files).
2. Call `spm_fmri_concatenate(SPM.mat, [n1 n2 n3])` **between specification and estimation**. The current scripts run specification and
   estimation in one batch (`matlabbatch{1}`/`{2}`), so this has to be split.
3. `spm_fmri_concatenate` replaces the single constant with one constant per run (`Sn(i) constant`), so **don't add run constants yourself**.
   It restores per-run high-pass filtering and per-run AR (`AR(1)` becomes `AR(0.2)` per run).

Pitfalls to handle:
- **4D files:** SPM expands a 4D NIfTI into volumes only when a session has exactly one file. With 3 files in one session it would
  count 3 scans. Expand each run with `spm_select('Expand', file)` and concatenate the volume lists.
- **Confounds:** each run's `_motion_with_dummies.txt` has a different number of columns (scrubbing dummies). The standard layout is block-diagonal
  (each run's confounds in their own columns, zero outside that run's rows), written to one file for `multi_reg`.
- **`points_feedback`** exists only in the learning runs: in one session it's simply a condition with learning onsets only.
- **Contrasts:** after concatenation `SPM.Sess` has one entry but `SPM.nscan` has three. The per-session contrast code (`min_cols`, `repl`)
  and `add_session_contrasts_glm2.m` / `submit_downstream.sh contrasts` don't apply. Plain contrasts on the single set of columns, with
  sessrep `'none'`, should.
- **HRF overhang** across run borders (the function's help mentions it): note it, likely minor.

## Open decisions (ask Hugo)

1. **Base model:** GLM2 chosen (`glm2_chosen_all_runs`, simplest; its March run is the reference) or preregistered GLM2 (`glm2_all_runs`)?
2. **Confound layout:** block-diagonal per run (standard, recommended) or merged columns?
3. **Pmod normalisation:** z-scoring across all runs is what the `_zscore` columns already do, and fits one concatenated regressor.
4. Whether this has to wait for PR #2 (`spm-cluster-port` → `main`). This branch is built on it, so rebase it after the merge.

## Validation idea

- sub-01 first (then sub-15, the most heavily scrubbed): check the design matrix (one session, 3 run constants, block-diagonal confounds,
  onsets at the right times; e.g. plot `SPM.xX.X`).
- VIFs: `scripts/vif_report.py` on the new output vs the per-session model's output. The per-session reference run is
  `spm_format/outputs/glm2_chosen_all_runs_scrubbed_2026-09-23-15-07` (sub-01, sub-15). Success means the Hval VIF drops below
  threshold without changing the modelling data.
- `scripts/compare_first_lvl.py` matches regressors by name, so it won't pair columns across the two designs. Compare contrast images instead.

## Constraints (from CLAUDE.md, repeated because they matter)

- All compute on the cluster via sbatch/srun, never on the login node; code reaches the cluster via push + `git pull` in `~/repos/lh-concat`.
- Record findings in a session note on this branch; vault updates follow the vault's CLAUDE.md, and every edit is logged in its edits log.
