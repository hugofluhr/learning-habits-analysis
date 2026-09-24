# Session log — new GLM2-diff-timing all-runs script + porting the SPM first-level pipeline to the cluster

**Date:** 2026-09-10

Started as a request to write one new MATLAB script
(`glm2_all_runs_diff_timing.m`). Turned into porting the entire SPM
first-level data-prep chain to the cluster, because that script (like every
`glm*.m` script) depends on a flat `sub-XX/func/` layout with unzipped BOLD +
`_motion_with_dummies.txt` confounds that only ever existed on the
decommissioned VM. Mid-session, discovered and closed two more undocumented
gaps (git-blessed but never run on the cluster): SPM12 itself, and the
physIO derivative.

**Then the cluster port stopped being the point.** The sub-01 smoke test of
the (VM-identical, unmodified) `glm2_all_runs.m` failed on the cluster with
"Onsets or Durations contain NaN values" — chased that down and it isn't a
porting bug at all: it's a **pre-existing correctness bug present in the
actual historical VM pipeline across ~24 first-level GLM scripts**, silently
producing a garbage `purple_frame` regressor for the test session in every
past run. See finding 6 — this is the important result of the session, the
cluster port itself is now secondary and paused pending Hugo's decision on
how to scope/handle it.

---

## Findings

### 1. `glm2_all_runs_diff_timing.m` combines two existing scripts; contrast weights verified against column layout

Model structure (one first-level model over all 3 sessions, `points_feedback`
only for learning runs, non-response handling) copied from `glm2_all_runs.m`;
condition/pmod spec (`first_stim` no pmods, `second_stim` carrying
Qval1/Hval1/Qval2/Hval2, `orth=0`) copied from `glm2_diff_timing.m`. Verified
by hand that design-matrix columns 1-8 (`first_stim, second_stim, Qval1,
Hval1, Qval2, Hval2, response, purple_frame`) are in the same order in every
session regardless of whether `points_feedback`/`nresp_screen` follow — so
the existing sum/diff contrast weight vectors (`[0 0 1 0 1 0]` etc.) and
`sessrep='repl'` are valid. See the comment block above `connames` in the
script itself for the full reasoning; no separate notebook needed for a
MATLAB batch-spec script like this.

### 2. Missing pipeline stage 1/3: SPM12 wasn't on the cluster at all

Rsynced from this Mac's local copy (`/Users/hugofluhr/code/spm12`, the same
version referenced by the VM's `glm2_diff_timing.m`/`glm2_all_runs.m`) to
`~/repos/spm12` on the cluster. 238M, verified `spm.m` present after
transfer.

### 3. Missing pipeline stage 2/3: `prepare_bids_spm.py`/`prepare_bids_spm_add_dummies.py`/`spm_smooth_data.m` were hardcoded to the VM

These are the scripts that unzip fmriprep BOLD/mask into the flat
`spm_format_noSDC/sub-XX/func/` layout, write `_motion_with_dummies.txt`
confounds, and smooth. All three had `base_dir`/`spmpath` hardcoded to
`/home/ubuntu/...` or a local Mac path. Parametrized (CLI `--base-dir` for
the Python scripts, injectable MATLAB vars for `spm_smooth_data.m`),
defaulting to cluster paths. Branch `spm-cluster-port`, pushed.

### 4. Missing pipeline stage 3/3: the physIO derivative didn't exist on the cluster either

`Subject.load_physio_regressors()` (`utils/data.py`) requires a precomputed
`<derivatives>/physIO/sub-XX/ses-1/func/*.tsv` directory. Only raw
`*_physio.log` files existed on the cluster; regenerating them needs the
TAPAS PhysIO SPM toolbox + `physio/physio_all_subjects.m`, which is itself
hardcoded to a network-share mount and local Mac paths — a fourth porting
job. Per Hugo's explicit call (given his past experience of silently mixing
confounds from different VM instances — see Open thread 1), copied the
**already-computed** physIO derivative from a freshly-booted `uzh.vm`
instead of regenerating it, to avoid a toolbox/version discrepancy:

```bash
# VM -> local (this Mac has both uzh.vm and uzh.cluster.cmd configured; VM's
# own SSH access to the cluster was untested, so routed through here)
rsync -az uzh.vm:/mnt/data/learning-habits/bids_dataset/derivatives/physIO/ \
    <local_scratch>/physIO/
# local -> cluster
rsync -az <local_scratch>/physIO/ \
    uzh.cluster.cmd:/home/hfluhr/shares-hare/ds-learning-habits/derivatives/physIO/
```

3.2GB, completed and verified (`sub-01` files present at the expected path
on the cluster). Local scratch copy deleted afterward.

### 5. Verified against the VM before trusting the port (partial)

With the VM up anyway, checked the two things most likely to silently
diverge:

```bash
# fmriprep variant the VM's spm_format actually used (from its own prep log)
ssh uzh.vm "head -3 /mnt/data/learning-habits/spm_format/prepare_bids_for_spm_log_*.txt"
# -> fmriprep-24.0.1-noSDC — matches what submit_spm_prep.sh points at on the cluster

# bbt.csv identical VM vs cluster
ssh uzh.vm "md5 /mnt/data/learning-habits/bbt.csv"
ssh uzh.cluster.cmd "md5sum /home/hfluhr/data/learninghabits/bbt.csv"
# -> both 0a767b37a6b20f10e306e07965580ac0
```

Both match. **Not yet done**: diffing the actual confound file contents
(`_motion_with_dummies.txt`) or smoothed BOLD voxel data between VM and
cluster output for the same subject — that needs the cluster's own prep run
to finish first (Open thread 1 / see memory
`project_spm_cluster_port_verify_confounds.md`).

### 6. Pre-existing bug: `purple_frame` regressor is garbage in the test session, across ~24 first-level GLM scripts, in every past run — not a porting issue

Every first-level script builds `purple_frame` duration as
`block_resp.t_points_feedback - block_resp.t_purple_frame`, unconditionally
for all 3 sessions (learning1/learning2/test). `t_points_feedback` is NaN
for every test-block trial by design (no points-feedback event in the test
phase) — confirmed unchanged across every archived `bbt.csv` on the VM,
including the oldest (`old_bbt.csv`):

```bash
for f in old_bbt.csv bbt_20260401.csv bbt_052026_combined.csv bbt_062026_mf_cols.csv; do
  scp uzh.vm:/mnt/data/learning-habits/$f <local_scratch>/bbt_versions/
done
conda run -n neuroim python3 -c "
import pandas as pd
for f in ['old_bbt.csv','bbt_20260401.csv','bbt_052026_combined.csv','bbt_062026_mf_cols.csv']:
    df = pd.read_csv('<local_scratch>/bbt_versions/' + f)
    sub = df[(df.sub_id=='sub-01') & (df.block=='test')]
    print(f, 'test rows:', len(sub), 'NaN t_points_feedback:', sub['t_points_feedback'].isna().sum())
"
# -> every version: 136/136 rows NaN. Data never changed.
```

And the script line itself has been byte-identical since the file was
created (`git log --follow -p -- matlab/first_lvl/glm2_all_runs.m`, single
commit `8a4ce80`, never touched again) — so neither data nor code changed
between Hugo's past successful runs and this session.

The actual explanation: SPM12 version. The cluster got a newer/different
SPM12 build (rsynced from this Mac's `/Users/hugofluhr/code/spm12`) whose
`spm_run_fmri_spec.m` added a hard `error('Onsets or Durations contain NaN
values.')` check — that's what failed the smoke test. The VM's real
historical SPM12, recovered from `/mnt/data/homefolder_backup/repos/spm12`
(fresh VM boots have a bare root disk; only the `/mnt/data` volume persists),
is **r7771 (13-Jan-2020)** and has no such check:

```bash
ssh uzh.vm "grep -n 'Onsets or Durations contain NaN' /mnt/data/homefolder_backup/repos/spm12/config/spm_run_fmri_spec.m"
# -> no match. r7771 silently accepts NaN durations.
```

Confirmed what it actually did with them by pulling the real saved design
matrix from a past run (`glm2_all_runs_scrubbed_2025-12-11-12-44`, sub-01)
and inspecting the `Sn(3) purple_frame*bf(1)` column (column 87 of 141,
0-indexed 86):

```bash
scp uzh.vm:/mnt/data/learning-habits/spm_format/outputs/glm2_all_runs_scrubbed_2025-12-11-12-44/sub-01/sub-01_design_matrix.csv <local_scratch>/vm_glm2_all_runs_ref/
scp uzh.vm:/mnt/data/learning-habits/spm_format/outputs/glm2_all_runs_scrubbed_2025-12-11-12-44/sub-01/sub-01_column_names.txt <local_scratch>/vm_glm2_all_runs_ref/
grep -n "Sn(3)" <local_scratch>/vm_glm2_all_runs_ref/sub-01_column_names.txt   # purple_frame is line 87 -> 0-indexed col 86
conda run -n neuroim python3 -c "
import pandas as pd
df = pd.read_csv('<local_scratch>/vm_glm2_all_runs_ref/sub-01_design_matrix.csv', header=None)
col = df.iloc[:, 86]
print('min', col.min(), 'max', col.max(), 'any NaN:', col.isna().any())
"
# -> min -1051.1, max 1.1, no NaN. Every other column in the file is in the
#    normal HRF-convolved range (~-1 to 1). This one is a numeric-garbage
#    artifact, not a clean or merely-absent regressor.
```

**Scope, established this session** (not fully quantified beyond this):
- Test session only (learning1/learning2 populate `t_points_feedback`
  normally).
- Every subject, deterministically — the NaN is present for 100% of
  test-block trials, not subject- or trial-specific.
- 24 of ~26 `matlab/first_lvl/*.m` scripts share the exact line
  (`grep -rl "t_points_feedback - block_resp.t_purple_frame" matlab/first_lvl/*.m`),
  including `glm2_diff_timing.m` (source script for finding 1's new
  `glm2_all_runs_diff_timing.m`, and itself already has a `test`-only model
  block with this bug).
- **Not established**: how much this biased other test-session regressors'
  beta estimates (shared design-matrix inversion — an extreme column can
  degrade conditioning beyond just its own contrast), or which specific past
  analyses/figures relied on affected test-session results. Hugo asked to
  stop and assess before going further — no fix, no rerun, no further
  cluster work happened after this was found.

---

## Code shipped

| File | Change | Git state |
|---|---|---|
| `matlab/first_lvl/glm2_all_runs_diff_timing.m` | New script (finding 1) | committed, branch `glm2-all-runs-diff-timing`, pushed |
| `scripts/prepare_bids_spm.py`, `scripts/prepare_bids_spm_add_dummies.py`, `matlab/spm_smooth_data.m` | Parametrized away from hardcoded VM paths (finding 3) | committed, branch `spm-cluster-port`, pushed |
| `scripts/submit_spm_prep.sh`, `scripts/submit_spm_smooth.sh`, `scripts/submit_glm2_all_runs_diff_timing.sh` | New sbatch wrappers, each with a subject-list override for smoke testing | committed, branch `spm-cluster-port`, pushed |

The two branches are deliberately separate (new analysis script vs.
cluster-infra port) and neither is merged to `main` yet.

## Data produced

- SPM12 copied to cluster `~/repos/spm12` (238M).
- physIO derivative copied to cluster
  `~/shares-hare/ds-learning-habits/derivatives/physIO` (3.2GB), sourced from
  `uzh.vm`.
- `sub-01` smoke test, stage 1/3 (`submit_spm_prep.sh 01`, SLURM job
  5725157) — **COMPLETED** (confirmed just as this note was being written).
  Unzipped BOLD+mask for all 3 runs and wrote `_motion.txt`/`_events.mat`/
  `_motion_with_dummies.txt` under
  `/home/hfluhr/data/learninghabits/spm_format_noSDC/sub-01/func/`. Not yet
  eyeballed in detail (e.g. confound file column count, dummy-regressor
  count) — do that before/while running stage 2. Ran on a local-only merge
  of both branches (`smoke-test-merge`, cluster repo only, not pushed —
  recreate with `git merge origin/glm2-all-runs-diff-timing` on top of
  `spm-cluster-port` if it's gone).

## Git state at session end

- Local (this Mac): on `spm-cluster-port`. Both `glm2-all-runs-diff-timing`
  and `spm-cluster-port` pushed to origin, both branched off `main` at
  `20b3512` (same commit — `main` untouched this session).
- Cluster (`uzh.cluster.cmd:~/repos/learning-habits-analysis`): on local
  branch `smoke-test-merge` (`spm-cluster-port` + `origin/glm2-all-runs-diff-timing`
  merged, cluster-local only, not pushed).

## Open threads — resume here on another machine

0. **[BLOCKING, highest priority] Decide how to handle finding 6** (the
   `purple_frame`/test-session garbage-regressor bug spanning ~24 first-level
   scripts and every past run). Concretely still unknown: (a) how much it
   actually biased other test-session betas/contrasts in affected scripts —
   would need re-running or re-inspecting a few past GLMs with a corrected
   `purple_frame` duration and diffing betas; (b) which downstream
   results/figures/session-notes drew on test-session output from these
   scripts and might need a caveat or rerun; (c) whether to fix the bug now
   (duration doesn't depend on `t_points_feedback` for `r==3`) or first
   quantify (a)/(b) before touching anything. Nothing else in this list
   should block on this, but no full-cohort cluster run should happen with
   the current buggy `purple_frame` logic still in any script being run.

1. **Finish the sub-01 smoke test.** Stage 1/3 (`submit_spm_prep.sh 01`, job
   5725157) COMPLETED — not yet inspected in detail. Run stages 2 and 3
   next, inspecting output after each before moving on:
   ```bash
   ssh uzh.cluster.cmd "cd ~/repos/learning-habits-analysis && bash scripts/submit_spm_smooth.sh"
   # after it completes, check smoothed_5mm_*.nii exist under sub-01/func/, then:
   ssh uzh.cluster.cmd "cd ~/repos/learning-habits-analysis && bash scripts/submit_glm2_all_runs_diff_timing.sh sub-01"
   # after it completes, inspect SPM.mat (design matrix sane, all contrasts
   # estimable — especially Qval_sum/Hval_sum/Qval_diff/Hval_diff and that
   # the test-session columns don't break the 'repl' contrasts, see finding 1)
   ```
   (cluster repo must be on `smoke-test-merge`, or recreate it — see Data
   produced above — since `submit_glm2_all_runs_diff_timing.sh` needs both
   branches' files present together.)

2. **Verify cluster-prepped data actually matches the VM's**, per Hugo's
   explicit concern about past cross-VM confound mismatches — not just
   fmriprep-version/bbt.csv (finding 5, done) but the actual
   `_motion_with_dummies.txt` content and smoothed BOLD for the same
   subject/run. Do this once sub-01's prep step (above) has produced output
   to diff against. `uzh.vm` was still up at session end if not yet shut
   down; see `project_spm_cluster_port_verify_confounds.md` memory.

3. **After sub-01 checks out**: merge `spm-cluster-port` into
   `glm2-all-runs-diff-timing` (or vice versa) for real, push, then decide
   whether to run `submit_spm_prep.sh` / `submit_spm_smooth.sh` /
   `submit_glm2_all_runs_diff_timing.sh` with no subject argument (full
   `participants_mvpa.tsv` cohort, ~59 subjects) or open PRs first.

4. Neither branch is merged to `main`. `glm2-all-runs-diff-timing`'s contrast
   `delete` flag was deliberately set to `1` (wipe-and-redefine) per Hugo's
   request, differing from `glm2_all_runs.m`'s `delete=0` — intentional, not
   an inconsistency to fix.
