# Session log — concatenated-design first level (`glm2_all_runs_concat.m`): build and validate on sub-01

**Date:** 2026-09-24
**Companion note:** [2026-09-24_glm-concat-brief.md](2026-09-24_glm-concat-brief.md) — the brief this session worked from.

Base model is `glm2_all_runs` (pmod on both stimuli), not `glm2_chosen_all_runs` as the brief suggested.
Started as "build it"; most of the session went into checking that the concat design is the same
model as the per-session one apart from the intended pooling.

---

## Findings (all sub-01, n=1)

### 1. The concat design reproduces the per-session design column by column

`scripts/check_concat_design.py`: task regressors (`first_stim`, `second_stim`, `response`, `purple_frame`,
`points_feedback`, `nresp_screen`) equal the sum of the per-session columns, max |diff| = 0, including the
volumes right after run borders (no HRF-overhang difference). Pmod columns are also identical, and constants
match. Confounds differ by up to 0.44 only by a constant per run (SPM centres them over the whole session);
after projecting out the run constants the diff is ≤4e-6, so the fitted model is the same.

**SPM25 does not mean-centre pmods** (per-run means of the per-session pmod columns are non-zero), contrary to
what was said early in the session. Onset shifts, duration handling and block-diagonal confounds are therefore right.

### 2. Sn(1) Hval VIF drops from 18 to 2.9

`scripts/compare_concat_vifs.py` (unfiltered design matrices, same `est_vifs` as `utils/vif.py`):

| regressor | per-session Sn(1) / Sn(2) / Sn(3) | concat |
|---|---|---|
| 1st × Hval | 18.2 / 3.5 / 4.0 | 2.9 |
| 2nd × Hval | 18.2 / 3.9 / 3.7 | 2.9 |
| 1st stim / 2nd stim | 63.9 / 39.9 (Sn1) | 8.8 / 7.2 |

Not like-for-like: the concat model has one Hval coefficient for all runs, the per-session model has three.

### 3. The concat Hval regressor carries large run-level offsets

`scripts/check_concat_pmods.m`: the z-scoring in the bbt is across runs, not per run. Between-run share of variance:
Hval 0.46 (first stim) / 0.42 (second stim), Qval 0.03 / 0.01. Run means of second-stim Hval are −0.90 / −0.05 / +0.67.
Correlation of `second_stimxHval` with `second_stim`: −0.846 in run 1 (the Sn(1) problem), −0.008 / +0.309 in runs 2 / 3,
−0.067 over the concat session. Whether the offsets are acceptable is a modelling decision (open thread 3).

### 4. Estimates agree where they should; response / purple_frame do not, because of collinearity

`scripts/compare_concat_estimates.m`: same mask (58,748 voxels), ResMS 1.283 vs 1.277 (voxel-wise r = 1.000),
residual df 1272 vs 1253. t-map r for the stim contrasts 0.85–0.88 (Qval 0.85–0.86, Hval 0.59–0.66).
`response` and `purple_frame` have r ≈ 0 because the per-session model cannot separate them in runs 1–2
(VIF ≈ 3000): `scripts/check_concat_response_betas.py` shows beta sd 25–58 there vs 3.4 / 1.6 in the concat model,
corr(response, purple) across voxels −0.986 / −0.988, and no replication across runs (r −0.26…+0.22). Known
collinearity; the per-session `response` / `purple_frame` contrasts are not a usable reference.

---

## Code shipped

| What | State |
|---|---|
| `matlab/first_lvl/glm2_all_runs_concat.m` | merged to `main` (PR #3, squash `a91f2c1`) |
| `scripts/check_concat_design.py`, `scripts/compare_concat_vifs.py` | merged to `main` (PR #3, `a91f2c1`) |
| `scripts/check_concat_pmods.m`, `scripts/compare_concat_estimates.m`, `scripts/check_concat_response_betas.py`, this note | merged to `main` (PR #3, `a91f2c1`) |

Everything in PR #3 is a new file; no existing tracked file is edited, to avoid conflicts with other work.

## Data produced (cluster, sub-01 only)

| What | Where |
|---|---|
| Concat first level (job 6448594) | `spm_format/outputs/glm2_all_runs_concat_scrubbed_2026-09-24-11-16/sub-01/` (+ `design_sub01.png`, exported design CSVs) |
| Per-session `glm2_all_runs` reference (job 6449978) | `spm_format/outputs/glm2_all_runs_scrubbed_2026-09-24-12-42/sub-01/` (+ exported design CSVs) |

Verified as above (findings 1–4). No other subjects run: only sub-01 and sub-15 have SPM-ready data.

## Git state at session end

PR #2 (`spm-cluster-port` → `main`) was squash-merged as `c020c29`. `glm-concat` was rebased onto it
(`git rebase --onto origin/main 2216712 glm-concat`) and PR #3 was squash-merged as `a91f2c1`. The branch and both
`lh-concat` worktrees (local and cluster) were then removed. This note was last updated directly on `main`.

## Open threads

1. Run sub-15 (per-session `glm2_all_runs` + concat) and repeat the design, VIF and beta checks.
2. Decide whether the run-level Hval offset (finding 3) is acceptable; if not, try a per-run z-scored Hval variant.
3. Group level: `add_session_contrasts_glm2.m` and `submit_downstream.sh contrasts` do not apply to this design, and base
   contrasts have one weight where the per-session model sums three (slope ≈ 3 in finding 4).
4. `srun` MATLAB failed on some nodes with a container `resolv.conf` mount error (L4 nodes and `u24-cva0000-129`);
   `--exclude` of those nodes worked.
