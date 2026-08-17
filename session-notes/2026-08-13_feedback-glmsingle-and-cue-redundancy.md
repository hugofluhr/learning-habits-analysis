# Session log — feedback-locked GLMsingle, and how redundant it is with the cue betas

**Date:** 2026-08-13
**Companion note:** [2026-08-11_glmsingle-conditions-and-reward-decoding.md](2026-08-11_glmsingle-conditions-and-reward-decoding.md)
— the analysis behind findings 1 and 2 below, with the derivations.

Started as "build a GLMsingle variant locked to reward-feedback onset". Detoured into whether
the GLMsingle condition choice explained the weak reward decoding, which turned up the more
consequential result (finding 2).

---

## Findings

### 1. The GLMsingle condition choice was not the problem

Condition labels affect exactly two things — the GLMdenoise PC count and the fracridge fraction
(verified against upstream `glmsingle.py` / `ssq/calcbadness.py`; beta estimation and HRF
selection ignore them entirely). `betas_qc_decoding.ipynb` bounds their total influence at
**~1.5 accuracy points** (B→D: +0.011 whole brain, +0.015 visual cortex; the ridge step alone is
~0.000, and `standardize=True` downstream largely undoes it anyway).

Using reward level as the condition would have been **worse**, not better: reward level is a
strict coarsening of stimulus identity, so identity conditions already preserve a superset of the
reward information. Stimulus identity is the right choice — it is the finest partition available
with adequate repeats.

### 2. Objective reward level is a deterministic function of stimulus identity

0.000 within-(subject × stimulus) variance, all 496 subject×stimulus cells. Every subject's four
categories take the value pairs `(1,5)`, `(2,3)`, `(2,4)`, `(3,4)`, so after `(run × category)`
demeaning the surviving low/high contrast **is** the individual-stimulus contrast.

**This suggests the reward decode in `qvalue_classification_results.ipynb` is identity decoding
under a value label**, contradicting that notebook's stated conclusion that the effect "survives
the category-confound control". It also predicts the exact observed profile: visual cortex 0.615
≫ whole brain 0.549, vmPFC/striatum flat at chance.

Not yet tested, and cheap: re-run the reward classification demeaning by **stimulus identity**
rather than category. Prediction: it collapses to chance.

### 3. The feedback betas are largely redundant with the cue betas

Predicted from timing (feedback onset is 1.95 s after cue onset, SD 0.16 s, at TR 2.334 s → 82.4%
of feedback events land exactly 1 TR after their cue's TR), then measured on n=59:

| whole brain | matched | shuffled | adjacent |
|---|---|---|---|
| type B | **0.782** (IQR 0.470–0.818) | −0.000 | 0.213 |
| type D | **0.773** (IQR 0.438–0.802) | −0.005 | 0.050 |

r² ≈ 0.60 shared variance, and that is a **lower** bound — two noisy estimates of the same
quantity correlate at their reliability, not at 1.

Side results: the beta-type confound that motivated `--beta-type` is negligible in practice
(corr(B,D) = 0.991 across subjects, D−B = −0.020 median). The adjacent baseline dropping 0.213→
0.050 from B to D is direct evidence GLMdenoise/ridge removes real temporal autocorrelation
between neighbouring trials.

**Consequence:** for ~2/3 of the sample the feedback betas re-express the cue response. Combined
with finding 2, a reward-at-feedback analysis on them is unlikely to yield anything not already
in the cue betas.

---

## Code shipped

| What | State |
|---|---|
| `multivariate/run_glmsingle_feedback.py` + `submit_glmsingle_feedback.sh` | **merged to `main`** (`579568d`) |
| Log-file fix — both GLMsingle runners | **merged to `main`** (`ed8ec32`) |
| `multivariate/compare_cue_vs_feedback_betas.py` + submit + `glmsingle_cue_vs_feedback_comparison.ipynb` | branch `cue-vs-feedback-comparison` (`397979f`), pushed, **unmerged** |

Design decisions worth remembering:

- **Feedback conditions are the 8 stimulus pairs**, not stimulus identity. The feedback screen
  shows the reward for *both* options, so its content is fully determined by the pair
  (`reward_chosen + reward_unchosen` has 0.000 within-pair variance). Pairs are also better
  conditioned for `calcbadness`: all 8 present in every run for every subject, min 3 repeats/run,
  no thin cells — chosen-stimulus identity gives 5.3% of cells <3 repeats.
- **Beta ordering is asserted**, not assumed. `run_glmsingle.py` recovers run labels by counting
  per-condition occurrences without ever checking; the feedback script rebuilds the expected
  condition sequence and compares it against `DESIGNINFO`'s `stimorder`. It passed for all 59.
- **The log-file bug**: `GLM_single.fit()` opens with `shutil.rmtree(outputdir)`, which unlinked
  the per-subject log created beforehand by logging's `FileHandler`. The handler kept writing to
  the unlinked inode, so nothing raised and stdout was unaffected — the file just didn't exist
  afterwards. That is why no production `glmsingle/sub-*/` has a per-subject log. Fixed by moving
  logs to `<output-dir>/logs/`.

## Data produced (cluster)

- `derivatives/glmsingle_feedback/` — 59 subjects, all COMPLETED. Verified: 8 pairs each,
  11 105 feedback events, 221 non-response trials correctly dropped, no TR collisions,
  beta-ordering assertion passed for every subject.
- `derivatives/cue_vs_feedback/` — 59 subjects × {B, D} × {whole brain, visual cortex}.
  CSVs also synced to `dev_sample/bids_dataset/derivatives/cue_vs_feedback/` (236 KB) so the
  notebook executes locally.

## Git state at session end

Local, cluster and origin all agree. `main` at `a02ce1f`. Cluster is checked out on
`cue-vs-feedback-comparison` with `glmsingle_qc.ipynb` edits sitting in `stash@{0}`.

---

## Open threads

1. **Merge `cue-vs-feedback-comparison`**; restore the cluster to `main` and pop that stash.
2. **Finding 2 is not in any notebook** and bears on a conclusion already written up. The
   identity-demeaning falsification test is the highest-value next step in this whole line.
3. **The multimodality in finding 3 is unexplained** — 38 subjects near 0.79, 18 near 0.42, 3
   near zero (`sub-05`, `sub-28`, `sub-70`), with clear gaps at 0.10–0.30 and 0.55–0.70. Ruled
   out: divergent GLMdenoise/ridge tuning (the split survives in type B, which has neither), data
   quality (`sub-05` has the *highest* median cue R² of the five checked), FRACvalue, HRF index,
   beta SD, noise-pool size. Untested: head motion, the fitted HRF index *distribution*, and RT
   variability — the only thing giving the two models any temporal separation.
4. **Gap in `glmsingle_cue_vs_feedback_comparison.ipynb`**: its sanity gate checks only the
   voxelwise shuffled column. The *trialwise* shuffled baseline is not clean for a few subjects
   (type B whole brain reaches 0.41), so trialwise matched values for those subjects should be
   read against their own shuffled value, not against zero. No headline number is affected — all
   of those are voxelwise.
5. The scratchpad scripts behind findings 1–2 were session-scoped and are gone; only the
   conclusions survive, in the companion note.
