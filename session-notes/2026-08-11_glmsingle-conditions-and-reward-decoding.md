# GLMsingle condition choice, the reward decode, and a feedback-locked model

**Date:** 2026-08-11
**Scope:** review of `multivariate/`, prompted by weak reward-level decoding relative to
stimulus-category decoding, and by a proposal to build a reward-feedback-locked GLMsingle model.

**Bottom line:** the GLMsingle condition definition is not the cause of the weak reward
results, and reward level as a condition would have been strictly worse. The more
consequential finding is that objective reward level is a deterministic function of stimulus
identity in this design, so the surviving "reward decode" in
`qvalue_classification_results.ipynb` is most likely identity decoding under a value label.

---

## 1. What GLMsingle actually does with the condition labels

Verified against the upstream source (`glmsingle/glmsingle.py`, `glmsingle/ssq/calcbadness.py`),
not from memory:

- **Beta estimation does not use them.** GLMsingle internally rebuilds `designSINGLE` with one
  column *per trial*, not per condition. The 8 columns we pass are collapsed to a `stimorder`
  vector.
- **HRF selection does not use them.** Type-B loops the 20-HRF library over the single-trial
  design and takes the per-voxel argmax R².
- Condition labels enter in exactly **two places**: choosing the number of GLMdenoise nuisance
  PCs, and choosing the per-voxel fracridge fraction. Both go through `calcbadness`, which for
  each held-out trial finds training trials with a *matching condition id* and scores squared
  error against them — i.e. leave-one-repeat-out condition reliability.

So the condition definition does tell GLMsingle "this is the signal worth preserving" and tunes
two shrinkage knobs to maximise its cross-run reproducibility. The mechanism is real.

### But the effect size is bounded by our own QC

`betas_qc_decoding.ipynb` measures the entire B→C→D pipeline — i.e. everything the condition
labels can influence — on category decoding, n=59:

| mask | B | C (+denoise) | D (+ridge) | B→D |
|---|---|---|---|---|
| wholebrain | 0.357 | 0.372 | 0.368 | **+0.011** |
| visualcortex | 0.468 | 0.482 | 0.483 | **+0.015** |

C→D — the fracridge step, where the per-voxel fraction lives — is ~0.000/+0.001. Expected: the
ridge fraction is close to a per-voxel scalar, and all our decoders use `standardize=True`,
which z-scores each voxel and largely undoes it (the README already notes this). Total available
leverage of the condition choice is therefore **~1.5 accuracy points**, nearly all of it in the
PC count. It cannot explain a weak reward result.

---

## 2. Would reward level as the condition have helped?

No — it would have hurt. **Reward level is a deterministic function of stimulus identity.**

```
within-(subject × first_stim_name) variance of first_stim_value: 0.0000
496 / 496  (subject × stimulus) cells have exactly 1 unique reward value
```

The 8 identities partition into the 5 levels as `{1, 2, 2, 3, 3, 4, 4, 5}`. Reward level is a
strict *coarsening* of identity, so the identity conditions already preserve a superset of the
reward information, and `calcbadness` under identity labels is optimising for strictly more.

Going the other way, with 5 reward conditions we would be telling `calcbadness` that
`face_female` and `house_1` (same level, different stimulus) are repeats of one another. Their
genuine pattern difference then counts as *badness*, pushing PC/fraction selection toward more
aggressive shrinkage of exactly the variance carrying the reward signal.

**Is stimulus identity the best choice? Yes.** GLMsingle's guidance is the finest condition
partition that still has adequate repeats. Identity is both the finest available here and
well-populated (~11–23 repeats per identity per run). There is nothing finer to use.

---

## 3. The reward decode is very likely identity decoding

`qvalue_classification_results.ipynb` currently concludes the reward decode "survives the
category-confound control, largely intact… there's genuine within-category variance carrying
reward information." That within-category variance appears to be **stimulus identity**.

The value assignment is fixed across all 62 subjects: each category always receives one of the
pairs `(1,5)`, `(2,3)`, `(2,4)`, `(3,4)` — the two stimuli within a category *never* share a
value. After dropping level 3, exactly **2 categories per subject straddle the low/high split,
for all 62 subjects**: the `(1,5)` category and the `(2,4)` one.

So after `(run × category)`-demeaning, the only trials still carrying a class contrast are those
two categories, and within each of them the low/high label **is** the individual-stimulus label.
The classifier is performing two binary identity discriminations.

This predicts exactly the observed profile — visual cortex 0.615 ≫ whole-brain 0.549, vmPFC and
striatum flat at chance — and explains why category-demeaning cost only ~1.5 points: it removes
the between-category component but leaves the within-category identity contrast untouched.

Objective reward level at cue cannot be dissociated from identity in this design. That is a
paradigm property, not a pipeline one; no GLMsingle setting changes it.

Variables that *are* dissociable (within-subject × first-stim-identity variance fraction,
learning runs):

| variable | dissociable variance |
|---|---|
| `first_stim_value` (objective) | **0.000** |
| `first_stim_value_rl` (learned Q) | 0.080 |
| `first_stim_choice_val` | 0.061 |
| `value_diff` | **0.979** |

---

## 4. Assessment of a reward-feedback-locked GLMsingle model

Two problems, both in the paradigm rather than the code.

### (a) Feedback is 1.95 s after cue onset (SD 0.16 s), with TR = 2.334 s

Measured on 11 683 responded learning trials. In TR bins: **82.4% of feedback events land
exactly 1 TR after the cue's TR, 17.5% in the same TR**, 0.03% at +2. There is only 0.16 s of RT
jitter available for deconvolution. A feedback-locked design is therefore a near-shifted copy of
the cue design: model feedback alone and the betas are a cue+feedback composite that will
closely resemble the existing ones; model both jointly and the variance split is near-singular
and largely arbitrary.

Supporting timings (learning runs, responded trials): feedback duration
(`t_iti_onset − t_points_feedback`) = 1.47 s ± 0.15; cue-to-cue ISI = 10.14 s ± 0.56; no two
feedback events fall in the same TR within a run (so GLMsingle would not error).

### (b) Nothing new varies at feedback

Feedback is shown for **both** the chosen and unchosen option, which makes this worse rather
than better. Rewards are deterministic per stimulus (495/495 chosen-stimulus cells have one
unique reward value), there are exactly **8 stimulus pairs per subject**, and the two options in
a pair always differ by exactly one level. The feedback *display* is therefore fully determined
by the pair — `rew_sum` has 0.000 within-pair variance — which is information already on screen
at cue.

The only within-pair variance at feedback:

| target | within-pair variance fraction | usability |
|---|---|---|
| `reward_chosen − reward_unchosen` | 0.888 | ±1 only, and 89.1% / 10.9% imbalanced — effectively an error-trial regressor |
| RPE (`reward_chosen − chosen_value_rl`) | 0.814 | 79.4% of trials have \|RPE\| < 0.01 — deterministic rewards, Q converges fast |

### If built anyway

The script itself is a straightforward variant of `multivariate/run_glmsingle.py`:

- onsets `t_first_stim` → `t_points_feedback`
- `RUNS = ['learning1', 'learning2']` (test has no feedback; `Block.block_type` is set by the
  presence of `raw_block.time.rewards_onset`), `sessionindicator = [[1, 1]]`
- `stimdur` = mean `t_iti_onset − t_points_feedback` ≈ 1.47 s
- drop non-response trials (`action` is NaN, 221/11904 ≈ 1.9%) — they have no feedback event
- conditions = the **8 stimulus pairs**, not stimulus identity

On that last point: the feedback display shows the reward for *both* options, so its content is
fully determined by the pair (`rew_sum` has 0.000 within-pair variance). Pairs are also better
conditioned for `calcbadness` than any identity-based alternative:

| condition | conditions present per run | repeats/cond/run (min, med) | cells with <3 repeats |
|---|---|---|---|
| **pair (8)** | 8 for every subject/run | 3, 10 | **0.0%** |
| chosen identity (8) | 7–8 | 1, 12 | 5.3% |
| pair × choice (16) | 8–16 | 1, 5 | 25.9% |

Given (a) and (b), expect it to reproduce the cue betas rather than reveal anything new.

### Measured, 2026-08-13 (n=59, `compare_cue_vs_feedback_betas.py`)

The prediction is confirmed. Voxelwise r between matched cue and feedback betas, whole brain:
**median 0.773**, IQR [0.438, 0.802]. Both baselines are flat — shuffled pairing median −0.005
(max |0.011|), adjacent cue-cue 0.050 — so this is genuine trial-by-trial coupling, not shared
spatial structure. At r = 0.77 that is r² ≈ 0.60 shared variance, and it is a *lower* bound:
single-trial betas are noisy, and two noisy estimates of the same quantity correlate at their
reliability, not at 1.

The distribution is multimodal rather than smooth: 38 subjects at ~0.79, 18 at ~0.42, 3 near
zero (`sub-05`, `sub-28`, `sub-70`), with almost nothing between 0.10–0.30 or 0.55–0.70.
Whole-brain and visual-cortex values correlate at 0.994 across subjects, so whatever drives the
spread is subject-level, not regional. The near-zero subjects are not explained by data quality
(`sub-05` has the *highest* median cue R² of the five first checked), nor by FRACvalue, HRF
index, beta SD, or noise-pool size. Unresolved.

**Consequence:** for roughly two-thirds of the sample the feedback betas largely re-express the
cue response. Combined with §3 — objective reward is a deterministic relabeling of stimulus
identity — a reward-at-feedback analysis built on these betas is unlikely to yield anything not
already in the cue betas.

---

## 5. Recommended next steps

1. **Re-run the reward classification demeaning by stimulus identity rather than category**, as a
   falsification check on the current result. Prediction: it collapses to chance. Cheap, and it
   determines whether the finding in `qvalue_classification_results.ipynb` stands.
2. **Switch the target from objective reward to model-derived value** — `first_stim_value_rl` or
   `value_diff` — with identity (not category) as the nuisance to demean. This is the only route
   to a value question this design can actually answer; `value_diff` at 0.979 dissociable
   variance is the strongest candidate.
3. If a feedback-locked model is still wanted, the honest targets are **RPE** (accepting the
   degenerate distribution) or a **chose-better vs chose-worse** contrast, with the cue/feedback
   collinearity stated explicitly.

---

## Reproducing the numbers

All figures above come from `bbt.csv` (62 subjects,
`/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bbt.csv`) and from the executed
outputs of `multivariate/betas_qc_decoding.ipynb` and
`multivariate/qvalue_classification_results.ipynb`. Key derivations:

- Deterministic identity→value: `bbt.groupby(['sub_id','first_stim_name'])['first_stim_value'].nunique()`
  → all 1.
- Category value pairs: `groupby(['sub_id', cat])['first_stim_value']` → `(1,5)`, `(2,3)`,
  `(2,4)`, `(3,4)` for every subject.
- Cue→feedback lag and TR binning: `t_points_feedback − t_first_stim` on learning blocks with
  `action.notna()`, binned by `floor((t − t_first_stim.min()) / 2.33384)`.
- Dissociable-variance fractions: variance of the within-group demeaned column divided by its
  total variance.
