# Session log — closing the frequency-effect confound sweep, and a verdict on the accumulation reading

**Date:** 2026-09-07
**Companion note:** [2026-09-03_rsa-searchlight-cluster-fwe-and-roi-method.md](2026-09-03_rsa-searchlight-cluster-fwe-and-roi-method.md)
(findings 1-25, the authoritative log for the frequency-effect investigation this
session continues; see especially finding 22, the early-`learning1` presence this
session was chasing, and the "Current best-supported account" section this session's
verdict supersedes)

Picked up from finding 22/24 of the companion note: β(frequency) is already full
strength in the first half of `learning1`, doesn't grow with exposure, and survives
every confound tested so far (stim-2 leakage, category-general artifact, learning-scope
collinearity, design-constant exemplar discriminability). Went candidate-hunting for
what else could produce an effect present that early — presentation-count imbalance
and partner-value-profile were retracted on reasoning alone (both already subsumed by
existing checks); temporal placement, screen side, and a possible pipeline
indexing/alignment bug were checked directly and came back clean. Session ended with
Hugo's explicit decision that the early presence is sufficient by itself to invalidate
a learning/habit-accumulation reading of the effect, regardless of whether its actual
source is ever found.

---

## Findings

### 1. Two candidates retracted on reasoning alone, no code needed

Partner-value *profile* (as opposed to the scalar partner-value-mean already in the
model as `second_stim_value`) was proposed, then retracted: value is a deterministic,
many-to-one function of partner identity, so a value-profile RDM is strictly a
coarsening of the partner-*identity* profile — and `s2_identity` (full profile) already
failed to kill β(frequency) in `rsa_roi_results.ipynb` §20 (companion note finding 20).
A coarser statistic can't succeed where a finer one didn't. Trial-difficulty/RT was
also retracted — Hugo confirmed the value-gap at choice is constant by design in
`learning`, not a candidate channel.

### 2. Temporal placement and screen side ruled out for the early-`learning1` presence

Trial position within run vs. frequency label: r=+0.004, p=0.47 (no front/back-loading).
Lag-1 autocorrelation of the label sequence: r=−0.047, p<0.0001 — labels alternate
*slightly more* than chance, which argues against, not for, a shared-local-noise
clustering account. Screen side (left vs right) by label: 3725 vs 3714, χ² p=0.90 —
balanced. Derivation: `rsa_design_checks.ipynb` §9 (new, executed).

### 3. Code audit: no indexing/alignment bug in `run_rsa_roi.py`'s condition pipeline

Checked whether a labeling mismatch between GLMsingle beta order and the frequency
label — differing between `learning` and `test` scopes — could fake the pattern
instead of a real confound. `cond_idx` is built once, globally, from
`trial_info['stim_name']` (`run_rsa_roi.py:380-381`) and reused unchanged across every
scope — pooled/learning1/learning2/test are boolean subsets of the same fixed indexing,
no per-scope reconstruction that could silently diverge. `load_target_from_bbt`
(`utils/data.py:70-127`) carries a hard `assert list(trial_info['stim_name']) == names`
alignment guard that would crash rather than silently misalign.
`run_rsa_learning_dynamics.py` reuses the same frozen `cond_idx`/`props` from
`load_stimuli()` for its early/late split rather than rebuilding anything. No bug
found — code-reading only, nothing new to run.

### 4. Decision: the early-`learning1` presence alone invalidates a learning/habit-accumulation reading of β(frequency)

Hugo's call: a full-strength effect already present in the first half of `learning1` —
before differential reinforcement could plausibly have accrued — is sufficient on its
own to reject "β(frequency) reflects accumulated choice-history/habit strength,"
independent of ever identifying what does produce it. Findings 2-3 above only closed
off two specific rival explanations (temporal/screen-side confound, pipeline bug) for
*why* it's already present early; neither panned out, so the source remains unknown,
but that no longer bears on this verdict. Consequence: the companion note's "leading
candidate" account (fast, non-accumulating role-tagging via value-driven attention,
closing its finding 24) is the only account among those considered still consistent
with the data — "the effect reflects the study's headline habit construct" is off the
table regardless.

---

## Code shipped

| File | Change | Git state |
|---|---|---|
| `multivariate/rsa_design_checks.ipynb` | New §9 (temporal placement + screen-side balance of the frequency label) + updated Summary table/prose | staged, not committed |
| `multivariate/presentation.md` | New caveat slide after the title + inline "⚠️ NOT habit" tags on every slide that calls the frequency effect a habit/learning signature | staged, not committed |
| `multivariate/rsa_roi_results.ipynb`, `rsa_searchlight_results.ipynb`, `frequency_decoding_results.ipynb`, `frequency_searchlight_results.ipynb`, `rsa_partner_context_results.ipynb`, `stim2_decoding_results.ipynb` | New warning markdown cell inserted right after the title cell, pointing to this note and the decision | staged, not committed |

Hugo asked (this same session) to make it unmissable everywhere that the frequency
result is a problem, not a habit/learning/choice-history finding, while he thinks about
next steps. `rsa_roi_results.ipynb` exceeds the Read tool's size cap (68k tokens), so
its warning cell was inserted via a small script that edits the notebook JSON directly
(same operation `NotebookEdit` performs) rather than via the tool; validated with
`nbformat.validate` afterward on all 6 notebooks — all still valid, cell counts as
expected.

## Data produced

None — everything this session ran against the local `bbt.csv` (dev_sample, n=62,
`/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bbt.csv`). No cluster access,
no new GLMsingle/RSA outputs.

## Git state at session end

On `main`, local and remote in sync at session start (`1b4bb71`, per companion note's
git log). This session's notebook edit is staged, not committed.

## Open threads

1. **What actually produces the early presence, if not any confound or bug checked so
   far?** Still open — findings 2-3 close two candidates, not the question itself. The
   companion note's role-tagging/attention account (finding 24's closing paragraph) is
   the leading remaining candidate but untested directly.
2. **Why the neural signature vanishes (not merely weakens) in `test`**, given the
   behavioral habit effect Hugo confirms persists there — unchanged from companion note
   open thread 8/16, the single biggest unresolved question in this whole investigation.
3. Given the decision in finding 4, consider whether any presentation-deck slides
   describing the frequency effect as a habit/accumulation signature
   (`multivariate/presentation.md`) need a caveat or rewrite.
