# Session log — RSA searchlight (frequency, value, interaction)

**Date:** 2026-08-31
**Companion note:** [2026-08-31_frequency-decoding-and-searchlight.md](2026-08-31_frequency-decoding-and-searchlight.md)

Started as "explain the frequency and interaction RSA findings again" and a
follow-up on whether the effects extend beyond visual cortex; ended with a new
RSA searchlight pipeline (main terms + interaction) run on the full cohort, one
caught-and-fixed collinearity bug, and a branch/history cleanup.

---

## Findings

### 1. RSA searchlight main effects: only frequency survives FDR

n=58, 6mm-radius spheres, 5-term regression (category, value, frequency,
second_stim_value, choice_rate). Frequency: 2,945 FDR-sig voxels (1.8% of
brain, +2,926/−19), bilateral occipitotemporal peak t=9.3. Value and category:
0 FDR-sig voxels each — consistent with the ROI-level null for value in the
joint model. Derivation: `rsa_searchlight_results.ipynb` §2–§4.

### 2. Striatum newly significant for frequency at the searchlight level

ROI mean β(frequency): fusiform +0.174***, VC +0.086*** (confirm ROI RSA),
**striatum +0.047\*** (NEW — n.s. at ROI level, β=+0.091 there). Searchlight's
local-neighborhood averaging is more sensitive than full-ROI averaging for
weak distributed effects. Derivation: `rsa_searchlight_results.ipynb` §5.

### 3. Value×frequency interaction reproduces at voxel level, same territory as frequency

Computed as a slope-difference (see finding 5 for why not a joint regressor):
3,643 FDR-sig voxels (2.2%, +3,596/−47), peak t=12.5 — stronger than the
frequency main effect alone (t=9.3). ROI means: fusiform +0.647***, VC
+0.314*** — same sign, same two ROIs as the ROI-level RSA (§8c of
`rsa_roi_results.ipynb`: VC +1.230***, fusiform +1.357***; different numeric
scale because searchlight spheres vs. full ROI, same qualitative pattern).
Derivation: `rsa_searchlight_results.ipynb` §9.

### 4. Spatial correlation with frequency decoding searchlight: r=0.733

RSA searchlight is more conservative (2,945 vs. 8,245 FDR voxels) because it
partials out category, value, and confounds; decoding classifies raw patterns.
Same peak regions. Derivation: `rsa_searchlight_results.ipynb` §6.

### 5. Interaction as a 6th joint regressor is severely collinear with frequency — must be computed separately

First attempt added value×frequency as a 6th RDM term in the same OLS as the
5 main effects. Verified via ad hoc synthetic checks (not previously saved —
rescued below) that this RDM is r=-0.89 correlated with the frequency
main-effect RDM in a realistic 6-stimulus/3-per-frequency-class design, because
the interaction's sign is dominated by the same same/different-frequency split
that defines the frequency RDM. Adding it visibly changed every other beta on
real data (sub-01 frequency β: +0.038 → −0.106). Fixed by computing the
interaction as a same-vs-different-frequency slope difference instead —
mirrors `rsa_roi_results.ipynb` §8b/§8c exactly, uses only within-subset mean
differences, cannot compete with the regression terms for shared variance.

```python
# Collinearity check that motivated the fix — realistic 6-stimulus design
# (values {2,3,4} only, since {1,5} are reserved for the figure category;
# see rsa_design_checks.ipynb for that constraint). Rerun to reproduce r=-0.89.
import numpy as np
from numpy.linalg import matrix_rank, cond
from multivariate.run_rsa_roi import abs_diff_rdm, different_rdm, _triu, _z

cat = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
value = np.array([1, 3, 5, 2, 4, 6], dtype=float)   # unconstrained value range
frequency = np.array([1, -1, 1, -1, 1, -1], dtype=float)

same_freq = (frequency[:, None] == frequency[None, :]).astype(float)
val_diff = abs_diff_rdm(value)
interaction = val_diff * (2 * same_freq - 1)
freq_rdm = abs_diff_rdm(frequency)
cat_rdm = different_rdm(cat)

terms = {'category': cat_rdm, 'value': val_diff, 'frequency': freq_rdm,
         'interaction': interaction}
X = np.column_stack([_z(_triu(v)) for v in terms.values()])
print('rank:', matrix_rank(X), 'of', X.shape[1], ' cond:', cond(X))
import pandas as pd
print(pd.DataFrame(X, columns=terms.keys()).corr().round(3))
# frequency vs interaction: r = -0.887

# Realistic value range {2,3,4} (matches actual non-figure stimulus values):
# same check gives rank=3, cond~1.75, no collinearity — this is what the
# fixed 5-term regression (frequency and value as separate, unmodified terms)
# actually runs on in run_rsa_searchlight.py.
value_real = np.array([2, 3, 3, 4, 2, 4], dtype=float)
X_real = np.column_stack([_z(_triu(different_rdm(cat))),
                          _z(_triu(abs_diff_rdm(value_real))),
                          _z(_triu(freq_rdm))])
print('\nreal-value-range rank:', matrix_rank(X_real), 'cond:', cond(X_real))
```

Full reasoning, the fix, and its verification against real cluster data are
in the `run_rsa_searchlight.py` module docstring and commit `82020f3`.

---

## Code shipped

| File | Change | Git state |
|------|--------|-----------|
| `multivariate/run_rsa_searchlight.py` | New: whole-brain RSA searchlight, 5-term regression + interaction slope-diff | `rsa-roi` (merged, `b3c5d89`) |
| `multivariate/submit_rsa_searchlight.sh` | New: SLURM array submit | `rsa-roi` (merged) |
| `multivariate/rsa_searchlight_results.ipynb` | New: executed with full n=58, including interaction (§9) | `rsa-roi` (merged) |
| `multivariate/presentation.md` | Added 3 interpretive slides (frequency-effect meaning, interaction meaning, RSA-vs-decoding caveats) | **uncommitted in the original checkout** — made via direct file edits before this session entered its worktree; not part of any commit in this branch's history, needs checking/committing separately |

## Data produced

- **RSA searchlight (job 5430956, final/correct version):** 58/58 subjects
  complete (`sub-46` fails as always — absent from BBT). Outputs at
  `derivatives/rsa_searchlight/sub-*/sub-*_rsa_searchlight_{beta_<term>,interaction_value_freq}.nii.gz`.
  Synced locally to `~/phd_local/data/LearningHabits/derivatives/rsa_searchlight/`.
- Two earlier cluster runs (5429419: pre-interaction 5-term only; 5430061:
  contaminated 6-term joint regression) were superseded. Contaminated
  `*_beta_value_x_frequency.nii.gz` files deleted from both cluster and local
  copies after the fix.

## Git state

Branch `worktree-rsa-searchlight` merged into `rsa-roi` (commit `b3c5d89`,
pushed to origin) and deleted (local + remote). `rsa-roi` is the current
active branch, still not merged to `main`. Squashed the buggy-interaction +
fix commit pair into one clean commit (`82020f3`) before merging — verified
byte-identical final tree via `git diff` before force-pushing.

Also deleted `qvalue-decoding` (fully merged into `main`, safe cleanup).
Left `coverage` and `backup/pre-rebase-20250915` untouched pending user
decision.

---

## Open threads

1. **`presentation.md` slide edits may be sitting uncommitted** in the
   original (non-worktree) checkout — the frequency-effect/interaction
   explanatory slides added earlier this session were never committed by this
   worktree session. Check `git status` there and commit if still present.
2. **Frontal/orbitofrontal/IFG clusters** in both the frequency and interaction
   searchlight maps — anatomical labeling and interpretation still needed
   (same open item as the prior frequency-searchlight note).
3. **Striatum interaction trends negative** (uncorrected p=0.064, opposite
   sign to fusiform/VC) — not significant but worth watching in a future
   analysis with more power or a targeted ROI test.
4. `rsa-roi` branch still not merged to `main`.
5. `coverage` and `backup/pre-rebase-20250915` branches — user hasn't decided
   whether to delete.
