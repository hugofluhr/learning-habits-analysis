# Session log — chosen-value RSA (new analysis) + does the GLMsingle model definition explain the frequency effect?

**Date:** 2026-09-07
**Companion notes:**
[2026-09-07_frequency-confound-sweep-and-verdict.md](2026-09-07_frequency-confound-sweep-and-verdict.md)
(the frequency-effect verdict this session's second thread extends),
[2026-08-26_rsa-design-and-roi-pipeline.md](2026-08-26_rsa-design-and-roi-pipeline.md)
(nonfigure subset, crossnobis pipeline this session builds on)

Started as a new analysis direction (chosen-value RSA, prompted by the
whole-trial-contamination finding in session-notes 2026-09-03 findings 18-19), shipped
and run on the cluster same-session for speed. Detoured mid-session into auditing whether
the GLMsingle model definition itself — not a downstream confound — could be behind the
still-unexplained frequency effect from the companion note.

---

## Findings

### 1. Chosen-value RSA: conditions relabeled by chosen identity, not cue identity — implemented, run, n=58

New `run_subject_chosen()` in `multivariate/run_rsa_roi.py` (`--condition-on chosen`,
`--chosen-scope {all,stim2}`), branch `chosen-value-rsa`. Conditions are the identity the
subject actually *chose*, fixed to the nonfigure universe (values {2,3,4}, 6 identities —
reuses the figure-confound boundary from 2026-08-26 finding 2) rather than dynamic
per-subject dropping, which eliminates a real sparsity problem: chosen-conditions are not
presentation-balanced across value the way stim1-conditions are, and the all-8-stimuli
universe leaves 33/62 dev_sample subjects short of the 4-trial CV floor vs. 0/62 for
nonfigure-only. Derivation of that check: see `run_subject_chosen()` docstring in
`run_rsa_roi.py`, and the offline dev_sample validation run inline this session (not
saved as a script — low-value to preserve, reproducible in two lines against `bbt.csv`
grouping by `chosen_stim`/`first_stim`/`second_stim`).

### 2. Chosen-value effect: null everywhere

No `beta_value` survives correction in either scope, any of 9 masks. Cleanest test
(`chosen_scope='stim2'`, pooled only) best case is striatum p=.27. `all`-scope nominal
hits (fusiform/learning2 p=.013, visualcortex/learning2 p=.038) don't replicate across
scopes and are negative-signed (more value-different → more similar), inconsistent with
a coding interpretation. Derivation:

```python
import pandas as pd, numpy as np, glob
from scipy import stats

def load(tree):
    files = glob.glob(f'/Users/hugofluhr/phd_local/data/LearningHabits/derivatives/{tree}/sub-*/sub-*_rsa_chosen_results.csv')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def group_stats(df, terms=('category','value','frequency','unchosen_value','role_fraction')):
    rows=[]
    for (mask,scope), g in df.groupby(['mask','scope']):
        row={'mask':mask,'scope':scope,'n':len(g)}
        for t in terms:
            col=f'beta_{t}'
            if col not in g or g[col].isna().all():
                continue
            x=g[col].dropna().values
            if len(x)<3: continue
            tstat,p = stats.ttest_1samp(x,0)
            row[f'{t}_mean']=x.mean(); row[f'{t}_t']=tstat; row[f'{t}_p']=p
        rows.append(row)
    return pd.DataFrame(rows)

stim2 = load('rsa_chosen_stim2')
allr  = load('rsa_chosen_all')
print(group_stats(stim2).sort_values('value_p').to_string(index=False))
print(group_stats(allr).sort_values('value_p').to_string(index=False))
```

Rerunnable as-is against the pulled result trees under
`/Users/hugofluhr/phd_local/data/LearningHabits/derivatives/{rsa_chosen_stim2,rsa_chosen_all}/`.
Not yet folded into a notebook — see Open threads.

### 3. Frequency effect replicates under this orthogonal condition-defining scheme, including the learning→test dissociation

Same `group_stats()` call as finding 2, on `frequency_t`/`frequency_p` instead of `value_*`.
Fusiform `pooled_chosen`: t=9.83, p=7e-14; even the maximally conservative `stim2`-only
scope still gives t=5.42, p=1e-6. Per-run breakdown (`all`-scope tree only) shows the same
learning-vs-test split as the companion note: fusiform p≤1e-9 in learning1/learning2, p=.15
in test; visualcortex p=6e-8/p=3e-6 in learning1/learning2, p=.20 in test. This is
independent evidence (different condition-definition entirely — chosen identity, not cue
identity) that the effect and its test-attenuation are not artifacts of the stim1-mode
pipeline specifically.

### 4. GLM definition IS a documented, deliberate cause of whole-trial contamination — confirmed, not new

Read `multivariate/run_glmsingle.py` and `multivariate/dev_glmsingle_stim_cat.ipynb` Steps
3-4 in response to Hugo asking whether the GLMsingle model definition could be behind these
results. At TR=2.33s, first-stim, second-stim (~0.8s later) and response (~1.7s later) fall
in the *same TR* under floor-division onset assignment — a "same-TR collision" (dev
notebook's own words) that forces only first-stim to be modeled, with only 8
identity-level conditions total. `stimdur` is then deliberately set to span
first-stim-onset-to-response specifically *"to cover both stimulus presentations"* (dev
notebook Step 3) — confirmed empirically at 1.42s mean (range 0.84-1.99s, n=20,013 response
trials, `action.notna()` filter matching `run_glmsingle.py:236` exactly):

```python
import pandas as pd
df = pd.read_csv('/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bbt.csv')
df = df[df['block'].isin(['learning1','learning2','test'])].copy()
df['rt'] = df['t_action'] - df['t_first_stim']
resp = df[df['action'].notna()]   # t_action is 0.0, not NaN, for no-response trials
print(resp['rt'].describe())
print(resp.groupby('block')['rt'].agg(['mean','std','count']))
```

This is the confirmed, sufficient, *by-design* mechanism behind the whole-trial
contamination already established (companion 2026-09-03 findings 18-19) — not a new
discovery, but it settles that it's structural (TR-collision at this acquisition's
resolution), not incidental. **Consequence for chosen-value RSA (finding 1-2 above):**
cue and choice can fall in the same or adjacent TR given the stimdur span, so "value of
the cued image" and "value of the eventually-chosen image" are not cleanly separable in
principle at the level of one trial's beta — a real interpretive ceiling on finding 2's
null result, not just a footnote.

### 5. One specific sub-hypothesis this raises — fixed-stimdur duration mismatch — tested and ruled out for the frequency effect

If a single RT-pooled `stimdur` created a duration-mismatch that correlated with
choice-frequency or value, that could in principle produce spurious RSA structure. Checked
directly:

```python
import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv('/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bbt.csv')
df = df[df['block'].isin(['learning1','learning2','test'])].copy()
df['rt'] = df['t_action'] - df['t_first_stim']
resp = df[df['action'].notna()]

def corr_by(col):
    corrs=[]
    for s,g in resp.groupby('sub_id'):
        if g[col].nunique()>1:
            r,_ = stats.pearsonr(g[col], g['rt'])
            corrs.append(r)
    corrs=np.array(corrs)
    t,p = stats.ttest_1samp(corrs,0)
    print(f'{col}: n={len(corrs)} mean_r={corrs.mean():.4f} t={t:.3f} p={p:.4f}')

corr_by('first_stim_frequ')  # r=0.0039, p=.71 pooled -- ns in every run too
corr_by('first_stim_value')  # r=-0.0477, p=.0001 pooled -- real but tiny (~tens of ms)
```

RT vs. choice-frequency label: r≈0.004-0.016, ns pooled and in every run separately —
**rules out duration-mismatch as an explanation for the frequency effect**, the one
that's been unexplained through the whole 2026-09-03/09-07 confound sweep. RT vs.
objective value: r≈-0.05 pooled (p=1e-4, strongest in learning1 r=-0.08) — real but too
small (implies tens-of-ms true duration differences against a ~1.4s±0.15s boxcar) to
plausibly produce effects of the size seen in the value RSA results via this mechanism.
Mean RT is also nearly flat across runs (1.44/1.40/1.42s learning1/learning2/test), so it
doesn't explain the frequency effect's learning→test attenuation either. **This candidate
was not covered by the companion note's confound sweep — now checked and ruled out for
the frequency effect specifically.**

---

## Code shipped

| File | Change | Git state |
|---|---|---|
| `multivariate/run_rsa_roi.py` | New `run_subject_chosen()`, `--condition-on {stim1,chosen}`, `--chosen-scope {all,stim2}`, `CHOSEN_MODEL_TERMS`, `MIN_TRIALS_PER_COND` | branch `chosen-value-rsa` (`10a011b`), pushed |
| `multivariate/submit_rsa_roi.sh` | `CONDITION_ON`/`CHOSEN_SCOPE` env vars, same separate-output-tree guardrail as SHUFFLE/REMOVE_MEAN/SYMMETRIC | branch `chosen-value-rsa` (`10a011b`), pushed |
| This session note | findings 2, 4, 5's group-stats/RT-correlation code inlined above (not yet moved into a notebook) | new, staged this checkpoint |

## Data produced

Two full cluster runs, both n=58 (`participants_mvpa.tsv` minus `sub-46`, missing from
`bbt.csv` — pre-existing, unrelated to this session's code):

- `derivatives/rsa_chosen_stim2/` — job 5637066, `--chosen-scope stim2`, `pooled_chosen`
  scope only (per-run stim2-only counts too sparse — checked offline pre-submission,
  32-57/62 dev_sample subjects short per run).
- `derivatives/rsa_chosen_all/` — job 5637067, `--chosen-scope all`, `pooled_chosen` +
  per-run scopes where a subject's counts support it (52-60/58 subjects per run).

Both pulled locally to
`/Users/hugofluhr/phd_local/data/LearningHabits/derivatives/{rsa_chosen_stim2,rsa_chosen_all}/`
(per-subject CSVs + `.npz` model RDMs; per-mask/scope `.npy` RDMs not pulled, small and
regenerable from cluster if needed).

## Git state at session end

Branch `chosen-value-rsa`, one commit ahead of `main` (`10a011b` on top of `8673fba`),
pushed to origin. This session note staged, not committed.

## Open threads

1. ~~Build `multivariate/rsa_chosen_results.ipynb`~~ — done, same session: findings 2-5
   above are now backed by an executed notebook (per-mask/scope tables, learning/test
   breakdown, GLM-definition + frequency-not-habit caveats up front). Validated with
   `nbformat.validate` and a full `nbconvert --execute` pass, no errors.
2. **rl/ck graded chosen-value variants** are not yet computed — `chosen_value_rl`/
   `chosen_value_ck` already exist in the BBT (see `run_subject_chosen()` docstring); only
   the `objective` variant was run this session.
3. **Companion note's open thread 2** (why the frequency signature vanishes in `test` while
   the behavioral habit effect persists) is now *doubly* unresolved — finding 3 above shows
   the same vanishing under a second, unrelated condition-definition, and finding 5 rules
   out one more candidate mechanism (GLM stimdur duration-mismatch) without finding the
   actual source. Still open.
4. **`unchosen_value`'s mild visual-cortex hit** (p=.003, `all`/`pooled_chosen`) wasn't
   investigated further — plausible low-level partner-value leakage, not follow-up priority
   unless it recurs.
