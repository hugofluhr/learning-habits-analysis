# Session log: bulletproofing the concat models

**Date:** 2026-09-25 (check 2 set up 2026-09-28)
**Companion note:** [2026-09-24_tier1-reruns-and-second-level-comparison.md](2026-09-24_tier1-reruns-and-second-level-comparison.md) (the concat models' results, open thread 1: bulletproof the GLM2 chosen concat and understand the Q-value loss)

Checks on whether the concat models' H-value effects could come from run-level differences rather than trial-by-trial H. Hugo's framing: the between-run differences of H are real signal (slow accumulation of choice history), so demeaning H within runs is not an option; the question is whether the data can separate that signal from anything else that changes across runs.

---

## Findings

### 1. More than half of the chosen-H regressor is between-run variance, and it is a near-linear function of run order
Second-level sample (59 subjects), share of trial-level variance between runs, median [IQR]: chosen H 0.55 [0.52–0.58] (min 0.43), GLM1 concat first/second-stim H 0.41 / 0.43, all Q modulators 0.00–0.02. Median chosen-H run means −1.03 / −0.04 / +0.76 (learning 1 / learning 2 / test), so the run-level part rises ~0.9 z-units per run in every subject; the rise from learning 1 to test barely varies across subjects (+1.79, IQR +1.74 to +1.83). Consequences: the H effect in the concat models is confounded with run order to the extent it rests on this component; a between-subject covariate test on the z-scored rise has almost no leverage; the Q loss is not explained by Q's own between-run variance (≈ 0). `python scripts/concat_between_run_variance.py --out <csv>` (commit `69cc7a5`); per-subject table on the cluster: `spm_outputs/compare_second_lvl_2026-09-24-14-56/concat_between_run_variance.csv`.

### 2. Check 2 (H split): the bilateral DLS effect is carried by within-run H; vmPFC mostly by the run-level part
`glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25` (N = 59): chosen H split into within-run (deviation from run mean) and between-run (run mean) modulators, which sum to H. VIF median/max: Q 1.5/2.0, H within 1.7/2.2, H between 4.1/5.5. H within alone: DLS right (27, 9, 6) k 15 t 4.81 cluster p .013, left (−27, 9, 2) k 14 t 4.01 p .015; M1 (−12, −33, 72) k 160 p < .001; vmPFC only 2 voxels (p .025). H between: DLS (30, 9, 2) k 26 p .003, (−27, 9, 6) k 16 p .011; vmPFC (−3, 42, −15) k 7 p .009. So the DLS and M1 results do not depend on run order; the vmPFC effect largely does. Q_chosen is unchanged by the split (right VS k 7 p .010, left VS 1 voxel p .035), so the Q loss comes from the concat design, not from how H is modelled. Clusters: `clusters('glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25', 'hval')` with the second code block in the 2026-09-24 note; VIF: `spm_outputs/glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25/vif_report/`.

### 3. The choice variable is ~97 % chosen Q, yet GLM3 shows no VS effect
Per-subject trial-level correlation of `chosen_choice_val_zscore` with chosen Q: median r 0.97 (0.78–1.00); with chosen H 0.60; Q–H 0.40. GLM3 has nothing in VS but strong posterior putamen (−33, −12, 2; k 30, t 5.59, p .002 in Guida), where GLM2 chosen Q_chosen has VS. Hypothesis (not tested): with H in the model (GLM2 chosen), Q's beta is its effect beyond H, and the VS Q effect may depend on H absorbing shared variance; GLM3's single regressor also picks up what it shares with H. Test: GLM2 chosen with Q only.

```python
# Finding 3: correlation of the chosen choice variable with chosen Q and H, per subject (second-level sample)
import pandas as pd
b = pd.read_csv('/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv')
b = b[b.block.isin(['learning1', 'learning2', 'test']) & b.action.notna() & ~b.sub_id.isin(['sub-04', 'sub-45', 'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'])]
r = b.groupby('sub_id').apply(lambda d: pd.Series({'rQ': d.chosen_choice_val_zscore.corr(d.chosen_value_rl_zscore),
                                                   'rH': d.chosen_choice_val_zscore.corr(d.chosen_value_ck_zscore),
                                                   'rQH': d.chosen_value_rl_zscore.corr(d.chosen_value_ck_zscore)}))
print(r.describe().round(2))
```

### 4. Model-free models: code consistent with GLM2 chosen, no bug; value shows the choice-variable pattern, frequency shows nothing
`glm2_mf_val.m` / `glm2_mf_frequ.m` differ from `glm2_chosen_all_runs.m` only in the modulator (`git diff --no-index` of the scripts). No NaN in either modulator (0/19,373 response trials), design matrices max |value| 0.55 (no Bug A–type values). `chosen_stim_frequ` is raw −1/0/+1, not z-scored (0 = stimuli outside the frequency manipulation, ~25 % of choices): changes beta scale only, not t-maps. Re-run results (`…_2026-09-24-14-56`): chosen value VS k 3 p .021, vmPFC k 3 p .019, DLS/posterior putamen (−33, −12, 2) k 30 t 5.52 p .002, parietal; test session alone vmPFC (0, 42, −4) k 102 t 5.26 p < .001. Chosen frequency: nothing in any ROI or whole brain, all runs or any session. Clusters: `clusters('glm2_mf_chosenval_2026-09-24-14-56', 'chosenval', prefix='')` with the second code block of the 2026-09-24 note.

```python
# Finding 4: model-free modulator columns and design-matrix sanity (run on the cluster)
import pandas as pd, numpy as np, glob
b = pd.read_csv('/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv')
b = b[b.block.isin(['learning1','learning2','test']) & ~b.sub_id.isin(['sub-04','sub-45'])]
r = b[b.action.notna()]
for c in ['chosen_stim_value_zscore', 'chosen_stim_frequ', 'chosen_value_rl_zscore']:
    x = r[c]
    print(f"{c}: NaN on response trials {x.isna().sum()} / {len(x)}; unique (first 8) {sorted(x.dropna().unique())[:8]}")
    g = r.groupby('sub_id')[c].agg(['mean', 'std'])
    print(f"   per-subject mean over response trials: median {g['mean'].median():.3f} [{g['mean'].min():.2f}, {g['mean'].max():.2f}], sd median {g['std'].median():.3f}")
# where does z-scoring happen: over all trials or response trials?
for c in ['chosen_stim_value_zscore', 'chosen_value_rl_zscore']:
    g = b.groupby('sub_id')[c].agg(['mean', 'std'])
    print(f"{c} over ALL trials: mean median {g['mean'].median():.3f}, sd median {g['std'].median():.3f}")
print('chosen_stim_frequ by run:', r.groupby('block').chosen_stim_frequ.value_counts().unstack().fillna(0).astype(int).to_dict('index'))
# design matrices: any extreme values in the pmod columns (Bug-A style)?
for m in ['glm2_mf_chosenval_2026-09-24-14-56', 'glm2_mf_chosenfrequ_2026-09-24-14-56']:
    worst = 0
    for f in glob.glob(f'/home/hfluhr/data/learninghabits/spm_format/outputs/{m}/sub-*/sub-*_design_matrix.csv'):
        X = pd.read_csv(f, header=None)
        names = [l.strip() for l in open(f.replace('_design_matrix.csv', '_column_names.txt'))]
        X.columns = names
        task = [n for n in names if 'bf(1)' in n]
        worst = max(worst, np.nanmax(np.abs(X[task].values)))
        if X[task].isna().any().any(): print('NaN in', f)
    print(m, 'max |value| over task columns, all subjects:', round(worst, 2))
```

### 5. Test-session vmPFC value effect, the same in model-free value and GLM2 chosen Q; never reported before
Session-03 second levels: model-free chosen value vmPFC (0, 42, −4) k 102 t 5.26 cluster p < .001; GLM2 chosen Q_chosen (−3, 42, −1) k 97 t 4.92 p < .001 (Sn3 H VIF ≤ 2.6, so interpretable). Absent in sessions 1–2, and 1–3 voxels at all-runs and Sn2+Sn3. Never reported: the Dec 2025 learning/test-split GLM2 chosen showed it and it was dismissed as a normalization artefact; the old per-session modulator contrasts were skipped for the model-free models (bug fixed in PR #2); May session-level maps were inspected for H only. Across all levels the two models differ consistently: VS bilateral and stronger with H in the model (GLM2 chosen), posterior putamen (−33, −12, 2) stronger with value alone (model-free), vmPFC identical. t-maps r 0.81–0.86 whole brain at every level. `python scripts/compare_models_second_lvl.py --a glm2_mf_chosenval_2026-09-24-14-56 second_stimxchosenval --b glm2_chosen_all_runs_scrubbed_2026-09-24-14-56 second_stimxqval_chosen --out <md>` (output: `spm_outputs/compare_second_lvl_2026-09-24-14-56/mfval_vs_glm2chosen_q.md`).

### 6. Test differs from learning in pair variety, not in chosen-value spread
Median per subject, learning 1 / learning 2 / test: distinct stimulus pairs 8 / 8 / 28; same-value pairs 0 / 0 / 26 %; SD chosen value 1.18 / 1.14 / 0.99; SD chosen Q (z) 1.06 / 1.05 / 0.91; r(chosen value, pair mean value) 0.95 / 0.99 / 0.82. So the test-session vmPFC effect is not an estimability gain (less spread, slightly higher VIF). Learning repeats 8 pairs (~12 times per run) and chosen value nearly relabels the pair; test decouples them. Remaining explanations for test-specificity, not separable here: repetition/adaptation of the 8 learning pairs, no feedback in test, converged values.

```python
# Finding 6: per-session descriptives of chosen value (second-level sample, response trials)
import pandas as pd, numpy as np
b = pd.read_csv('/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv')
b = b[b.block.isin(['learning1','learning2','test']) & b.action.notna() & ~b.sub_id.isin(['sub-04','sub-45','sub-44','sub-48','sub-68','sub-17','sub-31'])].copy()
b['pair'] = [tuple(sorted(p)) for p in zip(b.first_stim, b.second_stim)]
b['vmax'] = b[['first_stim_value','second_stim_value']].max(axis=1)
b['vmean'] = b[['first_stim_value','second_stim_value']].mean(axis=1)
b['same_val'] = b.first_stim_value == b.second_stim_value
b['chose_better'] = (b.chosen_stim_value == b.vmax) & ~b.same_val
def per(g):
    d = g[~g.same_val]
    return pd.Series({
        'n trials': len(g), 'n distinct pairs': g.pair.nunique(), '% same-value pairs': 100 * g.same_val.mean(),
        'SD chosen value (1-9)': g.chosen_stim_value.std(), 'SD chosen Q (z, across runs)': g.chosen_value_rl_zscore.std(),
        'accuracy (diff-value pairs)': d.chose_better.mean(),
        'r(chosen value, pair max)': g.chosen_stim_value.corr(g.vmax), 'r(chosen value, pair mean)': g.chosen_stim_value.corr(g.vmean),
        'r(chosen Q, pair mean value)': g.chosen_value_rl_zscore.corr(g.vmean)})
t = b.groupby(['block','sub_id']).apply(per).groupby('block').median().T
print(t.round(2).to_string())
print()
lv = b.groupby('block').chosen_stim_value.value_counts(normalize=True).unstack().round(2)
print('share of choices by chosen value (pooled):'); print(lv.to_string())
```

---

## Code shipped (branch `model-reruns`)

| Commit | What |
|---|---|
| `69cc7a5` | `scripts/concat_between_run_variance.py` |
| `48f5d64` | `glm2_chosen_all_runs_concat_hsplit.m` (check 2) and this note |
| (this commit) | `scripts/compare_models_second_lvl.py` (finding 5) |

## Open threads
1. ~~Check 2~~ done (finding 2). Next: GLM2 chosen with Q only (per-session and concat), to test whether the VS Q effect depends on H being in the model (finding 3) and to explain the Q loss.
2. Between-subject covariate test (does a larger H rise predict a larger increase in DLS stimulus response across runs): only on the raw H scale, if at all (finding 1).
3. Learning runs only (runs 1 + 2) variant, to remove the learning/test phase change as an alternative explanation.
4. The Q-value loss in both concat models: not caused by how H is modelled (finding 2); candidates are the shared onset regressor and the Q/H suppression in finding 3.
5. vmPFC H effect in the concat models rests mostly on the between-run component: treat as unconfirmed.
