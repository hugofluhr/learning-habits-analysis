# Session log: bulletproofing the concat models

**Date:** 2026-09-25 to 2026-09-28 (between-run check on the 25th; H split, model-free audit, model comparison and vmPFC on the 28th)
**Companion note:** [2026-09-24_tier1-reruns-and-second-level-comparison.md](2026-09-24_tier1-reruns-and-second-level-comparison.md) (the concat models' results, open thread 1: bulletproof the GLM2 chosen concat and understand the Q-value loss)

Checks on whether the concat models' H-value effects could come from run-level differences rather than trial-by-trial H; then a model-free audit, a full model-free vs GLM2 chosen comparison, and a test-session vmPFC value effect that pooled analyses had hidden. Hugo's framing: the between-run differences of H are real signal (slow accumulation of choice history), so demeaning H within runs is not an option; the question is whether the data can separate that signal from anything else that changes across runs.

---

## Findings

### 1. More than half of the chosen-H regressor is between-run variance, and it is a near-linear function of run order
Second-level sample (59 subjects), share of trial-level variance between runs, median [IQR]: chosen H 0.55 [0.52–0.58] (min 0.43), GLM1 concat first/second-stim H 0.41 / 0.43, all Q modulators 0.00–0.02. Median chosen-H run means −1.03 / −0.04 / +0.76 (learning 1 / learning 2 / test), so the run-level part rises ~0.9 z-units per run in every subject; the rise from learning 1 to test barely varies across subjects (+1.79, IQR +1.74 to +1.83). Consequences: the H effect in the concat models is confounded with run order to the extent it rests on this component; a between-subject covariate test on the z-scored rise has almost no leverage; the Q loss is not explained by Q's own between-run variance (≈ 0). `python scripts/concat_between_run_variance.py --out <csv>` (commit `69cc7a5`); per-subject table on the cluster: `spm_outputs/compare_second_lvl_2026-09-24-14-56/concat_between_run_variance.csv`.

### 2. Check 2 (H split): both parts of H relate to DLS; within-run H alone is sufficient; vmPFC mostly run-level
`glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25` (N = 59): chosen H split into within-run (deviation from run mean) and between-run (run mean) modulators, which sum to H. VIF median/max: Q 1.5/2.0, H within 1.7/2.2, H between 4.1/5.5. H within alone: DLS right (27, 9, 6) k 15 t 4.81 cluster p .013, left (−27, 9, 2) k 14 t 4.01 p .015; M1 (−12, −33, 72) k 160 p < .001; vmPFC only 2 voxels (p .025). H between: DLS (30, 9, 2) k 26 p .003, (−27, 9, 6) k 16 p .011; vmPFC (−3, 42, −15) k 7 p .009. So within-run H alone is sufficient for the DLS and M1 effects, i.e. they do not depend on the run-level component; but the run-level component also loads on DLS, more strongly (and is confounded with run order). *Corrected 2026-09-28: an earlier wording said the DLS effect is "carried by" within-run H, which wrongly implied the run-level part contributes nothing.* The vmPFC effect largely rests on the run-level part. Q_chosen is unchanged by the split (right VS k 7 p .010, left VS 1 voxel p .035), so the Q loss comes from the concat design, not from how H is modelled. Clusters: `clusters('glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25', 'hval')` with the second code block in the 2026-09-24 note; VIF: `spm_outputs/glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25/vif_report/`.

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

### 7. Pooled analyses hide the vmPFC effect through dilution plus a negative session-2 trend
In the vmPFC voxels significant in session 3 (98 / 102 voxels), group effect by level, GLM2 chosen Q / model-free value: session 1 −0.01 / +0.05, session 2 −0.23 / −0.23, session 3 +0.57 / +0.55, Sn2+Sn3 +0.17 / +0.16, all runs +0.33 / +0.37, GLM2 chosen concat +0.06. Session 2 as an ROI-average one-sample test (ROI from session 3, not circular for session 2): t(58) = −1.93, p = .058 (GLM2 chosen) and −2.45, p = .018 (model-free value); no voxel below t −3.23. A marginal negative trend. Dilution and cancellation both contribute (arithmetic on the group means, GLM2 chosen): with session 2 at 0, Sn2+Sn3 would be +0.285 (observed +0.17) and all runs +0.56 (observed +0.33); dilution alone roughly halves the effect, the session-2 trend removes ~40 % of the rest. t-values also depend on variance; the subject-level counterfactual (session 2 demeaned, noise kept) was not run.

```python
# Finding 7a: effect and t in the session-3 vmPFC cluster at each level
import nibabel as nib, numpy as np
from nilearn.image import resample_to_img
O = '/home/hfluhr/data/learninghabits/spm_outputs'
mask = nib.load('/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym/vmpfc_bartra2013_MNI152NLin2009cAsym.nii')
for name, run, con, lev_concat in [('GLM2 chosen Q', 'glm2_chosen_all_runs_scrubbed_2026-09-24-14-56', 'second_stimxqval_chosen', ('glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40', 'second_stimxqval_chosen')),
                                   ('model-free value', 'glm2_mf_chosenval_2026-09-24-14-56', 'second_stimxchosenval', None)]:
    ref = nib.load(f'{O}/{run}/second-lvl/session-03/{con}/spmT_0001.nii')
    t3 = np.squeeze(ref.get_fdata())
    m = np.squeeze(resample_to_img(mask, ref, interpolation='nearest').get_fdata()) > 0
    roi = m & (t3 > 3.23)   # vmPFC voxels significant at p < .001 in session 3 (df 58)
    print(f'== {name}: ROI = {roi.sum()} vmPFC voxels with session-3 t > 3.23')
    levels = [(l, f'{O}/{run}/second-lvl/{l}/{con}') for l in ['session-01', 'session-02', 'session-03', 'session-02-03', 'allruns']]
    if lev_concat: levels.append(('concat allruns', f'{O}/{lev_concat[0]}/second-lvl/allruns/{lev_concat[1]}'))
    for l, d in levels:
        t = np.squeeze(nib.load(f'{d}/spmT_0001.nii').get_fdata())
        c = np.squeeze(nib.load(f'{d}/con_0001.nii').get_fdata())
        print(f'  {l:15s} mean t {np.nanmean(t[roi]):5.2f}   max t {np.nanmax(t[roi]):5.2f}   mean group effect (con) {np.nanmean(c[roi]):+.3f}')
```

```python
# Finding 7b: subject-level ROI means per session, one-sample t-tests
import nibabel as nib, numpy as np, scipy.io as sio
from scipy import stats
from nilearn.image import resample_to_img
O = '/home/hfluhr/data/learninghabits/spm_outputs'
mask = nib.load('/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym/vmpfc_bartra2013_MNI152NLin2009cAsym.nii')
def inputs(spm_dir):
    S = sio.loadmat(f'{spm_dir}/SPM.mat', squeeze_me=True, struct_as_record=False)['SPM']
    return [str(p).strip().split(',')[0] for p in np.atleast_1d(S.xY.P)]
for name, run, con in [('GLM2 chosen Q', 'glm2_chosen_all_runs_scrubbed_2026-09-24-14-56', 'second_stimxqval_chosen'),
                       ('model-free value', 'glm2_mf_chosenval_2026-09-24-14-56', 'second_stimxchosenval')]:
    ref = nib.load(f'{O}/{run}/second-lvl/session-03/{con}/spmT_0001.nii')
    roi = (np.squeeze(resample_to_img(mask, ref, interpolation='nearest').get_fdata()) > 0) & (np.squeeze(ref.get_fdata()) > 3.23)
    print(f'== {name} (ROI {roi.sum()} voxels, from session 3)')
    for lev in ['session-01', 'session-02', 'session-03']:
        files = inputs(f'{O}/{run}/second-lvl/{lev}/{con}')
        v = np.array([np.nanmean(np.squeeze(nib.load(f).get_fdata())[roi]) for f in files])
        t, p = stats.ttest_1samp(v, 0)
        print(f'  {lev}: n {len(v)}, mean {v.mean():+.3f}, t({len(v)-1}) = {t:+.2f}, p two-sided = {p:.4f}' + ('  (ROI defined here: circular)' if lev == 'session-03' else ''))
    t2 = np.squeeze(nib.load(f'{O}/{run}/second-lvl/session-02/{con}/spmT_0001.nii').get_fdata())
    print(f'  session-02 voxel t in ROI: min {np.nanmin(t2[roi]):.2f}, voxels with t < -3.23: {(t2[roi] < -3.23).sum()}')
```

### 8. VS and vmPFC by session: VS in test strongest with H in the model; the test vmPFC effect is specific to chosen-value regressors
Test session (session 3), SVC: GLM2 chosen Q_chosen VS bilateral (9, 9, −4) k 10 p .006 and (−9, 9, −4) k 12 p .004, vmPFC k 97; GLM3 choice variable VS none, vmPFC (0, 42, −4) k 95 t 5.21 p < .001; model-free value VS k 3 p .021, vmPFC k 102; GLM1 per-stimulus Q VS small (first stim k 4 + 3, second stim k 5), vmPFC none. Learning sessions: VS 0–2 voxels in every model, so the all-runs VS effects come from pooling. So VS follows the Q-with-H pattern (finding 5) and the test vmPFC effect appears for every chosen-value regressor (three models) but not for per-stimulus Q.

```python
# Finding 8: VS and vmPFC SVC clusters per session and model (peak tables from svc_report.m)
import pandas as pd
SVC = '/home/hfluhr/data/learninghabits/spm_outputs/compare_second_lvl_2026-09-24-14-56/svc'
models = [('glm2_all_runs_scrubbed_2026-09-24-14-56', ['first_stimxqval', 'second_stimxqval']),
          ('glm2_chosen_all_runs_scrubbed_2026-09-24-14-56', ['second_stimxqval_chosen']),
          ('glm3_chosen_choice_var_scrubbed_2026-09-24-14-56', ['second_stimxchoiceval_chosen']),
          ('glm2_mf_chosenval_2026-09-24-14-56', ['second_stimxchosenval'])]
for run, cons in models:
    d = pd.read_csv(f'{SVC}/{run}.csv').dropna(subset=['peak_t']).drop_duplicates(['model', 'region', 'cluster_k', 'cluster_p_fwe'])
    for con in cons:
        for lev in ['session-01', 'session-02', 'session-03', 'allruns']:
            for reg in ['striatum_bartra', 'vmpfc_bartra']:
                x = d[(d.model == f'{lev}/{con}') & (d.region == reg) & ((d.cluster_p_fwe < .05) | (d.peak_p_fwe < .05))]
                print(run, con, lev, reg, x[['x', 'y', 'z', 'cluster_k', 'peak_t', 'cluster_p_fwe']].round(3).values.tolist())
```

### 9. Within-run H is largely trial order, especially in learning 1
Per subject and run, median r(chosen H, trial position within run): learning 1 0.83 [IQR 0.80–0.85], learning 2 0.46, test 0.37; Q: 0.16 / 0.00 / −0.02. So a within-run H effect cannot be separated from any drift of the stimulus response over the run without modelling time; the H split (finding 2) does not address this. Code: second half of the block below.

### 10. Per session, the DLS H effect is only in session 1; whole-mask averages are uninformative
Per-session GLM2 chosen, chosen H, DLS (Guida) SVC: session 1 (−30, −3, −1) k 24 cluster p .004 and (27, 6, 6) k 6 p .051; sessions 2, 3 and 2+3: no suprathreshold voxel. Session 1 is the VIF-inflated session where within-run H ≈ trial order (finding 9). Averages over the whole Guida mask (605 voxels) are ≈ 0 for every H beta (per-session, Sn2+3, concat within, concat unsplit; all |t| < 1.2), because the effect is focal: that check ("check 1") could not answer the question. Code: first half of the block below; per-session SVC from the peak tables.

```python
# Findings 9-10: whole-Guida-mask H betas per model/session, and within-run H vs trial order
import nibabel as nib, numpy as np, pandas as pd, scipy.io as sio
from scipy import stats
from nilearn.image import resample_to_img
O = '/home/hfluhr/data/learninghabits/spm_outputs'
mask = nib.load('/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym/habit_Guida2022_MNI152NLin2009cAsym.nii')
def inputs(d):
    S = sio.loadmat(f'{d}/SPM.mat', squeeze_me=True, struct_as_record=False)['SPM']
    return [str(p).strip().split(',')[0] for p in np.atleast_1d(S.xY.P)]
def sid(f):
    i = f.find('sub-'); return f[i:i + 6]
def roi_means(d):
    files = inputs(d)
    ref = nib.load(files[0])
    roi = np.squeeze(resample_to_img(mask, ref, interpolation='nearest').get_fdata()) > 0
    return pd.Series({sid(f): np.nanmean(np.squeeze(nib.load(f).get_fdata())[roi]) for f in files})

print('=== Check 1: chosen-H betas, subject means over the whole Guida mask (a priori)')
ps = 'glm2_chosen_all_runs_scrubbed_2026-09-24-14-56'
cols = {f'per-session {l}': f'{O}/{ps}/second-lvl/{l}/second_stimxhval_chosen' for l in ['session-01', 'session-02', 'session-03', 'session-02-03']}
cols['concat within-run H'] = f'{O}/glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25/second-lvl/allruns/second_stimxhvalwithin_chosen'
cols['concat H (unsplit)'] = f'{O}/glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40/second-lvl/allruns/second_stimxhval_chosen'
df = pd.DataFrame({k: roi_means(v) for k, v in cols.items()}).dropna()
print(f'n = {len(df)} subjects')
for c in df:
    x = df[c]; t, p = stats.ttest_1samp(x, 0)
    print(f'  {c:24s} mean {x.mean():+.4f}  SD {x.std():.4f}  t({len(x)-1}) = {t:+.2f}  p = {p:.4f}  % subjects > 0: {100*(x > 0).mean():.0f}')
s = df[['per-session session-01', 'per-session session-02', 'per-session session-03']]
m, se = s.mean(), s.std() / np.sqrt(len(s))
w = 1 / se**2
comb = (w * m).sum() / w.sum(); comb_se = 1 / np.sqrt(w.sum())
print(f'  inverse-variance-weighted combination of sessions 1-3: {comb:+.4f}, z = {comb/comb_se:+.2f}  (weights {dict((k.split()[-1], round(v / w.sum(), 2)) for k, v in w.items())})')
print('  correlation across subjects, concat within-run H vs per-session betas:', {c.split()[-1]: round(df['concat within-run H'].corr(df[c]), 2) for c in s})

print('\n=== Check 2a: within-run H (and Q) vs trial order, per subject and run')
b = pd.read_csv('/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv')
b = b[b.block.isin(['learning1', 'learning2', 'test']) & b.action.notna() & ~b.sub_id.isin(['sub-04', 'sub-45', 'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'])].copy()
b['trial'] = b.groupby(['sub_id', 'block']).t_second_stim.rank()
r = b.groupby(['block', 'sub_id']).apply(lambda g: pd.Series({'r(H, trial)': g.chosen_value_ck_zscore.corr(g.trial),
                                                               'r(Q, trial)': g.chosen_value_rl_zscore.corr(g.trial)}))
print(r.groupby('block').agg(['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).round(2).rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'}).to_string())
```

### 11. With within-run trial order in the model, within-run H loses the voxel-level DLS effect; trial order itself loads on DLS
`glm2_chosen_all_runs_concat_hsplit_time_scrubbed_2026-09-28-12-59`: the H-split model plus a second-stimulus modulator for trial rank within run, demeaned per run (commit `195380f`). VIF median/max: H within 1.9/2.3, trial order 3.0/3.5, H between 4.1/5.5, Q 1.5/2.0. SVC: H within no longer has any DLS/putamen cluster (keeps M1 (−12, −33, 72) and (15, −36, 72), p .014/.015, and parietal (15, −51, 62) k 59 p < .001); trial order: DLS (27, 12, 6) k 12 p .018, (−27, 3, 2) k 27 p .002, putamen (−30, 3, −1) k 26 p .002; H between unchanged in DLS (30, 9, 2) k 27 p .003 and vmPFC (k 7, p .009); Q_chosen VS 2 + 1 voxels. *My first reading ("the within-run DLS effect is explained by trial order") read a voxel-level null as a zero effect and was withdrawn after finding 12.*

### 12. Within-run H relates to DLS beyond a linear time trend (ROI-level): time explains ~25–30 % of it
Beta size in DLS, H split vs H split + time. First done in 6 mm spheres at the manuscript's GLM2 chosen peaks: ratio 0.66/0.69, remaining effect p ≈ .03; discarded because those peaks come from the invalid, session-1-driven per-session map (Hugo). Redone with a leave-one-subject-out ROI (6 mm sphere at the peak of the other 58 subjects' within-run H map, H split without time, inside the Guida mask; peaks (−27, 9, 2) in 56/59 folds, (27, 9, 6) in 59/59): left H within +0.233 (SE 0.076, p .003) → +0.177 with time (SE 0.076, 95 % CI [+0.03, +0.33], p .024); right +0.214 (p < .001) → +0.147 (CI [+0.03, +0.26], p .015); ratio 0.76 / 0.69, drop significant (paired p .028 / .004), SE unchanged (a real reduction, not lost power); trial order +0.005/trial (p .024 / .002). Both remaining effects survive Bonferroni for 2 hemispheres (just, on the left). So: DLS tracks H beyond a linear within-run time trend at ROI level, not voxel-level FWE; the run-level part of H stays confounded with run order; time control is linear only. `python scripts/loso_roi_time_check.py`.

---

## Code shipped (branch `model-reruns`)

| Commit | What |
|---|---|
| `69cc7a5` | `scripts/concat_between_run_variance.py` |
| `48f5d64` | `glm2_chosen_all_runs_concat_hsplit.m` (check 2) and this note |
| `45d4371` | `scripts/compare_models_second_lvl.py` (finding 5) |
| `195380f` | `glm2_chosen_all_runs_concat_hsplit_time.m` (finding 11) |
| (this commit) | `scripts/loso_roi_time_check.py` (finding 12) |

## Open threads
1. H vs time: the within-run H effect in DLS survives a linear time trend at ROI level (finding 12). Robustness: quadratic/nonlinear time; is the trial-order increase specific to the second stimulus (add time to first stimulus and response)?
2. The run-level part of H (55 % of its variance, stronger DLS clusters) cannot be separated from run order; the experiment-wide time regressor on the unsplit concat would show whether H and elapsed time are separable at all.
3. GLM2 chosen with Q only: does the VS Q effect depend on H being in the model (findings 3, 5)?
4. vmPFC test-session effect (findings 5, 8): how to report it; why learning 2 trends negative; repetition vs no feedback vs converged values.
5. Dilution vs cancellation of the vmPFC effect: subject-level counterfactual not run (finding 7 has the group-mean arithmetic).
6. The Q-value loss in both concat models: candidates are the shared onset regressor and the Q/H suppression.
7. vmPFC H effect in the concat models rests mostly on the between-run component: treat as unconfirmed.
