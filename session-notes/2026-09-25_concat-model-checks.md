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

---

## Code shipped (branch `model-reruns`)

| Commit | What |
|---|---|
| `69cc7a5` | `scripts/concat_between_run_variance.py` |
| `48f5d64` | `glm2_chosen_all_runs_concat_hsplit.m` (check 2) and this note |

## Open threads
1. ~~Check 2~~ done (finding 2). Next: GLM2 chosen with Q only (per-session and concat), to test whether the VS Q effect depends on H being in the model (finding 3) and to explain the Q loss.
2. Between-subject covariate test (does a larger H rise predict a larger increase in DLS stimulus response across runs): only on the raw H scale, if at all (finding 1).
3. Learning runs only (runs 1 + 2) variant, to remove the learning/test phase change as an alternative explanation.
4. The Q-value loss in both concat models: not caused by how H is modelled (finding 2); candidates are the shared onset regressor and the Q/H suppression in finding 3.
5. vmPFC H effect in the concat models rests mostly on the between-run component: treat as unconfirmed.
