# Session log: bulletproofing the concat models

**Date:** 2026-09-25 (check 2 set up 2026-09-28)
**Companion note:** [2026-09-24_tier1-reruns-and-second-level-comparison.md](2026-09-24_tier1-reruns-and-second-level-comparison.md) (the concat models' results, open thread 1: bulletproof the GLM2 chosen concat and understand the Q-value loss)

Checks on whether the concat models' H-value effects could come from run-level differences rather than trial-by-trial H. Hugo's framing: the between-run differences of H are real signal (slow accumulation of choice history), so demeaning H within runs is not an option; the question is whether the data can separate that signal from anything else that changes across runs.

---

## Findings

### 1. More than half of the chosen-H regressor is between-run variance, and it is a near-linear function of run order
Second-level sample (59 subjects), share of trial-level variance between runs, median [IQR]: chosen H 0.55 [0.52–0.58] (min 0.43), GLM1 concat first/second-stim H 0.41 / 0.43, all Q modulators 0.00–0.02. Median chosen-H run means −1.03 / −0.04 / +0.76 (learning 1 / learning 2 / test), so the run-level part rises ~0.9 z-units per run in every subject; the rise from learning 1 to test barely varies across subjects (+1.79, IQR +1.74 to +1.83). Consequences: the H effect in the concat models is confounded with run order to the extent it rests on this component; a between-subject covariate test on the z-scored rise has almost no leverage; the Q loss is not explained by Q's own between-run variance (≈ 0). `python scripts/concat_between_run_variance.py --out <csv>` (commit `69cc7a5`); per-subject table on the cluster: `spm_outputs/compare_second_lvl_2026-09-24-14-56/concat_between_run_variance.csv`.

---

## Code shipped (branch `model-reruns`)

| Commit | What |
|---|---|
| `69cc7a5` | `scripts/concat_between_run_variance.py` |

## Open threads
1. Check 2: split chosen H into within-run and run-mean components in the concat model (both kept, nothing removed) and see which carries the DLS effect.
2. Between-subject covariate test (does a larger H rise predict a larger increase in DLS stimulus response across runs): only on the raw H scale, if at all (finding 1).
3. Learning runs only (runs 1 + 2) variant, to remove the learning/test phase change as an alternative explanation.
4. The Q-value loss in both concat models: remaining hypothesis is the shared onset regressor (stimulus response forced equal across runs).
