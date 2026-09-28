"""How much of each concat-model parametric modulator is between-run variance?

In the concatenated designs (glm2_all_runs_concat.m, glm2_chosen_all_runs_concat.m) one onset regressor spans all three
runs, so the run means of a modulator are not absorbed by per-run onset regressors: the between-run part of the modulator
is confounded with anything else that changes from run to run. This computes, per subject and modulator, the share of the
trial-level variance that lies between runs (between-run sum of squares / total sum of squares), and the run means.

Trials are selected as in the first-level scripts: all trials for the GLM1 concat modulators (block_data), response trials
only for the chosen-value modulators (block_resp). sub-04 and sub-45 are skipped, as in the GLM scripts.

Usage:
    python scripts/concat_between_run_variance.py [--bbt PATH] [--out CSV]
"""
import argparse

import numpy as np
import pandas as pd

RUNS = ['learning1', 'learning2', 'test']
MODULATORS = {  # column -> (model, trial selection)
    'first_stim_value_rl_zscore': ('glm1_concat', 'all'),
    'first_stim_value_ck_zscore': ('glm1_concat', 'all'),
    'second_stim_value_rl_zscore': ('glm1_concat', 'all'),
    'second_stim_value_ck_zscore': ('glm1_concat', 'all'),
    'chosen_value_rl_zscore': ('glm2_chosen_concat', 'resp'),
    'chosen_value_ck_zscore': ('glm2_chosen_concat', 'resp'),
}
SKIP = {'sub-04', 'sub-45'}  # skipped by the GLM scripts
EXCLUDED_2ND = {'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'}  # second-level exclusions


def between_share(x, run):
    grand = x.mean()
    total = ((x - grand) ** 2).sum()
    between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in x.groupby(run))
    return between / total if total > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bbt', default='/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv')
    ap.add_argument('--out', default=None, help='per-subject CSV (optional)')
    args = ap.parse_args()

    bbt = pd.read_csv(args.bbt)
    bbt = bbt[bbt['block'].isin(RUNS) & ~bbt['sub_id'].isin(SKIP)]
    rows = []
    for sub, d in bbt.groupby('sub_id'):
        for col, (model, sel) in MODULATORS.items():
            dd = d if sel == 'all' else d[d['action'].notna()]
            dd = dd[dd[col].notna()]
            means = dd.groupby('block')[col].mean().reindex(RUNS)
            rows.append({'sub_id': sub, 'model': model, 'modulator': col, 'n_trials': len(dd),
                         'between_share': between_share(dd[col], dd['block']),
                         **{f'mean_{r}': means[r] for r in RUNS}, 'in_second_level': sub not in EXCLUDED_2ND})
    res = pd.DataFrame(rows)
    if args.out:
        res.to_csv(args.out, index=False)

    s = res[res['in_second_level']]
    print(f"Subjects: {s['sub_id'].nunique()} (second-level sample)\n")
    print('Between-run share of variance (median [IQR], min-max):')
    for (model, col), g in s.groupby(['model', 'modulator'], sort=False):
        b = g['between_share']
        print(f"  {model:18s} {col:28s} {b.median():.2f} [{b.quantile(.25):.2f}-{b.quantile(.75):.2f}]  {b.min():.2f}-{b.max():.2f}")
    print('\nRun means (median across subjects) and rise from learning1 to test (median [IQR]):')
    for (model, col), g in s.groupby(['model', 'modulator'], sort=False):
        rise = g['mean_test'] - g['mean_learning1']
        print(f"  {model:18s} {col:28s} " + ' '.join(f"{g[f'mean_{r}'].median():+.2f}" for r in RUNS)
              + f"   rise {rise.median():+.2f} [{rise.quantile(.25):+.2f} to {rise.quantile(.75):+.2f}]")


if __name__ == '__main__':
    main()
