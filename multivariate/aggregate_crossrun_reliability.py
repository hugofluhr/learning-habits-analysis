#!/usr/bin/env python3
"""
Aggregate `run_beta_crossrun_reliability.py` per-subject CSVs and test whether
learning-vs-test reliability is lower than learning1-vs-learning2 (session-notes
2026-09-03 finding 17).

Usage
-----
python multivariate/aggregate_crossrun_reliability.py \\
    --glmsingle-qc-dir /path/to/derivatives/glmsingle_qc
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glmsingle-qc-dir", required=True,
                        help="Directory containing sub-*/sub-*_crossrun_reliability.csv")
    args = parser.parse_args()

    files = glob.glob(str(Path(args.glmsingle_qc_dir) / "sub-*" / "sub-*_crossrun_reliability.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['pair'] = df.run_a + '_vs_' + df.run_b
    print(f"{df.subject.nunique()} subjects, {len(df)} rows")

    per_sub = df.groupby(['mask', 'pair', 'subject'])['r'].mean().reset_index()
    summary = per_sub.groupby(['mask', 'pair'])['r'].agg(['mean', 'std', 'count'])
    print(summary.round(3).to_string())
    print()

    for mask in ['wholebrain', 'visualcortex', 'fusiform']:
        sub = per_sub[per_sub['mask'] == mask].pivot(index='subject', columns='pair', values='r')
        ll = sub['learning1_vs_learning2']
        test_avg = (sub['learning1_vs_test'] + sub['learning2_vs_test']) / 2
        t, p = stats.ttest_rel(ll, test_avg)
        print(f"--- {mask} ---")
        print(f"  learning1_vs_learning2: {ll.mean():.3f} +- {ll.std():.3f}")
        print(f"  learning-vs-test (avg): {test_avg.mean():.3f} +- {test_avg.std():.3f}")
        print(f"  paired t-test (learning1_vs_learning2 vs learning-vs-test avg): "
              f"t={t:.2f}, p={p:.2e}")
        print()


if __name__ == "__main__":
    main()
