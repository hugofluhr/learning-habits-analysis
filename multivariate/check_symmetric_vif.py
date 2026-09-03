#!/usr/bin/env python3
"""
Pre-flight VIF check for `run_rsa_roi.py --symmetric` — no betas, no cluster.

Builds all 8 model RDMs (category, value, frequency, second_stim_value, choice_rate,
s2_category, s2_frequency, s2_identity) directly from the BBT for each subject/scope
and reports variance inflation factors on the non-figure subset (the pipeline's
primary readout — see run_rsa_roi.py MODEL_TERMS/SYMMETRIC_TERMS). Determines whether
s2_identity (or any other new predictor) causes problematic collinearity before
committing a `--symmetric` cluster run.

Runs in seconds locally: unlike `run_rsa_roi.py::load_stimuli`, this does NOT need the
GLMsingle info CSV for chronological alignment — it builds conditions and RDMs straight
from the BBT's own first_stim_name/block columns, which is all the regressors need.

Usage
-----
python multivariate/check_symmetric_vif.py --bbt /path/to/bbt.csv
python multivariate/check_symmetric_vif.py --bbt /path/to/bbt.csv --subjects 01 02 03
python multivariate/check_symmetric_vif.py --bbt /path/to/bbt.csv --n-subjects 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rsa_roi import (
    abs_diff_rdm, different_rdm, profile_dist_rdm, compute_vif, FIGURE_CAT,
    MODEL_TERMS, SYMMETRIC_TERMS,
)

RUNS = ['learning1', 'learning2', 'test']


def build_scope_rdms(sub_bbt, scope):
    """All 8 model RDMs for one subject/scope, built directly from BBT rows (no
    GLMsingle info CSV needed — see module docstring)."""
    block = sub_bbt if scope == 'pooled' else sub_bbt[sub_bbt['block'] == scope]
    if block.empty:
        return None

    stim_names = np.array(sorted(block['first_stim_name'].unique()))
    n_cond = len(stim_names)
    cond_idx = np.searchsorted(stim_names, block['first_stim_name'].values)
    obs = np.ones(len(block), dtype=bool)

    def per_stim(col):
        return (block.groupby('first_stim_name')[col].first()
                 .reindex(stim_names).values)

    def mean_per_cond(values):
        return (pd.Series(np.asarray(values, dtype=float))
                 .groupby(cond_idx).mean().reindex(range(n_cond)).values)

    cat = per_stim('first_stim_cat')
    value = per_stim('first_stim_value').astype(float)
    frequency = per_stim('first_stim_frequ').astype(float)

    chosen = (block['first_stim'].values == block['chosen_stim'].values).astype(float)
    s2cat_categories = sorted(block['second_stim_cat'].unique())

    model_rdms = {
        'category':          different_rdm(cat),
        'value':             abs_diff_rdm(value),
        'frequency':         abs_diff_rdm(frequency),
        'second_stim_value': abs_diff_rdm(mean_per_cond(block['second_stim_value'].values)),
        'choice_rate':       abs_diff_rdm(mean_per_cond(chosen)),
        's2_category':       profile_dist_rdm(block['second_stim_cat'].values, cond_idx,
                                               obs, n_cond, s2cat_categories),
        's2_frequency':      abs_diff_rdm(mean_per_cond(block['second_stim_frequ'].values)),
        's2_identity':       profile_dist_rdm(block['second_stim_name'].values, cond_idx,
                                               obs, n_cond, stim_names),
    }
    non_figure = cat != FIGURE_CAT
    return model_rdms, non_figure


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight VIF check for run_rsa_roi.py --symmetric.")
    parser.add_argument("--bbt", required=True, help="Path to the Big Behavior Table CSV")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Specific subject IDs (without 'sub-'); default: all in BBT")
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Cap the number of subjects checked (first N in BBT order)")
    parser.add_argument("--vif-threshold", type=float, default=10.0,
                        help="Flag predictors exceeding this VIF (default 10)")
    args = parser.parse_args()

    bbt = pd.read_csv(args.bbt)
    all_terms = MODEL_TERMS + SYMMETRIC_TERMS

    if args.subjects:
        sub_ids = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]
    else:
        sub_ids = sorted(bbt['sub_id'].unique())
        if args.n_subjects:
            sub_ids = sub_ids[:args.n_subjects]

    rows = []
    for sub_id in sub_ids:
        sub_bbt = bbt[bbt['sub_id'] == sub_id]
        if sub_bbt.empty:
            print(f"{sub_id}: absent from BBT, skipped")
            continue
        for scope in RUNS + ['pooled']:
            built = build_scope_rdms(sub_bbt, scope)
            if built is None:
                continue
            model_rdms, non_figure = built
            for subset_name, keep in [('all', np.ones_like(non_figure)),
                                       ('nonfigure', non_figure)]:
                vifs = compute_vif(model_rdms, keep)
                rows.append({'sub_id': sub_id, 'scope': scope, 'subset': subset_name,
                            **{f'vif_{k}': v for k, v in vifs.items()}})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No subjects/scopes produced a result — check --bbt / --subjects.")
        sys.exit(1)

    print(f"\n{len(sub_ids)} subjects x {len(RUNS)+1} scopes x 2 subsets "
          f"= {len(df)} rows\n")
    print("Median / max VIF per predictor, by scope x subset:\n")
    vif_cols = [f'vif_{t}' for t in all_terms]
    summary = df.groupby(['scope', 'subset'])[vif_cols].agg(['median', 'max'])
    with pd.option_context('display.width', 160, 'display.max_columns', None):
        print(summary)

    flagged = df[(df[vif_cols] > args.vif_threshold).any(axis=1)]
    print(f"\n{len(flagged)}/{len(df)} (subject, scope, subset) rows have at least one "
          f"predictor VIF > {args.vif_threshold}:")
    if len(flagged):
        worst = summary.xs('max', axis=1, level=1).max().sort_values(ascending=False)
        print("\nWorst-case (max over all rows) VIF per predictor:")
        print(worst)
    else:
        print("None — all predictors stay under threshold everywhere checked.")


if __name__ == "__main__":
    main()
