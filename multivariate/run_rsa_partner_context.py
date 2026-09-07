#!/usr/bin/env python3
"""
RSA partner-context pollution test on GLMsingle cue betas — one subject.

Targeted, neural version of `rsa_design_checks.ipynb` §8/§8b (session-notes
2026-09-03, findings 16-18): during `learning`, every stimulus has one fixed
*target* (dominant) partner it's paired with in 18/24 pair-level trials (75%),
and one other ("rest") partner in the other 6/24 -- the choice-frequency label's
6:18 repetition ratio. During `test`, the same stimulus is paired with all 7
other stimuli, but its same-value partner is shown 3x as often as any other
(12 vs 4 pair-level trials) -- a much weaker, more diffuse version of the same
lopsided-partner structure. Hugo's hypothesis: `run_rsa_roi.py` conditions are
per-stimulus trial-pool means, so a stimulus's beta is partly a "this stimulus,
in the context of its most-associated partner" trace, not a clean identity
trace -- and this pollution should be large in `learning` (one partner
dominates) and much smaller in `test` (no partner dominates; finding 16).

`--scope {learning,test}` runs the same underlying test in either phase:
  * learning: target = the single most-frequent partner (2 partners total per
    stimulus); rest = the other one.
  * test: target = the single most-frequent partner (asserted to be the
    same-value partner -- the design's only 3x-oversampled pair); rest = the
    pooled trials against the other 6 partners.

Design note that shaped this script (found via a local dry run against `bbt.csv`
before touching the cluster -- see session-notes finding 18): GLMsingle betas are
locked to FIRST-stimulus onset only, so a stimulus's beta condition only ever
sees the subset of its pair-level appearances where it happened to be shown
first -- roughly half, unevenly. The realized per-subject target/rest split is
therefore noisier and more variable than the clean pair-level ratios (learning
mean target-partner count 6.0 trials sd 1.8 range 1-11, pooled over 496 subject x
stimulus cells). Some subjects have a target or rest count of 1-2 for a given
stimulus -- too sparse to split into 2 CV folds. Handled by: (a) testing each
stimulus's target-vs-rest distance independently (a sparse stimulus doesn't sink
the subject's other 7), (b) an interleaved-trial-order 2-fold split pooling all
of the scope's trials (needs only >=2 trials total per condition, not >=1 per
block/run), and (c) a per-stimulus minimum-trial-count gate (>=3 on the smaller
side) that drops the stimulus (NaN) rather than including a degenerate estimate.

IMPORTANT construct-validity note (session-notes 2026-09-03, added after the first
`test` run came back with LARGER partner_distance than `learning`, the opposite of
the naive prediction): `learning`'s target/rest split isolates pure REPETITION
FREQUENCY of an otherwise-arbitrary partner (same stimulus, same partner, just
seen 3x more). `test`'s target/rest split isolates a VALUE TIE -- a same-value
comparison is a qualitatively different decision problem (no objectively correct
choice), not merely a more-repeated one. The two `partner_distance` numbers are
therefore NOT measuring the same construct and should not be compared directly
as "did pollution shrink in test" -- a large `test` value most plausibly reflects
a genuine decision-conflict/tied-value signature, not repetition-driven context
pollution.

A first attempt at a construct-matched `test` null (`find_floor_split`: an
arbitrary partner-identity split among the 6 different-value partners) backfired
instructively: it came back LARGER than everything else (finding 19), because an
arbitrary id-sorted 3-vs-3 split has no control over the CATEGORY composition of
the two groups, and category is a big, uncontrolled confound here. That result
motivated `find_category_split`/`category_distance` below -- the direct, honest
version: group by the suspected confound (second-stimulus category) instead of
hoping an arbitrary split averages it away. Computed for BOTH scopes (not
test-only), since the underlying concern -- does the co-present stimulus's
category leak into `s`'s own beta at all, independent of value-tie or
repetition-frequency structure -- applies equally to `learning` and `test`.

Numbers per stimulus:
  * partner_distance     = crossnobis distance between the SAME stimulus's
                           target-partner-trials pattern and rest-partner-trials
                           pattern (see scope-specific definitions above).
                           Crossnobis is unbiased around 0 when two conditions
                           don't differ, so a group-level one-sample t-test of
                           this value against 0 is already the test for "does
                           partner context leave a residual signature after
                           conditioning on identity" -- no permutation/null
                           needed.
  * category_distance    = the same crossnobis test between `s`'s own two most
                           numerous second-stimulus-CATEGORY subgroups (e.g.
                           "partner was a face" vs "partner was a house") --
                           tests the more basic category-leak concern directly,
                           in both scopes.
  * betweenstim_distance = the ordinary 8-condition (identity-only) crossnobis
                           RDM restricted to this scope's trials (2-fold
                           learning1-vs-learning2 for `learning`; interleaved
                           within-run pseudo-halves for `test`), mean
                           off-diagonal -- the standard between-identity RSA
                           signal, for a same-units magnitude comparison
                           against partner_distance/category_distance.

Usage
-----
python multivariate/run_rsa_partner_context.py --subject 01 --scope learning \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir  .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/rsa_partner_context \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz \\
    --roi-mask fusiform .../fusiform_mask_MNI152NLin2009cAsym.nii

Outputs to <output-dir>/sub-<id>/
    sub-<id>_partner_context_<scope>.csv    one row per (mask, stim_name):
                                     partner_distance, target_n, rest_n, usable
    sub-<id>_partner_context_<scope>_summary.csv   one row per mask: mean
                                     partner_distance (over usable stimuli),
                                     betweenstim_distance, ratio
    rsa_partner_context_<scope>_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for utils.*
sys.path.insert(0, str(Path(__file__).resolve().parent))      # this dir, for run_rsa_roi
from utils.data import load_target_from_bbt, load_string_target_from_bbt
from run_rsa_roi import (  # noqa: E402  (path inserts must come first)
    load_stimuli, build_masks, compute_noise_sd, _crossnobis_loo, within_run_folds,
)

LEARNING_RUNS = ['learning1', 'learning2']
SCOPE_RUNS = {'learning': LEARNING_RUNS, 'test': ['test']}
N_EXPECTED_PARTNERS = {'learning': 2, 'test': 7}
MIN_MINOR_TRIALS = 3  # per-stimulus inclusion gate; see module docstring


def find_target_partner(trial_info, second_stim, second_stim_value, stim_value, stim_names,
                         scope_mask, scope):
    """Per stimulus: boolean target/rest trial masks (over the full trial array),
    trial counts, and whether the smaller side clears MIN_MINOR_TRIALS.

    The two scopes define "target" differently, and NOT symmetrically by empirical
    frequency -- an earlier version picked the empirically most-frequent partner in
    both scopes and broke on `test`: (a) figure_circle/figure_triangle hold the only
    two unique reward values in this design (`rsa_design_checks.ipynb` §3), so they
    have NO same-value partner in `test` at all -- "most frequent partner" is then
    just sampling noise, not a real oversampled pair; (b) even for non-figure
    stimuli, the true same-value partner is sometimes edged out by chance given only
    ~2-4 first-stimulus-locked trials per partner. Both are avoided by defining
    `test`'s target group directly from value equality (`second_stim_value`),
    which is known a priori and needs no frequency inference:
      * learning: target = the single empirically most-frequent partner (this
        *is* frequency-defined by design -- the 18:6 dominant/minor split).
      * test: target = trials whose partner has the SAME reward value as `s`
        (the design's only 3x-oversampled pair); figure stimuli have none and are
        marked unusable, matching how `run_rsa_roi.py` already treats them.
    """
    partners = {}
    for s in stim_names:
        sel = scope_mask & (trial_info['stim_name'].values == s)

        if scope == 'learning':
            counts = pd.Series(second_stim[sel]).value_counts()
            assert len(counts) == N_EXPECTED_PARTNERS['learning'], (
                f"stimulus {s}: expected 2 partners in `learning`, "
                f"got {len(counts)}: {counts.to_dict()}"
            )
            dominant = counts.index[0]
            target_mask = sel & (second_stim == dominant)
            rest_mask = sel & ~(second_stim == dominant)
        else:  # test
            target_mask = sel & (second_stim_value == stim_value[s])
            rest_mask = sel & (second_stim_value != stim_value[s])

        target_n, rest_n = int(target_mask.sum()), int(rest_mask.sum())
        partners[s] = dict(target_mask=target_mask, rest_mask=rest_mask,
                           target_n=target_n, rest_n=rest_n,
                           usable=(target_n > 0 and rest_n > 0 and
                                   min(target_n, rest_n) >= MIN_MINOR_TRIALS))
    return partners


def find_category_split(trial_info, second_stim_cat, stim_names, scope_mask):
    """Per stimulus: boolean masks for its two most numerous second-stimulus-CATEGORY
    subgroups, holding first-stimulus identity fixed (same-stimulus, own trials only).

    Tests the more basic concern behind findings 16-19 directly: does the CATEGORY of
    whichever stimulus was shown alongside `s` leave a residual pattern signature in
    `s`'s own beta, independent of value-tie or repetition-frequency structure?
    Superseded the `test`-only "floor" arbitrary-partner-id split (session-notes
    2026-09-03 finding 19): that split's own huge, unexpected magnitude turned out to
    be exactly this -- an uncontrolled partner-CATEGORY imbalance between its two
    arbitrary id-sorted halves, not a clean null. This is the direct, honest version
    of that test: group by the confound itself instead of hoping an arbitrary split
    averages it away.
    """
    splits = {}
    for s in stim_names:
        sel = scope_mask & (trial_info['stim_name'].values == s)
        counts = pd.Series(second_stim_cat[sel]).value_counts()
        if len(counts) < 2:
            splits[s] = dict(a_mask=None, b_mask=None, a_n=0, b_n=0,
                             cat_a=None, cat_b=None, usable=False)
            continue
        cat_a, cat_b = counts.index[0], counts.index[1]
        a_mask = sel & (second_stim_cat == cat_a)
        b_mask = sel & (second_stim_cat == cat_b)
        a_n, b_n = int(a_mask.sum()), int(b_mask.sum())
        splits[s] = dict(a_mask=a_mask, b_mask=b_mask, a_n=a_n, b_n=b_n,
                         cat_a=cat_a, cat_b=cat_b,
                         usable=(a_n > 0 and b_n > 0 and min(a_n, b_n) >= MIN_MINOR_TRIALS))
    return splits


def per_stimulus_split_distance(X, trial_order, mask_a, mask_b):
    """Crossnobis distance between two arbitrary (mutually exclusive) trial-mask
    conditions for one stimulus, cross-validated via an interleaved-trial-order
    2-fold split pooling all of the scope's trials (not a block/run split -- see
    module docstring for why that fails on sparse partner counts)."""
    sel = mask_a | mask_b
    lab = mask_b[sel].astype(int)  # 0=a, 1=b
    order = trial_order[sel]

    fold = within_run_folds(lab, order, mode='interleaved')
    Xs = X[sel]
    sd = compute_noise_sd(Xs, lab, np.zeros_like(lab))  # per-condition demeaning only
    good = sd > 0
    Xw = Xs[:, good] / sd[good]
    rdm2 = _crossnobis_loo(Xw, lab, fold, n_cond=2)
    return float(rdm2[0, 1])


def betweenstim_distance(X, cond_idx, run_labels, trial_order, scope_mask, scope, n_cond):
    """Ordinary 8-condition crossnobis RDM restricted to this scope's trials --
    the magnitude-comparison baseline. `learning` cross-validates over its 2
    blocks; `test` (a single run) uses an interleaved within-run 2-fold split,
    same convention as `run_rsa_roi.py`'s per-run scopes."""
    Xs = X[scope_mask]
    conds = cond_idx[scope_mask]
    if scope == 'learning':
        fold = run_labels[scope_mask]
    else:
        fold = within_run_folds(conds, trial_order[scope_mask], mode='interleaved')
    sd = compute_noise_sd(Xs, conds, fold)
    good = sd > 0
    Xw = Xs[:, good] / sd[good]
    rdm = _crossnobis_loo(Xw, conds, fold, n_cond)
    return float(rdm[np.triu_indices(n_cond, 1)].mean())


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                 roi_masks, scope, overwrite=False):
    subject_output = output_dir / f"sub-{subject}"
    out_csv = subject_output / f"sub-{subject}_partner_context_{scope}.csv"
    out_summary_csv = subject_output / f"sub-{subject}_partner_context_{scope}_summary.csv"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        logging.info(f"sub-{subject}: {scope} output exists, skipping (pass --overwrite to rerun)")
        return

    (betas_img, trial_info, cond_idx, run_labels, props, *_rest) = load_stimuli(
        subject, glmsingle_dir, bbt_path)
    stim_names = list(props['names'])
    n_cond = len(stim_names)
    stim_value = dict(zip(props['names'], props['value']))

    second_stim = load_target_from_bbt(subject, bbt_path, trial_info, target_col='second_stim')
    second_stim_value = load_target_from_bbt(subject, bbt_path, trial_info,
                                             target_col='second_stim_value')
    second_stim_cat = load_string_target_from_bbt(subject, bbt_path, trial_info,
                                                  target_col='second_stim_cat')
    trial_order = np.asarray(trial_info.index.values, dtype=float)  # chronological
    scope_mask = np.isin(run_labels, SCOPE_RUNS[scope])

    partners = find_target_partner(trial_info, second_stim, second_stim_value, stim_value,
                                   stim_names, scope_mask, scope)
    n_usable = sum(p['usable'] for p in partners.values())
    logging.info(f"sub-{subject}: [{scope}] {n_usable}/{len(stim_names)} stimuli usable "
                 f"(min_n>={MIN_MINOR_TRIALS}) -- " +
                 ", ".join(f"{s}:{p['target_n']}/{p['rest_n']}" for s, p in partners.items()))
    if n_usable == 0:
        logging.warning(f"sub-{subject}: [{scope}] 0 usable stimuli, nothing to compute")
        return

    cat_splits = find_category_split(trial_info, second_stim_cat, stim_names, scope_mask)
    n_cat_usable = sum(c['usable'] for c in cat_splits.values())
    logging.info(f"sub-{subject}: [{scope}] category split {n_cat_usable}/{len(stim_names)} "
                 f"stimuli usable -- " +
                 ", ".join(f"{s}:{c['cat_a']}={c['a_n']}/{c['cat_b']}={c['b_n']}"
                          for s, c in cat_splits.items() if c['cat_a'] is not None))

    X_wb, masks = build_masks(subject, base_dir, bids_dir, betas_img, roi_masks)

    per_stim_rows, summary_rows = [], []
    for mask_name, mask_idx in masks:
        X = X_wb[:, mask_idx]

        stim_rows = []
        for s in stim_names:
            p = partners[s]
            d = (per_stimulus_split_distance(X, trial_order, p['target_mask'], p['rest_mask'])
                 if p['usable'] else float('nan'))
            row = dict(stim_name=s, partner_distance=d,
                      target_n=p['target_n'], rest_n=p['rest_n'], usable=p['usable'])
            c = cat_splits[s]
            cd = (per_stimulus_split_distance(X, trial_order, c['a_mask'], c['b_mask'])
                 if c['usable'] else float('nan'))
            row.update(category_distance=cd, cat_a=c['cat_a'], cat_b=c['cat_b'],
                      cat_a_n=c['a_n'], cat_b_n=c['b_n'], category_usable=c['usable'])
            stim_rows.append(row)
        per_stim = pd.DataFrame(stim_rows)
        per_stim['mask'] = mask_name
        per_stim['subject'] = f'sub-{subject}'
        per_stim['scope'] = scope
        per_stim_rows.append(per_stim)

        between = betweenstim_distance(X, cond_idx, run_labels, trial_order, scope_mask,
                                       scope, n_cond)
        partner_mean = float(per_stim.loc[per_stim.usable, 'partner_distance'].mean())
        summary = dict(mask=mask_name, subject=f'sub-{subject}', scope=scope,
                       n_usable=int(per_stim.usable.sum()),
                       partner_distance_mean=partner_mean,
                       betweenstim_distance=between,
                       ratio=(partner_mean / between) if between != 0 else float('nan'))
        cat_mean = float(per_stim.loc[per_stim.category_usable, 'category_distance'].mean())
        summary['category_distance_mean'] = cat_mean
        summary['n_category_usable'] = int(per_stim.category_usable.sum())
        summary_rows.append(summary)
        logging.info(f"sub-{subject}: [{scope}] {mask_name} partner_distance_mean="
                     f"{partner_mean:+.4f} (n={summary['n_usable']})  "
                     f"betweenstim_distance={between:+.4f}  ratio={summary['ratio']:+.3f}  "
                     f"category_distance_mean={cat_mean:+.4f} (n={summary['n_category_usable']})")

    pd.concat(per_stim_rows, ignore_index=True).to_csv(out_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}, {out_summary_csv.name}")


def main():
    parser = argparse.ArgumentParser(
        description="RSA partner-context pollution test on GLMsingle cue betas, one subject.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--scope", choices=['learning', 'test'], default='learning')
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--bids-dir", required=True)
    parser.add_argument("--glmsingle-dir", required=True)
    parser.add_argument("--bbt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-mask", nargs=2, action="append", default=[],
                        metavar=("NAME", "PATH"),
                        help="Add an ROI mask (repeatable): --roi-mask name /path/to/mask.nii.gz")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    subject_output = output_dir / f"sub-{args.subject}"
    subject_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                subject_output / f"rsa_partner_context_{args.scope}_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        base_dir      = Path(args.base_dir),
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        output_dir    = output_dir,
        bbt_path      = args.bbt,
        roi_masks     = args.roi_mask,
        scope         = args.scope,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
