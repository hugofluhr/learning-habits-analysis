#!/usr/bin/env python3
"""
RSA partner-context pollution test on GLMsingle cue betas — one subject.

Targeted, neural version of `rsa_design_checks.ipynb` §8/§8b (session-notes
2026-09-03, findings 16-17): during `learning`, every stimulus has one fixed
*dominant* partner it's paired with in 18/24 pair-level trials (75%), and a
*minor* partner in the other 6/24 -- the choice-frequency label's 6:18
repetition ratio. Hugo's hypothesis: `run_rsa_roi.py` conditions are per-stimulus
trial-pool means, so a stimulus's `learning` beta is partly a "this stimulus, in
the context of its majority partner" trace, not a clean identity trace --
pollution that has nothing comparable to average toward in `test` (no partner
dominates there; finding 16).

Design note that shaped this script (found via a local dry run against `bbt.csv`
before touching the cluster -- see session-notes finding 18): GLMsingle betas are
locked to FIRST-stimulus onset only, so a stimulus's beta condition only ever
sees the subset of its pair-level appearances where it happened to be shown
first -- roughly half, unevenly. The realized per-subject dominant/minor split is
therefore noisier and more variable than the clean 18:6 pair-level ratio (mean
minor-partner count 6.0 trials, sd 1.8, range 1-11 across 496 subject x stimulus
cells). ~10/62 subjects have a minor-partner count of 1-2 for at least one
stimulus -- too sparse to split into 2 CV folds. Handled by: (a) testing each
stimulus's dominant-vs-minor distance independently (a sparse stimulus doesn't
sink the subject's other 7), (b) an interleaved-trial-order 2-fold split pooling
both learning runs (needs only >=2 trials total per condition, not >=1 per
block), and (c) a per-stimulus minimum-trial-count gate (>=3, keeps 485/496
cells) that drops the stimulus (NaN) rather than including a degenerate estimate.

Two numbers per stimulus:
  * partner_distance     = crossnobis distance between the SAME stimulus's
                           dominant-partner-trials pattern and minor-partner-
                           trials pattern. Crossnobis is unbiased around 0 when
                           two conditions don't differ, so a group-level
                           one-sample t-test of this value against 0 is already
                           the test for "does partner context leave a residual
                           signature after conditioning on identity" -- no
                           permutation/null needed.
  * betweenstim_distance = the ordinary 8-condition (identity-only), 2-fold
                           (learning1 vs learning2) crossnobis RDM restricted to
                           `learning`, mean off-diagonal -- the standard
                           between-identity RSA signal, for a same-units
                           magnitude comparison against partner_distance. No
                           sparsity concerns here (each stimulus's ordinary
                           condition uses ~24 trials/block).

Usage
-----
python multivariate/run_rsa_partner_context.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir  .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/rsa_partner_context \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz \\
    --roi-mask fusiform .../fusiform_mask_MNI152NLin2009cAsym.nii

Outputs to <output-dir>/sub-<id>/
    sub-<id>_partner_context.csv    one row per (mask, stim_name): partner_distance,
                                     dom_n, min_n, usable
    sub-<id>_partner_context_summary.csv   one row per mask: mean partner_distance
                                     (over usable stimuli), betweenstim_distance, ratio
    rsa_partner_context_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for utils.*
sys.path.insert(0, str(Path(__file__).resolve().parent))      # this dir, for run_rsa_roi
from utils.data import load_target_from_bbt
from run_rsa_roi import (  # noqa: E402  (path inserts must come first)
    load_stimuli, build_masks, compute_noise_sd, _crossnobis_loo, within_run_folds,
)

LEARNING_RUNS = ['learning1', 'learning2']
MIN_MINOR_TRIALS = 3  # per-stimulus inclusion gate; see module docstring


def find_dominant_minor_partners(trial_info, second_stim, stim_names, learning_mask):
    """Per stimulus: dominant/minor partner id, trial counts, and whether the minor
    count clears MIN_MINOR_TRIALS (see module docstring for why this is needed and
    why the threshold is set where it is)."""
    partners = {}
    for s in stim_names:
        sel = learning_mask & (trial_info['stim_name'].values == s)
        counts = pd.Series(second_stim[sel]).value_counts()
        assert len(counts) == 2, (
            f"stimulus {s}: expected exactly 2 partners in `learning`, "
            f"got {len(counts)}: {counts.to_dict()}"
        )
        dom_partner, min_partner = counts.index[0], counts.index[1]
        dom_n, min_n = int(counts.iloc[0]), int(counts.iloc[1])
        assert dom_n > min_n, (
            f"stimulus {s}: expected a strict dominant/minor split, got {dom_n}/{min_n}"
        )
        partners[s] = dict(dominant=dom_partner, minor=min_partner, dom_n=dom_n, min_n=min_n,
                           usable=(min_n >= MIN_MINOR_TRIALS))
    return partners


def per_stimulus_partner_distance(X, trial_info, second_stim, trial_order, s, p):
    """Crossnobis distance between stimulus `s`'s dominant- and minor-partner-trial
    patterns, cross-validated via an interleaved-trial-order 2-fold split pooling
    both learning runs (not a learning1-vs-learning2 block split -- see module
    docstring for why that fails on sparse minor-partner counts)."""
    sel = (trial_info['stim_name'].values == s) & (
        (second_stim == p['dominant']) | (second_stim == p['minor']))
    lab = (second_stim[sel] == p['minor']).astype(int)  # 0=dominant, 1=minor
    order = trial_order[sel]

    fold = within_run_folds(lab, order, mode='interleaved')
    Xs = X[sel]
    sd = compute_noise_sd(Xs, lab, np.zeros_like(lab))  # per-condition demeaning only
    good = sd > 0
    Xw = Xs[:, good] / sd[good]
    rdm2 = _crossnobis_loo(Xw, lab, fold, n_cond=2)
    return float(rdm2[0, 1])


def betweenstim_distance_learning(X, cond_idx, run_labels, learning_mask, n_cond):
    """Ordinary 8-condition, 2-fold (learning1 vs learning2) crossnobis RDM
    restricted to `learning` trials -- the magnitude-comparison baseline."""
    Xl = X[learning_mask]
    condl = cond_idx[learning_mask]
    foldl = run_labels[learning_mask]
    sd = compute_noise_sd(Xl, condl, foldl)
    good = sd > 0
    Xw = Xl[:, good] / sd[good]
    rdm = _crossnobis_loo(Xw, condl, foldl, n_cond)
    return float(rdm[np.triu_indices(n_cond, 1)].mean())


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                 roi_masks, overwrite=False):
    subject_output = output_dir / f"sub-{subject}"
    out_csv = subject_output / f"sub-{subject}_partner_context.csv"
    out_summary_csv = subject_output / f"sub-{subject}_partner_context_summary.csv"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        logging.info(f"sub-{subject}: output exists, skipping (pass --overwrite to rerun)")
        return

    (betas_img, trial_info, cond_idx, run_labels, props, *_rest) = load_stimuli(
        subject, glmsingle_dir, bbt_path)
    stim_names = list(props['names'])
    n_cond = len(stim_names)

    second_stim = load_target_from_bbt(subject, bbt_path, trial_info, target_col='second_stim')
    trial_order = np.asarray(trial_info.index.values, dtype=float)  # chronological
    learning_mask = np.isin(run_labels, LEARNING_RUNS)

    partners = find_dominant_minor_partners(trial_info, second_stim, stim_names, learning_mask)
    n_usable = sum(p['usable'] for p in partners.values())
    logging.info(f"sub-{subject}: {n_usable}/{len(stim_names)} stimuli usable "
                 f"(min_n>={MIN_MINOR_TRIALS}) -- " +
                 ", ".join(f"{s}:{p['dom_n']}/{p['min_n']}" for s, p in partners.items()))
    if n_usable == 0:
        logging.warning(f"sub-{subject}: 0 usable stimuli, nothing to compute")
        return

    X_wb, masks = build_masks(subject, base_dir, bids_dir, betas_img, roi_masks)

    per_stim_rows, summary_rows = [], []
    for mask_name, mask_idx in masks:
        X = X_wb[:, mask_idx]

        stim_rows = []
        for s in stim_names:
            p = partners[s]
            d = (per_stimulus_partner_distance(X, trial_info, second_stim, trial_order, s, p)
                 if p['usable'] else float('nan'))
            stim_rows.append(dict(stim_name=s, partner_distance=d,
                                  dom_n=p['dom_n'], min_n=p['min_n'], usable=p['usable']))
        per_stim = pd.DataFrame(stim_rows)
        per_stim['mask'] = mask_name
        per_stim['subject'] = f'sub-{subject}'
        per_stim_rows.append(per_stim)

        betweenstim = betweenstim_distance_learning(X, cond_idx, run_labels, learning_mask, n_cond)
        partner_mean = float(per_stim.loc[per_stim.usable, 'partner_distance'].mean())
        summary = dict(mask=mask_name, subject=f'sub-{subject}',
                       n_usable=int(per_stim.usable.sum()),
                       partner_distance_mean=partner_mean,
                       betweenstim_distance=betweenstim,
                       ratio=(partner_mean / betweenstim) if betweenstim != 0 else float('nan'))
        summary_rows.append(summary)
        logging.info(f"sub-{subject}: {mask_name} partner_distance_mean={partner_mean:+.4f} "
                     f"(n={summary['n_usable']})  betweenstim_distance={betweenstim:+.4f}  "
                     f"ratio={summary['ratio']:+.3f}")

    pd.concat(per_stim_rows, ignore_index=True).to_csv(out_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}, {out_summary_csv.name}")


def main():
    parser = argparse.ArgumentParser(
        description="RSA partner-context pollution test on GLMsingle cue betas, one subject.")
    parser.add_argument("--subject", required=True)
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
            logging.FileHandler(subject_output / f"rsa_partner_context_sub-{args.subject}.log"),
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
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
