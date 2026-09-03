#!/usr/bin/env python3
"""
Early-half vs. late-half RSA within learning1/learning2 — one subject.

Direct test of Hugo's objection to the "genuine accumulating habit" reading of
session-notes 2026-09-03 finding 12 (beta(frequency) flat learning1 ~= learning2,
collapses in test): if the frequency manipulation needs repeated exposure/choice
reinforcement to build a habitual representation, beta(frequency) should be visibly
weaker in the FIRST half of learning1 (before much has been reinforced) than in the
SECOND half. If instead the effect is already present early and doesn't grow within
a run either, that argues for a structural/pairing-context account (the choice-
frequency label is a FIXED partner assignment held constant for the whole learning
phase -- finding 16 -- not something revealed gradually), consistent with finding 8a
(the graded choice-kernel H-value barely reconstructs the design label's structure).

Splits each of learning1/learning2 at the median trial time (GLOBAL median across
that run's trials, not per-stimulus) into 'early'/'late' halves, then computes an
independent crossnobis RDM within each half (interleaved 2-fold CV, same logic as
run_rsa_roi.py's within-run scopes, just on a trial subset). Per-stimulus trial
counts in a half are thin (dev_sample: nonfigure minimum ~3-6/stimulus/half) --
a subject/half/mask is skipped (NaN row, not a crash) if any non-figure stimulus
has fewer than MIN_TRIALS trials in a fold, mirroring run_rsa_partner_context.py's
minimum-count gate.

Only the primary 3-predictor regression (category, value, frequency) plus the
second_stim_value/choice_rate confound controls are reported -- same MODEL_TERMS
as run_rsa_roi.py, on the nonfigure subset only (the primary readout).

Usage
-----
python multivariate/run_rsa_learning_dynamics.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir  .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/rsa_learning_dynamics \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz

Outputs to <output-dir>/sub-<id>/
    sub-<id>_rsa_learning_dynamics.csv   one row per (mask, run, half)
    rsa_learning_dynamics_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for utils.*
sys.path.insert(0, str(Path(__file__).resolve().parent))      # this dir, for run_rsa_roi
from run_rsa_roi import (  # noqa: E402  (path inserts must come first)
    load_stimuli, build_masks, compute_noise_sd, _crossnobis_loo, within_run_folds,
    abs_diff_rdm, different_rdm, fit_rdm_regression, FIGURE_CAT, MODEL_TERMS,
)

LEARNING_RUNS = ['learning1', 'learning2']
MIN_TRIALS_PER_FOLD = 2   # each condition needs >=1 obs/fold; require >=2 for stability


def split_early_late(run_mask, trial_order):
    """Boolean (early_mask, late_mask) splitting this run's trials at the global
    median chronological position -- not per-stimulus, so 'early'/'late' means
    'early/late in wall-clock run time', matching the exposure/reinforcement
    argument directly."""
    idx = np.flatnonzero(run_mask)
    order = idx[np.argsort(trial_order[idx])]
    half = len(order) // 2
    early = np.zeros_like(run_mask)
    late = np.zeros_like(run_mask)
    early[order[:half]] = True
    late[order[half:]] = True
    return early, late


def min_fold_count(cond_idx, fold_idx, obs_mask, keep_conditions):
    """Minimum (condition, fold) observation count over the kept conditions -- the
    binding constraint for whether crossnobis can even run on this subset."""
    counts = []
    for c in keep_conditions:
        for f in np.unique(fold_idx[obs_mask]):
            counts.append(int(((cond_idx == c) & (fold_idx == f) & obs_mask).sum()))
    return min(counts) if counts else 0


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                roi_masks=None, overwrite=False):

    roi_masks = roi_masks or []
    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_rsa_learning_dynamics.csv"
    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return
    subject_output.mkdir(parents=True, exist_ok=True)

    (betas_img, trial_info, cond_idx, run_labels, props,
     rl_per_trial, ck_per_trial,
     s2v_per_trial, chosen_per_trial,
     s2f_per_trial, s2cat_per_trial, s2name_per_trial) = load_stimuli(
        subject, glmsingle_dir, bbt_path, shuffle_seed=None)
    stim_names, n_cond = props['names'], len(props['names'])
    non_figure = props['cat'] != FIGURE_CAT
    keep_idx = np.flatnonzero(non_figure)

    model_rdms_base = {'category':  different_rdm(props['cat']),
                        'value':     abs_diff_rdm(props['value']),
                        'frequency': abs_diff_rdm(props['frequency'])}

    X_wb, masks = build_masks(subject, base_dir, bids_dir, betas_img, roi_masks)
    trial_order = np.asarray(trial_info.index.values, dtype=float)

    rows = []
    for run in LEARNING_RUNS:
        run_mask = run_labels == run
        early_mask, late_mask = split_early_late(run_mask, trial_order)

        for half_name, half_mask in [('early', early_mask), ('late', late_mask)]:
            folds_full = np.full(len(cond_idx), -1, dtype=int)
            folds_full[half_mask] = within_run_folds(
                cond_idx[half_mask], trial_order[half_mask], mode='interleaved')

            # second_stim_value / choice_rate confound RDMs, this half only.
            s2v_rdm = abs_diff_rdm(pd.Series(s2v_per_trial[half_mask])
                                   .groupby(cond_idx[half_mask]).mean()
                                   .reindex(range(n_cond)).values)
            cr_rdm = abs_diff_rdm(pd.Series(chosen_per_trial[half_mask])
                                  .groupby(cond_idx[half_mask]).mean()
                                  .reindex(range(n_cond)).values)
            models = {**model_rdms_base, 'second_stim_value': s2v_rdm, 'choice_rate': cr_rdm}

            min_n = min_fold_count(cond_idx, folds_full, half_mask, keep_idx)
            usable = min_n >= MIN_TRIALS_PER_FOLD

            for mask_name, vox in masks:
                if not usable:
                    rows.append({'subject': f"sub-{subject}", 'mask': mask_name,
                                'run': run, 'half': half_name, 'usable': False,
                                'min_fold_n': min_n, 'n_trials': int(half_mask.sum())})
                    continue
                X = X_wb[:, vox]
                sd = compute_noise_sd(X, cond_idx, run_labels)
                good = sd > 0
                Xw = X[:, good] / sd[good]
                try:
                    rdm = _crossnobis_loo(Xw[half_mask], cond_idx[half_mask],
                                          folds_full[half_mask], n_cond)
                except ValueError as e:
                    logging.warning(f"sub-{subject}/{mask_name}/{run}/{half_name}: "
                                    f"crossnobis failed ({e}), marking unusable")
                    rows.append({'subject': f"sub-{subject}", 'mask': mask_name,
                                'run': run, 'half': half_name, 'usable': False,
                                'min_fold_n': min_n, 'n_trials': int(half_mask.sum())})
                    continue

                betas, corrs, n_pairs = fit_rdm_regression(rdm, models, non_figure)
                rows.append({
                    'subject': f"sub-{subject}", 'mask': mask_name, 'run': run,
                    'half': half_name, 'usable': True, 'min_fold_n': min_n,
                    'n_trials': int(half_mask.sum()), 'n_pairs': n_pairs,
                    **{f'beta_{k}': v for k, v in betas.items()},
                })
                logging.info(f"  {mask_name}/{run}/{half_name}: n_trials={int(half_mask.sum())} "
                            f"min_fold_n={min_n} beta_frq={betas['frequency']:+.3f} "
                            f"beta_val={betas['value']:+.3f}")

    pd.DataFrame(rows).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done — {len(rows)} rows -> {done_flag}")


def main():
    parser = argparse.ArgumentParser(
        description="Early-half vs. late-half RSA within learning1/learning2 for one subject.")
    parser.add_argument("--subject", required=True, help="ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--bids-dir", required=True)
    parser.add_argument("--glmsingle-dir", required=True)
    parser.add_argument("--bbt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-mask", action="append", nargs=2, metavar=("NAME", "PATH"),
                        default=[], help="ROI mask as NAME PATH; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    subject_output = output_dir / f"sub-{args.subject}"
    subject_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(subject_output / f"rsa_learning_dynamics_sub-{args.subject}.log")],
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
