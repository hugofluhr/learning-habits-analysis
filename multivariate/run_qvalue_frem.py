#!/usr/bin/env python3
"""
Reward-value (Q-value) FREM decoding — one subject.

Regression counterpart of run_frem.py. Loads the same GLMsingle type-D CUES betas
and fits a whole-brain FREMRegressor (Fast Regularized Ensemble of Models: ANOVA
screening + ReNA clustering + ensembling of regularized SVRs) to a continuous
per-trial target — by default the objective reward level of the first stimulus
(`first_stim_value`, integers 1-5) — sourced from the Big Behavior Table (the same
table used to fit the SPM first-levels).

Complementary to run_qvalue_searchlight.py:
  searchlight -> "where can reward be read out locally" (correlation map)
  FREM        -> "which voxels the global model uses to predict reward" (weight map)
                 + cross-validated r^2 (baseline 0).

Why reward level (not the model RL Q)? The 8 stimulus identities map to reward
levels {1,2,2,3,3,4,4,5}: three levels are each shared by two *different*
identities, so the model cannot predict reward purely from stimulus identity.

Caveat: the FREM weight map is a discriminative/coefficient map, NOT a response
map; for neuroscientific interpretation it should be Haufe-transformed (Haufe et
al., 2014). Report the cross-validated r^2 as the quantitative measure.

Usage
-----
python multivariate/run_qvalue_frem.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /mnt/data/learning-habits/bbt.csv \\
    --output-dir .../derivatives/frem \\
    --n-jobs 1

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_frem_reward_coef.nii.gz   — whole-brain weight map
    sub-<id>_frem_reward_cvscores.csv  — cross-validated r^2 (mean + per-fold), baseline 0
    qvalue_frem_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.decoding import FREMRegressor
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import load_target_from_bbt


def _single(d):
    """FREM regressor stores coef_img_/cv_scores_ as a 1-key dict (key 'beta').
    Return that single value, tolerating either a dict or a bare object."""
    if isinstance(d, dict):
        return d.get('beta', next(iter(d.values())))
    return d


def run_subject(subject, bids_dir, glmsingle_dir, bbt_path, output_dir,
                target_col='first_stim_value', n_jobs=1, overwrite=False):

    tag = 'reward' if target_col == 'first_stim_value' else target_col
    subject_output = output_dir / f"sub-{subject}"
    scores_out = subject_output / f"sub-{subject}_frem_{tag}_cvscores.csv"
    coef_out   = subject_output / f"sub-{subject}_frem_{tag}_coef.nii.gz"
    subject_output.mkdir(parents=True, exist_ok=True)

    if scores_out.exists() and not overwrite:
        logging.info(f"sub-{subject}: FREM outputs exist, skipping "
                     f"(pass --overwrite to rerun)")
        return

    # --- Load betas and trial info ---
    betas_path = glmsingle_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"
    info_path  = glmsingle_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES_info.csv"

    betas_img  = nib.load(betas_path)
    trial_info = pd.read_csv(info_path, index_col='trial_id')

    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    # --- Continuous target from the BBT, aligned to the beta order ---
    y = load_target_from_bbt(subject, bbt_path, trial_info, target_col=target_col)
    groups = trial_info['run'].values
    logging.info(f"sub-{subject}: target '{target_col}' -> {len(y)} values, "
                 f"range [{y.min():.3g}, {y.max():.3g}]")

    # --- Brain mask: fMRIPrep functional brain mask (first run, MNI space) ---
    # rglob handles session-structured layouts (ses-1/func/) transparently
    mask_candidates = sorted(
        m for m in (bids_dir / f"sub-{subject}").rglob(
            f"sub-{subject}_*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        )
        if m.parent.name == "func"
    )
    if not mask_candidates:
        raise FileNotFoundError(f"No fMRIPrep brain mask found for sub-{subject} in {bids_dir}")
    brain_mask_img = nib.load(mask_candidates[0])
    logging.info(f"sub-{subject}: brain mask {mask_candidates[0].name}")

    # --- Fit FREM (LeaveOneGroupOut over runs — run-independent, no leakage) ---
    logging.info(f"sub-{subject}: fitting FREMRegressor (n_jobs={n_jobs})")
    dec = FREMRegressor(
        estimator='svr',
        mask=brain_mask_img,
        cv=LeaveOneGroupOut(),
        screening_percentile=20,
        clustering_percentile=10,
        scoring='r2',
        standardize=True,
        n_jobs=n_jobs,
        verbose=1,
    )
    dec.fit(betas_img, y, groups=groups)

    # --- Save the whole-brain weight map ---
    _single(dec.coef_img_).to_filename(str(coef_out))
    logging.info(f"sub-{subject}: saved {coef_out.name}")

    # --- Save cross-validated r^2 (per-fold, one per run) ---
    fold_scores = np.asarray(_single(dec.cv_scores_), dtype=float)
    row = {'target': target_col, 'mean_r2': float(np.nanmean(fold_scores))}
    for i, s in enumerate(fold_scores):
        row[f'fold_{i}'] = float(s)
    pd.DataFrame([row]).to_csv(scores_out, index=False)
    logging.info(f"sub-{subject}: saved {scores_out.name} "
                 f"(mean r2 = {row['mean_r2']:.3f}, baseline 0)")
    logging.info(f"sub-{subject}: FREM complete")


def main():
    parser = argparse.ArgumentParser(description="Run reward-value FREM regression for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV holding the target column "
                             "(same table used to fit the SPM first-levels)")
    parser.add_argument("--target-col", default="first_stim_value",
                        help="BBT column to decode (default: first_stim_value, objective reward)")
    parser.add_argument("--output-dir",    default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/frem")
    parser.add_argument("--n-jobs",  type=int,   default=1,
                        help="Parallel jobs for FREM (default: 1)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir     = Path(args.output_dir)
    subject_output = output_dir / f"sub-{args.subject}"
    subject_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(subject_output / f"qvalue_frem_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        bbt_path      = args.bbt,
        output_dir    = output_dir,
        target_col    = args.target_col,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
