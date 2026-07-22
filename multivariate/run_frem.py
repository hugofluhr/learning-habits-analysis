#!/usr/bin/env python3
"""
Stimulus category FREM decoding — one subject.

Loads GLMsingle type-D betas and fits a whole-brain FREMClassifier (Fast
Regularized Ensemble of Models: ANOVA screening + ReNA clustering + ensembling
of regularized SVCs). Unlike the searchlight (a local decodability map), FREM is
a single global model whose per-class weight maps show which voxels the model
relies on to identify each category, plus cross-validated per-class ROC-AUC.

Complementary to run_searchlight.py:
  searchlight -> "where can category k be read out locally" (accuracy/recall map)
  FREM        -> "which voxels the model uses to identify category k" (weight map)
                 + per-class AUC (chance 0.5).

Caveat: FREM weight maps are discriminative coefficients, NOT response maps; for
neuroscientific interpretation they should be Haufe-transformed (Haufe et al.,
2014). Report the per-class AUC as the quantitative measure.

Usage
-----
python multivariate/run_frem.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --output-dir .../derivatives/frem \\
    --n-jobs 1

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_frem_coef_<cat>.nii.gz   — per-class weight map (one per category)
    sub-<id>_frem_cvscores.csv        — per-class ROC-AUC (mean + per-fold), chance 0.5
    frem_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.decoding import FREMClassifier
from sklearn.model_selection import LeaveOneGroupOut


def run_subject(subject, bids_dir, glmsingle_dir, output_dir,
                n_jobs=1, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    scores_out = subject_output / f"sub-{subject}_frem_cvscores.csv"
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

    y      = trial_info['stim_cat'].values
    groups = trial_info['run'].values

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
    # NOTE: FREMClassifier has no class_weight param (estimator is a string); the
    # 4-way problem is balanced and scoring='roc_auc' is imbalance-robust, so the
    # internal one-vs-all fits are fine without it.
    logging.info(f"sub-{subject}: fitting FREMClassifier (n_jobs={n_jobs})")
    dec = FREMClassifier(
        estimator='svc',
        mask=brain_mask_img,
        cv=LeaveOneGroupOut(),
        screening_percentile=20,
        clustering_percentile=10,
        scoring='roc_auc',
        standardize=True,
        n_jobs=n_jobs,
        verbose=1,
    )
    dec.fit(betas_img, y, groups=groups)

    # --- Save per-class weight maps ---
    for cat, coef_img in dec.coef_img_.items():
        coef_out = subject_output / f"sub-{subject}_frem_coef_{cat}.nii.gz"
        coef_img.to_filename(str(coef_out))
        logging.info(f"sub-{subject}: saved {coef_out.name}")

    # --- Save per-class cross-validated ROC-AUC ---
    # cv_scores_ maps each class label -> list of per-fold scores (one per run).
    rows = []
    for cat, fold_scores in dec.cv_scores_.items():
        fold_scores = np.asarray(fold_scores, dtype=float)
        row = {'category': cat, 'mean_auc': float(np.nanmean(fold_scores))}
        for i, s in enumerate(fold_scores):
            row[f'fold_{i}'] = float(s)
        rows.append(row)
    pd.DataFrame(rows).to_csv(scores_out, index=False)
    logging.info(f"sub-{subject}: saved {scores_out.name}")
    logging.info(f"sub-{subject}: FREM complete")


def main():
    parser = argparse.ArgumentParser(description="Run stimulus category FREM decoding for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
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
            logging.FileHandler(subject_output / f"frem_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        output_dir    = output_dir,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
