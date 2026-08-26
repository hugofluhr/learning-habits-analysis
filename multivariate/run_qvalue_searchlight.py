#!/usr/bin/env python3
"""
Reward-value (Q-value) searchlight decoding — one subject.

Regression counterpart of run_searchlight.py. Loads the same GLMsingle type-D
CUES betas, but instead of classifying stimulus category it *regresses* a
continuous per-trial target — by default the objective reward level of the first
stimulus (`first_stim_value`, integers 1-5) — sourced from the Big Behavior Table
(the same table used to fit the SPM first-levels). Ridge estimator, leave-one-run-out
CV, 6mm radius; saves one NIfTI map.

Why reward level (not the model RL Q)? The 8 stimulus identities map to reward
levels {1,2,2,3,3,4,4,5}: three levels are each shared by two *different*
identities, so a sphere that reads out reward cannot be doing pure identity
decoding. The map therefore asks "where can stimulus *value* be read out locally".

The map is per-voxel cross-validated Pearson correlation between predicted and true
reward (baseline 0), NOT r^2 — searchlight r^2 goes sharply negative in
uninformative spheres and produces unreadable maps.

Usage
-----
python multivariate/run_qvalue_searchlight.py --subject 01 \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/searchlight \\
    --n-jobs 8

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_searchlight_reward.nii.gz   — per-voxel CV correlation map (baseline 0)
    qvalue_searchlight_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.decoding import SearchLight
from nilearn.image import new_img_like
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import load_target_from_bbt


def _corr_scorer(estimator, X, y_true):
    """Callable scorer -> Pearson r between predicted and true target.

    A plain (estimator, X, y) callable is what nilearn's SearchLight `scoring`
    expects. Pearson r (baseline 0) is used instead of r^2 because r^2 is sharply
    negative in uninformative spheres, swamping the map; r stays in [-1, 1] and
    reads directly as local decodability. A constant prediction (zero variance)
    scores 0 rather than NaN.
    """
    pred = estimator.predict(X)
    if np.std(pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, pred)[0, 1])


def run_subject(subject, bids_dir, glmsingle_dir, bbt_path, output_dir,
                target_col='first_stim_value', radius=6., n_jobs=1, overwrite=False):

    tag = 'reward' if target_col == 'first_stim_value' else target_col
    subject_output = output_dir / f"sub-{subject}"
    out_path = subject_output / f"sub-{subject}_searchlight_{tag}.nii.gz"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        logging.info(f"sub-{subject}: output exists, skipping (pass --overwrite to rerun)")
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

    # --- Searchlight regression (Ridge, LeaveOneGroupOut over runs) ---
    logging.info(f"sub-{subject}: running reward searchlight "
                 f"(radius={radius}mm, n_jobs={n_jobs})")
    sl = SearchLight(
        mask_img=brain_mask_img,
        radius=radius,
        estimator=Ridge(),
        cv=LeaveOneGroupOut(),
        scoring=_corr_scorer,
        n_jobs=n_jobs,
        verbose=1,
    )
    sl.fit(betas_img, y, groups=groups)
    new_img_like(brain_mask_img, sl.scores_).to_filename(str(out_path))
    logging.info(f"sub-{subject}: saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run reward-value searchlight regression for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True,
                        help="GLMsingle betas directory")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV holding the target column "
                             "(same table used to fit the SPM first-levels)")
    parser.add_argument("--target-col", default="first_stim_value",
                        help="BBT column to decode (default: first_stim_value, objective reward)")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--radius",  type=float, default=6.,
                        help="Searchlight sphere radius in mm (default: 6)")
    parser.add_argument("--n-jobs",  type=int,   default=1,
                        help="Parallel jobs for searchlight (default: 1)")
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
            logging.FileHandler(subject_output / f"qvalue_searchlight_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        bbt_path      = args.bbt,
        output_dir    = output_dir,
        target_col    = args.target_col,
        radius        = args.radius,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
