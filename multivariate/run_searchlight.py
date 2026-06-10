#!/usr/bin/env python3
"""
Stimulus category searchlight decoding — one subject.

Loads GLMsingle type-D betas, runs a whole-brain searchlight (LinearSVC,
leave-one-run-out CV, 6mm radius), saves a NIfTI accuracy map.

Usage
-----
python multivariate/run_searchlight.py --subject 01 \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/searchlight \\
    --n-jobs 8

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_searchlight_stim_cat.nii.gz   — voxel-wise decoding accuracy map
    searchlight_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd
from nilearn.decoding import SearchLight
from nilearn.image import new_img_like
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC


def run_subject(subject, bids_dir, glmsingle_dir, output_dir,
                radius=6., n_jobs=1, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_searchlight_stim_cat.nii.gz"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: output exists, skipping (pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)

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

    logging.info(f"sub-{subject}: running searchlight (radius={radius}mm, n_jobs={n_jobs})")

    sl = SearchLight(
        mask_img=brain_mask_img,
        radius=radius,
        estimator=LinearSVC(max_iter=10000, dual='auto'),
        cv=LeaveOneGroupOut(),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
    )
    sl.fit(betas_img, y, groups=groups)

    acc_img = new_img_like(brain_mask_img, sl.scores_)
    acc_img.to_filename(str(done_flag))
    logging.info(f"sub-{subject}: done — saved {done_flag.name}")


def main():
    parser = argparse.ArgumentParser(description="Run stimulus category searchlight for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
    parser.add_argument("--output-dir",    default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/searchlight")
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
            logging.FileHandler(subject_output / f"searchlight_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        output_dir    = output_dir,
        radius        = args.radius,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
