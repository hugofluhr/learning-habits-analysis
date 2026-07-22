#!/usr/bin/env python3
"""
Stimulus category searchlight decoding — one subject.

Loads GLMsingle type-D betas, runs a whole-brain searchlight (LinearSVC,
leave-one-run-out CV, 6mm radius), saves NIfTI maps.

The overall map is 4-class accuracy (chance 0.25). The per-category maps are
per-class *recall* extracted from the single competitive 4-way model (chance
0.25) — NOT one-vs-rest accuracy. One-vs-rest accuracy suffers a 1:3 class
imbalance (~80 positive vs ~245 negative per subject), sitting on a hidden 0.75
floor and dominated by the majority class; per-class recall from the balanced
4-way model reads directly as "how reliably this sphere identifies category k
against the other 3."

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
    sub-<id>_searchlight_stim_cat.nii.gz              — overall 4-class accuracy map (chance 0.25)
    sub-<id>_searchlight_stim_cat_recall_<cat>.nii.gz — per-class recall from the 4-way model (chance 0.25)
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
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC


def _recall_scorer(cat):
    """Callable scorer -> recall (sensitivity) for one class label.

    A plain (estimator, X, y) callable is what nilearn's SearchLight `scoring`
    expects. We avoid sklearn's make_scorer here because, for a string-labelled
    multiclass problem, make_scorer injects a binary pos_label=1 that isn't a
    valid label and raises. Keying on the label value (labels=[cat]) is
    independent of the estimator's classes_ ordering; zero_division=0 turns any
    0/0 fold into a truthful 0 rather than NaN.
    """
    def _score(estimator, X, y_true):
        return recall_score(y_true, estimator.predict(X),
                            labels=[cat], average='macro', zero_division=0)
    return _score


def run_subject(subject, bids_dir, glmsingle_dir, output_dir,
                radius=6., n_jobs=1, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    overall_out = subject_output / f"sub-{subject}_searchlight_stim_cat.nii.gz"

    subject_output.mkdir(parents=True, exist_ok=True)

    # --- Load betas and trial info ---
    betas_path = glmsingle_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"
    info_path  = glmsingle_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES_info.csv"

    betas_img  = nib.load(betas_path)
    trial_info = pd.read_csv(info_path, index_col='trial_id')

    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    y      = trial_info['stim_cat'].values
    groups = trial_info['run'].values
    categories = sorted(trial_info['stim_cat'].unique())

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

    def _make_searchlight(scoring='accuracy'):
        # class_weight='balanced' is a near-no-op on the balanced 4-way problem
        # but is correct for the per-class recall passes below.
        return SearchLight(
            mask_img=brain_mask_img,
            radius=radius,
            estimator=LinearSVC(max_iter=10000, dual='auto', class_weight='balanced'),
            cv=LeaveOneGroupOut(),
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
        )

    # --- Overall 4-class searchlight ---
    if overall_out.exists() and not overwrite:
        logging.info(f"sub-{subject}: overall output exists, skipping")
    else:
        logging.info(f"sub-{subject}: running overall searchlight (radius={radius}mm, n_jobs={n_jobs})")
        sl = _make_searchlight()
        sl.fit(betas_img, y, groups=groups)
        new_img_like(brain_mask_img, sl.scores_).to_filename(str(overall_out))
        logging.info(f"sub-{subject}: saved {overall_out.name}")

    # --- Per-category recall searchlights (from the competitive 4-way model) ---
    # Each pass fits the full 4-class problem but scores only the recall
    # (sensitivity) of one category via _recall_scorer(cat). See that function.
    for cat in categories:
        cat_out = subject_output / f"sub-{subject}_searchlight_stim_cat_recall_{cat}.nii.gz"
        if cat_out.exists() and not overwrite:
            logging.info(f"sub-{subject}: {cat} recall output exists, skipping")
            continue
        logging.info(f"sub-{subject}: running per-class recall searchlight for '{cat}'")
        sl = _make_searchlight(scoring=_recall_scorer(cat))
        sl.fit(betas_img, y, groups=groups)
        new_img_like(brain_mask_img, sl.scores_).to_filename(str(cat_out))
        logging.info(f"sub-{subject}: saved {cat_out.name}")

    logging.info(f"sub-{subject}: all searchlights complete")


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
