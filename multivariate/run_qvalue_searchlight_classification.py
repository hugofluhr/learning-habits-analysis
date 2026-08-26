#!/usr/bin/env python3
"""
Reward-value (Q-value) high/low classification searchlight — one subject.

Searchlight counterpart of run_qvalue_classification.py, not of the existing
run_qvalue_searchlight.py (that script regresses a continuous reward level; it was
never run on the cluster and has no results notebook). We localize the *classification*
analysis instead: it's the better-behaved of the two whole-ROI reward analyses —
binary high/low split around the middle reward level (chance = 0.5, LinearSVC,
LeaveOneGroupOut over runs), sidestepping the between-run CV-arithmetic artifact that
affects the regression version (see run_qvalue_decoding.py's docstring). Whole-brain/
visual-cortex decode reward genuinely at whole-ROI resolution (run_qvalue_classification.py:
acc=0.549/0.615, surviving a category-confound control); vmPFC/striatum are flat at
chance either way. RidgeCV regularizing those ~120-voxel ROIs almost to a constant in the
regression version is circumstantial evidence a spatially localized signal could be
getting washed out by whole-ROI averaging — this searchlight is the direct way to check.

Feature preprocessing (single variant, not two)
------------------------------------------------
run_qvalue_classification.py reports two feature variants — run-demeaned and
run×category-demeaned — and found they tell nearly the same story (wholebrain
0.558->0.549, vmPFC 0.498->0.499). Since stimulus category is a near-deterministic
confound of the low/high split for every subject (chi-square p in 1e-15-1e-22 range,
checked there), category-demeaning is needed for validity regardless of whether it
changes the numbers much — so this script computes only that single, already-primary
variant rather than paying for two full whole-brain SearchLight.fit() calls per subject.
Each voxel's own per (run x category)-cell mean is subtracted, computed from ALL trials
before the middle-level drop — leakage-safe, since it uses only known run/category group
membership, never the low/high label.

No standardize=True. Unlike the ~65k-voxel whole-brain/ROI decoders (where standardizing
matters because L2-penalized LinearSVC is otherwise biased toward high-magnitude voxels),
each searchlight sphere here holds only ~30-90 voxels (6mm radius) — spatially adjacent,
tissue-similar voxels with far less across-voxel scale heterogeneity by construction. Same
conclusion reached for the category searchlight (run_searchlight.py): standardization and
run-demeaning were both investigated there and found unwarranted at this local scale.

Usage
-----
python multivariate/run_qvalue_searchlight_classification.py --subject 01 \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/searchlight \\
    --n-jobs 8

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_searchlight_reward_classification.nii.gz   — per-voxel CV accuracy map (chance 0.5)
    qvalue_searchlight_classification_sub-<id>.log
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
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import load_target_from_bbt


def run_subject(subject, bids_dir, glmsingle_dir, bbt_path, output_dir,
                target_col='first_stim_value', low_max=2.0, high_min=4.0,
                radius=6., n_jobs=1, overwrite=False):

    tag = 'reward' if target_col == 'first_stim_value' else target_col
    subject_output = output_dir / f"sub-{subject}"
    out_path = subject_output / f"sub-{subject}_searchlight_{tag}_classification.nii.gz"
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

    # --- Continuous target from the BBT, then split into low/high, dropping the middle ---
    y_cont      = load_target_from_bbt(subject, bbt_path, trial_info, target_col=target_col)
    groups_full = trial_info['run'].values
    cat_full    = trial_info['stim_cat'].values
    logging.info(f"sub-{subject}: target '{target_col}' -> {len(y_cont)} values, "
                 f"range [{y_cont.min():.3g}, {y_cont.max():.3g}]")

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

    # --- Run x category demeaning (single variant — see module docstring), computed
    # from ALL trials before the low/high filter, leakage-safe (run/category group
    # membership only, never the label). No standardize (see module docstring). ---
    masker = NiftiMasker(mask_img=brain_mask_img, standardize=False).fit()
    X_full = masker.transform(betas_img)
    cell_full = np.array([f"{r}__{c}" for r, c in zip(groups_full, cat_full)])
    for cell in np.unique(cell_full):
        m = cell_full == cell
        X_full[m] -= X_full[m].mean(axis=0, keepdims=True)

    # --- Split into low/high, dropping the middle reward level ---
    keep = (y_cont <= low_max) | (y_cont >= high_min)
    y      = np.where(y_cont[keep] <= low_max, 'low', 'high')
    groups = groups_full[keep]
    n_low, n_high = int((y == 'low').sum()), int((y == 'high').sum())
    logging.info(f"sub-{subject}: kept {keep.sum()}/{len(y_cont)} trials "
                 f"(low<={low_max:g}: {n_low}, high>={high_min:g}: {n_high})")

    betas_proc_img = masker.inverse_transform(X_full[keep])

    # --- Searchlight classification (LinearSVC, LeaveOneGroupOut over runs) ---
    logging.info(f"sub-{subject}: running reward-classification searchlight "
                 f"(radius={radius}mm, n_jobs={n_jobs})")
    sl = SearchLight(
        mask_img=brain_mask_img,
        radius=radius,
        estimator=LinearSVC(max_iter=10000, dual='auto'),
        cv=LeaveOneGroupOut(),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
    )
    sl.fit(betas_proc_img, y, groups=groups)
    new_img_like(brain_mask_img, sl.scores_).to_filename(str(out_path))
    logging.info(f"sub-{subject}: saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run reward high/low classification searchlight for one subject.")
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
    parser.add_argument("--low-max",  type=float, default=2.0,
                        help="Trials with target <= this value are labeled 'low' (default: 2.0)")
    parser.add_argument("--high-min", type=float, default=4.0,
                        help="Trials with target >= this value are labeled 'high' (default: 4.0)")
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
            logging.FileHandler(subject_output / f"qvalue_searchlight_classification_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        bbt_path      = args.bbt,
        output_dir    = output_dir,
        target_col    = args.target_col,
        low_max       = args.low_max,
        high_min      = args.high_min,
        radius        = args.radius,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
