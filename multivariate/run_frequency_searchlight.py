#!/usr/bin/env python3
"""
Choice-frequency (±1) classification searchlight — one subject.

Searchlight counterpart of run_frequency_decoding.py. Localizes where in the brain
choice-frequency information is decodable from single-trial GLMsingle betas, at
voxel-level resolution. Complements the ROI-level RSA findings (run_rsa_roi.py) which
establish representational geometry but are confined to predefined masks.

Only non-figure trials are used (freq ≠ 0): the non-figure subset gives a clean binary
±1 split with 3 stimuli per class, perfectly balanced across categories and value levels.

Feature preprocessing (single variant)
---------------------------------------
Run-demeaning only: each voxel's per-run mean is subtracted, computed from ALL trials
before the freq=0 filter (leakage-safe — uses only run membership, never the label).

No run×category demeaning: unlike the value searchlight (run_qvalue_searchlight_
classification.py), where category was a severe confound of the high/low split,
frequency is perfectly balanced across categories on the non-figure subset by design.
The ROI-level results (run_frequency_decoding.py) will confirm the two demeaning
variants produce identical accuracy, validating this choice.

No standardize=True: searchlight spheres hold only ~30-90 spatially adjacent voxels
with minimal across-voxel scale heterogeneity (same reasoning as the category and
value searchlights).

Usage
-----
python multivariate/run_frequency_searchlight.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/searchlight \\
    --n-jobs 8

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_searchlight_frequency.nii.gz       — per-voxel CV accuracy map (chance 0.5)
    frequency_searchlight_sub-<id>.log
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
                radius=6., n_jobs=1, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    out_path = subject_output / f"sub-{subject}_searchlight_frequency.nii.gz"
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

    # --- Choice-frequency label from the BBT ---
    y_freq      = load_target_from_bbt(subject, bbt_path, trial_info,
                                       target_col='first_stim_frequ')
    groups_full = trial_info['run'].values
    logging.info(f"sub-{subject}: target 'first_stim_frequ' -> {len(y_freq)} values, "
                 f"unique: {np.unique(y_freq)}")

    # --- Brain mask: fMRIPrep functional brain mask (first run, MNI space) ---
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

    # --- Run-demeaning (single variant — see module docstring), computed from ALL
    # trials before the freq=0 filter, leakage-safe. No standardize. ---
    masker = NiftiMasker(mask_img=brain_mask_img, standardize=False).fit()
    X_full = masker.transform(betas_img)
    for r_id in np.unique(groups_full):
        m = groups_full == r_id
        X_full[m] -= X_full[m].mean(axis=0, keepdims=True)

    # --- Drop figure stimuli (freq == 0) ---
    keep = y_freq != 0
    y      = y_freq[keep].astype(int)
    groups = groups_full[keep]
    n_neg  = int((y == -1).sum())
    n_pos  = int((y == +1).sum())
    logging.info(f"sub-{subject}: kept {keep.sum()}/{len(y_freq)} trials "
                 f"(freq=-1: {n_neg}, freq=+1: {n_pos}, dropped freq=0: {(~keep).sum()})")

    betas_proc_img = masker.inverse_transform(X_full[keep])

    # --- Searchlight classification (LinearSVC, LeaveOneGroupOut over runs) ---
    logging.info(f"sub-{subject}: running frequency-classification searchlight "
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
    parser = argparse.ArgumentParser(
        description="Run choice-frequency (±1) classification searchlight for one subject.")
    parser.add_argument("--subject", required=True,
                        help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True,
                        help="GLMsingle betas directory")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--radius", type=float, default=6.,
                        help="Searchlight sphere radius in mm (default: 6)")
    parser.add_argument("--n-jobs", type=int, default=1,
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
            logging.FileHandler(subject_output / f"frequency_searchlight_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        bbt_path      = args.bbt,
        output_dir    = output_dir,
        radius        = args.radius,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
