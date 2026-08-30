#!/usr/bin/env python3
"""
Choice-frequency (±1) classification from GLMsingle single-trial betas — one subject.

Binary classification of the choice-frequency label (first_stim_frequ = ±1) on the
non-figure stimulus subset (6 stimuli, 3 per class). Figure stimuli (freq=0) are
excluded because:
  - They hold the extreme value levels {1, 5} by design, confounding value with category.
  - Their frequency label is 0 (neither high nor low choice rate), making them a third
    class that would dilute the binary test.
  - The RSA pipeline's primary readout uses the same non-figure subset.

On the non-figure subset, category is perfectly balanced with frequency (one +1 and
one −1 per category: face, hand, house), so no category confound exists. This is a
structural design property, not an empirical accident.

Two feature variants are reported to confirm this empirically:
  - `run_demeaned`: per-voxel run-mean subtracted (removes session drift)
  - `run_cat_demeaned`: per (run × category)-cell mean subtracted
If the two produce similar accuracy, the category-balance is confirmed. If they diverge,
category structure is contributing — which would be unexpected given the design.

Masks: whole-brain (subject's own functional brain mask, always included) plus any
number of ROI masks passed via repeated --roi-mask NAME PATH, matching the RSA pipeline's
9 ROIs for direct comparison.

Usage
-----
python multivariate/run_frequency_decoding.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/frequency_decoding \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_frequency_decoding.csv                            — mask, n_voxels,
        accuracy_run_demeaned, accuracy_run_cat_demeaned, n_trials, n_neg, n_pos
    sub-<id>_frequency_decoding_confusion_<mask>_<variant>.npy — 2x2, labels=[-1, +1]
    frequency_decoding_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, math_img, index_img
from nilearn.maskers import NiftiMasker
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_target_from_bbt

LABELS = [-1, +1]


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                roi_masks=None, overwrite=False):

    roi_masks = roi_masks or []

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_frequency_decoding.csv"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)

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
    cat_full    = trial_info['stim_cat'].values
    cell_full   = np.array([f"{r}__{c}" for r, c in zip(groups_full, cat_full)])

    # Drop figure stimuli (freq == 0) — non-figure subset only
    keep = y_freq != 0
    y      = y_freq[keep].astype(int)
    groups = groups_full[keep]
    n_neg  = int((y == -1).sum())
    n_pos  = int((y == +1).sum())
    logging.info(f"sub-{subject}: kept {keep.sum()}/{len(y_freq)} trials "
                 f"(freq=-1: {n_neg}, freq=+1: {n_pos}, dropped freq=0: {(~keep).sum()})")

    # --- Brain mask (3D, used as spatial reference) ---
    sub = Subject(
        base_dir=str(base_dir),
        subject_id=subject,
        include_imaging=True,
        bids_dir=str(bids_dir),
    )
    brain_mask_img = nib.load(sub.brain_mask['learning1'])

    # --- ROI masks: resample each atlas mask to subject functional space ---
    masks = [('wholebrain', brain_mask_img)]
    for name, path in roi_masks:
        roi_mni  = nib.load(str(path))
        roi_func = resample_to_img(roi_mni, brain_mask_img, interpolation='nearest')
        # Some masks in 3mm space have a trailing singleton 4th dim (53,65,48,1);
        # squeeze it so math_img can broadcast against the 3D brain mask.
        if roi_func.ndim == 4 and roi_func.shape[3] == 1:
            roi_func = index_img(roi_func, 0)
        roi_mask = math_img('(v > 0) & (b > 0)', v=roi_func, b=brain_mask_img)
        masks.append((name, roi_mask))

    # --- Classify for each mask ---
    logo = LeaveOneGroupOut()
    results = []

    for mask_name, mask_img in masks:
        masker = NiftiMasker(mask_img=mask_img, standardize=True).fit()
        X_full   = masker.transform(betas_img)
        n_voxels = X_full.shape[1]
        logging.info(f"  {mask_name}: {n_voxels:,} voxels")

        # Two feature variants, both computed from ALL trials (before the freq=0
        # filter) on top of the global standardize=True.

        # Variant 1: run-demeaned (per-voxel run-mean subtracted)
        X_run = X_full.copy()
        for r_id in np.unique(groups_full):
            m = groups_full == r_id
            X_run[m] -= X_run[m].mean(axis=0, keepdims=True)

        # Variant 2: run × category demeaned (per-voxel cell-mean subtracted)
        X_cat = X_full.copy()
        for cell in np.unique(cell_full):
            m = cell_full == cell
            X_cat[m] -= X_cat[m].mean(axis=0, keepdims=True)

        result = {'mask': mask_name, 'n_voxels': n_voxels, 'n_trials': len(y),
                  'n_neg': n_neg, 'n_pos': n_pos}

        for variant, X_variant in [('run_demeaned', X_run), ('run_cat_demeaned', X_cat)]:
            X = X_variant[keep]
            y_pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'), X, y,
                                       cv=logo, groups=groups)
            acc = float((y_pred == y).mean())
            result[f'accuracy_{variant}'] = acc
            cm = confusion_matrix(y, y_pred, labels=LABELS)
            np.save(subject_output / f"sub-{subject}_frequency_decoding_confusion_{mask_name}_{variant}.npy", cm)

        logging.info(f"  {mask_name}: accuracy(run-demeaned) = {result['accuracy_run_demeaned']:.3f}  "
                     f"accuracy(run+cat-demeaned) = {result['accuracy_run_cat_demeaned']:.3f}  (chance = 0.5)")
        results.append(result)

    pd.DataFrame(results).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done")


def main():
    parser = argparse.ArgumentParser(
        description="Run choice-frequency (±1) classification for one subject.")
    parser.add_argument("--subject", required=True,
                        help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", required=True,
                        help="Root data directory")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True,
                        help="GLMsingle betas directory")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--roi-mask", action="append", nargs=2, metavar=("NAME", "PATH"),
                        default=[],
                        help="ROI mask to classify, given as NAME PATH; repeatable. "
                             "Whole-brain is always included automatically.")
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
            logging.FileHandler(subject_output / f"frequency_decoding_sub-{args.subject}.log"),
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
