#!/usr/bin/env python3
"""
Second-stimulus category decoding — one subject.

Existence proof for finding 19 (session-notes/2026-09-03): GLMsingle cue-locked betas
are locked to first-stimulus onset, but first stim, second stim, decision and response
all fall within one TR (~2.33s), so the single-trial beta is a mixed-event
representation. This script asks the direct question: can the second stimulus's
CATEGORY be decoded from a beta nominally "about" the first stimulus?

4-class (face/hand/house/figure) LinearSVC, leave-one-run-out CV, same masks/task
structure as `run_decoding.py` (which does the equivalent stim-1 decoding — run that
separately and compare CSVs, not duplicated here).

Two feature variants, mirroring `run_frequency_decoding.py`'s demeaning logic:
  - raw (global standardize=True only)
  - `run_s1cat_demeaned`: per (run x stim-1-category) cell-mean subtracted — proves
    any above-chance stim-2 accuracy isn't just stim-1 category patterns leaking
    through (stim-1 category is fully confounded with run x stim-1-category cell
    membership, so this demeaning removes all stim-1-category-driven variance).

All trials are kept (every trial has a valid second-stimulus category, unlike the
frequency decoder which drops figure stimuli).

Usage
-----
python multivariate/run_stim2_decoding.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/stim2_decoding \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_stim2_decoding.csv                                 — mask, n_voxels,
        accuracy, accuracy_s1cat_demeaned, n_trials, chance
    sub-<id>_stim2_decoding_confusion_<mask>_<variant>.npy      — 4x4
    sub-<id>_stim2_decoding_labels.npy                          — category order
    stim2_decoding_sub-<id>.log
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
from utils.data import Subject, load_string_target_from_bbt


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                roi_masks=None, overwrite=False):

    roi_masks = roi_masks or []

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_stim2_decoding.csv"

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

    # --- Second-stimulus category label from the BBT (string column) ---
    y = load_string_target_from_bbt(subject, bbt_path, trial_info, target_col='second_stim_cat')
    groups   = trial_info['run'].values
    s1cat    = trial_info['stim_cat'].values
    cell_full = np.array([f"{r}__{c}" for r, c in zip(groups, s1cat)])
    cats = sorted(set(y))
    logging.info(f"sub-{subject}: {len(y)} trials, second-stim categories = {cats}")

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
        if roi_func.ndim == 4 and roi_func.shape[3] == 1:
            roi_func = index_img(roi_func, 0)
        roi_mask = math_img('(v > 0) & (b > 0)', v=roi_func, b=brain_mask_img)
        masks.append((name, roi_mask))

    # --- Decode for each mask ---
    logo    = LeaveOneGroupOut()
    results = []

    for mask_name, mask_img in masks:
        masker = NiftiMasker(mask_img=mask_img, standardize=True).fit()
        X_full   = masker.transform(betas_img)
        n_voxels = X_full.shape[1]
        logging.info(f"  {mask_name}: {n_voxels:,} voxels")

        # Variant 2: run x stim-1-category demeaned (per-voxel cell-mean subtracted)
        X_s1cat = X_full.copy()
        for cell in np.unique(cell_full):
            m = cell_full == cell
            X_s1cat[m] -= X_s1cat[m].mean(axis=0, keepdims=True)

        result = {'mask': mask_name, 'n_voxels': n_voxels, 'n_trials': len(y),
                  'chance': 1.0 / len(cats)}

        for variant, X_variant in [('raw', X_full), ('s1cat_demeaned', X_s1cat)]:
            y_pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'), X_variant, y,
                                       cv=logo, groups=groups)
            acc = float((y_pred == y).mean())
            key = 'accuracy' if variant == 'raw' else f'accuracy_{variant}'
            result[key] = acc
            cm = confusion_matrix(y, y_pred, labels=cats)
            np.save(subject_output / f"sub-{subject}_stim2_decoding_confusion_{mask_name}_{variant}.npy", cm)

        logging.info(f"  {mask_name}: accuracy(raw) = {result['accuracy']:.3f}  "
                     f"accuracy(s1cat-demeaned) = {result['accuracy_s1cat_demeaned']:.3f}  "
                     f"(chance = {result['chance']:.3f})")
        results.append(result)

    np.save(subject_output / f"sub-{subject}_stim2_decoding_labels.npy", np.array(cats))
    pd.DataFrame(results).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done")


def main():
    parser = argparse.ArgumentParser(
        description="Run second-stimulus category decoding for one subject.")
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
            logging.FileHandler(subject_output / f"stim2_decoding_sub-{args.subject}.log"),
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
