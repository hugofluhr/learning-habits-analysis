#!/usr/bin/env python3
"""
Reward-value (Q-value) high/low classification — one subject.

Simpler counterpart of run_qvalue_decoding.py: instead of regressing the continuous
reward level, split it into two extremes ("low" = levels <= --low-max, "high" = levels
>= --high-min; --low-max/--high-min default to 2/4, dropping the middle level-3 trials)
and classify, mirroring run_decoding.py's category-decoding recipe exactly (LinearSVC,
LeaveOneGroupOut, chance = 0.5 for a balanced binary split). Reward levels {1,2,2,3,3,4,4,5}
give a natural near-balanced split around the middle level.

Why start here instead of trusting the regression numbers: this sidesteps the
between-run LeaveOneGroupOut CV artifact documented in run_qvalue_decoding.py (reward
level is ~flat across runs by design, which mechanically anti-correlates a heavily-
regularized regressor's predictions with the held-out run's true mean). A degenerate/
over-regularized *classifier* doesn't have that failure mode the same way: if it just
predicts the training folds' majority class, it lands at ~chance accuracy on a balanced
held-out set regardless of that run's true label balance — no systematic directional
bias the way a collapsed regression produces a signed correlation. Chance is a clean,
model-independent 0.5, same as the category decoder's chance=0.25.

Masks: whole-brain (subject's own functional brain mask, always included) plus any
number of ROI masks passed via repeated --roi-mask NAME PATH, e.g.:
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz
    --roi-mask vmpfc        .../vmpfc_bartra2013_MNI152NLin2009cAsym.nii
    --roi-mask striatum     .../striatum_bartra2013_MNI152NLin2009cAsym.nii
Each ROI mask is resampled (nearest-neighbor) to the subject's functional space and
ANDed with the brain mask, mirroring run_decoding.py's visual-cortex handling.

Features are run-demeaned (each voxel's own per-run mean subtracted, computed from ALL
trials in that run before dropping the middle-level ones, on top of the existing global
standardize=True) — removes session-level drift as a nuisance confound, leakage-safe
since it only uses `run` membership, never the label. Carried over from
run_qvalue_decoding.py, where it measurably helped wholebrain/visual-cortex signal.

**Category confound.** Checked empirically across all 58 successfully-run subjects:
stimulus category is essentially deterministic of the low/high split for every single
subject (chi-square p in the 1e-15 to 1e-22 range) — most categories are *entirely* low
or *entirely* high for a given subject (not a tendency, a near-total split), because
reward level is tied to a fixed slot and, within a subject, whole categories tend to
land on one side of that slot's value tier. Since category decodes very well from
visual cortex/whole-brain (see `decoding_results.ipynb`), a classifier could hit high
"reward" accuracy purely by re-detecting category, with zero real value-coding signal.
Every mask/subject therefore also gets a **(run x category)-demeaned** feature variant
(subtract each run/category cell's own mean, on top of the run-only demeaning) —
strips any category-driven mean-pattern signal, leaving only within-category variance
for the classifier. Both `accuracy_run_demeaned` (the original, confound-able version)
and `accuracy_run_cat_demeaned` (the control) are reported side by side.

Usage
-----
python multivariate/run_qvalue_classification.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding \\
    --roi-mask visualcortex /home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz \\
    --roi-mask vmpfc /home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym/vmpfc_bartra2013_MNI152NLin2009cAsym.nii \\
    --roi-mask striatum /home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym/striatum_bartra2013_MNI152NLin2009cAsym.nii

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_qvalue_classification_<tag>.csv                                  — mask,
        n_voxels, accuracy_run_demeaned, accuracy_run_cat_demeaned, n_trials, n_low, n_high
    sub-<id>_qvalue_classification_confusion_<mask>_<variant>_<tag>.npy       — 2x2,
        labels=[low,high]; <variant> is run_demeaned or run_cat_demeaned
    qvalue_classification_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, math_img
from nilearn.maskers import NiftiMasker
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_target_from_bbt

LABELS = ['low', 'high']


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                target_col='first_stim_value', low_max=2.0, high_min=4.0,
                roi_masks=None, overwrite=False):

    roi_masks = roi_masks or []
    tag = 'reward' if target_col == 'first_stim_value' else target_col

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_qvalue_classification_{tag}.csv"

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

    # --- Continuous target from the BBT, then split into low/high, dropping the middle ---
    y_cont      = load_target_from_bbt(subject, bbt_path, trial_info, target_col=target_col)
    groups_full = trial_info['run'].values
    cat_full    = trial_info['stim_cat'].values
    cell_full   = np.array([f"{r}__{c}" for r, c in zip(groups_full, cat_full)])

    keep = (y_cont <= low_max) | (y_cont >= high_min)
    y      = np.where(y_cont[keep] <= low_max, 'low', 'high')
    groups = groups_full[keep]
    n_low, n_high = int((y == 'low').sum()), int((y == 'high').sum())
    logging.info(f"sub-{subject}: kept {keep.sum()}/{len(y_cont)} trials "
                 f"(low<={low_max:g}: {n_low}, high>={high_min:g}: {n_high})")

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
        roi_mask = math_img('(v > 0) & (b > 0)', v=roi_func, b=brain_mask_img)
        masks.append((name, roi_mask))

    # --- Classify for each mask ---
    logo = LeaveOneGroupOut()
    results = []

    for mask_name, mask_img in masks:
        # standardize=True: same rationale as run_decoding.py — widely varying voxel
        # signal scale otherwise biases the LinearSVC penalty toward high-magnitude voxels.
        masker = NiftiMasker(mask_img=mask_img, standardize=True).fit()
        X_full   = masker.transform(betas_img)
        n_voxels = X_full.shape[1]
        logging.info(f"  {mask_name}: {n_voxels:,} voxels")

        # Two feature variants, both computed from ALL trials (before the low/high
        # filter) on top of the global standardize=True — see module docstring.
        X_run = X_full.copy()
        for r_id in np.unique(groups_full):
            m = groups_full == r_id
            X_run[m] -= X_run[m].mean(axis=0, keepdims=True)

        X_cat = X_full.copy()
        for cell in np.unique(cell_full):
            m = cell_full == cell
            X_cat[m] -= X_cat[m].mean(axis=0, keepdims=True)

        result = {'mask': mask_name, 'n_voxels': n_voxels, 'n_trials': len(y),
                  'n_low': n_low, 'n_high': n_high}

        for variant, X_variant in [('run_demeaned', X_run), ('run_cat_demeaned', X_cat)]:
            X = X_variant[keep]
            y_pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'), X, y,
                                        cv=logo, groups=groups)
            acc = float((y_pred == y).mean())
            result[f'accuracy_{variant}'] = acc
            cm = confusion_matrix(y, y_pred, labels=LABELS)
            np.save(subject_output / f"sub-{subject}_qvalue_classification_confusion_{mask_name}_{variant}_{tag}.npy", cm)

        logging.info(f"  {mask_name}: accuracy(run-demeaned) = {result['accuracy_run_demeaned']:.3f}  "
                     f"accuracy(run+cat-demeaned) = {result['accuracy_run_cat_demeaned']:.3f}  (chance = 0.5)")
        results.append(result)

    pd.DataFrame(results).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done")


def main():
    parser = argparse.ArgumentParser(description="Run reward high/low classification for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", required=True,
                        help="Root data directory")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True,
                        help="GLMsingle betas directory")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV holding the target column "
                             "(same table used to fit the SPM first-levels)")
    parser.add_argument("--target-col", default="first_stim_value",
                        help="BBT column to split (default: first_stim_value, objective reward)")
    parser.add_argument("--low-max", type=float, default=2.0,
                        help="Trials with target <= this are labeled 'low' (default: 2)")
    parser.add_argument("--high-min", type=float, default=4.0,
                        help="Trials with target >= this are labeled 'high' (default: 4)")
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
            logging.FileHandler(subject_output / f"qvalue_classification_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        base_dir      = Path(args.base_dir),
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        output_dir    = output_dir,
        bbt_path      = args.bbt,
        target_col    = args.target_col,
        low_max       = args.low_max,
        high_min      = args.high_min,
        roi_masks     = args.roi_mask,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
