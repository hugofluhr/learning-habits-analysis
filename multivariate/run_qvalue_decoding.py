#!/usr/bin/env python3
"""
Reward-value (Q-value) whole-brain + ROI decoding — one subject.

Regression counterpart of run_decoding.py: instead of classifying stimulus category,
regresses a continuous per-trial target (default first_stim_value, the objective
reward level of the first stimulus, integers 1-5) sourced from the Big Behavior Table
(the same table used to fit the SPM first-levels). This is the "plain decoding"
baseline alongside run_qvalue_searchlight.py (local spatial map) and
run_qvalue_frem.py (whole-brain sparse ensemble + weight map): a single per-mask
regression score, no spatial map — the cheapest, most direct "can reward level be
read out of these patterns at all" check, worth running before spending compute on
the spatial analyses.

Masks: whole-brain (subject's own functional brain mask, always included) plus any
number of ROI masks passed via repeated --roi-mask NAME PATH, e.g.:
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz
    --roi-mask vmpfc        .../vmpfc_bartra2013_MNI152NLin2009cAsym.nii
    --roi-mask striatum     .../striatum_bartra2013_MNI152NLin2009cAsym.nii
Each ROI mask is resampled (nearest-neighbor) to the subject's functional space and
ANDed with the brain mask, mirroring run_decoding.py's visual-cortex handling.

Estimator: RidgeCV (log-spaced alpha grid, refit via sklearn's efficient built-in
generalized-CV alpha selection within each training fold) — a single fixed alpha
doesn't suit both a ~65k-voxel wholebrain mask and a ~120-voxel ROI equally well;
RidgeCV adapts per mask automatically. LeaveOneGroupOut CV over runs, standardize=True
features (matches run_decoding.py's rationale: widely varying voxel signal scale
otherwise biases the penalty toward high-magnitude voxels).

Run-level correction (both leakage-safe — neither uses y from a run to transform that
same run's own X, nor lets the model see a held-out run's true mean at fit time):
  - **Features** are additionally demeaned per run (each voxel's own per-run mean
    subtracted, on top of the existing global standardize=True) — removes session-level
    drift/baseline shift as a nuisance confound before fitting. Uses only `run` group
    membership, never y, so it's safe to apply before the CV split.
  - **Scoring**: reward level is ~flat across runs by design (levels {1,2,2,3,3,4,4,5}
    average to exactly 3.0 per run), so with only 3 LeaveOneGroupOut folds a heavily-
    regularized model's predictions get anchored near the *training*-fold mean, which is
    mechanically anti-correlated with the *held-out* run's mean (removing an
    above-average run from a fixed-ish total necessarily raises the remaining mean) —
    a CV arithmetic artifact, not signal. It's most visible for small, low-SNR ROIs
    (vmPFC/striatum) where there's little real within-run signal to outweigh it.
    r/r2 are therefore computed on predictions and targets demeaned by their own run's
    mean *after* cross_val_predict (a post-hoc scoring transform — doesn't feed anything
    back into fitting), isolating the within-run relationship this analysis actually
    cares about. Raw (non-demeaned) r/r2 are kept alongside as r_raw/r2_raw for
    transparency.

Usage
-----
python multivariate/run_qvalue_decoding.py --subject 01 \\
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
    sub-<id>_qvalue_decoding_<tag>.csv               — mask, n_voxels, r, r2, r_raw, r2_raw,
                                                        n_trials (r/r2 = run-demeaned; see above)
    sub-<id>_qvalue_decoding_predictions_<tag>.csv    — mask, run, y_true, y_pred (long, raw)
    qvalue_decoding_sub-<id>.log
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_target_from_bbt

ALPHAS = np.logspace(-3, 5, 17)


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                target_col='first_stim_value', roi_masks=None, overwrite=False):

    roi_masks = roi_masks or []
    tag = 'reward' if target_col == 'first_stim_value' else target_col

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_qvalue_decoding_{tag}.csv"

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

    # --- Continuous target from the BBT, aligned to the beta order ---
    y      = load_target_from_bbt(subject, bbt_path, trial_info, target_col=target_col)
    groups = trial_info['run'].values
    logging.info(f"sub-{subject}: target '{target_col}' -> {len(y)} values, "
                 f"range [{y.min():.3g}, {y.max():.3g}]")

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

    # --- Decode for each mask ---
    logo = LeaveOneGroupOut()
    results, pred_rows = [], []

    for mask_name, mask_img in masks:
        # standardize=True: same rationale as run_decoding.py — widely varying voxel
        # signal scale otherwise biases the Ridge penalty toward high-magnitude voxels.
        masker = NiftiMasker(mask_img=mask_img, standardize=True).fit()
        X      = masker.transform(betas_img)
        n_voxels = X.shape[1]

        # Run-demean features on top of the global standardize=True: subtracts each
        # voxel's own per-run mean, removing session-level drift as a nuisance
        # confound. Uses only `run` membership (never y) so it's leakage-safe to do
        # once before the CV split.
        X = X.copy()
        for r_id in np.unique(groups):
            m = groups == r_id
            X[m] -= X[m].mean(axis=0, keepdims=True)

        logging.info(f"  {mask_name}: {n_voxels:,} voxels")

        # RidgeCV: a single fixed alpha doesn't suit both a ~65k-voxel wholebrain mask
        # and a ~120-voxel ROI equally well — let each mask pick its own regularization
        # strength via sklearn's efficient built-in generalized-CV.
        estimator = RidgeCV(alphas=ALPHAS)
        y_pred = cross_val_predict(estimator, X, y, cv=logo, groups=groups)

        r_raw  = float(np.corrcoef(y, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
        r2_raw = float(r2_score(y, y_pred))

        # Run-demeaned scoring (post-hoc, doesn't touch fitting): reward level is
        # ~flat across runs by design, so with only 3 LeaveOneGroupOut folds a
        # heavily-regularized model's between-run prediction spread is dominated by a
        # CV arithmetic artifact (see module docstring), not signal — most visible for
        # small ROIs with little real within-run signal to outweigh it. Demeaning both
        # y and y_pred by their own run's mean isolates the within-run relationship.
        y_s      = pd.Series(y)
        y_pred_s = pd.Series(y_pred)
        run_s    = pd.Series(groups)
        y_dm      = (y_s - y_s.groupby(run_s).transform('mean')).values
        y_pred_dm = (y_pred_s - y_pred_s.groupby(run_s).transform('mean')).values

        r  = float(np.corrcoef(y_dm, y_pred_dm)[0, 1]) if np.std(y_pred_dm) > 0 else 0.0
        r2 = float(r2_score(y_dm, y_pred_dm))
        logging.info(f"  {mask_name}: r = {r:.3f}, r2 = {r2:.3f}  "
                     f"(raw: r = {r_raw:.3f}, r2 = {r2_raw:.3f})  (null r = 0)")

        results.append({'mask': mask_name, 'n_voxels': n_voxels, 'r': r, 'r2': r2,
                         'r_raw': r_raw, 'r2_raw': r2_raw, 'n_trials': len(y)})
        for run_id, yt, yp in zip(groups, y, y_pred):
            pred_rows.append({'mask': mask_name, 'run': run_id,
                               'y_true': float(yt), 'y_pred': float(yp)})

    pd.DataFrame(results).to_csv(done_flag, index=False)
    pd.DataFrame(pred_rows).to_csv(
        subject_output / f"sub-{subject}_qvalue_decoding_predictions_{tag}.csv", index=False
    )
    logging.info(f"sub-{subject}: done")


def main():
    parser = argparse.ArgumentParser(description="Run reward-value whole-brain + ROI decoding for one subject.")
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
                        help="BBT column to decode (default: first_stim_value, objective reward)")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--roi-mask", action="append", nargs=2, metavar=("NAME", "PATH"),
                        default=[],
                        help="ROI mask to decode, given as NAME PATH; repeatable. "
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
            logging.FileHandler(subject_output / f"qvalue_decoding_sub-{args.subject}.log"),
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
        roi_masks     = args.roi_mask,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
