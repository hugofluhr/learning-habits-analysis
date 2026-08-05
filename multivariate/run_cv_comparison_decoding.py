#!/usr/bin/env python3
"""
CV-scheme comparison for category decoding — one subject.

Empirical check of a design choice: `run_decoding.py` / `run_beta_qc_decoding.py`
both cross-validate the category-decoding validation probe with
`LeaveOneGroupOut` over the 3 runs/sessions (`learning1`, `learning2`, `test`) —
a *between-run* CV scheme chosen to avoid temporal-autocorrelation leakage
within a run. Nobody has checked whether that actually matters here.

This script runs the *same* standardized decoder (type-D betas, LinearSVC,
standardize=True) on the *same* subject/mask/labels, but under two CV schemes:

- ``between_run_logo``  — LeaveOneGroupOut over runs (3 folds). The production
  scheme, reproduced here unchanged.
- ``within_run_kfold``  — StratifiedKFold(n_splits=3, shuffle=True), ignoring
  run boundaries entirely: trials are pooled across all 3 runs and split by
  category-stratified random assignment. Fold count matches LOGO (3) so
  training-set size is comparable between schemes.

If temporal autocorrelation within a run were inflating decodability, the
within-run scheme should show systematically higher accuracy than the
between-run scheme (same subject, same features, same labels — only the fold
boundaries differ). If the two are statistically indistinguishable, the
leakage concern is more theoretical than empirical for this probe.

Scope: category decoding only (the validation probe). Does not touch
reward/value decoding.

Reuses `run_beta_qc_decoding.py`'s self-contained mask construction (glob for
the fMRIPrep brain mask, resample the visual-cortex atlas mask to it — no
`utils.data.Subject` / behavioral-data dependency) and `run_decoding.py`'s
direct load of the type-D `_CUES.nii.gz` (GLMsingle's best/production beta
version — this is not a beta-version comparison, so only type D is used).

Usage
-----
python multivariate/run_cv_comparison_decoding.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --output-dir .../derivatives/cv_comparison \\
    --visual-cortex-mask .../derivatives/decoding/visual_cortex_mask.nii.gz

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_cv_comparison_decoding.csv   — tidy: subject, mask, scheme, target, metric, value, baseline
    cv_comparison_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import math_img, resample_to_img
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC

# Same fold count as the production LeaveOneGroupOut (3 runs) so training-set
# size is comparable across schemes — the only thing that should differ is
# whether fold boundaries respect run identity.
N_FOLDS = 3
RANDOM_STATE = 42


def run_subject(subject, bids_dir, glmsingle_dir, output_dir, visual_cortex_mask_path,
                 overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    out_csv = subject_output / f"sub-{subject}_cv_comparison_decoding.csv"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        logging.info(f"sub-{subject}: output exists, skipping (pass --overwrite to rerun)")
        return

    glm_dir    = glmsingle_dir / f"sub-{subject}"
    info_path  = glm_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv"
    betas_path = glm_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"

    betas_img  = nib.load(betas_path)
    trial_info = pd.read_csv(info_path, index_col='trial_id')
    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    y      = trial_info['stim_cat'].values
    groups = trial_info['run'].values

    # --- Brain mask (fMRIPrep functional brain mask, first run, MNI space) ---
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

    # --- Visual cortex mask: resample atlas mask to subject functional space ---
    vis_mask_mni  = nib.load(str(visual_cortex_mask_path))
    vis_mask_func = resample_to_img(vis_mask_mni, brain_mask_img, interpolation='nearest')
    vis_mask_img  = math_img('(v > 0) & (b > 0)', v=vis_mask_func, b=brain_mask_img)

    schemes = {
        'between_run_logo': LeaveOneGroupOut(),
        'within_run_kfold': StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                             random_state=RANDOM_STATE),
    }

    rows = []
    for mask_name, mask_img in [('wholebrain', brain_mask_img), ('visualcortex', vis_mask_img)]:
        masker = NiftiMasker(mask_img=mask_img, standardize=True)  # neutralize ridge scaling
        X = masker.fit_transform(betas_img)
        logging.info(f"sub-{subject}: {mask_name} -> X {X.shape}")

        for scheme_name, cv in schemes.items():
            cv_kwargs = {'groups': groups} if scheme_name == 'between_run_logo' else {}
            pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'),
                                     X, y, cv=cv, **cv_kwargs)
            acc = float((pred == y).mean())
            rows.append({'subject': f'sub-{subject}', 'mask': mask_name, 'scheme': scheme_name,
                         'target': 'category', 'metric': 'accuracy',
                         'value': acc, 'baseline': 0.25})
            logging.info(f"sub-{subject}: {mask_name} {scheme_name} category accuracy = {acc:.3f}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare between-run LOGO vs within-run k-fold CV for category decoding, one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
    parser.add_argument("--output-dir",    default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/cv_comparison")
    parser.add_argument("--visual-cortex-mask", required=True,
                        help="Path to pre-built visual cortex mask NIfTI "
                             "(from build_visual_cortex_mask.py)")
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
            logging.FileHandler(subject_output / f"cv_comparison_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject                 = args.subject,
        bids_dir                = Path(args.bids_dir),
        glmsingle_dir           = Path(args.glmsingle_dir),
        output_dir              = output_dir,
        visual_cortex_mask_path = Path(args.visual_cortex_mask),
        overwrite               = args.overwrite,
    )


if __name__ == "__main__":
    main()
