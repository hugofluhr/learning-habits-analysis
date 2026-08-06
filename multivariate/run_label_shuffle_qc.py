#!/usr/bin/env python3
"""
Label-shuffle robustness check for category decoding — one subject.

Negative-control / permutation-test sanity check on the *production* decoding
pipeline (run_decoding.py): decode stimulus category from type-D GLMsingle
betas with the true labels once, then repeat with `stim_cat` permuted at
random, `--n-permutations` times, keeping everything else identical (same
betas, same masks, same LeaveOneGroupOut-over-runs CV). If real decoding
reflects genuine category information, shuffled accuracy should collapse to
chance (0.25); if it doesn't, that's a leakage red flag (standardization
fit before the CV split, temporal autocorrelation between adjacent trials,
motion/physio confounds correlated with condition order, etc.).

Labels are shuffled *globally* across all trials (not within-run) before
being handed to the same grouped CV — this matches the convention used by
sklearn.model_selection.permutation_test_score and fully destroys the true
label-feature mapping while keeping the CV scheme identical to production.

This does not modify or re-run GLMsingle or run_decoding.py; it reads the
same type-D betas + info CSV and reconstructs the same masks (brain mask
glob + visual-cortex ROI resample), same pattern as run_beta_qc_decoding.py.

One deliberate deviation from run_decoding.py: standardize=True (not False).
Unstandardized whole-brain betas (~65k voxels) make LinearSVC/liblinear
converge very slowly -- a one-off cost run_decoding.py absorbs for its single
fit, but fatal here where each mask needs `n_permutations + 1` refits on the
same X. True and shuffled permutations share the identical standardized X, so
the true-vs-null comparison stays apples-to-apples; only the raw true-accuracy
value may differ slightly from run_decoding.py's unstandardized number.

Usage
-----
python multivariate/run_label_shuffle_qc.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --output-dir .../derivatives/glmsingle_qc \\
    --visual-cortex-mask .../derivatives/decoding/visual_cortex_mask.nii.gz \\
    --n-permutations 100

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_label_shuffle_qc.csv   — tidy: subject, mask, target, permutation
                                       (0=true, 1..N=shuffled), accuracy, baseline
    label_shuffle_qc_sub-<id>.log
"""

import argparse
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import math_img, resample_to_img
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.svm import LinearSVC


def run_subject(subject, bids_dir, glmsingle_dir, output_dir, visual_cortex_mask_path,
                 n_permutations=100, seed=0, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    out_csv = subject_output / f"sub-{subject}_label_shuffle_qc.csv"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        logging.info(f"sub-{subject}: label-shuffle QC output exists, skipping (pass --overwrite to rerun)")
        return

    glm_dir    = glmsingle_dir / f"sub-{subject}"
    info_path  = glm_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv"
    betas_path = glm_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"

    trial_info = pd.read_csv(info_path, index_col='trial_id')
    y_true     = trial_info['stim_cat'].values
    groups     = trial_info['run'].values
    betas_img  = nib.load(betas_path)
    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

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

    # --- Visual cortex mask: resample atlas mask to subject functional space,
    # same construction as run_decoding.py / run_beta_qc_decoding.py ---
    vis_mask_mni  = nib.load(str(visual_cortex_mask_path))
    vis_mask_func = resample_to_img(vis_mask_mni, brain_mask_img, interpolation='nearest')
    vis_mask_img  = math_img('(v > 0) & (b > 0)', v=vis_mask_func, b=brain_mask_img)

    logo = LeaveOneGroupOut()
    rows = []

    for mask_idx, (mask_name, mask_img) in enumerate(
        [('wholebrain', brain_mask_img), ('visualcortex', vis_mask_img)]
    ):
        # standardize=True (unlike run_decoding.py's standardize=False): with
        # unstandardized whole-brain betas (65k+ voxels), LinearSVC/liblinear
        # converges very slowly (empirically: still not converged after 20+ min
        # on a single fit, see git history for the smoke-test finding) -- fine
        # for run_decoding.py's one-off fit, not for the 101 refits/mask this
        # script needs. Both true and shuffled permutations use the identical
        # standardized X, so the true-vs-null comparison stays apples-to-apples;
        # only the raw true-accuracy value may differ slightly from
        # run_decoding.py's unstandardized number.
        masker = NiftiMasker(mask_img=mask_img, standardize=True).fit()
        X = masker.transform(betas_img)  # transform once, reused across all permutations
        logging.info(f"sub-{subject}: {mask_name} -> X {X.shape}")

        # permutation 0 = true labels (this reproduces run_decoding.py's accuracy).
        # Seed via SeedSequence over plain ints only — Python's built-in hash()
        # is salted per-process for str objects (PYTHONHASHSEED), so it would
        # silently break --seed reproducibility across reruns.
        mask_rows = []
        for perm in range(n_permutations + 1):
            if perm == 0:
                y = y_true
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, zlib.crc32(subject.encode()), mask_idx, perm])
                )
                y = rng.permutation(y_true)

            pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'),
                                     X, y, cv=logo, groups=groups)
            acc = float((pred == y).mean())
            mask_rows.append({'subject': f'sub-{subject}', 'mask': mask_name, 'target': 'category',
                              'permutation': perm, 'shuffled': perm != 0,
                              'metric': 'accuracy', 'value': acc, 'baseline': 0.25})

            if perm == 0:
                logging.info(f"sub-{subject}: {mask_name} true accuracy = {acc:.3f}")
            elif perm % 20 == 0:
                logging.info(f"sub-{subject}: {mask_name} permutation {perm}/{n_permutations}")

        true_acc = mask_rows[0]['value']
        shuffled_acc = [r['value'] for r in mask_rows[1:]]
        n_geq = sum(a >= true_acc for a in shuffled_acc)
        pval = (n_geq + 1) / (n_permutations + 1)
        logging.info(f"sub-{subject}: {mask_name} empirical p-value = {pval:.4f} "
                     f"({n_geq}/{n_permutations} shuffles >= true accuracy)")
        rows.extend(mask_rows)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Label-shuffle robustness check for category decoding, one subject."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
    parser.add_argument("--output-dir",    default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle_qc")
    parser.add_argument("--visual-cortex-mask", required=True,
                        help="Path to pre-built visual cortex mask NIfTI "
                             "(from build_visual_cortex_mask.py)")
    parser.add_argument("--n-permutations", type=int, default=100,
                        help="Number of label-shuffle iterations per mask (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for label shuffling")
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
            logging.FileHandler(subject_output / f"label_shuffle_qc_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject                 = args.subject,
        bids_dir                = Path(args.bids_dir),
        glmsingle_dir           = Path(args.glmsingle_dir),
        output_dir              = output_dir,
        visual_cortex_mask_path = Path(args.visual_cortex_mask),
        n_permutations          = args.n_permutations,
        seed                    = args.seed,
        overwrite               = args.overwrite,
    )


if __name__ == "__main__":
    main()
