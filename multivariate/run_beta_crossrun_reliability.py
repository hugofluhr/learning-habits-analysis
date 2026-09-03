#!/usr/bin/env python3
"""
GLMsingle cue-beta cross-run pattern reliability — one subject.

Finishes `glmsingle_qc.ipynb` §8 (per-stimulus, whole-brain cross-run correlation of
mean beta patterns), which was written but never run to completion / persisted — no
`qc_group_summary.csv` exists locally or on the cluster. Extended here to also report
the visual-cortex and fusiform ROI masks that `run_rsa_roi.py` actually uses, since
that's the more diagnostic comparison for the RSA test-phase collapse question
(session-notes 2026-09-03, open thread 8): if `learning1_vs_test`/`learning2_vs_test`
per-stimulus reliability is already visibly lower than `learning1_vs_learning2` in
these ROIs, that's independent, coarse evidence pointing toward the same mechanism as
the pairing-structure analysis in `rsa_design_checks.ipynb` §8 — before committing to
the finer, partner-conditioned test.

For each mask and each pair of runs, computes the per-stimulus Pearson correlation
between that run's mean (across-trial) beta pattern for the stimulus and the other
run's, then reports both the per-stimulus values and their mean.

Usage
-----
python multivariate/run_beta_crossrun_reliability.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --output-dir .../derivatives/glmsingle_qc \\
    --visual-cortex-mask .../decoding/visual_cortex_mask.nii.gz \\
    --fusiform-mask .../masks/fusiform_mask_MNI152NLin2009cAsym.nii

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_crossrun_reliability.csv   tidy: subject, mask, run_a, run_b, stim_name, r
    crossrun_reliability_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, math_img
from nilearn.maskers import NiftiMasker

RUNS = ['learning1', 'learning2', 'test']


def run_subject(subject, bids_dir, glmsingle_dir, output_dir,
                 visual_cortex_mask_path, fusiform_mask_path, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    out_csv = subject_output / f"sub-{subject}_crossrun_reliability.csv"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        logging.info(f"sub-{subject}: output exists, skipping (pass --overwrite to rerun)")
        return

    glm_dir    = glmsingle_dir / f"sub-{subject}"
    info_path  = glm_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv"
    betas_path = glm_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"

    trial_info = pd.read_csv(info_path, index_col='trial_id')
    betas_img  = nib.load(betas_path)
    stim_names = sorted(trial_info['stim_name'].unique())
    logging.info(f"sub-{subject}: {betas_img.shape[-1]} cue betas, "
                 f"{len(trial_info)} info rows, {len(stim_names)} stimuli")

    # --- brain mask (fMRIPrep functional brain mask, first run, MNI space) ---
    mask_candidates = sorted(
        m for m in (bids_dir / f"sub-{subject}").rglob(
            f"sub-{subject}_*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        )
        if m.parent.name == "func"
    )
    if not mask_candidates:
        raise FileNotFoundError(f"No fMRIPrep brain mask found for sub-{subject} in {bids_dir}")
    brain_mask_img = nib.load(mask_candidates[0])

    roi_masks = [
        ('visualcortex', visual_cortex_mask_path),
        ('fusiform', fusiform_mask_path),
    ]
    masks = [('wholebrain', brain_mask_img)]
    for name, path in roi_masks:
        roi_mni  = nib.load(str(path))
        roi_func = resample_to_img(roi_mni, brain_mask_img, interpolation='nearest')
        masks.append((name, math_img('(r > 0) & (b > 0)', r=roi_func, b=brain_mask_img)))

    rows = []
    for mask_name, mask_img in masks:
        masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()
        X = masker.transform(betas_img)   # (n_trials, n_voxels)
        logging.info(f"sub-{subject}: {mask_name} {X.shape[1]:,} voxels")

        run_patterns = {}
        for run in RUNS:
            idx = np.where(trial_info['run'].values == run)[0]
            if len(idx) == 0:
                continue
            run_stim = trial_info['stim_name'].values[idx]
            run_patterns[run] = {
                s: X[idx[run_stim == s]].mean(axis=0) for s in stim_names
            }

        for run_a, run_b in combinations(RUNS, 2):
            if run_a not in run_patterns or run_b not in run_patterns:
                continue
            for s in stim_names:
                pa, pb = run_patterns[run_a][s], run_patterns[run_b][s]
                r = float(np.corrcoef(pa, pb)[0, 1])
                rows.append({'subject': f'sub-{subject}', 'mask': mask_name,
                             'run_a': run_a, 'run_b': run_b, 'stim_name': s, 'r': r})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}")
    summary = out_df.groupby(['mask', 'run_a', 'run_b'])['r'].mean()
    logging.info(f"sub-{subject}: mean per-stimulus reliability -\n{summary.to_string()}")


def main():
    parser = argparse.ArgumentParser(
        description="GLMsingle cue-beta cross-run per-stimulus pattern reliability, one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir", required=True, help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True, help="GLMsingle betas directory")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--visual-cortex-mask", required=True,
                        help="Path to pre-built visual cortex mask NIfTI")
    parser.add_argument("--fusiform-mask", required=True,
                        help="Path to pre-built fusiform mask NIfTI")
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
            logging.FileHandler(subject_output / f"crossrun_reliability_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject                 = args.subject,
        bids_dir                = Path(args.bids_dir),
        glmsingle_dir           = Path(args.glmsingle_dir),
        output_dir              = output_dir,
        visual_cortex_mask_path = Path(args.visual_cortex_mask),
        fusiform_mask_path      = Path(args.fusiform_mask),
        overwrite               = args.overwrite,
    )


if __name__ == "__main__":
    main()
