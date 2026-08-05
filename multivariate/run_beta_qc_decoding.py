#!/usr/bin/env python3
"""
GLMsingle beta-version QC via decoding — one subject.

Runs the *same* standardized whole-brain decoder on each GLMsingle model type
(A: ONOFF, B: FITHRF, C: +GLMDENOISE, D: +ridge) and reports one scalar per
version, so you can see whether the successive denoising/ridge steps actually
improve decodability on *this* dataset (the validation GLMsingle's paper runs on
its benchmark data — Prince et al. 2022).

Type A (ONOFF) is written but *skipped* in the comparison: it pools every event
into a single on/off beta per voxel (no per-trial dimension), so there is nothing
to decode. The real comparison is B -> C -> D.

Target = stimulus category (LinearSVC, accuracy, chance 0.25). Category decoding is
purely a pipeline-validation probe — a high-SNR signal that is sensitive enough to
rank the beta versions. It is NOT a result of interest; the analyses of interest
(reward/value) live in run_qvalue_*.py.

This reads TYPE{A,B,C,D}.npy directly (all saved by run_glmsingle with
wantfileoutputs=[1,1,1,1]) and reuses the existing type-D info CSV for trial
order/labels — the design is identical across types, so the ordering is too. It
does NOT modify or re-run the production GLMsingle pipeline.

Notes on fairness
-----------------
- standardize=True everywhere: type-D betas are ridge-shrunk (fracridge scales
  them toward zero), so per-feature standardization is required or the comparison
  just measures beta magnitude.
- LeaveOneGroupOut over runs: run-independent, no temporal leakage.
- A->D is usually but not guaranteed monotonic; a dip is informative (e.g.
  GLMdenoise removing condition-correlated variance), not a bug.

Usage
-----
python multivariate/run_beta_qc_decoding.py --subject 01 \\
    --bids-dir .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --output-dir .../derivatives/glmsingle_qc

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_beta_qc_decoding.csv   — tidy: subject, beta_type, target, metric, value, baseline
    beta_qc_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import new_img_like
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.svm import LinearSVC

# GLMsingle model types, in denoising order. The 4th dim of betasmd holds one
# beta per event; the first 8 conditions are the first-stimulus identities
# (cue_mask = stimorder < 8, mirroring run_glmsingle.extract_betas).
BETA_FILES = {
    'A': 'TYPEA_ONOFF.npy',
    'B': 'TYPEB_FITHRF.npy',
    'C': 'TYPEC_FITHRF_GLMDENOISE.npy',
    'D': 'TYPED_FITHRF_GLMDENOISE_RR.npy',
}
N_CUE_CONDITIONS = 8  # len(run_glmsingle.STIM_NAMES)


def _load_cue_betas(subject_dir, beta_type, cue_mask, ref_img):
    """Load one model type's betas, keep the cue volumes, wrap as a NIfTI image
    on the reference grid. Returns None if that type wasn't written, or if it
    isn't single-trial output (type A/ONOFF collapses every event into one
    pooled on-off beta per voxel — there's no per-trial signal to decode)."""
    npy = subject_dir / BETA_FILES[beta_type]
    if not npy.exists():
        return None
    betasmd = np.load(npy, allow_pickle=True).item()['betasmd']
    if betasmd.shape[-1] != cue_mask.shape[0]:
        logging.warning(
            f"type-{beta_type}: betasmd has {betasmd.shape[-1]} volume(s), expected "
            f"{cue_mask.shape[0]} (one per trial) — not single-trial output, skipping "
            "from decoding (ONOFF pools all trials into one beta by design)"
        )
        return None
    return new_img_like(ref_img, betasmd[..., cue_mask])


def run_subject(subject, bids_dir, glmsingle_dir, output_dir, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    out_csv = subject_output / f"sub-{subject}_beta_qc_decoding.csv"
    subject_output.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        logging.info(f"sub-{subject}: QC output exists, skipping (pass --overwrite to rerun)")
        return

    glm_dir    = glmsingle_dir / f"sub-{subject}"
    info_path  = glm_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv"
    typed_nii  = glm_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"
    designinfo = glm_dir / "DESIGNINFO.npy"

    trial_info = pd.read_csv(info_path, index_col='trial_id')
    y          = trial_info['stim_cat'].values
    groups     = trial_info['run'].values
    ref_img    = nib.load(typed_nii)   # reference grid/affine for A/B/C rewrap
    n_cues     = ref_img.shape[-1]
    logging.info(f"sub-{subject}: {n_cues} cue betas, {len(trial_info)} info rows")

    stimorder = np.array(np.load(designinfo, allow_pickle=True).item()['stimorder'])
    cue_mask  = stimorder < N_CUE_CONDITIONS
    assert cue_mask.sum() == n_cues, (
        f"sub-{subject}: cue_mask ({cue_mask.sum()}) != type-D cue volumes ({n_cues})"
    )

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

    logo   = LeaveOneGroupOut()
    masker = NiftiMasker(mask_img=brain_mask_img, standardize=True)  # neutralize ridge scaling
    masker.fit()

    rows = []
    for btype in BETA_FILES:
        betas_img = _load_cue_betas(glm_dir, btype, cue_mask, ref_img)
        if betas_img is None:
            logging.warning(f"sub-{subject}: type-{btype} betas missing, skipping")
            continue
        X = masker.transform(betas_img)
        logging.info(f"sub-{subject}: type-{btype} -> X {X.shape}")

        pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'),
                                 X, y, cv=logo, groups=groups)
        acc = float((pred == y).mean())
        rows.append({'subject': f'sub-{subject}', 'beta_type': btype,
                     'target': 'category', 'metric': 'accuracy',
                     'value': acc, 'baseline': 0.25})
        logging.info(f"sub-{subject}: type-{btype} category accuracy = {acc:.3f}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}")


def main():
    parser = argparse.ArgumentParser(description="GLMsingle beta-version QC via whole-brain category decoding, one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
    parser.add_argument("--output-dir",    default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle_qc")
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
            logging.FileHandler(subject_output / f"beta_qc_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        output_dir    = output_dir,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
