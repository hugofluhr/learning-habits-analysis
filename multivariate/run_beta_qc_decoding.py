#!/usr/bin/env python3
"""
GLMsingle beta-version QC via decoding — one subject.

Runs the *same* standardized whole-brain decoder on each GLMsingle model type
(A: ONOFF, B: FITHRF, C: +GLMDENOISE, D: +ridge) and reports one scalar per
version, so you can see whether the successive denoising/ridge steps actually
improve decodability on *this* dataset (the validation GLMsingle's paper runs on
its benchmark data — Prince et al. 2022).

Two targets, both decoded from every beta version with identical mask / CV /
standardization so differences reflect only the betas:
  - category : stimulus category (LinearSVC, accuracy, chance 0.25). NOT a result
               of interest — a high-SNR probe that validates the decoding setup and
               is sensitive enough to rank beta versions.
  - reward   : objective reward level of the first stimulus (Ridge, Pearson r,
               baseline 0), sourced from the BBT. The quantity we actually care
               about; included so the ranking is judged on a relevant signal too.

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
    --bbt /mnt/data/learning-habits/bbt.csv \\
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
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import load_target_from_bbt

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
    on the reference grid. Returns None if that type wasn't written."""
    npy = subject_dir / BETA_FILES[beta_type]
    if not npy.exists():
        return None
    betasmd = np.load(npy, allow_pickle=True).item()['betasmd']
    return new_img_like(ref_img, betasmd[..., cue_mask])


def run_subject(subject, bids_dir, glmsingle_dir, bbt_path, output_dir,
                target_col='first_stim_value', targets=('category', 'reward'),
                overwrite=False):

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
    groups     = trial_info['run'].values
    ref_img    = nib.load(typed_nii)   # reference grid/affine for A/B/C rewrap
    n_cues     = ref_img.shape[-1]
    logging.info(f"sub-{subject}: {n_cues} cue betas, {len(trial_info)} info rows")

    stimorder = np.array(np.load(designinfo, allow_pickle=True).item()['stimorder'])
    cue_mask  = stimorder < N_CUE_CONDITIONS
    assert cue_mask.sum() == n_cues, (
        f"sub-{subject}: cue_mask ({cue_mask.sum()}) != type-D cue volumes ({n_cues})"
    )

    # --- Targets ---
    y = {}
    if 'category' in targets:
        y['category'] = trial_info['stim_cat'].values
    if 'reward' in targets:
        y['reward'] = load_target_from_bbt(subject, bbt_path, trial_info, target_col=target_col)

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

        if 'category' in y:
            pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'),
                                     X, y['category'], cv=logo, groups=groups)
            acc = float((pred == y['category']).mean())
            rows.append({'subject': f'sub-{subject}', 'beta_type': btype,
                         'target': 'category', 'metric': 'accuracy',
                         'value': acc, 'baseline': 0.25})
            logging.info(f"sub-{subject}: type-{btype} category accuracy = {acc:.3f}")

        if 'reward' in y:
            pred = cross_val_predict(Ridge(), X, y['reward'], cv=logo, groups=groups)
            r = float(np.corrcoef(y['reward'], pred)[0, 1]) if np.std(pred) else 0.0
            rows.append({'subject': f'sub-{subject}', 'beta_type': btype,
                         'target': 'reward', 'metric': 'pearson_r',
                         'value': r, 'baseline': 0.0})
            logging.info(f"sub-{subject}: type-{btype} reward pearson r = {r:.3f}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"sub-{subject}: saved {out_csv.name}")


def main():
    parser = argparse.ArgumentParser(description="GLMsingle beta-version QC via whole-brain decoding, one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--bids-dir",      default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir", default="/home/ubuntu/data/learning-habits/bids_dataset"
                                                    "/derivatives/glmsingle")
    parser.add_argument("--bbt", required=True,
                        help="BBT CSV holding the reward target column (same table as the SPM first-levels)")
    parser.add_argument("--target-col", default="first_stim_value",
                        help="BBT column for the 'reward' target (default: first_stim_value)")
    parser.add_argument("--targets", default="category,reward",
                        help="Comma-separated targets to decode (default: category,reward)")
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
        bbt_path      = args.bbt,
        output_dir    = output_dir,
        target_col    = args.target_col,
        targets       = tuple(t.strip() for t in args.targets.split(',') if t.strip()),
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
