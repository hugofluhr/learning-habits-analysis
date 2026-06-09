#!/usr/bin/env python3
"""
Stimulus category decoding — one subject.

Loads GLMsingle type-D betas, applies whole-brain and visual cortex masks,
runs LinearSVC with leave-one-run-out CV, saves accuracy and confusion matrices.

Usage
-----
python multivariate/run_decoding.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding \\
    --visual-cortex-mask /home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_decoding_accuracy.csv          — mask, accuracy
    sub-<id>_decoding_confusion_wholebrain.npy
    sub-<id>_decoding_confusion_visualcortex.npy
    sub-<id>_decoding_labels.npy            — category order for confusion matrices
    decoding_sub-<id>.log
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
from utils.data import Subject


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir,
                visual_cortex_mask_path, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_decoding_accuracy.csv"

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

    y      = trial_info['stim_cat'].values
    groups = trial_info['run'].values

    # --- Brain mask (3D, used as spatial reference) ---
    sub = Subject(
        base_dir=str(base_dir),
        subject_id=subject,
        include_imaging=True,
        bids_dir=str(bids_dir),
    )
    brain_mask_img = nib.load(sub.brain_mask['learning1'])

    # --- Visual cortex mask: resample atlas mask to subject functional space ---
    vis_mask_mni  = nib.load(str(visual_cortex_mask_path))
    vis_mask_func = resample_to_img(vis_mask_mni, brain_mask_img, interpolation='nearest')
    vis_mask      = math_img('(v > 0) & (b > 0)', v=vis_mask_func, b=brain_mask_img)

    # --- Decode for each mask ---
    logo    = LeaveOneGroupOut()
    results = []
    cats    = sorted(set(y))

    for mask_name, mask_img in [('wholebrain', brain_mask_img), ('visualcortex', vis_mask)]:
        masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()
        X      = masker.transform(betas_img)
        logging.info(f"  {mask_name}: {X.shape[1]:,} voxels")

        y_pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'), X, y, cv=logo, groups=groups)
        acc    = float((y_pred == y).mean())
        logging.info(f"  {mask_name}: accuracy = {acc:.3f}  (chance = 0.25)")

        results.append({'mask': mask_name, 'accuracy': acc})
        cm = confusion_matrix(y, y_pred, labels=cats)
        np.save(subject_output / f"sub-{subject}_decoding_confusion_{mask_name}.npy", cm)

    np.save(subject_output / f"sub-{subject}_decoding_labels.npy", np.array(cats))
    pd.DataFrame(results).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done")


def main():
    parser = argparse.ArgumentParser(description="Run stimulus category decoding for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir",
                        default="/home/ubuntu/data/learning-habits")
    parser.add_argument("--bids-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/glmsingle")
    parser.add_argument("--output-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/decoding")
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
            logging.FileHandler(subject_output / f"decoding_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject              = args.subject,
        base_dir             = Path(args.base_dir),
        bids_dir             = Path(args.bids_dir),
        glmsingle_dir        = Path(args.glmsingle_dir),
        output_dir           = output_dir,
        visual_cortex_mask_path = Path(args.visual_cortex_mask),
        overwrite            = args.overwrite,
    )


if __name__ == "__main__":
    main()
