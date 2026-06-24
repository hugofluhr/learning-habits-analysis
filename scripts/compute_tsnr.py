#!/usr/bin/env python3
"""
Compute voxelwise tSNR from fMRIPrep MNI-space BOLD timeseries.

    tSNR = mean(signal) / std(linearly-detrended signal)

Outputs
-------
Per subject (OUTPUT_DIR/sub-XX/):
    tsnr_run-{1,2,3}.nii.gz   per-run tSNR map (zeros outside brain mask)
    tsnr_mean.nii.gz           mean across available runs

Group (OUTPUT_DIR/):
    group_mean_tsnr.nii.gz     voxelwise mean across subjects (NaN-safe)
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.signal import detrend

FMRIPREP_DIR = "/mnt/data2/learning-habits/bids_dataset/derivatives/fmriprep-24.0.1-noSDC"
PARTICIPANTS_TSV = "/home/ubuntu/data/learning-habits/participants_mvpa.tsv"
OUTPUT_DIR = "/home/ubuntu/data/learning-habits/tsnr_fmriprep"

RUNS = [
    ("learning", 1),
    ("learning", 2),
    ("test",     3),
]


def compute_tsnr(bold_path, mask_path):
    """Return a 3D tSNR NIfTI (zeros outside mask)."""
    bold_img = nib.load(bold_path)
    mask = nib.load(mask_path).get_fdata(dtype=np.float32) > 0

    data = bold_img.get_fdata(dtype=np.float32)  # (x, y, z, t)
    ts = data[mask]                               # (n_vox, t)

    mean_sig = ts.mean(axis=1)
    std_det  = detrend(ts, axis=1).std(axis=1, ddof=1)  # linear detrend

    tsnr_vals = np.where(std_det > 0, mean_sig / std_det, 0.0)

    vol = np.zeros(data.shape[:3], dtype=np.float32)
    vol[mask] = tsnr_vals

    return nib.Nifti1Image(vol, bold_img.affine, bold_img.header)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subjects = (
        pd.read_csv(PARTICIPANTS_TSV, header=None, dtype=str)[0]
        .str.strip()
        .tolist()
    )

    sub_means = []   # list of 3D arrays for group mean
    ref_img   = None

    for sub_id in subjects:
        sub      = f"sub-{sub_id}"
        func_dir = os.path.join(FMRIPREP_DIR, sub, "ses-1", "func")
        sub_out  = os.path.join(OUTPUT_DIR, sub)
        os.makedirs(sub_out, exist_ok=True)

        run_vols = []
        for task, run_no in RUNS:
            prefix    = f"{sub}_ses-1_task-{task}_run-{run_no}_space-MNI152NLin2009cAsym"
            bold_path = os.path.join(func_dir, f"{prefix}_desc-preproc_bold.nii.gz")
            mask_path = os.path.join(func_dir, f"{prefix}_desc-brain_mask.nii.gz")

            if not os.path.exists(bold_path):
                print(f"  [SKIP] {sub} run-{run_no}: BOLD not found", flush=True)
                continue
            if not os.path.exists(mask_path):
                print(f"  [SKIP] {sub} run-{run_no}: mask not found", flush=True)
                continue

            print(f"  {sub} task-{task} run-{run_no} ...", flush=True)
            tsnr_img = compute_tsnr(bold_path, mask_path)

            out_path = os.path.join(sub_out, f"tsnr_run-{run_no}.nii.gz")
            tsnr_img.to_filename(out_path)

            vol = tsnr_img.get_fdata(dtype=np.float32)
            run_vols.append(vol)
            if ref_img is None:
                ref_img = tsnr_img

        if not run_vols:
            print(f"  [WARN] {sub}: no runs processed, skipping", flush=True)
            continue

        # Per-subject mean: NaN-safe across runs (zeros = out-of-mask)
        stack = np.stack(run_vols, axis=-1)
        stack[stack == 0] = np.nan
        mean_vol = np.nanmean(stack, axis=-1)
        mean_vol = np.nan_to_num(mean_vol, nan=0.0)

        mean_img = nib.Nifti1Image(mean_vol, ref_img.affine, ref_img.header)
        mean_img.to_filename(os.path.join(sub_out, "tsnr_mean.nii.gz"))

        wb_mean = mean_vol[mean_vol > 0].mean()
        print(f"  {sub} done  (whole-brain mean tSNR = {wb_mean:.1f})", flush=True)

        sub_means.append(mean_vol)

    if not sub_means:
        print("No subjects processed. Check paths.")
        sys.exit(1)

    # Group mean: NaN-safe across subjects
    group_stack = np.stack(sub_means, axis=-1).astype(np.float32)
    group_stack[group_stack == 0] = np.nan
    group_mean = np.nanmean(group_stack, axis=-1)
    group_mean = np.nan_to_num(group_mean, nan=0.0)

    out = os.path.join(OUTPUT_DIR, "group_mean_tsnr.nii.gz")
    nib.Nifti1Image(group_mean, ref_img.affine, ref_img.header).to_filename(out)
    print(f"\nGroup mean tSNR saved → {out}")
    print(f"Whole-brain group mean tSNR = {group_mean[group_mean > 0].mean():.1f}")


if __name__ == "__main__":
    main()
