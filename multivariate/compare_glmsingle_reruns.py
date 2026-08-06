#!/usr/bin/env python3
"""
Compare two GLMsingle output directories (e.g. old vs. a refactored rerun) and
range-check the new QC-diagnostic maps.

This is a validation tool, not a pipeline step: it exists specifically to turn
a full end-to-end GLMsingle rerun into a real regression/determinism check on
`run_glmsingle.py`, rather than a tautology — see multivariate/README.md and
the plan that introduced multivariate/run_glmsingle.py's save_qc_maps().

Usage
-----
python multivariate/compare_glmsingle_reruns.py \
    --old-dir /path/to/derivatives/glmsingle \
    --new-dir /path/to/derivatives/glmsingle_extended \
    --base-dir /home/hfluhr/data/learninghabits \
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \
    --output-csv /path/to/derivatives/glmsingle_extended/compare_old_vs_new.csv

Subject list defaults to participants_mvpa.tsv under --base-dir (via
utils.data.load_participant_list), same convention as run_glmsingle.py.

Output CSV columns
-------------------
Betas comparison (old vs new sub-<id>_glmSingle_betas_CUES.nii.gz):
    shape_match, max_abs_diff, mean_abs_diff, pearson_r, pct_allclose,
    n_nan_old, n_nan_new

QC-map checks:
    frac_in_range, frac_min, frac_max                    (new dir; GLMsingle's
        FRACvalue is expected to sit in [0.05, 1.0] regardless of old/new)
    hrfindex_in_range, hrfindex_min, hrfindex_max          (new dir; sanity range)
    n_nan_r2_old, n_nan_r2_new, r2_nan_count_match         (old vs new — R2 having
        NaN at a handful of degenerate-variance mask-edge voxels is expected and
        NOT itself a problem; what matters is whether the *count* changed between
        the old and new run, which would indicate a real regression)
    n_nan_r2run_old, n_nan_r2run_new, r2run_nan_count_match
    noisepool_prop                                         (new dir; informational
        only — this dataset runs ~50-70%, no principled a-priori "normal" range)
    n_noisepool_old, n_noisepool_new, noisepool_count_match (old vs new)

flag: derived triage column — True means "look at this row", not "this failed".
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_participant_list

N_HRFS = 20          # size of GLMsingle's canonical HRF library
FRAC_MIN = 0.05      # GLMsingle's minimum FRACvalue (max regularization)
ALLCLOSE_RTOL = 1e-3
ALLCLOSE_ATOL = 1e-4


def load_nii(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32) if Path(path).exists() else None


def load_old_typed_key(old_dir, subject, key):
    """Old dir predates save_qc_maps() — no qc_*.nii.gz files exist there, so
    R2/R2run must be pulled from the raw TYPED_FITHRF_GLMDENOISE_RR.npy dict."""
    path = old_dir / f"sub-{subject}" / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True).item().get(key)


def compare_betas(old_dir, new_dir, subject):
    old_path = old_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"
    new_path = new_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"

    row = {}
    if not old_path.exists() or not new_path.exists():
        row['shape_match'] = np.nan
        return row

    old = load_nii(old_path)
    new = load_nii(new_path)

    if old.shape != new.shape:
        logging.warning(f"sub-{subject}: shape mismatch old={old.shape} new={new.shape}")
        row['shape_match'] = False
        return row
    row['shape_match'] = True

    old64, new64 = old.astype(np.float64), new.astype(np.float64)
    diff = np.abs(old64 - new64)
    finite = np.isfinite(old64) & np.isfinite(new64)

    row['max_abs_diff']  = float(np.nanmax(diff[finite])) if finite.any() else np.nan
    row['mean_abs_diff'] = float(np.nanmean(diff[finite])) if finite.any() else np.nan
    row['pearson_r']     = (float(np.corrcoef(old64[finite], new64[finite])[0, 1])
                             if finite.sum() > 1 else np.nan)
    row['pct_allclose']  = (float(np.isclose(old64[finite], new64[finite],
                                              rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL).mean())
                             if finite.any() else np.nan)
    row['n_nan_old'] = int(np.isnan(old64).sum())
    row['n_nan_new'] = int(np.isnan(new64).sum())
    return row


def check_qc_maps(old_dir, new_dir, subject, mask_arr):
    sub_dir = new_dir / f"sub-{subject}"
    row = {}

    frac = load_nii(sub_dir / f"sub-{subject}_glmSingle_qc_FRACvalue.nii.gz")
    if frac is not None:
        frac_masked = frac[mask_arr]
        row['frac_min'] = float(np.nanmin(frac_masked))
        row['frac_max'] = float(np.nanmax(frac_masked))
        row['frac_in_range'] = bool(
            (frac_masked >= FRAC_MIN - 1e-6).all() and (frac_masked <= 1.0 + 1e-6).all()
        )
    else:
        row['frac_in_range'] = np.nan

    # noisepool_prop: no principled a-priori "normal" range for this dataset was
    # ever empirically checked (the [0.10, 0.50] used earlier was carried over
    # from generic docs, not this data — actual values run ~50-73% here, which
    # just looks like this dataset's characteristic pool size). Compare against
    # the old run instead, same as R2/R2run above — a real regression check.
    noisepool_new = load_nii(sub_dir / f"sub-{subject}_glmSingle_qc_noisepool.nii.gz")
    noisepool_old = load_old_typed_key(old_dir, subject, 'noisepool')
    n_brain = int(mask_arr.sum())
    if noisepool_new is not None:
        row['noisepool_prop'] = float((noisepool_new[mask_arr] > 0).sum() / n_brain) if n_brain else np.nan
    else:
        row['noisepool_prop'] = np.nan
    if noisepool_new is not None and noisepool_old is not None and n_brain:
        n_old = int((noisepool_old[mask_arr] > 0).sum())
        n_new = int((noisepool_new[mask_arr] > 0).sum())
        row['n_noisepool_old'] = n_old
        row['n_noisepool_new'] = n_new
        row['noisepool_count_match'] = n_old == n_new
    else:
        row['noisepool_count_match'] = np.nan

    # R2/R2run: a handful of NaN at degenerate-variance mask-edge voxels is a
    # normal, pre-existing GLMsingle characteristic (confirmed against sub-01:
    # the old run already had the same NaN voxels). Presence alone is not a
    # regression signal — compare the *count* between old and new instead.
    r2_new = load_nii(sub_dir / f"sub-{subject}_glmSingle_qc_R2.nii.gz")
    r2_old = load_old_typed_key(old_dir, subject, 'R2')
    if r2_new is not None and r2_old is not None:
        row['n_nan_r2_old'] = int(np.isnan(r2_old[mask_arr]).sum())
        row['n_nan_r2_new'] = int(np.isnan(r2_new[mask_arr]).sum())
        row['r2_nan_count_match'] = row['n_nan_r2_old'] == row['n_nan_r2_new']
    else:
        row['r2_nan_count_match'] = np.nan

    r2run_new = load_nii(sub_dir / f"sub-{subject}_glmSingle_qc_R2run.nii.gz")
    r2run_old = load_old_typed_key(old_dir, subject, 'R2run')
    if r2run_new is not None and r2run_old is not None:
        row['n_nan_r2run_old'] = int(np.isnan(r2run_old[mask_arr]).sum())
        row['n_nan_r2run_new'] = int(np.isnan(r2run_new[mask_arr]).sum())
        row['r2run_nan_count_match'] = row['n_nan_r2run_old'] == row['n_nan_r2run_new']
    else:
        row['r2run_nan_count_match'] = np.nan

    hrf = load_nii(sub_dir / f"sub-{subject}_glmSingle_qc_HRFindex.nii.gz")
    if hrf is not None:
        hrf_masked = hrf[mask_arr]
        row['hrfindex_min'] = float(np.nanmin(hrf_masked))
        row['hrfindex_max'] = float(np.nanmax(hrf_masked))
        row['hrfindex_in_range'] = bool(
            (hrf_masked >= 0).all() and (hrf_masked <= N_HRFS - 1).all()
        )
    else:
        row['hrfindex_in_range'] = np.nan

    return row


def flag_row(row):
    checks = [
        row.get('shape_match') is False,
        isinstance(row.get('pearson_r'), float) and row['pearson_r'] < 0.999,
        isinstance(row.get('pct_allclose'), float) and row['pct_allclose'] < 0.99,
        row.get('n_nan_old', 0) != row.get('n_nan_new', 0),
        row.get('frac_in_range') is False,
        row.get('hrfindex_in_range') is False,
        row.get('r2_nan_count_match') is False,
        row.get('r2run_nan_count_match') is False,
        row.get('noisepool_count_match') is False,
    ]
    return any(checks)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two GLMsingle output directories and range-check new QC maps."
    )
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--base-dir", required=True,
                        help="Root data dir containing participants_mvpa.tsv")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory (for brain mask lookup)")
    parser.add_argument("--participants-file", default="participants_mvpa.tsv")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    subjects = load_participant_list(args.base_dir, file_name=args.participants_file)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    logging.info(f"Comparing {len(subjects)} subjects: {old_dir} vs {new_dir}")

    rows = []
    for subject in subjects:
        row = {'subject': subject}
        row.update(compare_betas(old_dir, new_dir, subject))

        try:
            sub = Subject(base_dir=args.base_dir, subject_id=subject,
                          include_imaging=True, bids_dir=args.bids_dir)
            mask_path = sub.get_brain_mask('learning1')
            mask_arr = nib.load(mask_path).get_fdata() > 0
            row.update(check_qc_maps(old_dir, new_dir, subject, mask_arr))
        except FileNotFoundError as e:
            logging.warning(f"sub-{subject}: brain mask unavailable — {e}")

        row['flag'] = flag_row(row)
        if row['flag']:
            logging.warning(f"sub-{subject}: FLAGGED — {row}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved comparison CSV: {args.output_csv}")
    logging.info(f"{df['flag'].sum()}/{len(df)} subjects flagged for review")
    print(df.describe(include='all').round(4).to_string())


if __name__ == "__main__":
    main()
