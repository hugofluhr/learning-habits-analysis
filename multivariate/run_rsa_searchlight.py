#!/usr/bin/env python3
"""
RSA searchlight — one subject.

For each voxel, compute a crossnobis RDM over a local sphere (radius 6 mm) on
the 8 stimulus conditions, then regress it on the same 5-term model used in the
ROI analysis (category + value + frequency + second_stim_value + choice_rate).
Output: one brain map per regression coefficient (beta_category, beta_value,
beta_frequency, beta_second_stim_value, beta_choice_rate).

Searchlight counterpart of run_rsa_roi.py. Uses the non-figure subset (6
stimuli) to avoid the figure-category confound. Operates on the 'pooled' scope
(cross-validates over the 3 runs, same as the main ROI result).

The crossnobis distance is unbiased under the null, so the per-voxel betas can
be tested against 0 at the group level with one-sample t-tests, exactly as in
the ROI analysis and the decoding searchlights.

Usage
-----
python multivariate/run_rsa_searchlight.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir  .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/rsa_searchlight \\
    --n-jobs 8

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_rsa_searchlight_beta_<term>.nii.gz   — per-voxel beta map (5 files)
    rsa_searchlight_sub-<id>.log
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
from nilearn.image import new_img_like
from nilearn.image.resampling import coord_transform
from nilearn.maskers import NiftiMasker
from nilearn.masking import load_mask_img

# nilearn renamed this between 0.10 and 0.13
try:
    from nilearn.maskers.nifti_spheres_masker import apply_mask_and_get_affinity
except ImportError:
    from nilearn.maskers.nifti_spheres_masker import (
        _apply_mask_and_get_affinity as apply_mask_and_get_affinity,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_target_from_bbt
from multivariate.run_rsa_roi import (
    _crossnobis_loo, compute_noise_sd, _triu, _z, RUNS, FIGURE_CAT,
    abs_diff_rdm, different_rdm,
)

MODEL_TERMS = ['category', 'value', 'frequency', 'second_stim_value', 'choice_rate']


# ---------------------------------------------------------------------------
# Sphere adjacency — reuse nilearn's machinery
# ---------------------------------------------------------------------------
def _build_adjacency(brain_mask_img, radius):
    """Build the voxel adjacency matrix.

    Returns A where A.rows[i] gives the list of neighbouring voxel indices
    (in the masked space) for seed voxel i.
    """
    process_mask, process_mask_affine = load_mask_img(brain_mask_img)
    process_mask_coords = np.where(process_mask != 0)
    process_mask_coords = coord_transform(
        process_mask_coords[0],
        process_mask_coords[1],
        process_mask_coords[2],
        process_mask_affine,
    )
    process_mask_coords = np.asarray(process_mask_coords).T

    # apply_mask_and_get_affinity wants a 4D image but we only need the
    # adjacency matrix A, not X — we'll mask data ourselves.
    dummy_4d = new_img_like(brain_mask_img,
                            np.zeros(brain_mask_img.shape + (1,),
                                     dtype=np.float32))
    _, A = apply_mask_and_get_affinity(
        process_mask_coords, dummy_4d, radius, True,
        mask_img=brain_mask_img,
    )
    return A


# ---------------------------------------------------------------------------
# Per-sphere RSA regression
# ---------------------------------------------------------------------------
def _sphere_rsa(voxel_indices, X_pw, cond_idx, fold_idx, n_cond,
                model_rdms_triu, keep):
    """Compute crossnobis RDM on a sphere and regress on model RDMs.

    Parameters
    ----------
    voxel_indices : array-like
        Column indices into X_pw for this sphere.
    X_pw : ndarray (n_trials, n_all_voxels)
        Pre-whitened pattern matrix (full brain).
    cond_idx, fold_idx : arrays
        Condition and fold labels per trial.
    n_cond : int
    model_rdms_triu : dict {name: 1D array}
        z-scored upper-triangular model RDMs, precomputed on the kept subset.
    keep : bool array
        Which of the n_cond conditions to include (non-figure mask).

    Returns
    -------
    betas : dict {term: float}
    """
    sphere_data = X_pw[:, voxel_indices]
    if sphere_data.shape[1] < 2:
        return {t: np.nan for t in MODEL_TERMS}

    try:
        rdm = _crossnobis_loo(sphere_data, cond_idx, fold_idx, n_cond)
    except ValueError:
        return {t: np.nan for t in MODEL_TERMS}

    sub = np.ix_(keep, keep)
    y = _z(_triu(rdm[sub]))
    if y.std() == 0:
        return {t: np.nan for t in MODEL_TERMS}

    # Build design matrix from pre-computed model RDM upper triangulars
    cols = []
    active_terms = []
    for name in MODEL_TERMS:
        v = model_rdms_triu[name]
        if np.ptp(v) > 0:
            active_terms.append(name)
            cols.append(_z(v))

    betas = {t: np.nan for t in MODEL_TERMS}
    if cols:
        design = np.column_stack([np.ones_like(y)] + cols)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        for i, name in enumerate(active_terms):
            betas[name] = float(coef[i + 1])

    return betas


def _process_chunk(rows_chunk, chunk_indices, X_pw, cond_idx, fold_idx,
                   n_cond, model_rdms_triu, keep, thread_id, total, verbose):
    """Process a chunk of voxels (for joblib parallelism)."""
    n = len(rows_chunk)
    results = np.full((n, len(MODEL_TERMS)), np.nan, dtype=np.float64)
    t0 = time.time()

    for i, row in enumerate(rows_chunk):
        betas = _sphere_rsa(row, X_pw, cond_idx, fold_idx, n_cond,
                            model_rdms_triu, keep)
        for j, term in enumerate(MODEL_TERMS):
            results[i, j] = betas[term]

        if verbose > 0 and i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            pct = i / n * 100
            remaining = (n - i) / i * elapsed
            logging.info(f"  thread {thread_id}: {i}/{n} voxels "
                         f"({pct:.1f}%, ~{remaining:.0f}s remaining)")

    return chunk_indices, results


# ---------------------------------------------------------------------------
# Main per-subject routine
# ---------------------------------------------------------------------------
def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir,
                bbt_path, radius=6., n_jobs=1, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    # Check if all output files exist
    out_paths = {t: subject_output / f"sub-{subject}_rsa_searchlight_beta_{t}.nii.gz"
                 for t in MODEL_TERMS}
    if all(p.exists() for p in out_paths.values()) and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return
    subject_output.mkdir(parents=True, exist_ok=True)

    # --- Load betas and trial info ---
    sub_dir = glmsingle_dir / f"sub-{subject}"
    betas_img = nib.load(sub_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz")
    trial_info = pd.read_csv(
        sub_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv",
        index_col='trial_id')
    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    # --- Load per-trial properties from BBT ---
    per_trial = {}
    for col in ['first_stim_value', 'first_stim_frequ',
                'second_stim_value', 'first_stim', 'chosen_stim']:
        per_trial[col] = load_target_from_bbt(
            subject, bbt_path, trial_info, target_col=col)
    per_trial['first_stim_chosen'] = (
        per_trial['first_stim'] == per_trial['chosen_stim']).astype(float)

    stim_labels = trial_info['stim_name'].values
    run_labels = trial_info['run'].values
    stim_names = np.array(sorted(pd.unique(trial_info['stim_name'])))
    cond_idx = np.searchsorted(stim_names, stim_labels)
    n_cond = len(stim_names)

    # Per-stimulus properties
    def per_stim(series):
        grp = pd.Series(series, index=trial_info['stim_name'].values).groupby(level=0)
        return grp.first().reindex(stim_names).values

    cat = per_stim(trial_info['stim_cat'].values)
    value = per_stim(per_trial['first_stim_value']).astype(float)
    frequency = per_stim(per_trial['first_stim_frequ']).astype(float)

    # Non-figure subset
    non_figure = cat != FIGURE_CAT
    logging.info(f"sub-{subject}: {non_figure.sum()} non-figure stimuli")

    # --- Build model RDMs and precompute upper-triangle vectors ---
    # Scope-specific confound RDMs (pooled scope: all trials)
    obs_mask = np.ones(len(cond_idx), dtype=bool)

    s2v_per_stim = pd.Series(per_trial['second_stim_value'][obs_mask]).groupby(
        cond_idx[obs_mask]).mean().reindex(range(n_cond)).values
    cr_per_stim = pd.Series(per_trial['first_stim_chosen'][obs_mask]).groupby(
        cond_idx[obs_mask]).mean().reindex(range(n_cond)).values

    model_rdms_full = {
        'category': different_rdm(cat),
        'value': abs_diff_rdm(value),
        'frequency': abs_diff_rdm(frequency),
        'second_stim_value': abs_diff_rdm(s2v_per_stim),
        'choice_rate': abs_diff_rdm(cr_per_stim),
    }

    # Precompute upper-triangular vectors on the non-figure subset
    sub = np.ix_(non_figure, non_figure)
    model_rdms_triu = {}
    for name, mrdm in model_rdms_full.items():
        model_rdms_triu[name] = _triu(mrdm[sub])

    # --- Brain mask ---
    sub_obj = Subject(base_dir=str(base_dir), subject_id=subject,
                      include_imaging=True, bids_dir=str(bids_dir))
    brain_mask_img = nib.load(sub_obj.brain_mask['learning1'])

    # --- Mask betas and prewhiten ---
    masker = NiftiMasker(mask_img=brain_mask_img, standardize=False).fit()
    X = masker.transform(betas_img)
    n_voxels = X.shape[1]
    logging.info(f"sub-{subject}: {n_voxels:,} voxels in brain mask")

    # Noise normalisation (univariate, same as ROI analysis)
    sd = compute_noise_sd(X, cond_idx, run_labels)
    good = sd > 0
    if not good.all():
        logging.warning(f"sub-{subject}: {(~good).sum():,} voxels with zero SD "
                        "— they will produce NaN in any sphere that includes them")
    # Pre-whiten: divide by SD where possible, leave zeros in place
    X_pw = np.zeros_like(X)
    X_pw[:, good] = X[:, good] / sd[good]

    # Fold indices for pooled scope (3 runs)
    fold_idx = np.searchsorted(np.array(RUNS), run_labels)

    # --- Build sphere adjacency ---
    logging.info(f"sub-{subject}: building adjacency (radius={radius}mm)...")
    A = _build_adjacency(brain_mask_img, radius)
    n_seeds = A.shape[0]
    assert n_seeds == n_voxels, f"adjacency {n_seeds} != masked {n_voxels}"
    logging.info(f"sub-{subject}: {n_seeds:,} searchlight seeds, "
                 f"median sphere size {np.median([len(r) for r in A.rows]):.0f} voxels")

    # --- Run searchlight RSA ---
    logging.info(f"sub-{subject}: running RSA searchlight ({n_jobs} jobs)...")
    t_start = time.time()

    if n_jobs == 1:
        # Single-threaded: simple loop
        all_betas = np.full((n_seeds, len(MODEL_TERMS)), np.nan, dtype=np.float64)
        for i, row in enumerate(A.rows):
            betas = _sphere_rsa(row, X_pw, cond_idx, fold_idx, n_cond,
                                model_rdms_triu, non_figure)
            for j, term in enumerate(MODEL_TERMS):
                all_betas[i, j] = betas[term]
            if i % 2000 == 0 and i > 0:
                elapsed = time.time() - t_start
                remaining = (n_seeds - i) / i * elapsed
                logging.info(f"  {i}/{n_seeds} ({i/n_seeds*100:.1f}%, "
                             f"~{remaining:.0f}s remaining)")
    else:
        # Parallel: chunk the voxels for joblib
        from nilearn.decoding.searchlight import GroupIterator
        group_iter = GroupIterator(n_seeds, n_jobs)
        chunks = list(group_iter)

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_chunk)(
                [A.rows[i] for i in chunk],
                chunk,
                X_pw, cond_idx, fold_idx, n_cond,
                model_rdms_triu, non_figure,
                thread_id + 1, n_seeds, 1,
            )
            for thread_id, chunk in enumerate(chunks)
        )

        all_betas = np.full((n_seeds, len(MODEL_TERMS)), np.nan, dtype=np.float64)
        for indices, chunk_results in results:
            all_betas[indices] = chunk_results

    elapsed = time.time() - t_start
    logging.info(f"sub-{subject}: searchlight done in {elapsed:.0f}s "
                 f"({elapsed/60:.1f} min)")

    # --- Save beta maps ---
    for j, term in enumerate(MODEL_TERMS):
        out_img = masker.inverse_transform(all_betas[:, j:j+1].T)
        out_path = out_paths[term]
        out_img.to_filename(str(out_path))
        valid = np.isfinite(all_betas[:, j])
        logging.info(f"  saved {out_path.name} "
                     f"(valid: {valid.sum():,}/{n_seeds:,}, "
                     f"mean={np.nanmean(all_betas[:, j]):+.4f}, "
                     f"sd={np.nanstd(all_betas[:, j]):.4f})")

    logging.info(f"sub-{subject}: all done")


def main():
    parser = argparse.ArgumentParser(
        description="RSA searchlight for one subject (5-term regression per sphere).")
    parser.add_argument("--subject", required=True,
                        help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", required=True,
                        help="Root data directory (for brain mask via Subject)")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True,
                        help="GLMsingle betas directory")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--radius", type=float, default=6.,
                        help="Searchlight sphere radius in mm (default: 6)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for searchlight (default: 1)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    subject_output = output_dir / f"sub-{args.subject}"
    subject_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                subject_output / f"rsa_searchlight_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject       = args.subject,
        base_dir      = Path(args.base_dir),
        bids_dir      = Path(args.bids_dir),
        glmsingle_dir = Path(args.glmsingle_dir),
        output_dir    = output_dir,
        bbt_path      = args.bbt,
        radius        = args.radius,
        n_jobs        = args.n_jobs,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
