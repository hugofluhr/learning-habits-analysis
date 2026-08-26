#!/usr/bin/env python3
"""
ROI representational similarity analysis on GLMsingle cue betas — one subject.

Why RSA, after the reward-decoding line stalled
-----------------------------------------------
Objective reward level is a strict *coarsening* of stimulus identity: within a
(subject x first_stim_name) cell the reward value has 0.000 variance (496/496 cells,
see session-notes/2026-08-11_glmsingle-conditions-and-reward-decoding.md). A high/low
classifier therefore cannot be made to discriminate value without also discriminating
identity, and category-demeaning does not fix it (the residual contrast *is* the
individual-stimulus contrast).

RSA sidesteps the split entirely. At the 8-condition level the question becomes
"are stimuli with similar objective value represented more similarly?", and the
value model RDM makes a prediction no identity/category model can mimic: the three
pairs of stimuli that *share* a reward level (values 2, 3 and 4 each sit on two
stimuli) should be extra-similar even though they are different images in different
categories. No binarisation, no `value_diff` trial regressor — just the per-stimulus
objective value expressed as pairwise structure.

Design facts this script relies on (all verified against bbt.csv, n=62)
-----------------------------------------------------------------------
* 8 stimuli / 4 categories / 2 per category; reward levels {1,2,2,3,3,4,4,5};
  each stimulus has exactly one value and one frequency level (0 non-unique cells).
* The image -> value mapping IS counterbalanced: 12 distinct assignments across the
  62 subjects. A group-level value effect therefore cannot be an artefact of a fixed
  idiosyncratic visual similarity between two particular images — which is precisely
  what makes RSA defensible here where the classification framing was not.
* BUT values {1,5} always land on the `figure` category (62/62 subjects). "Extreme
  value" is thus perfectly confounded with figure-vs-rest at the group level. Values
  {2,3,4} rotate freely across face/hand/house (~40/62 each). Every analysis is
  therefore run twice: on all 8 stimuli, and on the 6-stimulus **non-figure subset**,
  which is the clean, counterbalanced test.
* All 3 same-value pairs are cross-category, and all 3 live in the non-figure subset.
* Frequency runs *against* the value hypothesis in the targeted contrast: all 3
  same-value pairs have the maximal |dfreq| = 2, versus a mean of 1.0 for the other
  non-figure pairs. corr(value, frequency) = -0.346 and corr(category, value) = -0.179,
  identical for every subject; corr(category, frequency) varies (-0.232..0.310).
  Frequency is carried as a covariate in the regression for exactly this reason.

What is computed
----------------
Distances are **crossnobis** (cross-validated Mahalanobis with univariate noise
normalisation): unbiased around 0 under the null, which matters here because plain
cross-run pattern reliability is only r~0.31 (glmsingle_qc.ipynb S8). Noise
normalisation divides each voxel by its residual SD, residuals taken around the
(stimulus x run) cell means (dof = n_trials - n_cells), estimated once on all trials
and reused for every scope so that per-run RDMs stay on a common scale.

The crossnobis kernel is implemented locally rather than via
`rsatoolbox.rdm.calc_rdm(..., method='crossnobis')`: that function materialises a
dense `n_channel x n_channel` identity when no noise precision is passed, which is
~20 GB at whole-brain scale. `_crossnobis_loo` below reproduces rsatoolbox 0.3.2's
`_calc_rdm_crossnobis_single` exactly (same leave-one-fold-out structure, same
pooled-training-average, same division by n_channels) without ever forming that
matrix; `--validate-against-rsatoolbox` asserts the two agree on the smallest ROI.

Note that a run-constant additive offset cancels in the crossnobis kernel (it enters
both patterns of the difference m_i - m_j identically), so no run-demeaning of the
features is needed — unlike the decoding scripts.

Scopes
------
* `pooled`   — crossnobis cross-validated over the 3 runs (learning1/learning2/test),
               the same leakage-safe partition every decoding script uses.
* per-run    — one RDM per run, cross-validated over two interleaved within-run
               pseudo-halves (trials alternated within each (stimulus, run) cell, so
               both halves are balanced). Legitimate because adjacent-trial beta
               correlation is only 0.05 for type-D betas (2026-08-13 note, adjacent
               baseline). Use `--within-run-split blocked` for the first-half /
               second-half alternative. This is the learning-dynamics readout:
               learning1 -> learning2 -> test, where `test` carries no feedback.

Models regressed against each RDM (z-scored predictors, z-scored RDM, so the
coefficients are standardised partial weights comparable across subjects and ROIs):
`category` (0 same / 1 different), `value` (|dvalue|), `frequency` (|dfrequency|).
No identity regressor is needed — every off-diagonal cell is a different-identity
cell, so the identity model *is* the intercept. A second regression swaps the
objective value for the model-derived RL value (|d mean first_stim_value_rl|,
computed within scope), reported as model='rl'.

Also reported per scope/subset: `contrast_value` = mean(cross-category different-value
cells) - mean(cross-category same-value cells), on the z-scored RDM. Positive means
same-value stimuli are represented more similarly, i.e. value coding.

Usage
-----
python multivariate/run_rsa_roi.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/rsa \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz \\
    --roi-mask vmpfc .../vmpfc_bartra2013_MNI152NLin2009cAsym.nii \\
    --roi-mask striatum .../striatum_bartra2013_MNI152NLin2009cAsym.nii

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_rsa_results.csv                     — long table, one row per
        (mask, scope, subset, model); see COLUMNS note at the bottom of run_subject.
    sub-<id>_rsa_rdm_<mask>_<scope>.npy          — raw 8x8 crossnobis RDM,
        rows/cols ordered by `stim_names` in the npz below.
    sub-<id>_rsa_model_rdms.npz                  — stim_names, stim_cat, stim_value,
        stim_frequency, category/value/frequency 8x8 model RDMs, rl_value_<scope>,
        and the non-figure boolean stimulus mask.
    rsa_roi_sub-<id>.log
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_target_from_bbt

RUNS = ['learning1', 'learning2', 'test']
FIGURE_CAT = 'figure'


# ---------------------------------------------------------------------------
# Crossnobis
# ---------------------------------------------------------------------------
def _crossnobis_loo(patterns, cond_idx, fold_idx, n_cond):
    """Leave-one-fold-out crossnobis RDM.

    Reproduces rsatoolbox 0.3.2 `calc_rdm_crossnobis` with an identity noise
    precision (i.e. after the data have already been univariately prewhitened),
    without materialising the dense n_voxels x n_voxels identity that function
    builds. For each fold f: the test patterns are the per-condition means within
    f, the train patterns are the per-condition means over *all observations* of
    the remaining folds pooled; then

        d_ij = < m_train_i - m_train_j , m_test_i - m_test_j > / n_voxels

    averaged over folds. Unbiased around 0 when conditions do not differ.

    Parameters
    ----------
    patterns : (n_obs, n_voxels) float array, already prewhitened.
    cond_idx : (n_obs,) int array in [0, n_cond).
    fold_idx : (n_obs,) int array; cross-validation fold of each observation.
    n_cond : int

    Returns
    -------
    (n_cond, n_cond) float array with a zero diagonal.
    """
    folds = np.unique(fold_idx)
    if len(folds) < 2:
        raise ValueError(f"crossnobis needs >=2 folds, got {len(folds)}")
    n_voxels = patterns.shape[1]
    acc = np.zeros((n_cond, n_cond), dtype=float)

    for f in folds:
        test_m = _condition_means(patterns, cond_idx, fold_idx == f, n_cond)
        train_m = _condition_means(patterns, cond_idx, fold_idx != f, n_cond)
        kernel = train_m @ test_m.T
        diag = np.diag(kernel)
        acc += (diag[None, :] + diag[:, None] - kernel - kernel.T) / n_voxels

    rdm = acc / len(folds)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _condition_means(patterns, cond_idx, obs_mask, n_cond):
    """(n_cond, n_voxels) mean pattern per condition over the selected observations."""
    out = np.empty((n_cond, patterns.shape[1]), dtype=float)
    for c in range(n_cond):
        sel = obs_mask & (cond_idx == c)
        if not sel.any():
            raise ValueError(f"condition {c} has no observations in this fold")
        out[c] = patterns[sel].mean(axis=0)
    return out


def compute_noise_sd(patterns, cond_idx, run_idx):
    """Per-voxel residual SD around the (condition x run) cell means.

    This is the univariate noise normalisation term: dividing by it turns the
    crossnobis kernel into a cross-validated Mahalanobis distance under a diagonal
    noise covariance assumption (a full covariance is not estimable here —
    n_trials ~ 328 << n_voxels). Estimated once on all trials and reused for every
    scope so that per-run RDMs remain on a common scale and are comparable across
    learning1 / learning2 / test.
    """
    resid = patterns.copy()
    n_cells = 0
    for c in np.unique(cond_idx):
        for r in np.unique(run_idx):
            sel = (cond_idx == c) & (run_idx == r)
            if sel.any():
                resid[sel] -= resid[sel].mean(axis=0, keepdims=True)
                n_cells += 1
    dof = max(patterns.shape[0] - n_cells, 1)
    return np.sqrt((resid ** 2).sum(axis=0) / dof)


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------
def within_run_folds(cond_idx, trial_order, mode='interleaved'):
    """Split trials of a single run into 2 balanced pseudo-halves.

    'interleaved' alternates trials within each condition in chronological order,
    so both halves hold ~the same number of repeats of every stimulus and both
    span the whole run (no drift/learning difference between halves).
    'blocked' splits each condition's trials into first half / second half.
    """
    fold = np.empty(len(cond_idx), dtype=int)
    for c in np.unique(cond_idx):
        sel = np.flatnonzero(cond_idx == c)
        sel = sel[np.argsort(trial_order[sel])]
        if mode == 'interleaved':
            fold[sel] = np.arange(len(sel)) % 2
        elif mode == 'blocked':
            fold[sel] = (np.arange(len(sel)) >= len(sel) / 2).astype(int)
        else:
            raise ValueError(f"unknown within-run split mode: {mode}")
    return fold


# ---------------------------------------------------------------------------
# Model RDMs and regression
# ---------------------------------------------------------------------------
def abs_diff_rdm(values):
    v = np.asarray(values, dtype=float)
    return np.abs(v[:, None] - v[None, :])


def different_rdm(labels):
    lab = np.asarray(labels)
    return (lab[:, None] != lab[None, :]).astype(float)


def _triu(mat):
    return mat[np.triu_indices(mat.shape[0], 1)]


def _z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


def fit_rdm_regression(rdm, model_rdms, keep):
    """Regress the (z-scored) data RDM on z-scored model RDMs.

    `keep` is a boolean stimulus mask (all 8, or the 6 non-figure stimuli). Model
    RDMs whose vector is constant over the retained cells are dropped and reported
    as NaN — nothing to estimate. Returns (coefficients dict, correlations dict,
    z-scored data RDM vector, dict of z-scored model vectors).
    """
    sub = np.ix_(keep, keep)
    y = _z(_triu(rdm[sub]))

    names, cols = [], []
    for name, mrdm in model_rdms.items():
        v = _triu(mrdm[sub])
        if np.ptp(v) == 0:
            continue
        names.append(name)
        cols.append(_z(v))

    corrs = {n: float(np.corrcoef(y, c)[0, 1]) for n, c in zip(names, cols)}
    betas = {n: float('nan') for n in model_rdms}
    if names:
        X = np.column_stack([np.ones_like(y)] + cols)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        for n, b in zip(names, coef[1:]):
            betas[n] = float(b)
    return betas, corrs, y, dict(zip(names, cols))


def value_contrast(rdm, keep, cat, value):
    """mean(cross-category, different-value) - mean(cross-category, same-value).

    Computed on the z-scored RDM so it is comparable across subjects and ROIs.
    Positive = stimuli sharing a reward level are represented more similarly.
    Returns (contrast, n_same_value_cells, n_different_value_cells).
    """
    sub = np.ix_(keep, keep)
    y = _z(_triu(rdm[sub]))
    cross = _triu(different_rdm(cat[keep])) > 0
    same_v = _triu(abs_diff_rdm(value[keep])) == 0
    a, b = cross & same_v, cross & ~same_v
    if not a.any() or not b.any():
        return float('nan'), int(a.sum()), int(b.sum())
    return float(y[b].mean() - y[a].mean()), int(a.sum()), int(b.sum())


# ---------------------------------------------------------------------------
# Main per-subject routine
# ---------------------------------------------------------------------------
def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                roi_masks=None, within_run_split='interleaved', shuffle_seed=None,
                validate_rsatoolbox=False, overwrite=False):

    roi_masks = roi_masks or []
    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_rsa_results.csv"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)

    # --- Betas and trial info ------------------------------------------------
    sub_dir = glmsingle_dir / f"sub-{subject}"
    betas_img = nib.load(sub_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz")
    trial_info = pd.read_csv(sub_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv",
                             index_col='trial_id')
    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    # --- Per-stimulus value / frequency / category from the BBT --------------
    # load_target_from_bbt asserts the BBT identity sequence matches the info CSV,
    # which is our guard against silent beta/label misalignment.
    value_per_trial = load_target_from_bbt(subject, bbt_path, trial_info,
                                           target_col='first_stim_value')
    frequ_per_trial = load_target_from_bbt(subject, bbt_path, trial_info,
                                           target_col='first_stim_frequ')
    rl_per_trial = load_target_from_bbt(subject, bbt_path, trial_info,
                                        target_col='first_stim_value_rl')

    stim_labels = trial_info['stim_name'].values
    run_labels = trial_info['run'].values
    if shuffle_seed is not None:
        # Negative control: permute the stimulus labels within each run, leaving
        # every other aspect of the pipeline untouched. All model fits should
        # collapse to ~0.
        rng = np.random.default_rng(shuffle_seed)
        stim_labels = stim_labels.copy()
        for r in np.unique(run_labels):
            m = run_labels == r
            stim_labels[m] = rng.permutation(stim_labels[m])
        logging.warning(f"sub-{subject}: SHUFFLE CONTROL active (seed={shuffle_seed})")

    stim_names = np.array(sorted(pd.unique(trial_info['stim_name'])))
    n_cond = len(stim_names)
    cond_idx = np.searchsorted(stim_names, stim_labels)

    def per_stim(series):
        tab = pd.Series(series, index=trial_info['stim_name'].values).groupby(level=0)
        uniq = tab.nunique()
        assert (uniq == 1).all(), f"sub-{subject}: non-unique per-stimulus values: {uniq.to_dict()}"
        return tab.first().reindex(stim_names).values

    stim_cat = per_stim(trial_info['stim_cat'].values)
    stim_value = per_stim(value_per_trial).astype(float)
    stim_frequ = per_stim(frequ_per_trial).astype(float)
    logging.info(f"sub-{subject}: stimuli " + ", ".join(
        f"{n}({c},v={v:.0f},f={f:+.0f})"
        for n, c, v, f in zip(stim_names, stim_cat, stim_value, stim_frequ)))

    non_figure = stim_cat != FIGURE_CAT
    subsets = {'all': np.ones(n_cond, dtype=bool), 'nonfigure': non_figure}
    logging.info(f"sub-{subject}: non-figure subset has {non_figure.sum()} stimuli")

    model_rdms = {
        'category':  different_rdm(stim_cat),
        'value':     abs_diff_rdm(stim_value),
        'frequency': abs_diff_rdm(stim_frequ),
    }

    # --- Masks ---------------------------------------------------------------
    sub = Subject(base_dir=str(base_dir), subject_id=subject,
                  include_imaging=True, bids_dir=str(bids_dir))
    brain_mask_img = nib.load(sub.brain_mask['learning1'])

    # Transform the betas once at whole-brain and index ROIs into that array —
    # every ROI is (roi & brain_mask), i.e. a subset of the whole-brain voxels.
    wb_masker = NiftiMasker(mask_img=brain_mask_img, standardize=False).fit()
    X_wb = wb_masker.transform(betas_img)
    logging.info(f"sub-{subject}: wholebrain {X_wb.shape[1]:,} voxels")

    masks = [('wholebrain', np.ones(X_wb.shape[1], dtype=bool))]
    for name, path in roi_masks:
        roi_func = resample_to_img(nib.load(str(path)), brain_mask_img,
                                   interpolation='nearest')
        roi_img = math_img('(v > 0) & (b > 0)', v=roi_func, b=brain_mask_img)
        sel = wb_masker.transform(roi_img)[0] > 0
        if sel.sum() == 0:
            logging.warning(f"sub-{subject}: ROI '{name}' empty after masking, skipped")
            continue
        masks.append((name, sel))
        logging.info(f"sub-{subject}: ROI '{name}' {sel.sum():,} voxels")

    # Validate the local crossnobis against rsatoolbox on the smallest ROI only —
    # rsatoolbox builds a dense n_voxels x n_voxels identity, so this is only
    # tractable for a small mask.
    smallest_mask = min(masks, key=lambda m: m[1].sum())[0]

    # --- Cross-validation folds per scope ------------------------------------
    trial_order = np.asarray(trial_info.index.values, dtype=float)  # chronological
    scopes = {'pooled': (np.ones(len(cond_idx), dtype=bool),
                         np.searchsorted(np.array(RUNS), run_labels))}
    for r in RUNS:
        m = run_labels == r
        if not m.any():
            logging.warning(f"sub-{subject}: run '{r}' absent, skipping that scope")
            continue
        folds = np.full(len(cond_idx), -1, dtype=int)
        folds[m] = within_run_folds(cond_idx[m], trial_order[m], mode=within_run_split)
        scopes[r] = (m, folds)

    # RL value model RDM per scope (Q evolves with learning, so it is scope-specific).
    rl_rdms = {}
    for scope, (obs, _) in scopes.items():
        means = pd.Series(rl_per_trial[obs]).groupby(cond_idx[obs]).mean()
        rl_rdms[scope] = abs_diff_rdm(means.reindex(range(n_cond)).values)

    # --- Noise normalisation, then one RDM per (mask, scope) -----------------
    rows = []
    for mask_name, vox in masks:
        X = X_wb[:, vox]
        sd = compute_noise_sd(X, cond_idx, run_labels)
        good = sd > 0
        if not good.all():
            logging.warning(f"sub-{subject}/{mask_name}: dropping "
                            f"{(~good).sum():,} voxels with zero residual SD")
        Xw = X[:, good] / sd[good]
        n_voxels = int(good.sum())

        for scope, (obs, folds) in scopes.items():
            rdm = _crossnobis_loo(Xw[obs], cond_idx[obs], folds[obs], n_cond)
            np.save(subject_output / f"sub-{subject}_rsa_rdm_{mask_name}_{scope}.npy", rdm)

            if validate_rsatoolbox and mask_name == smallest_mask:
                _validate_against_rsatoolbox(Xw[obs], cond_idx[obs], folds[obs],
                                             stim_names, rdm)

            for subset, keep in subsets.items():
                contrast, n_same, n_diff = value_contrast(rdm, keep, stim_cat, stim_value)
                for model_name, value_rdm in [('objective', model_rdms['value']),
                                              ('rl', rl_rdms[scope])]:
                    models = {'category': model_rdms['category'],
                              'value': value_rdm,
                              'frequency': model_rdms['frequency']}
                    betas, corrs, y, _ = fit_rdm_regression(rdm, models, keep)
                    rows.append({
                        'subject': f"sub-{subject}", 'mask': mask_name,
                        'n_voxels': n_voxels, 'scope': scope, 'subset': subset,
                        'model': model_name, 'n_stim': int(keep.sum()),
                        'n_pairs': len(y), 'n_trials': int(obs.sum()),
                        **{f'beta_{k}': v for k, v in betas.items()},
                        **{f'r_{k}': corrs.get(k, float('nan')) for k in models},
                        'contrast_value': contrast,
                        'n_same_value_pairs': n_same, 'n_diff_value_pairs': n_diff,
                        'rdm_mean': float(_triu(rdm[np.ix_(keep, keep)]).mean()),
                        'rdm_sd': float(_triu(rdm[np.ix_(keep, keep)]).std()),
                        'within_run_split': within_run_split,
                        'shuffle_seed': -1 if shuffle_seed is None else shuffle_seed,
                    })

            b = [r for r in rows if r['mask'] == mask_name and r['scope'] == scope
                 and r['subset'] == 'all' and r['model'] == 'objective'][0]
            logging.info(
                f"  {mask_name}/{scope}: beta_cat={b['beta_category']:+.3f} "
                f"beta_val={b['beta_value']:+.3f} beta_frq={b['beta_frequency']:+.3f} "
                f"contrast={b['contrast_value']:+.3f}")

    np.savez(subject_output / f"sub-{subject}_rsa_model_rdms.npz",
             stim_names=stim_names, stim_cat=stim_cat, stim_value=stim_value,
             stim_frequency=stim_frequ, non_figure=non_figure,
             **model_rdms, **{f'rl_value_{k}': v for k, v in rl_rdms.items()})

    pd.DataFrame(rows).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done — {len(rows)} rows -> {done_flag}")


def _validate_against_rsatoolbox(patterns, cond_idx, fold_idx, stim_names, rdm_ours):
    """Assert `_crossnobis_loo` matches rsatoolbox's crossnobis on this ROI."""
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm

    ds = Dataset(measurements=patterns,
                 obs_descriptors={'stim': np.asarray(stim_names)[cond_idx],
                                  'fold': fold_idx.astype(str)})
    ref = calc_rdm(ds, method='crossnobis', descriptor='stim', cv_descriptor='fold')
    ref_mat = ref.get_matrices()[0]
    order = np.argsort(np.asarray(ref.pattern_descriptors['stim']))
    ref_mat = ref_mat[np.ix_(order, order)]
    max_dev = np.abs(ref_mat - rdm_ours).max()
    assert np.allclose(ref_mat, rdm_ours, rtol=1e-6, atol=1e-10), \
        f"crossnobis mismatch vs rsatoolbox, max |dev| = {max_dev:.3e}"
    logging.info(f"  rsatoolbox validation OK (max |dev| = {max_dev:.3e})")


def main():
    parser = argparse.ArgumentParser(description="ROI crossnobis RSA for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", default="/home/ubuntu/data/learning-habits")
    parser.add_argument("--bids-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/fmriprep-24.0.1-noSDC")
    parser.add_argument("--glmsingle-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/glmsingle")
    parser.add_argument("--bbt", required=True,
                        help="Path to the Big Behavior Table CSV (first_stim_value, "
                             "first_stim_frequ, first_stim_value_rl)")
    parser.add_argument("--output-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/rsa")
    parser.add_argument("--roi-mask", action="append", nargs=2, metavar=("NAME", "PATH"),
                        default=[],
                        help="ROI mask given as NAME PATH; repeatable. Whole-brain is "
                             "always included automatically.")
    parser.add_argument("--within-run-split", choices=['interleaved', 'blocked'],
                        default='interleaved',
                        help="How to form the two within-run crossnobis folds for the "
                             "per-run RDMs (default: interleaved)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Negative control: permute stimulus labels within run. "
                             "Write to a separate --output-dir.")
    parser.add_argument("--validate-against-rsatoolbox", action="store_true",
                        help="Assert the local crossnobis matches rsatoolbox on the "
                             "smallest ROI (needs rsatoolbox installed)")
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
            logging.FileHandler(subject_output / f"rsa_roi_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject             = args.subject,
        base_dir            = Path(args.base_dir),
        bids_dir            = Path(args.bids_dir),
        glmsingle_dir       = Path(args.glmsingle_dir),
        output_dir          = output_dir,
        bbt_path            = args.bbt,
        roi_masks           = args.roi_mask,
        within_run_split    = args.within_run_split,
        shuffle_seed        = args.shuffle_seed,
        validate_rsatoolbox = args.validate_against_rsatoolbox,
        overwrite           = args.overwrite,
    )


if __name__ == "__main__":
    main()
