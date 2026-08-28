#!/usr/bin/env python3
"""
ROI crossnobis RSA on GLMsingle cue betas — one subject.

Conditions are the 8 stimulus identities. Distances are crossnobis (cross-validated
Mahalanobis, univariate noise normalisation), computed pooled over the 3 runs and
per-run for the learning-dynamics readout. Each RDM is regressed on category / value /
choice-frequency model RDMs, and a targeted same-value contrast is reported.

`first_stim_frequ` is CHOICE frequency, not presentation frequency: every stimulus is
presented equally often (first-stim counts balanced across the +-1 labels, per-subject
r=-0.05 ns — see `rsa_design_checks.ipynb`); what the +-1 label encodes is how often the
subject *chose* the stimulus during learning, manipulated by selectively pairing it with
higher- or lower-valued alternatives. The graded behavioural counterpart is the
choice-kernel H-value (`first_stim_value_ck`), used by the `ck` model variant.

Notes that live nowhere else:
  * No identity regressor is needed — every off-diagonal cell is a different-identity
    cell, so the identity model *is* the intercept.
  * No run-demeaning of features, unlike the decoding scripts: a run-constant offset
    cancels inside the crossnobis kernel (it enters both patterns of `m_i - m_j`).
  * The noise-normalisation term is estimated once on all trials and reused for every
    scope, so per-run RDMs stay on a common scale and are comparable across runs.

Usage
-----
# Precondition check only — needs just --bbt, no NIfTI/cluster access:
python multivariate/run_rsa_roi.py --subject 01 --bbt /path/to/bbt.csv --dry-run

python multivariate/run_rsa_roi.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir  .../derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir .../derivatives/glmsingle \\
    --bbt /home/hfluhr/data/learninghabits/bbt.csv \\
    --output-dir .../derivatives/rsa \\
    --roi-mask visualcortex .../visual_cortex_mask.nii.gz

Outputs to <output-dir>/sub-<id>/
    sub-<id>_rsa_results.csv              one row per (mask, scope, subset, model)
    sub-<id>_rsa_rdm_<mask>_<scope>.npy   8x8 crossnobis RDM, rows ordered as stim_names
    sub-<id>_rsa_model_rdms.npz           model RDMs + per-stimulus properties
    rsa_roi_sub-<id>.log
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import index_img, resample_to_img, math_img
from nilearn.maskers import NiftiMasker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_target_from_bbt

RUNS = ['learning1', 'learning2', 'test']
FIGURE_CAT = 'figure'
MODEL_TERMS = ['category', 'value', 'frequency']


# ---------------------------------------------------------------------------
# Crossnobis
# ---------------------------------------------------------------------------
def _crossnobis_loo(patterns, cond_idx, fold_idx, n_cond):
    """Leave-one-fold-out crossnobis RDM over already-prewhitened patterns.

    For each fold f, d_ij = <m_train_i - m_train_j, m_test_i - m_test_j> / n_voxels,
    where test means come from f and train means from all remaining folds pooled;
    averaged over folds. Unbiased around 0 when conditions do not differ.

    Reproduces rsatoolbox 0.2.0 `calc_rdm(..., method='crossnobis')` with an identity
    noise precision, without building the dense identity that function requires — see
    `crossnobis_validation.ipynb` §1.
    """
    folds = np.unique(fold_idx)
    if len(folds) < 2:
        raise ValueError(f"crossnobis needs >=2 folds, got {len(folds)}")
    acc = np.zeros((n_cond, n_cond), dtype=float)
    for f in folds:
        test_m = _condition_means(patterns, cond_idx, fold_idx == f, n_cond)
        train_m = _condition_means(patterns, cond_idx, fold_idx != f, n_cond)
        kernel = train_m @ test_m.T
        diag = np.diag(kernel)
        acc += (diag[None, :] + diag[:, None] - kernel - kernel.T) / patterns.shape[1]
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

    The univariate noise-normalisation term; a full covariance is not estimable here
    (n_trials ~ 328 << n_voxels). See `crossnobis_validation.ipynb` §5.
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


def within_run_folds(cond_idx, trial_order, mode='interleaved'):
    """Split one run's trials into 2 balanced pseudo-halves.

    'interleaved' alternates within each condition in chronological order, so both
    halves span the run; 'blocked' splits each condition first-half / second-half.
    See `crossnobis_validation.ipynb` §4 for how much the choice matters.
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
    """Regress the z-scored data RDM on z-scored model RDMs over the `keep` stimuli.

    Coefficients are standardised partial weights, comparable across subjects and ROIs.
    Models constant over the retained cells are dropped and reported NaN.
    Returns (betas, correlations, n_pairs).
    """
    sub = np.ix_(keep, keep)
    y = _z(_triu(rdm[sub]))

    names, cols = [], []
    for name, mrdm in model_rdms.items():
        v = _triu(mrdm[sub])
        if np.ptp(v) > 0:
            names.append(name)
            cols.append(_z(v))

    corrs = {n: float(np.corrcoef(y, c)[0, 1]) for n, c in zip(names, cols)}
    betas = {n: float('nan') for n in model_rdms}
    if names:
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(y)] + cols), y, rcond=None)
        betas.update(dict(zip(names, (float(b) for b in coef[1:]))))
    return betas, corrs, len(y)


def value_contrast(rdm, keep, cat, value):
    """mean(cross-category, different-value) - mean(cross-category, same-value).

    On the z-scored RDM. Positive = stimuli sharing a reward level are more similar.
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
# Dry-run precondition check
# ---------------------------------------------------------------------------
def check_bbt_only(subject, bbt_path):
    """Whether the BBT has a well-formed 8-stimulus record for `subject`, with no
    NIfTI/info-CSV access at all.

    This is a *necessary but not sufficient* precondition: it catches a subject
    entirely missing from the BBT (the sub-46 failure mode below) or one whose
    stimuli aren't uniquely valued, using only the ~200 KB bbt.csv read — no
    cluster access needed. It does NOT check that the BBT's identity sequence
    matches the beta volume order; that's `load_target_from_bbt`'s job at runtime,
    and it needs the info CSV that only exists once GLMsingle has produced betas.
    Passing this check does not guarantee `run_subject` will succeed; failing it
    guarantees `run_subject` will crash on this subject.

    Returns (ok: bool, message: str).
    """
    sub_id = f"sub-{subject}"
    bbt = pd.read_csv(bbt_path, usecols=['sub_id', 'block', 'first_stim_name',
                                         'first_stim_cat', 'first_stim_value',
                                         'first_stim_frequ', 'first_stim_value_rl'])
    sub_bbt = bbt[bbt['sub_id'] == sub_id]
    if sub_bbt.empty:
        return False, f"{sub_id}: absent from BBT ({bbt_path})"

    for run in RUNS:
        if not (sub_bbt['block'] == run).any():
            return False, f"{sub_id}: no rows for run '{run}'"

    grp = sub_bbt.groupby('first_stim_name')
    n_stim = grp.ngroups
    if n_stim != 8:
        return False, f"{sub_id}: {n_stim} distinct stimuli, expected 8"

    bad = grp.agg(n_cat=('first_stim_cat', 'nunique'),
                  n_val=('first_stim_value', 'nunique'),
                  n_frq=('first_stim_frequ', 'nunique'))
    bad = bad[(bad > 1).any(axis=1)]
    if len(bad):
        return False, f"{sub_id}: non-unique value/frequency/category for {list(bad.index)}"

    return True, f"{sub_id}: OK — 8 stimuli, all 3 runs present, values well-formed"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_stimuli(subject, glmsingle_dir, bbt_path, shuffle_seed=None):
    """Betas image, trial info, condition indices, and per-stimulus properties.

    `load_target_from_bbt` asserts the BBT identity sequence matches the info CSV — the
    guard against silent beta/label misalignment.
    """
    sub_dir = glmsingle_dir / f"sub-{subject}"
    betas_img = nib.load(sub_dir / f"sub-{subject}_glmSingle_betas_CUES.nii.gz")
    trial_info = pd.read_csv(sub_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv",
                             index_col='trial_id')
    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    per_trial = {col: load_target_from_bbt(subject, bbt_path, trial_info, target_col=col)
                 for col in ['first_stim_value', 'first_stim_frequ',
                             'first_stim_value_rl', 'first_stim_value_ck']}

    stim_labels = trial_info['stim_name'].values
    run_labels = trial_info['run'].values
    if shuffle_seed is not None:
        # Negative control: permute stimulus labels within run, nothing else touched.
        rng = np.random.default_rng(shuffle_seed)
        stim_labels = stim_labels.copy()
        for r in np.unique(run_labels):
            m = run_labels == r
            stim_labels[m] = rng.permutation(stim_labels[m])
        logging.warning(f"sub-{subject}: SHUFFLE CONTROL active (seed={shuffle_seed})")

    stim_names = np.array(sorted(pd.unique(trial_info['stim_name'])))
    cond_idx = np.searchsorted(stim_names, stim_labels)

    def per_stim(series):
        grp = pd.Series(series, index=trial_info['stim_name'].values).groupby(level=0)
        uniq = grp.nunique()
        assert (uniq == 1).all(), \
            f"sub-{subject}: non-unique per-stimulus values: {uniq[uniq > 1].to_dict()}"
        return grp.first().reindex(stim_names).values

    props = dict(
        names=stim_names,
        cat=per_stim(trial_info['stim_cat'].values),
        value=per_stim(per_trial['first_stim_value']).astype(float),
        frequency=per_stim(per_trial['first_stim_frequ']).astype(float),
    )
    logging.info(f"sub-{subject}: stimuli " + ", ".join(
        f"{n}({c},v={v:.0f},f={f:+.0f})" for n, c, v, f
        in zip(props['names'], props['cat'], props['value'], props['frequency'])))

    return (betas_img, trial_info, cond_idx, run_labels, props,
            per_trial['first_stim_value_rl'], per_trial['first_stim_value_ck'])


def build_masks(subject, base_dir, bids_dir, betas_img, roi_masks):
    """(masked betas at whole-brain, [(name, voxel index into that array), ...]).

    The betas are transformed once at whole-brain and ROIs are indexed into that array:
    every ROI is (roi & brain_mask), i.e. a subset of the whole-brain voxels.
    """
    sub = Subject(base_dir=str(base_dir), subject_id=subject,
                  include_imaging=True, bids_dir=str(bids_dir))
    brain_mask_img = nib.load(sub.brain_mask['learning1'])

    wb_masker = NiftiMasker(mask_img=brain_mask_img, standardize=False).fit()
    X_wb = wb_masker.transform(betas_img)
    logging.info(f"sub-{subject}: wholebrain {X_wb.shape[1]:,} voxels")

    masks = [('wholebrain', np.ones(X_wb.shape[1], dtype=bool))]
    for name, path in roi_masks:
        roi_func = resample_to_img(nib.load(str(path)), brain_mask_img,
                                   interpolation='nearest')
        # Some masks carry a singleton 4th dimension (e.g. 53,65,48,1 in 3mm
        # space); squeeze it so math_img can broadcast against the 3D brain mask.
        if roi_func.ndim == 4 and roi_func.shape[3] == 1:
            roi_func = index_img(roi_func, 0)
        roi_img = math_img('(v > 0) & (b > 0)', v=roi_func, b=brain_mask_img)
        sel = wb_masker.transform(roi_img)[0] > 0
        if sel.sum() == 0:
            logging.warning(f"sub-{subject}: ROI '{name}' empty after masking, skipped")
            continue
        masks.append((name, sel))
        logging.info(f"sub-{subject}: ROI '{name}' {sel.sum():,} voxels")
    return X_wb, masks


def build_scopes(subject, cond_idx, run_labels, trial_order, within_run_split):
    """{scope: (observation mask, per-observation CV fold)}.

    'pooled' cross-validates over the 3 runs; each run scope cross-validates over two
    within-run pseudo-halves.
    """
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
    return scopes


# ---------------------------------------------------------------------------
# Main per-subject routine
# ---------------------------------------------------------------------------
def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir, bbt_path,
                roi_masks=None, within_run_split='interleaved', shuffle_seed=None,
                remove_mean=False, validate_rsatoolbox=False, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_rsa_results.csv"
    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return
    subject_output.mkdir(parents=True, exist_ok=True)

    (betas_img, trial_info, cond_idx, run_labels, props,
     rl_per_trial, ck_per_trial) = load_stimuli(
        subject, glmsingle_dir, bbt_path, shuffle_seed)
    stim_names, n_cond = props['names'], len(props['names'])

    non_figure = props['cat'] != FIGURE_CAT
    subsets = {'all': np.ones(n_cond, dtype=bool), 'nonfigure': non_figure}
    logging.info(f"sub-{subject}: non-figure subset has {non_figure.sum()} stimuli")

    model_rdms = {'category':  different_rdm(props['cat']),
                  'value':     abs_diff_rdm(props['value']),
                  'frequency': abs_diff_rdm(props['frequency'])}

    X_wb, masks = build_masks(subject, base_dir, bids_dir, betas_img, roi_masks or [])
    # rsatoolbox needs a dense n_voxels^2 identity, so validate on the smallest ROI only.
    smallest_mask = min(masks, key=lambda m: m[1].sum())[0]

    trial_order = np.asarray(trial_info.index.values, dtype=float)   # chronological
    scopes = build_scopes(subject, cond_idx, run_labels, trial_order, within_run_split)

    # Q and H evolve with learning, so their model RDMs are scope-specific.
    rl_rdms = {scope: abs_diff_rdm(pd.Series(rl_per_trial[obs]).groupby(cond_idx[obs])
                                   .mean().reindex(range(n_cond)).values)
               for scope, (obs, _) in scopes.items()}
    ck_rdms = {scope: abs_diff_rdm(pd.Series(ck_per_trial[obs]).groupby(cond_idx[obs])
                                   .mean().reindex(range(n_cond)).values)
               for scope, (obs, _) in scopes.items()}

    rows, amp_rows = [], []
    for mask_name, vox in masks:
        X = X_wb[:, vox]
        sd = compute_noise_sd(X, cond_idx, run_labels)
        good = sd > 0
        if not good.all():
            logging.warning(f"sub-{subject}/{mask_name}: dropping {(~good).sum():,} "
                            "voxels with zero residual SD")
        Xw = X[:, good] / sd[good]
        n_voxels = int(good.sum())

        # Per-(stimulus, run) mean amplitude of the whitened pattern — the direct probe
        # for a global-gain alternative to a choice-frequency-geometry effect (e.g.
        # frequently-chosen stimuli responding uniformly stronger/weaker via attention
        # or choice-related gain; presentation counts are equal, so classic repetition
        # suppression is not a candidate). Always computed from the NON-mean-removed
        # patterns, before the optional removal below.
        amp = Xw.mean(axis=1)   # (n_trials,) spatial mean per trial
        for r in RUNS:
            for c, name in enumerate(stim_names):
                sel = (run_labels == r) & (cond_idx == c)
                if sel.any():
                    amp_rows.append({'mask': mask_name, 'stim_name': name, 'run': r,
                                     'n_trials': int(sel.sum()),
                                     'mean_amp': float(amp[sel].mean())})

        if remove_mean:
            # Subtract each trial pattern's spatial mean (rsatoolbox's remove_mean
            # semantics): any purely global amplitude difference between conditions
            # can no longer contribute to the crossnobis distances.
            Xw = Xw - Xw.mean(axis=1, keepdims=True)

        for scope, (obs, folds) in scopes.items():
            rdm = _crossnobis_loo(Xw[obs], cond_idx[obs], folds[obs], n_cond)
            np.save(subject_output / f"sub-{subject}_rsa_rdm_{mask_name}_{scope}.npy", rdm)

            if validate_rsatoolbox and mask_name == smallest_mask:
                _validate_against_rsatoolbox(Xw[obs], cond_idx[obs], folds[obs],
                                             stim_names, rdm)

            for subset, keep in subsets.items():
                contrast, n_same, n_diff = value_contrast(rdm, keep, props['cat'],
                                                          props['value'])
                cells = _triu(rdm[np.ix_(keep, keep)])
                # 'objective'/'rl' swap the VALUE regressor (objective |dv| vs |dQ|);
                # 'ck' instead swaps the FREQUENCY regressor: the graded per-subject
                # choice-kernel |dH| replaces the categorical +-1 choice-frequency
                # label, testing whether behaviourally expressed habit strength
                # explains the geometry better than the design condition.
                for model_name, value_rdm, freq_rdm in [
                        ('objective', model_rdms['value'], model_rdms['frequency']),
                        ('rl',        rl_rdms[scope],      model_rdms['frequency']),
                        ('ck',        model_rdms['value'], ck_rdms[scope])]:
                    models = {'category': model_rdms['category'], 'value': value_rdm,
                              'frequency': freq_rdm}
                    betas, corrs, n_pairs = fit_rdm_regression(rdm, models, keep)
                    rows.append({
                        'subject': f"sub-{subject}", 'mask': mask_name,
                        'n_voxels': n_voxels, 'scope': scope, 'subset': subset,
                        'model': model_name, 'n_stim': int(keep.sum()),
                        'n_pairs': n_pairs, 'n_trials': int(obs.sum()),
                        **{f'beta_{k}': v for k, v in betas.items()},
                        **{f'r_{k}': corrs.get(k, float('nan')) for k in MODEL_TERMS},
                        'contrast_value': contrast,
                        'n_same_value_pairs': n_same, 'n_diff_value_pairs': n_diff,
                        'rdm_mean': float(cells.mean()), 'rdm_sd': float(cells.std()),
                        'within_run_split': within_run_split,
                        'remove_mean': bool(remove_mean),
                        'shuffle_seed': -1 if shuffle_seed is None else shuffle_seed,
                    })
                    if subset == 'all' and model_name == 'objective':
                        logging.info(
                            f"  {mask_name}/{scope}: "
                            f"beta_cat={betas['category']:+.3f} "
                            f"beta_val={betas['value']:+.3f} "
                            f"beta_frq={betas['frequency']:+.3f} "
                            f"contrast={contrast:+.3f}")

    np.savez(subject_output / f"sub-{subject}_rsa_model_rdms.npz",
             stim_names=stim_names, stim_cat=props['cat'], stim_value=props['value'],
             stim_frequency=props['frequency'], non_figure=non_figure,
             **model_rdms, **{f'rl_value_{k}': v for k, v in rl_rdms.items()},
             **{f'ck_value_{k}': v for k, v in ck_rdms.items()})

    pd.DataFrame(amp_rows).to_csv(
        subject_output / f"sub-{subject}_rsa_amplitude.csv", index=False)
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
    order = np.argsort(np.asarray(ref.pattern_descriptors['stim']))
    ref_mat = ref.get_matrices()[0][np.ix_(order, order)]
    max_dev = np.abs(ref_mat - rdm_ours).max()
    assert np.allclose(ref_mat, rdm_ours, rtol=1e-6, atol=1e-10), \
        f"crossnobis mismatch vs rsatoolbox, max |dev| = {max_dev:.3e}"
    logging.info(f"  rsatoolbox validation OK (max |dev| = {max_dev:.3e})")


def main():
    parser = argparse.ArgumentParser(description="ROI crossnobis RSA for one subject.")
    parser.add_argument("--subject", required=True, help="ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", help="Root data directory")
    parser.add_argument("--bids-dir", help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", help="GLMsingle betas directory")
    parser.add_argument("--bbt", required=True,
                        help="Big Behavior Table CSV (first_stim_value, first_stim_frequ, "
                             "first_stim_value_rl)")
    parser.add_argument("--output-dir", help="Output root directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check only that the BBT has a well-formed record for this "
                             "subject (presence, 8 stimuli, all 3 runs, unique values) and "
                             "exit. No NIfTI/mask access, no cluster paths needed — see "
                             "check_bbt_only(). Exit code 1 on failure, for shell loops.")
    parser.add_argument("--roi-mask", action="append", nargs=2, metavar=("NAME", "PATH"),
                        default=[],
                        help="ROI mask as NAME PATH; repeatable. Whole-brain always included.")
    parser.add_argument("--within-run-split", choices=['interleaved', 'blocked'],
                        default='interleaved',
                        help="How to form the two within-run folds for per-run RDMs")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Negative control: permute stimulus labels within run. "
                             "Write to a separate --output-dir.")
    parser.add_argument("--remove-mean", action="store_true",
                        help="Subtract each trial pattern's spatial mean before crossnobis "
                             "(amplitude-confound control: a global response-magnitude "
                             "difference between conditions, e.g. attention/choice-related "
                             "gain for frequently-chosen stimuli, can then no longer drive "
                             "the distances). Write to a separate --output-dir.")
    parser.add_argument("--validate-against-rsatoolbox", action="store_true",
                        help="Assert the local crossnobis matches rsatoolbox on the "
                             "smallest ROI (needs rsatoolbox installed)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        ok, message = check_bbt_only(args.subject, args.bbt)
        logging.info(("PASS  " if ok else "FAIL  ") + message)
        sys.exit(0 if ok else 1)

    missing = [f"--{name.replace('_', '-')}" for name in
              ('base_dir', 'bids_dir', 'glmsingle_dir', 'output_dir')
              if getattr(args, name) is None]
    if missing:
        parser.error(f"the following arguments are required (unless --dry-run): "
                     f"{', '.join(missing)}")

    output_dir = Path(args.output_dir)
    subject_output = output_dir / f"sub-{args.subject}"
    subject_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(subject_output / f"rsa_roi_sub-{args.subject}.log")],
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
        remove_mean         = args.remove_mean,
        validate_rsatoolbox = args.validate_against_rsatoolbox,
        overwrite           = args.overwrite,
    )


if __name__ == "__main__":
    main()
