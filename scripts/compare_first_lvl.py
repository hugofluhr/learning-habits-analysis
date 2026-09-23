"""
Compare one subject's first-level SPM output against a reference run of the same model.

Built to validate the cluster port (2026-09): a new run of a first-level script vs the VM's
earlier run of the same script. It checks
  1. the design matrix (SPM.xX.X, unfiltered), matched column by column by regressor name;
  2. the analysis masks;
  3. betas and ResMS, matched by regressor name, within the voxels both masks share.

Usage (on the cluster, in the learning-habits env):
    python scripts/compare_first_lvl.py --new <new_run>/sub-01 --ref <reference_run>/sub-01 \
        --out <dir>/sub-01_compare.csv

Writes one CSV row per regressor and prints a per-session summary.
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io


def load_spm(sub_dir):
    spm = scipy.io.loadmat(os.path.join(sub_dir, 'SPM.mat'), struct_as_record=False,
                           squeeze_me=True)['SPM']
    names = [str(n) for n in np.atleast_1d(spm.xX.name)]
    X = np.atleast_2d(spm.xX.X)
    sess = np.atleast_1d(spm.Sess)
    col_to_sess = {}
    for i, s in enumerate(sess, start=1):
        for c in np.atleast_1d(s.col):
            col_to_sess[int(c) - 1] = i          # SPM columns are 1-based
    return names, X, col_to_sess


def load_img(path):
    return np.asarray(nib.load(path).dataobj, dtype=np.float64)


def kind(name):
    # "Sn(1) R12" = nuisance column from the confound file; "Sn(1) constant" = session mean
    if re.search(r'\) R\d+$', name):
        return 'confound'
    if name.endswith('constant'):
        return 'constant'
    return 'task'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--new', required=True, help='new subject folder (contains SPM.mat, beta_*.nii)')
    p.add_argument('--ref', required=True, help='reference subject folder')
    p.add_argument('--out', required=True, help='output CSV path')
    p.add_argument('--dm-atol', type=float, default=1e-8,
                   help='max abs difference for a design-matrix column to count as identical')
    args = p.parse_args()

    names_new, X_new, sess_new = load_spm(args.new)
    names_ref, X_ref, sess_ref = load_spm(args.ref)

    print(f'new: {X_new.shape[1]} columns, {X_new.shape[0]} scans   ref: {X_ref.shape[1]} columns, {X_ref.shape[0]} scans')
    only_new = sorted(set(names_new) - set(names_ref))
    only_ref = sorted(set(names_ref) - set(names_new))
    if only_new or only_ref:
        print(f'!! column names differ. only in new: {only_new}   only in ref: {only_ref}')
    if len(set(names_new)) != len(names_new) or len(set(names_ref)) != len(names_ref):
        raise SystemExit('duplicate regressor names; matching by name is not possible')
    if X_new.shape[0] != X_ref.shape[0]:
        raise SystemExit('different number of scans; design matrices are not comparable')

    # Masks
    mask_new = load_img(os.path.join(args.new, 'mask.nii')) > 0
    mask_ref = load_img(os.path.join(args.ref, 'mask.nii')) > 0
    both = mask_new & mask_ref
    dice = 2 * both.sum() / (mask_new.sum() + mask_ref.sum())
    print(f'mask voxels: new {mask_new.sum()}  ref {mask_ref.sum()}  shared {both.sum()}  dice {dice:.6f}')

    rows = []
    idx_ref = {n: i for i, n in enumerate(names_ref)}
    for i_new, name in enumerate(names_new):
        if name not in idx_ref:
            continue
        i_ref = idx_ref[name]
        dm_diff = np.max(np.abs(X_new[:, i_new] - X_ref[:, i_ref]))

        b_new = load_img(os.path.join(args.new, f'beta_{i_new + 1:04d}.nii'))[both]
        b_ref = load_img(os.path.join(args.ref, f'beta_{i_ref + 1:04d}.nii'))[both]
        ok = np.isfinite(b_new) & np.isfinite(b_ref)
        b_new, b_ref = b_new[ok], b_ref[ok]
        ref_sd = b_ref.std()
        rows.append({
            'name': name,
            'session': sess_new.get(i_new),
            'kind': kind(name),
            'dm_max_abs_diff': dm_diff,
            'dm_identical': dm_diff <= args.dm_atol,
            'beta_r': np.corrcoef(b_new, b_ref)[0, 1] if ref_sd > 0 else np.nan,
            'beta_max_abs_diff': np.max(np.abs(b_new - b_ref)),
            'beta_mean_abs_diff_over_ref_sd': np.mean(np.abs(b_new - b_ref)) / ref_sd if ref_sd > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    # ResMS
    r_new = load_img(os.path.join(args.new, 'ResMS.nii'))[both]
    r_ref = load_img(os.path.join(args.ref, 'ResMS.nii'))[both]
    ok = np.isfinite(r_new) & np.isfinite(r_ref) & (r_ref > 0)
    rel = np.abs(r_new[ok] - r_ref[ok]) / r_ref[ok]
    print(f'ResMS: r = {np.corrcoef(r_new[ok], r_ref[ok])[0, 1]:.6f}, '
          f'median |rel diff| = {np.median(rel):.3g}, max |rel diff| = {rel.max():.3g}')

    print('\nDesign-matrix columns that are NOT identical:')
    diff = df[~df.dm_identical]
    print(diff[['name', 'dm_max_abs_diff']].to_string(index=False) if len(diff) else '  none')

    print('\nBetas by session and regressor kind (min r, max mean|diff|/sd):')
    summ = (df.groupby(['session', 'kind'], dropna=False)   # constants have no Sess.col -> session NaN
              .agg(n=('name', 'size'), min_r=('beta_r', 'min'),
                   max_mad_over_sd=('beta_mean_abs_diff_over_ref_sd', 'max'))
              .reset_index())
    print(summ.to_string(index=False))

    print('\nTask regressors:')
    print(df[df.kind == 'task'][['name', 'dm_identical', 'beta_r', 'beta_mean_abs_diff_over_ref_sd']]
          .to_string(index=False))
    print(f'\nWritten: {args.out}')


if __name__ == '__main__':
    main()
