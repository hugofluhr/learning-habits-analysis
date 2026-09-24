#!/usr/bin/env python
"""Compare the `response` and `purple_frame` betas of a per-session and a concatenated first level.

These two regressors can be nearly collinear in a per-session design (the purple frame starts at the
response). When they are, their betas are large, anti-correlated across voxels and unstable across runs,
so contrasts built from them are mostly noise and are a poor reference for the concatenated model.

For each model, over the voxels in the brain mask, prints per session:
  * mean and sd of the `response` and `purple_frame` betas
  * correlation of the two betas across voxels
  * sd of their sum (a better-determined quantity when the two are collinear)
For the per-session model it also prints the correlation of the betas between runs (1-2, 1-3, 2-3).

Reads beta_*.nii and mask.nii from each subject folder, and the column names written next to SPM.mat by
``matlab/export_spm_dms.m`` (``sub-XX_column_names.txt``); run the exporter first.

Example:
    python scripts/check_concat_response_betas.py <concat_sub_dir> <per_session_sub_dir>
"""

import glob
import sys

import numpy as np
import nibabel as nib

RUN_PAIRS = [(0, 1), (0, 2), (1, 2)]  # index pairs for the between-run correlations


def beta_reader(sub_dir):
    """Return (get, mask): get(name) gives that regressor's beta image within the brain mask."""
    with open(glob.glob(f'{sub_dir}/*_column_names.txt')[0]) as f:
        names = [line.strip() for line in f]
    mask = nib.load(f'{sub_dir}/mask.nii').get_fdata() > 0

    def get(name):
        idx = names.index(name) + 1  # beta_0001.nii is the first design column
        return nib.load(f'{sub_dir}/beta_{idx:04d}.nii').get_fdata()[mask]

    return get, mask


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def between_run_corrs(betas):
    """Correlation across voxels of one regressor's betas between pairs of runs."""
    return [f'{corr(betas[a], betas[b]):+.2f}' for a, b in RUN_PAIRS]


def main():
    concat_dir, session_dir = sys.argv[1], sys.argv[2]
    # the concatenated model has a single session, so its columns are all named Sn(1)
    models = {'per-session': (session_dir, [1, 2, 3]), 'concat': (concat_dir, [1])}

    for label, (sub_dir, sessions) in models.items():
        get, mask = beta_reader(sub_dir)
        print(f'== {label}  ({mask.sum()} voxels in mask) ==')
        resp, purp = {}, {}
        for k in sessions:
            resp[k] = get(f'Sn({k}) response*bf(1)')
            purp[k] = get(f'Sn({k}) purple_frame*bf(1)')
            total = resp[k] + purp[k]
            print(f'Sn({k}): response mean={resp[k].mean():+.3f} sd={resp[k].std():.3f} | '
                  f'purple mean={purp[k].mean():+.3f} sd={purp[k].std():.3f} | '
                  f'corr(response,purple) across voxels={corr(resp[k], purp[k]):+.3f} | '
                  f'sd(response+purple)={total.std():.3f}')
        if len(sessions) > 1:
            r = [resp[k] for k in sessions]
            p = [purp[k] for k in sessions]
            s = [a + b for a, b in zip(r, p)]
            print('corr of response betas between runs (1-2, 1-3, 2-3):', between_run_corrs(r))
            print('corr of purple betas between runs (1-2, 1-3, 2-3):  ', between_run_corrs(p))
            print('corr of (response+purple) between runs (1-2, 1-3, 2-3):', between_run_corrs(s))


if __name__ == '__main__':
    main()
