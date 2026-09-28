"""Leave-one-subject-out DLS ROI: does within-run chosen H relate to DLS beyond a linear within-run time trend?

For each subject, the ROI is a 6 mm sphere at the peak of the within-run H map computed from the other subjects
(H-split concat model, no time regressor), inside the Guida 2022 habit mask, per hemisphere. In that ROI the subject's
betas are read from the H-split model and from the same model with a within-run trial-order modulator, so the size of the
H effect with and without time can be compared without selection bias. Peaks are recomputed per fold; their locations
across folds are printed.

Usage (on the cluster):
    python scripts/loso_roi_time_check.py [--hsplit RUN] [--hsplit-time RUN]
"""
import argparse
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from nilearn.image import resample_to_img
from scipy import stats

OUTPUTS = '/home/hfluhr/data/learninghabits/spm_outputs'
MASK = '/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym/habit_Guida2022_MNI152NLin2009cAsym.nii'
RADIUS_MM = 6


def inputs(spm_dir):
    """Subject -> first-level contrast image, from a second-level SPM.mat."""
    S = sio.loadmat(f'{spm_dir}/SPM.mat', squeeze_me=True, struct_as_record=False)['SPM']
    files = (str(p).strip().split(',')[0] for p in np.atleast_1d(S.xY.P))
    return {f[f.find('sub-'):f.find('sub-') + 6]: f for f in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hsplit', default='glm2_chosen_all_runs_concat_hsplit_scrubbed_2026-09-28-10-25')
    ap.add_argument('--hsplit-time', default='glm2_chosen_all_runs_concat_hsplit_time_scrubbed_2026-09-28-12-59')
    args = ap.parse_args()
    models = {'H within (H split)': f'{OUTPUTS}/{args.hsplit}/second-lvl/allruns/second_stimxhvalwithin_chosen',
              'H within (H split + time)': f'{OUTPUTS}/{args.hsplit_time}/second-lvl/allruns/second_stimxhvalwithin_chosen',
              'trial order (time model)': f'{OUTPUTS}/{args.hsplit_time}/second-lvl/allruns/second_stimxtrialwithin'}

    files = {k: inputs(v) for k, v in models.items()}
    subs = sorted(set.intersection(*(set(f) for f in files.values())))
    ref = nib.load(files['H within (H split)'][subs[0]])
    data = {k: np.stack([np.squeeze(nib.load(files[k][s]).get_fdata()) for s in subs]) for k in models}
    guida = np.squeeze(resample_to_img(nib.load(MASK), ref, interpolation='nearest').get_fdata()) > 0
    ijk = np.indices(guida.shape).reshape(3, -1).T
    xyz = (ref.affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T.reshape(guida.shape + (3,))
    sel = data['H within (H split)']  # the selection map never includes the time model
    valid = np.all(np.isfinite(sel), axis=0)

    out, peaks = {h: [] for h in ('left', 'right')}, {h: [] for h in ('left', 'right')}
    for i in range(len(subs)):
        others = np.delete(sel, i, axis=0)
        t = others.mean(0) / (others.std(0, ddof=1) / np.sqrt(len(others)))
        for hemi, side in (('left', xyz[..., 0] < 0), ('right', xyz[..., 0] > 0)):
            peak = np.unravel_index(np.argmax(np.where(guida & side & valid, t, -np.inf)), t.shape)
            centre = xyz[peak]
            peaks[hemi].append(tuple(np.round(centre).astype(int)))
            sphere = (np.linalg.norm(xyz - centre, axis=-1) <= RADIUS_MM) & valid
            out[hemi].append({k: np.nanmean(v[i][sphere]) for k, v in data.items()})

    print(f'n = {len(subs)}; ROI = {RADIUS_MM} mm sphere at the leave-one-out peak of within-run H (no time) in the Guida mask')
    for hemi in ('left', 'right'):
        df = pd.DataFrame(out[hemi], index=subs)
        print(f'\n== {hemi} DLS; peaks across folds: ' + ', '.join(f'{k} x{v}' for k, v in Counter(peaks[hemi]).most_common(4)))
        for c in df:
            x = df[c]
            t, p = stats.ttest_1samp(x, 0)
            se = x.std() / np.sqrt(len(x))
            print(f'  {c:28s} mean {x.mean():+.4f}  SE {se:.4f}  95% CI [{x.mean() - 1.96 * se:+.4f}, {x.mean() + 1.96 * se:+.4f}]  t = {t:+.2f}  p = {p:.4f}')
        a, b = df['H within (H split)'], df['H within (H split + time)']
        t, p = stats.ttest_rel(a, b)
        print(f'  ratio of means (with time / without) {b.mean() / a.mean():.2f}; paired difference t = {t:+.2f}, p = {p:.4f}; r = {a.corr(b):.2f}')


if __name__ == '__main__':
    main()
