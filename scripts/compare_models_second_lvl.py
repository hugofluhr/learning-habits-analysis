"""Compare one contrast between two models, at every second-level folder they share (allruns, session-0X, session-02-03).

For each level: t-map correlation between the models (whole brain within both masks, and per ROI), and the clusters that
survive FWE p < 0.05 at peak or cluster level in each ROI and at whole brain (cluster p < 0.05), side by side. Reads the
peak tables written by matlab/second_lvl/svc_report.m and the spmT images.

Usage (on the cluster):
    python scripts/compare_models_second_lvl.py --a RUN_A CONTRAST_A --b RUN_B CONTRAST_B --out report.md
e.g. --a glm2_mf_chosenval_2026-09-24-14-56 second_stimxchosenval --b glm2_chosen_all_runs_scrubbed_2026-09-24-14-56 second_stimxqval_chosen
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

OUTPUTS = '/home/hfluhr/data/learninghabits/spm_outputs'
SVC_DIR = f'{OUTPUTS}/compare_second_lvl_2026-09-24-14-56/svc'
MASK_DIR = '/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym'
MASKS = {
    'striatum_bartra': ('VS', 'striatum_bartra2013_MNI152NLin2009cAsym.nii'),
    'vmpfc_bartra': ('vmPFC', 'vmpfc_bartra2013_MNI152NLin2009cAsym.nii'),
    'guida': ('DLS', 'habit_Guida2022_MNI152NLin2009cAsym.nii'),
    'putamen_aal': ('putamen', 'putamen_AAL_MNI152NLin2009cAsym.nii'),
    'motor_hmat': ('motor', 'motor_HMAT_MNI152NLin2009cAsym.nii'),
    'm1_hmat': ('M1', 'motor_M1only_HMAT_MNI152NLin2009cAsym.nii'),
    'premotor_hmat': ('premotor', 'premotor_HMAT_MNI152NLin2009cAsym.nii'),
    'parietal_aal': ('parietal', 'parietal_AAL_MNI152NLin2009cAsym.nii'),
}
LEVELS = ['allruns', 'session-01', 'session-02', 'session-03', 'session-02-03']


def clusters(svc, level, contrast, region):
    d = svc[(svc['model'] == f'{level}/{contrast}') & (svc['region'] == region)].dropna(subset=['peak_t'])
    d = d.drop_duplicates(['cluster_k', 'cluster_p_fwe'])  # first row of a cluster is its strongest peak
    keep = (d['cluster_p_fwe'] < .05) if region == 'wholebrain' else ((d['cluster_p_fwe'] < .05) | (d['peak_p_fwe'] < .05))
    return d[keep]


def fmt(d):
    if not len(d):
        return '–'
    p = lambda v: '<.001' if v < .001 else f'{v:.3f}'.lstrip('0')
    return '<br>'.join(f"({r.x:.0f}, {r.y:.0f}, {r.z:.0f}) k {r.cluster_k:.0f}, t {r.peak_t:.2f}, p<sub>clu</sub> {p(r.cluster_p_fwe)}, p<sub>pk</sub> {p(r.peak_p_fwe)}"
                       for r in d.itertuples())


def tmap_corr(dir_a, dir_b, rois):
    ta, tb = nib.load(f'{dir_a}/spmT_0001.nii'), nib.load(f'{dir_b}/spmT_0001.nii')
    a, b = np.squeeze(ta.get_fdata()), np.squeeze(tb.get_fdata())
    both = (np.squeeze(nib.load(f'{dir_a}/mask.nii').get_fdata()) > 0) & (np.squeeze(nib.load(f'{dir_b}/mask.nii').get_fdata()) > 0) & np.isfinite(a) & np.isfinite(b)
    out = {'whole brain': np.corrcoef(a[both], b[both])[0, 1]}
    for key, (label, img) in rois.items():
        sel = both & (np.squeeze(resample_to_img(img, ta, interpolation='nearest').get_fdata()) > 0)
        out[label] = np.corrcoef(a[sel], b[sel])[0, 1] if sel.sum() > 2 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', nargs=2, required=True, metavar=('RUN', 'CONTRAST'))
    ap.add_argument('--b', nargs=2, required=True, metavar=('RUN', 'CONTRAST'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    (run_a, con_a), (run_b, con_b) = args.a, args.b
    svc_a, svc_b = pd.read_csv(f'{SVC_DIR}/{run_a}.csv'), pd.read_csv(f'{SVC_DIR}/{run_b}.csv')
    rois = {k: (lab, nib.load(os.path.join(MASK_DIR, f))) for k, (lab, f) in MASKS.items()}

    out = [f'# {con_a} ({run_a}) vs {con_b} ({run_b})', '',
           'A = first model, B = second. Clusters surviving FWE p < 0.05 (peak or cluster level; whole brain: cluster level), '
           'cluster-forming p < 0.001, one row per cluster at its strongest peak.', '']
    corr_rows = []
    for level in LEVELS:
        da, db = f'{OUTPUTS}/{run_a}/second-lvl/{level}/{con_a}', f'{OUTPUTS}/{run_b}/second-lvl/{level}/{con_b}'
        if not (os.path.isdir(da) and os.path.isdir(db)):
            continue
        r = tmap_corr(da, db, rois)
        corr_rows.append(f"| {level} | " + ' | '.join(f'{v:.2f}' for v in r.values()) + ' |')
        out += [f'## {level}', '', '| Region | A | B |', '|---|---|---|']
        for region in list(MASKS) + ['wholebrain']:
            ca, cb = clusters(svc_a, level, con_a, region), clusters(svc_b, level, con_b, region)
            if len(ca) or len(cb):
                out.append(f"| {MASKS.get(region, ('whole brain',))[0]} | {fmt(ca)} | {fmt(cb)} |")
        out.append('')
    out = out[:4] + ['## t-map correlation A vs B', '', '| Level | whole brain | ' + ' | '.join(l for l, _ in MASKS.values()) + ' |',
                     '|---|---|' + '---|' * len(MASKS)] + corr_rows + [''] + out[4:]
    with open(args.out, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
