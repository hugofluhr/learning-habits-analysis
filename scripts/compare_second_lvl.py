"""Compare re-run second levels with the old ones, and check the claims in scripts/second_lvl_claims.csv.

Reads the peak tables written by matlab/second_lvl/svc_report.m (one CSV per run, <svc-dir>/<run>.csv) and the spmT
images, and writes a Markdown report that lists only what changed. A region "survives" at peak level if any local
maximum has peak p(FWE) < 0.05, and at cluster level if any cluster has cluster p(FWE) < 0.05. A result is flagged when
it flips at either level; peak and cluster disagreeing is marked separately.

Usage (see scripts/submit_compare_second_lvl.sh):
    python scripts/compare_second_lvl.py --svc-dir DIR --old-root DIR --new-root DIR \
        --pair glm2_all_runs OLD_RUN NEW_RUN [--pair ...] --claims scripts/second_lvl_claims.csv --out report.md
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

ALPHA = 0.05
MATCH_MM = 8.0  # peaks closer than this are the same peak
DT_FLAG = 0.5  # flag a matched significant peak whose t changed by more than this
MASK_DIR = '/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym'
MASKS = {
    'striatum_bartra': 'striatum_bartra2013_MNI152NLin2009cAsym.nii',
    'vmpfc_bartra': 'vmpfc_bartra2013_MNI152NLin2009cAsym.nii',
    'guida': 'habit_Guida2022_MNI152NLin2009cAsym.nii',
    'putamen_aal': 'putamen_AAL_MNI152NLin2009cAsym.nii',
    'motor_hmat': 'motor_HMAT_MNI152NLin2009cAsym.nii',
    'm1_hmat': 'motor_M1only_HMAT_MNI152NLin2009cAsym.nii',
    'premotor_hmat': 'premotor_HMAT_MNI152NLin2009cAsym.nii',
    'parietal_aal': 'parietal_AAL_MNI152NLin2009cAsym.nii',
}


def contrast_key(model_path):
    """Second-level folder relative to second-lvl/, lowercased, with the "allruns/" level dropped."""
    key = model_path.lower().replace('\\', '/')
    return key[len('allruns/'):] if key.startswith('allruns/') else key


def load_svc(svc_dir, run):
    d = pd.read_csv(os.path.join(svc_dir, f'{run}.csv'))
    d['key'] = d['model'].map(contrast_key)
    return d[d['con_idx'] == 1]


def status(rows):
    """Significance of one region: peak level, cluster level, and the strongest peak."""
    rows = rows.dropna(subset=['peak_t'])
    peak = bool((rows['peak_p_fwe'] < ALPHA).any())
    clu = bool((rows['cluster_p_fwe'] < ALPHA).any())
    top = rows.loc[rows['peak_t'].idxmax()] if len(rows) else None
    return peak, clu, top


def sig_peaks(rows):
    rows = rows.dropna(subset=['peak_t'])
    return rows[(rows['peak_p_fwe'] < ALPHA) | (rows['cluster_p_fwe'] < ALPHA)]


def nearest(rows, xyz):
    rows = rows.dropna(subset=['peak_t'])
    if not len(rows):
        return None, np.inf
    d = np.sqrt(((rows[['x', 'y', 'z']].to_numpy(float) - np.asarray(xyz, float)) ** 2).sum(axis=1))
    i = int(np.argmin(d))
    return rows.iloc[i], d[i]


def fmt_sig(peak, clu):
    s = {(True, True): 'yes', (False, False): 'no', (True, False): 'peak only', (False, True): 'cluster only'}
    return s[(peak, clu)]


def fmt_peak(r):
    if r is None:
        return '–'
    return (f"t={r['peak_t']:.2f} ({r['x']:.0f}, {r['y']:.0f}, {r['z']:.0f}) k={r['cluster_k']:.0f} "
            f"p_peak={r['peak_p_fwe']:.3f} p_clu={r['cluster_p_fwe']:.3f}")


def compare_region(old_rows, new_rows):
    """Return (flags, details) for one contrast x region; flags is empty when nothing changed."""
    op, oc, otop = status(old_rows)
    np_, nc, ntop = status(new_rows)
    flags = []
    if op != np_:
        flags.append(f"peak-level {'lost' if op else 'gained'}")
    if oc != nc:
        flags.append(f"cluster-level {'lost' if oc else 'gained'}")
    details = []
    for _, r in sig_peaks(old_rows).iterrows():
        m, d = nearest(new_rows, (r['x'], r['y'], r['z']))
        if m is None or d > MATCH_MM:
            flags.append(f"peak ({r['x']:.0f}, {r['y']:.0f}, {r['z']:.0f}) gone")
        elif abs(m['peak_t'] - r['peak_t']) > DT_FLAG:
            flags.append(f"peak ({r['x']:.0f}, {r['y']:.0f}, {r['z']:.0f}) t {r['peak_t']:.2f} -> {m['peak_t']:.2f}")
    for _, r in sig_peaks(new_rows).iterrows():
        m, d = nearest(old_rows, (r['x'], r['y'], r['z']))
        if m is None or d > MATCH_MM:
            flags.append(f"new peak ({r['x']:.0f}, {r['y']:.0f}, {r['z']:.0f}) t={r['peak_t']:.2f}")
    details = dict(old=fmt_sig(op, oc), new=fmt_sig(np_, nc), old_top=fmt_peak(otop), new_top=fmt_peak(ntop),
                   disagree_new=(np_ != nc))
    return list(dict.fromkeys(flags)), details


def check_claim(c, rows):
    """Does the claim hold in these rows? Returns (holds, note), judged at the claim's level."""
    level_col = 'cluster_p_fwe' if c['level'] == 'cluster' else 'peak_p_fwe'
    other_col = 'peak_p_fwe' if c['level'] == 'cluster' else 'cluster_p_fwe'
    rows = rows.dropna(subset=['peak_t'])
    if c['expect'] == 'none':
        sig = rows[rows[level_col] < ALPHA]
        other = rows[rows[other_col] < ALPHA]
        note = 'nothing' if not len(sig) else f"{len(sig)} significant peak(s), strongest {fmt_peak(sig.loc[sig['peak_t'].idxmax()])}"
        if not len(sig) and len(other):
            note += f"; but {c['level'] == 'cluster' and 'peak' or 'cluster'} level has {fmt_peak(other.loc[other['peak_t'].idxmax()])}"
        return not len(sig), note
    if pd.notna(c['x']):
        m, d = nearest(rows, (c['x'], c['y'], c['z']))
        if m is None or d > MATCH_MM:
            return False, 'no peak within 8 mm'
        return bool(m[level_col] < ALPHA), f"{fmt_peak(m)} ({d:.1f} mm away)"
    sig = rows[rows[level_col] < ALPHA]
    if not len(sig):
        top = rows.loc[rows['peak_t'].idxmax()] if len(rows) else None
        return False, f"nothing significant; strongest {fmt_peak(top)}"
    return True, fmt_peak(sig.loc[sig['peak_t'].idxmax()])


def tmap_corr(old_dir, new_dir, rois):
    """Pearson r between old and new spmT_0001 within both analysis masks, whole brain and per ROI."""
    try:
        ot, nt = nib.load(os.path.join(old_dir, 'spmT_0001.nii')), nib.load(os.path.join(new_dir, 'spmT_0001.nii'))
        om, nm = nib.load(os.path.join(old_dir, 'mask.nii')), nib.load(os.path.join(new_dir, 'mask.nii'))
    except FileNotFoundError:
        return {}
    # np.squeeze: some images are stored with a trailing singleton 4th dimension
    o, n = np.squeeze(ot.get_fdata()), np.squeeze(nt.get_fdata())
    both = (np.squeeze(om.get_fdata()) > 0) & (np.squeeze(nm.get_fdata()) > 0) & np.isfinite(o) & np.isfinite(n)
    out = {'wholebrain': np.corrcoef(o[both], n[both])[0, 1]}
    for name, img in rois.items():
        r = np.squeeze(resample_to_img(img, ot, interpolation='nearest').get_fdata()) > 0
        sel = both & r
        out[name] = np.corrcoef(o[sel], n[sel])[0, 1] if sel.sum() > 2 else np.nan
    return out


def subjects(d):
    s = d['subjects'].dropna()
    return set(s.iloc[0].split(';')) if len(s) else set()


def peak_dict(r):
    if r is None:
        return None
    return {k: (None if pd.isna(r[k]) else float(r[k])) for k in ['peak_t', 'x', 'y', 'z', 'cluster_k', 'peak_p_fwe', 'cluster_p_fwe']}


def run_dir(root_old, root_new, run, model_path):
    """Old runs live in reference/second_lvl/<run>/; a run can also be a re-run (<new root>/<run>/second-lvl/)."""
    old_style = os.path.join(root_old, run)
    base = old_style if os.path.isdir(old_style) else os.path.join(root_new, run, 'second-lvl')
    return os.path.join(base, *model_path.split('/'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--svc-dir', required=True)
    ap.add_argument('--old-root', required=True, help='folder holding <old run>/ second levels')
    ap.add_argument('--new-root', required=True, help='folder holding <new run>/second-lvl/')
    ap.add_argument('--pair', nargs=3, action='append', required=True, metavar=('MODEL', 'OLD_RUN', 'NEW_RUN'),
                    help='MODEL names the claims to check; MODEL=CLAIMS_MODEL uses another model\'s claims (e.g. the concat model)')
    ap.add_argument('--claims', required=True)
    ap.add_argument('--out', required=True, help='Markdown report; a .json with the same content is written next to it')
    args = ap.parse_args()

    claims = pd.read_csv(args.claims)
    rois = {k: nib.load(os.path.join(MASK_DIR, v)) for k, v in MASKS.items()}
    report = {'alpha': ALPHA, 'match_mm': MATCH_MM, 'dt_flag': DT_FLAG, 'regions': ['wholebrain'] + list(MASKS),
              'runs': [], 'claims': [], 'regions_compared': [], 'tmap_corr': []}

    for label, old_run, new_run in args.pair:
        model, _, claims_model = label.partition('=')
        claims_model = claims_model or model
        old, new = load_svc(args.svc_dir, old_run), load_svc(args.svc_dir, new_run)
        so, sn = subjects(old), subjects(new)
        keys = sorted(set(old['key']) & set(new['key']))
        report['runs'].append({'model': model, 'old_run': old_run, 'new_run': new_run, 'n_old': len(so), 'n_new': len(sn),
                               'added': sorted(sn - so), 'removed': sorted(so - sn), 'contrasts_compared': keys,
                               'only_new': sorted(set(new['key']) - set(old['key'])), 'only_old': sorted(set(old['key']) - set(new['key']))})

        for _, c in claims[claims['model'] == claims_model].iterrows():
            o_ok, o_note = check_claim(c, old[(old['key'] == c['contrast']) & (old['region'] == c['region'])])
            n_ok, n_note = check_claim(c, new[(new['key'] == c['contrast']) & (new['region'] == c['region'])])
            report['claims'].append({'model': model, 'contrast': c['contrast'], 'region': c['region'], 'expect': c['expect'],
                                     'xyz': None if pd.isna(c['x']) else [float(c['x']), float(c['y']), float(c['z'])],
                                     't': None if pd.isna(c['t']) else float(c['t']), 'level': c['level'], 'source': c['source'],
                                     'old_ok': o_ok, 'old_note': o_note, 'new_ok': n_ok, 'new_note': n_note})

        for key in keys:
            for region in report['regions']:
                o = old[(old['key'] == key) & (old['region'] == region)]
                n = new[(new['key'] == key) & (new['region'] == region)]
                flags, det = compare_region(o, n)
                (op, oc, otop), (np_, nc, ntop) = status(o), status(n)
                report['regions_compared'].append({'model': model, 'contrast': key, 'region': region, 'flags': flags,
                                                   'old_peak': op, 'old_cluster': oc, 'new_peak': np_, 'new_cluster': nc,
                                                   'old_top': peak_dict(otop), 'new_top': peak_dict(ntop)})
            r = tmap_corr(run_dir(args.old_root, args.new_root, old_run, old[old['key'] == key]['model'].iloc[0]),
                          run_dir(args.old_root, args.new_root, new_run, new[new['key'] == key]['model'].iloc[0]), rois)
            if r:
                report['tmap_corr'].append({'model': model, 'contrast': key,
                                            **{k: (None if np.isnan(v) else float(v)) for k, v in r.items()}})

    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(report, f, indent=1)
    write_markdown(report, args.out)
    print(f'Wrote {args.out} and its .json')


def write_markdown(rep, path):
    def top(p):
        return fmt_peak(None if p is None else pd.Series(p))

    def sig(pk, cl):
        return fmt_sig(pk, cl)

    out = ['# Second-level comparison: re-run vs old', '',
           f"Significance: FWE p < {rep['alpha']} (whole brain, or small-volume corrected within the ROI), cluster-forming p < 0.001. "
           f"Peaks within {rep['match_mm']:.0f} mm are treated as the same peak; matched significant peaks are flagged when |Δt| > {rep['dt_flag']}.", '',
           '## Runs', '', '| Model | Old | New | Subjects added | Subjects removed | Contrasts compared | Contrasts only in new |', '|---|---|---|---|---|---|---|']
    for r in rep['runs']:
        out.append(f"| {r['model']} | `{r['old_run']}` (N={r['n_old']}) | `{r['new_run']}` (N={r['n_new']}) | {', '.join(r['added']) or '–'} | "
                   f"{', '.join(r['removed']) or '–'} | {len(r['contrasts_compared'])} | {len(r['only_new'])} |")
    out += ['', '## Claims', '', '✓/✗ at the level the source used (the manuscript reports cluster-level FWE).', '',
            '| Model | Contrast | Region | Expected | Old run | Re-run | Source |', '|---|---|---|---|---|---|---|']
    for c in rep['claims']:
        where = f" at ({c['xyz'][0]:.0f}, {c['xyz'][1]:.0f}, {c['xyz'][2]:.0f})" if c['xyz'] else ''
        out.append(f"| {c['model']} | {c['contrast']} | {c['region']} | {c['expect']}{where} | {'✓' if c['old_ok'] else '✗'} {c['old_note']} | "
                   f"{'✓' if c['new_ok'] else '✗'} {c['new_note']} | {c['source']} |")
    out += ['', '## What changed', '', '| Model | Contrast | Region | Survives old → new | Flags | Old strongest peak | New strongest peak |', '|---|---|---|---|---|---|---|']
    for r in rep['regions_compared']:
        if r['flags']:
            out.append(f"| {r['model']} | {r['contrast']} | {r['region']} | {sig(r['old_peak'], r['old_cluster'])} → {sig(r['new_peak'], r['new_cluster'])} | "
                       f"{'; '.join(r['flags'])} | {top(r['old_top'])} | {top(r['new_top'])} |")
    out += ['', '## t-map correlation, old vs re-run', '', '| Model | Contrast | ' + ' | '.join(rep['regions']) + ' |', '|---|---|' + '---|' * len(rep['regions'])]
    for r in rep['tmap_corr']:
        out.append(f"| {r['model']} | {r['contrast']} | " + ' | '.join('–' if r.get(k) is None else f"{r[k]:.3f}" for k in rep['regions']) + ' |')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
