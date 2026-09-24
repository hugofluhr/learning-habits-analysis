#!/usr/bin/env python
"""Check that a concatenated-design SPM first level reproduces the per-session one.

Both designs are read from the CSVs written by ``matlab/export_spm_dms.m``
(``sub-XX_design_matrix.csv`` + ``sub-XX_column_names.txt``). The per-session
design has ``Sn(k) ...`` columns that are non-zero only in run k's rows, so:

  * conditions without a pmod: the concat column must equal the sum of the
    per-session columns of that condition;
  * pmod columns: they differ only because SPM mean-centres the pmod over the
    whole session instead of per run, so
    ``concat - sum_k session_k = sum_k c_k * parent_k`` for some constants c_k;
  * confounds: concat ``Sn(1) R{off+j}`` must equal per-session ``Sn(k) R{j}`` up to
    a per-run constant (SPM mean-centres user regressors over the whole session, so
    the concat columns are offset; the run constants absorb that);
  * constants: concat ``Sn(k) constant`` must equal per-session ``Sn(k) constant``.

HRF overhang across run borders is expected to differ in the first volumes of
run 2 and 3, so differences are also reported with those rows excluded.

Example:
    python scripts/check_concat_design.py <concat_dir> <session_dir> sub-01
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

TR = 2.33384
OVERHANG_S = 32  # rows within this many seconds after a run border are reported separately


def load(model_dir, sub):
    with open(f'{model_dir}/{sub}_column_names.txt') as f:
        names = [l.strip() for l in f]
    return pd.read_csv(f'{model_dir}/{sub}_design_matrix.csv', names=names, header=None)


def summarise(label, diff, borders, overhang_rows):
    """One line: max abs diff overall, away from run borders, and where the overall max sits."""
    keep = np.ones(len(diff), bool)
    for b in borders[1:]:
        keep[b:b + overhang_rows] = False
    worst = int(np.abs(diff).argmax())
    return (f'{label:<44s} max|diff|={np.abs(diff).max():.2e}  '
            f'excl. border={np.abs(diff[keep]).max():.2e}  (worst at row {worst})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('concat_dir')
    ap.add_argument('session_dir')
    ap.add_argument('sub')
    args = ap.parse_args()

    C = load(args.concat_dir, args.sub)
    S = load(args.session_dir, args.sub)
    print(f'concat: {C.shape}   per-session: {S.shape}')

    # run borders, from the per-run constants of the concat design
    consts = [c for c in C.columns if re.fullmatch(r'Sn\(\d+\) constant', c)]
    nscan = [int((C[c] != 0).sum()) for c in consts]
    borders = [0] + list(np.cumsum(nscan)[:-1])
    n_sess = len(nscan)
    overhang_rows = int(np.ceil(OVERHANG_S / TR))
    print(f'runs: nscan={nscan}, borders={borders}\n')

    def cols_of(df, stem):
        """Per-session columns for a condition/pmod stem, e.g. 'response*bf(1)'; {k: name}."""
        out = {}
        for c in df.columns:
            m = re.fullmatch(r'Sn\((\d+)\) ' + re.escape(stem), c)
            if m:
                out[int(m.group(1))] = c
        return out

    # --- conditions without pmod
    print('== conditions without pmod: concat vs sum of per-session columns ==')
    for stem in ['first_stim*bf(1)', 'second_stim*bf(1)', 'response*bf(1)', 'purple_frame*bf(1)',
                 'points_feedback*bf(1)', 'nresp_screen*bf(1)']:
        cc = cols_of(C, stem)
        if not cc:
            print(f'{stem:<44s} absent from concat design')
            continue
        ss = cols_of(S, stem)
        diff = C[cc[1]].to_numpy() - S[list(ss.values())].sum(axis=1).to_numpy()
        print(summarise(stem, diff, borders, overhang_rows))

    # --- pmods
    print('\n== pmod columns: concat - sum(per-session) explained by per-session parent columns ==')
    for parent, pm in [('first_stim', 'Qval'), ('first_stim', 'Hval'), ('second_stim', 'Qval'), ('second_stim', 'Hval')]:
        stem = f'{parent}x{pm}^1*bf(1)'
        cc = cols_of(C, stem)
        ss = cols_of(S, stem)
        diff = C[cc[1]].to_numpy() - S[list(ss.values())].sum(axis=1).to_numpy()
        par = cols_of(S, f'{parent}*bf(1)')
        P = S[[par[k] for k in sorted(par)]].to_numpy()
        coef, *_ = np.linalg.lstsq(P, diff, rcond=None)
        resid = diff - P @ coef
        print(summarise(f'{stem} (raw diff)', diff, borders, overhang_rows))
        print(summarise(f'{stem} (after parent fit)', resid, borders, overhang_rows)
              + f'   fitted per-run offsets={np.round(coef, 3)}')

    # --- confounds
    print('\n== confounds: concat R columns vs per-session R columns (block by run) ==')
    K = C[consts].to_numpy()  # run constants: differences inside their span do not change the fit
    r_concat = [c for c in C.columns if re.fullmatch(r'Sn\(1\) R\d+', c)]
    off = 0
    for k in range(1, n_sess + 1):
        r_k = [c for c in S.columns if re.fullmatch(rf'Sn\({k}\) R\d+', c)]
        a = C[r_concat[off:off + len(r_k)]].to_numpy()
        b = S[r_k].to_numpy()
        d = a - b
        d_proj = d - K @ np.linalg.lstsq(K, d, rcond=None)[0]
        print(f'run {k}: {len(r_k)} cols, max|diff|={np.abs(d).max():.2e}, '
              f'after projecting out run constants={np.abs(d_proj).max():.2e}')
        off += len(r_k)
    print(f'concat has {len(r_concat)} R columns, per-session has {off}')

    # --- constants
    print('\n== constants ==')
    for k in range(1, n_sess + 1):
        d = C[f'Sn({k}) constant'].to_numpy() - S[f'Sn({k}) constant'].to_numpy()
        print(f'Sn({k}) constant: max|diff|={np.abs(d).max():.2e}')


if __name__ == '__main__':
    sys.exit(main())
