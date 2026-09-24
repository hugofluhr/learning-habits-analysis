#!/usr/bin/env python
"""Compare task-regressor VIFs: per-session SPM design vs concatenated-design SPM.

Both designs are read from the CSVs written by ``matlab/export_spm_dms.m``.

* per-session model: ``utils.vif.compute_vifs`` as used elsewhere, i.e. one VIF
  per task regressor per session (regressed on that session's own columns).
* concat model: one session for all three runs, so every column of the design
  (task, confounds and the three run constants) is used for the VIF.

Example:
    python scripts/compare_concat_vifs.py <concat_dir> <session_dir> sub-01
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.analysis import est_vifs  # noqa: E402
from utils.vif import compute_vifs, load_design_matrices, shorten_pm  # noqa: E402


def concat_vifs(dm):
    """VIFs of the task regressors of a single-session (concatenated) design."""
    task_cols = [c for c in dm.columns
                 if not re.search(r'\bR\d+$', c) and 'constant' not in c.lower() and dm[c].nunique() > 1]
    vifs = est_vifs(dm, task_cols)
    return {k.replace('Sn(1) ', ''): v for k, v in vifs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('concat_dir')
    ap.add_argument('session_dir')
    ap.add_argument('sub')
    args = ap.parse_args()

    dm_concat = load_design_matrices(args.concat_dir)[args.sub]
    dm_sess = {args.sub: load_design_matrices(args.session_dir)[args.sub]}

    per_session = compute_vifs(dm_sess).loc[args.sub]           # index: session
    per_session.index = [f'per-session {s}' for s in per_session.index]
    concat = pd.DataFrame([concat_vifs(dm_concat)], index=['concat'])

    table = pd.concat([per_session, concat]).T
    table.index = [shorten_pm(i) for i in table.index]
    pd.set_option('display.width', 160)
    print(table.round(2).to_string())


if __name__ == '__main__':
    main()
