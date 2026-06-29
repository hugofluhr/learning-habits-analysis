"""Reusable VIF computation and report building for SPM "all-runs" GLMs.

This module factors out the per-model logic that used to be hand-copied into
``notebooks/glm/inspect_VIFs_allruns.ipynb`` so it can be driven from a CLI
(``scripts/vif_report.py``) given a single GLM outputs directory.

Scope: "all-runs" style GLM directories laid out as ``<glm_dir>/sub-XX/SPM.mat``
with sessions encoded as ``Sn(1) ``, ``Sn(2) `` ... column prefixes. Design
matrices are read from ``sub-XX/sub-XX_design_matrix.csv`` (+ ``_column_names.txt``),
as written by ``matlab/export_spm_dms.m``.
"""

import os
import re

import pandas as pd

from .analysis import est_vifs


# --------------------------------------------------------------------------- #
# Loading / computation
# --------------------------------------------------------------------------- #
def load_design_matrices(model_dir):
    """Load design matrices from ``model_dir/sub-XX/sub-XX_design_matrix.csv``.

    Returns a dict ``{sub_id: DataFrame}`` (columns named from the matching
    ``sub-XX_column_names.txt``). Subjects missing either file are skipped.
    """
    subjects = sorted(d for d in os.listdir(model_dir) if d.startswith('sub-'))
    DMs = {}
    for sub in subjects:
        dm_path = os.path.join(model_dir, sub, f'{sub}_design_matrix.csv')
        col_path = os.path.join(model_dir, sub, f'{sub}_column_names.txt')
        if not (os.path.exists(dm_path) and os.path.exists(col_path)):
            continue
        with open(col_path) as f:
            col_names = [l.strip() for l in f]
        DMs[sub] = pd.read_csv(dm_path, names=col_names, header=None)
    return DMs


def detect_n_sessions(DMs):
    """Largest ``Sn(N)`` index present across all design matrices (>=1)."""
    max_sn = 0
    for dm in DMs.values():
        for c in dm.columns:
            m = re.match(r'Sn\((\d+)\)', c)
            if m:
                max_sn = max(max_sn, int(m.group(1)))
    return max_sn


def compute_vifs(DMs, n_sessions=None):
    """Compute per-subject, per-session VIFs for task regressors.

    Task regressors = everything that is not a motion/nuisance ``R\\d+`` column
    or 'constant', and not constant-valued (e.g. all-zero ``nresp_screen``
    columns for subjects with no non-response trials).

    ``n_sessions`` is auto-detected from the ``Sn(N)`` prefixes when None.
    Returns a DataFrame indexed by ``(subject, session)``.
    """
    if n_sessions is None:
        n_sessions = detect_n_sessions(DMs)

    records = []
    for sub, dm in DMs.items():
        for sn in range(1, n_sessions + 1):
            prefix = f'Sn({sn}) '
            sess_cols = [c for c in dm.columns if c.startswith(prefix)]
            if not sess_cols:
                continue
            dm_sess = dm[sess_cols]

            task_cols = [
                c for c in sess_cols
                if not re.search(r'\bR\d+$', c)
                and 'constant' not in c.lower()
                and dm_sess[c].nunique() > 1
            ]
            if not task_cols:
                continue

            try:
                vifs = est_vifs(dm_sess, task_cols)
            except Exception as e:  # pragma: no cover - diagnostic only
                print(f'  Warning: VIF failed for {sub} Sn({sn}): {e}')
                continue

            vifs_clean = {k.replace(prefix, ''): v for k, v in vifs.items()}
            vifs_clean['subject'] = sub
            vifs_clean['session'] = f'Sn({sn})'
            records.append(vifs_clean)

    return pd.DataFrame(records).set_index(['subject', 'session'])


def shorten_pm(name):
    """Shorten an SPM regressor name for display (drop ``*bf(1)``, ``^1`` etc.)."""
    name = re.sub(r'\*bf\(\d+\)', '', name)
    name = re.sub(r'\^\d+', '', name)
    name = name.replace('first_stim', '1st').replace('second_stim', '2nd')
    return name


def pm_columns(vifs):
    """Parametric-modulator columns (interaction terms contain an 'x')."""
    return [c for c in vifs.columns if 'x' in c]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def summary_table(vifs):
    """``describe()`` over all task regressors, rounded."""
    return vifs.describe().round(2)


def pm_vif_max_per_session(vifs):
    """Max VIF per session over parametric-modulator columns (None if no PMs)."""
    pm_cols = pm_columns(vifs)
    if not pm_cols:
        return None
    return vifs[pm_cols].groupby('session').max().round(1)


def exclusion_counts(vifs, thresholds=(10, 5), regressor_filter='x'):
    """Subject exclusion counts at VIF thresholds for the selected regressors.

    A subject is flagged for a session if *any* selected regressor exceeds the
    threshold that session; flagged overall if it exceeds in *any* session
    (max across sessions). Returns a DataFrame indexed by threshold with one
    column per session plus ``any_session``, ``n_total`` and ``pct_excluded``.
    """
    sel_cols = [c for c in vifs.columns if regressor_filter in c]
    if not sel_cols:
        return None

    sessions = list(vifs.index.get_level_values('session').unique())
    n_total = vifs.index.get_level_values('subject').nunique()

    rows = {}
    for thr in thresholds:
        flagged = vifs[sel_cols].gt(thr).any(axis=1)          # bool per (sub, sess)
        per_sess = flagged.groupby(level='session').sum()
        worst = vifs[sel_cols].groupby(level='subject').max().gt(thr).any(axis=1)
        n_excl = int(worst.sum())
        row = {s: int(per_sess.get(s, 0)) for s in sessions}
        row['any_session'] = n_excl
        row['n_total'] = n_total
        row['pct_excluded'] = round(100 * n_excl / n_total, 1) if n_total else 0.0
        rows[f'VIF > {thr}'] = row

    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def vif_boxplot(vifs, title, pm_only=False):
    """Boxplot of VIF by session (hue = regressor), log y, 5/10 reference lines.

    Returns the matplotlib Figure (caller saves/closes).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = pm_columns(vifs) if pm_only else list(vifs.columns)
    if not cols:
        return None

    melted = vifs[cols].reset_index().melt(
        id_vars=['subject', 'session'], var_name='regressor', value_name='VIF'
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.boxplot(data=melted, x='session', y='VIF', hue='regressor',
                palette='tab10', ax=ax)
    ax.axhline(5, color='orange', linestyle='--', linewidth=0.8)
    ax.axhline(10, color='red', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.legend(title='regressor', fontsize=8, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    return fig
