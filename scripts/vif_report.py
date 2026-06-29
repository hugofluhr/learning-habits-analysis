#!/usr/bin/env python
"""Generate a self-contained VIF report (HTML) for a single SPM "all-runs" GLM.

Given a GLM outputs directory (``<glm_dir>/sub-XX/SPM.mat``), this:
  1. ensures per-subject design-matrix CSVs exist (running the MATLAB exporter
     ``export_spm_dms`` when any are missing),
  2. computes per-subject / per-session VIFs for the task regressors,
  3. writes tables (CSV) and figures (PNG) and assembles them into
     ``<out>/report.html``.

Examples
--------
    # CSVs already present (fast path)
    python scripts/vif_report.py /mnt/.../glm2_merged_stim_... --skip-export

    # full path: export design matrices via MATLAB, then report
    python scripts/vif_report.py /mnt/.../glm2_some_model_without_csvs
"""

import argparse
import base64
import datetime as _dt
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')  # headless; must precede pyplot import

sys.path.append('/home/ubuntu/repos/learning-habits-analysis')

from utils.vif import (  # noqa: E402
    compute_vifs,
    detect_n_sessions,
    exclusion_counts,
    load_design_matrices,
    pm_vif_max_per_session,
    summary_table,
    vif_boxplot,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_out_dir(glm_dir):
    """Default report location: the parallel ``spm_outputs/<model>`` directory.

    Raw GLMs live under ``.../spm_format/outputs/<model>`` while their derived
    exports (allruns/, session-*/, second-lvl/) live under the parallel
    ``.../spm_outputs/<model>``. Drop the VIF report there too. Falls back to
    ``<glm_dir>/vif_report`` when the path doesn't follow that convention.
    """
    norm = os.path.normpath(glm_dir)
    marker = os.path.join('spm_format', 'outputs')
    if marker in norm:
        return os.path.join(norm.replace(marker, 'spm_outputs'), 'vif_report')
    return os.path.join(glm_dir, 'vif_report')


# --------------------------------------------------------------------------- #
# Step 1 — ensure design-matrix CSVs exist
# --------------------------------------------------------------------------- #
def _subjects_missing_csv(glm_dir):
    """sub-XX dirs that contain an SPM.mat but no design_matrix.csv."""
    missing = []
    for sub in sorted(d for d in os.listdir(glm_dir) if d.startswith('sub-')):
        sub_dir = os.path.join(glm_dir, sub)
        if not os.path.isfile(os.path.join(sub_dir, 'SPM.mat')):
            continue
        if not os.path.isfile(os.path.join(sub_dir, f'{sub}_design_matrix.csv')):
            missing.append(sub)
    return missing


def ensure_csvs(glm_dir, matlab_module, spm_path, overwrite):
    """Run the MATLAB design-matrix exporter when CSVs are missing."""
    if not overwrite and not _subjects_missing_csv(glm_dir):
        print('Design-matrix CSVs already present — skipping MATLAB export.')
        return

    over = 'true' if overwrite else 'false'
    matlab_cmd = (
        f"addpath('{spm_path}'); addpath('{os.path.join(REPO_ROOT, 'matlab')}'); "
        f"export_spm_dms('{glm_dir}', 'Overwrite', {over}); exit"
    )
    shell_cmd = (
        'source /etc/profile.d/z00-lmod.sh && '
        f'module load {matlab_module} && '
        f'matlab -nodisplay -nosplash -nodesktop -r "{matlab_cmd}"'
    )
    print(f'Exporting design matrices via MATLAB ({matlab_module}) ...')
    subprocess.run(['bash', '-lc', shell_cmd], check=True)


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }
h1 { border-bottom: 2px solid #444; padding-bottom: .3rem; }
h2 { margin-top: 2rem; color: #333; }
table { border-collapse: collapse; margin: .5rem 0; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }
th { background: #f0f0f0; }
img { max-width: 100%; height: auto; margin: .5rem 0; }
.meta { color: #666; font-size: 13px; }
"""


def _img_tag(png_path):
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}" />'


def build_html(model_name, meta, tables, figure_paths, out_path):
    """Assemble report.html: metadata + tables (to_html) + embedded figures."""
    parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        f'<title>VIF report — {model_name}</title>',
        f'<style>{_CSS}</style></head><body>',
        f'<h1>VIF report — {model_name}</h1>',
        '<p class="meta">'
        + ' &nbsp;|&nbsp; '.join(f'{k}: {v}' for k, v in meta.items())
        + '</p>',
    ]
    for title, df in tables:
        parts.append(f'<h2>{title}</h2>')
        if df is None:
            parts.append('<p class="meta">(not applicable for this model)</p>')
        else:
            parts.append(df.to_html(border=0))
    for title, path in figure_paths:
        if path is None:
            continue
        parts.append(f'<h2>{title}</h2>')
        parts.append(_img_tag(path))
    parts.append('</body></html>')

    with open(out_path, 'w') as f:
        f.write('\n'.join(parts))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glm_dir', help='GLM outputs directory (contains sub-XX/SPM.mat)')
    p.add_argument('--out', default=None,
                   help='output directory (default: the parallel '
                        'spm_outputs/<model>/vif_report)')
    p.add_argument('--n-sessions', type=int, default=None,
                   help='number of sessions (default: auto-detect from Sn(N))')
    p.add_argument('--thresholds', type=int, nargs='+', default=[10, 5],
                   help='VIF thresholds for exclusion counts (default: 10 5)')
    p.add_argument('--exclusion-regressor', default='Hval',
                   help="substring selecting regressors that drive exclusion "
                        "counts (default: 'Hval', matching the notebook; use "
                        "'x' for all parametric modulators)")
    p.add_argument('--skip-export', action='store_true',
                   help='do not run the MATLAB export even if CSVs are missing')
    p.add_argument('--overwrite-export', action='store_true',
                   help='re-export design matrices even if CSVs already exist')
    p.add_argument('--matlab-module', default='matlab/r2023a')
    p.add_argument('--spm-path', default='/home/ubuntu/repos/spm12')
    args = p.parse_args()

    glm_dir = os.path.abspath(args.glm_dir)
    if not os.path.isdir(glm_dir):
        p.error(f'glm_dir not found: {glm_dir}')
    model_name = os.path.basename(glm_dir.rstrip('/'))
    out_dir = args.out or default_out_dir(glm_dir)
    tables_dir = os.path.join(out_dir, 'tables')
    figs_dir = os.path.join(out_dir, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # 1. CSVs
    if not args.skip_export:
        ensure_csvs(glm_dir, args.matlab_module, args.spm_path, args.overwrite_export)

    # 2. compute
    print('Loading design matrices ...')
    DMs = load_design_matrices(glm_dir)
    if not DMs:
        p.error(f'No design-matrix CSVs found under {glm_dir} '
                f'(run without --skip-export to export them).')
    n_sessions = args.n_sessions or detect_n_sessions(DMs)
    print(f'  {len(DMs)} subjects, {n_sessions} sessions')
    vifs = compute_vifs(DMs, n_sessions=n_sessions)
    print(f'  VIFs: {vifs.shape[0]} rows, {vifs.shape[1]} regressors')

    # 3. tables
    summary = summary_table(vifs)
    pm_max = pm_vif_max_per_session(vifs)
    excl = exclusion_counts(vifs, thresholds=tuple(args.thresholds),
                            regressor_filter=args.exclusion_regressor)

    vifs.round(3).to_csv(os.path.join(tables_dir, 'vifs_per_subject_session.csv'))
    summary.to_csv(os.path.join(tables_dir, 'summary_describe.csv'))
    if pm_max is not None:
        pm_max.to_csv(os.path.join(tables_dir, 'pm_vif_max_per_session.csv'))
    if excl is not None:
        excl.to_csv(os.path.join(tables_dir, 'exclusion_counts.csv'))

    # 4. figures
    fig_all = vif_boxplot(vifs, f'{model_name} — VIF distributions by session')
    all_path = os.path.join(figs_dir, 'vif_boxplot_all.png')
    fig_all.savefig(all_path, dpi=120, bbox_inches='tight')

    pm_path = None
    fig_pm = vif_boxplot(vifs, f'{model_name} — PM VIF distributions by session',
                         pm_only=True)
    if fig_pm is not None:
        pm_path = os.path.join(figs_dir, 'vif_boxplot_pm.png')
        fig_pm.savefig(pm_path, dpi=120, bbox_inches='tight')

    # 5. HTML
    meta = {
        'subjects': len(DMs),
        'sessions': n_sessions,
        'regressors': vifs.shape[1],
        'generated': _dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    tables = [
        ('VIF summary (describe)', summary),
        ('PM VIF max per session', pm_max),
        (f"Subject exclusion counts — '{args.exclusion_regressor}' regressors "
         f"(VIF thresholds {args.thresholds})", excl),
    ]
    figures = [
        ('VIF by session — all task regressors', all_path),
        ('VIF by session — parametric modulators only', pm_path),
    ]
    report_path = os.path.join(out_dir, 'report.html')
    build_html(model_name, meta, tables, figures, report_path)
    print(f'\nReport written: {report_path}')


if __name__ == '__main__':
    main()
