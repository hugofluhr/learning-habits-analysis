#!/usr/bin/env python3
"""
GLMsingle single-trial beta estimation — one subject.

Adapted from dev_glmsingle_stim_cat.ipynb. Core pipeline structure follows
multivariate/references/glmsingle_pipeline_withOutcomes.py (Rishabh & Sarah).

Usage
-----
# Single subject (local default paths)
python multivariate/run_glmsingle.py --subject 01

# All subjects from participants_mvpa.tsv (sequential)
python multivariate/run_glmsingle.py

# Cluster (override paths, single subject — called from submit_glmsingle.sh)
python multivariate/run_glmsingle.py --subject 01 \
    --base-dir /mnt/data/learning-habits \
    --bids-dir /mnt/data/learning-habits/bids_dataset/derivatives/fmriprep-24.0.1-noSDC \
    --output-dir /mnt/data/learning-habits/bids_dataset/derivatives/glmsingle

Subject list
------------
The canonical MVPA analysis sample is read from participants_mvpa.tsv (relative to
--base-dir) via utils.data.load_participant_list. Override with --participants-file.

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    TYPEA_ONOFF.npy
    TYPEB_FITHRF.npy
    TYPEC_FITHRF_GLMDENOISE.npy
    TYPED_FITHRF_GLMDENOISE_RR.npy      <- type-D: best model (HRF + denoise + ridge)
    DESIGNINFO.npy
    RUNWISEFIR.npy
    figures/
    sub-<id>_glmSingle_betas_CUES.nii.gz   <- (x,y,z, n_trials) type-D betas
    sub-<id>_glmSingle_betas_CUES_info.csv  <- trial→condition/run mapping
    glmsingle_sub-<id>.log

Design choices
--------------
- 8 first-stimulus identity conditions; within-trial events excluded (TR-collision
  at TR=2.33 s — see dev notebook Step 4).
- extra_regressors not used: GLMsingle has a bug when runs have unequal volumes
  (FIR diagnostic always picks run-0 regressors). GLMdenoise handles noise.
- sessionindicator = [1,1,1]: all runs pooled for GLMdenoise noise estimation.
- stimdur: mean(t_action - t_first_stim) over response-only trials.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from glmsingle.glmsingle import GLM_single

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_participant_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TR = 2.33384  # confirmed in matlab/first_lvl/glm2_chosen.m

RUNS = ['learning1', 'learning2', 'test']

STIM_NAMES = [
    'face_female', 'face_male',
    'figure_circle', 'figure_triangle',
    'hand_back', 'hand_palm',
    'house_1', 'house_2',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_design_matrix(trials, n_volumes, run_name):
    """Binary TR-indexed design matrix: (n_volumes, 8). Floor-division onset
    assignment follows the reference pipeline convention."""
    stim2col = {s: i for i, s in enumerate(STIM_NAMES)}
    dm = np.zeros((n_volumes, len(STIM_NAMES)))
    scan_start = trials['t_first_stim'].min()
    out_of_range = 0
    for _, row in trials.iterrows():
        tr_idx = int((row['t_first_stim'] - scan_start) / TR)
        if 0 <= tr_idx < n_volumes:
            dm[tr_idx, stim2col[row['first_stim_name']]] = 1.0
        else:
            out_of_range += 1
    if out_of_range:
        logging.warning(f"{run_name}: {out_of_range} onsets outside [0, {n_volumes})")
    logging.info(f"  {run_name}: DM {dm.shape}, {int(dm.sum())} events marked")
    return dm


def extract_betas(subject_output, run_trials):
    """Load type-D betas, build trial-info DataFrame."""
    typed      = np.load(subject_output / "TYPED_FITHRF_GLMDENOISE_RR.npy",
                         allow_pickle=True).item()
    designinfo = np.load(subject_output / "DESIGNINFO.npy",
                         allow_pickle=True).item()

    betas      = typed['betasmd']
    stimorder  = np.array(designinfo['stimorder'])
    cue_mask   = stimorder < len(STIM_NAMES)
    betas_cues = betas[..., cue_mask]

    # Build chronological index of all stimulus presentations across runs.
    # GLMsingle emits betas in this same order (run 1 → run 2 → run 3, time-sorted).
    all_stim_indices = []
    for run in RUNS:
        for _, row in run_trials[run].sort_values('t_first_stim').iterrows():
            all_stim_indices.append((STIM_NAMES.index(row['first_stim_name']), run))

    stim_counter, rows = {}, []
    for beta_vol_idx in np.where(cue_mask)[0]:
        col_idx = int(stimorder[beta_vol_idx])
        count   = stim_counter.get(col_idx, 0)
        stim_counter[col_idx] = count + 1
        occurrence_runs = [r for c, r in all_stim_indices if c == col_idx]
        stim_name = STIM_NAMES[col_idx]
        rows.append({
            'beta_vol_idx': int(beta_vol_idx),
            'stim_col':     col_idx,
            'stim_name':    stim_name,
            'stim_cat':     stim_name.split('_')[0],
            'run':          occurrence_runs[count],
        })

    trial_info = pd.DataFrame(rows)
    trial_info.index.name = 'trial_id'
    return betas_cues, trial_info


# ---------------------------------------------------------------------------
# Main subject function
# ---------------------------------------------------------------------------

def run_subject(subject, base_dir, bids_dir, output_dir, overwrite=False):
    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / "TYPED_FITHRF_GLMDENOISE_RR.npy"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping "
                     f"(pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)
    figures_dir = subject_output / "figures"
    figures_dir.mkdir(exist_ok=True)

    # --- Load subject ---
    logging.info(f"Loading sub-{subject}")
    sub = Subject(
        base_dir=str(base_dir),
        subject_id=subject,
        include_imaging=True,
        bids_dir=str(bids_dir),
    )
    run_trials = {
        'learning1': sub.learning1.trials,
        'learning2': sub.learning2.trials,
        'test':      sub.test.trials,
    }

    # --- stimdur ---
    all_trials  = pd.concat(run_trials.values())
    resp_trials = all_trials[all_trials['action'].notna()]
    stimdur = (resp_trials['t_action'] - resp_trials['t_first_stim']).mean()
    logging.info(f"stimdur = {stimdur:.3f}s  "
                 f"({len(resp_trials)}/{len(all_trials)} response trials)")

    # --- Load BOLD + design matrices ---
    fmri_data, design_matrices = [], []
    for run in RUNS:
        img = nib.load(sub.get_img_path(run))
        logging.info(f"  {run}: {img.shape}")
        fmri_data.append(img.get_fdata(dtype=np.float32))
        design_matrices.append(
            build_design_matrix(run_trials[run], img.shape[-1], run)
        )

    # --- Fit GLMsingle ---
    opt = dict(
        sessionindicator = np.ones((1, len(RUNS)), dtype=int),
        wantlibrary      = 1,
        wantglmdenoise   = 1,
        wantfracridge    = 1,
        wantfileoutputs  = [1, 1, 1, 1],
    )
    logging.info("Fitting GLMsingle...")
    GLM_single(opt).fit(
        design_matrices, fmri_data, stimdur, TR,
        outputdir=str(subject_output),
        figuredir=str(figures_dir),
    )
    logging.info("Fitting complete.")

    # --- Extract and save type-D betas ---
    betas_cues, trial_info = extract_betas(subject_output, run_trials)

    ref_img = nib.load(sub.get_img_path('learning1'))
    image.new_img_like(ref_img, betas_cues).to_filename(
        str(subject_output / f"sub-{subject}_glmSingle_betas_CUES.nii.gz")
    )
    trial_info.to_csv(
        str(subject_output / f"sub-{subject}_glmSingle_betas_CUES_info.csv")
    )

    logging.info(f"sub-{subject}: done — betas shape {betas_cues.shape}, "
                 f"{len(trial_info)} trials")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run GLMsingle single-trial estimation. "
                    "If --subject is omitted, all subjects in --participants-file are run."
    )
    parser.add_argument("--subject", default=None,
                        help="Subject ID without 'sub-' prefix, e.g. 01. "
                             "If omitted, all subjects in --participants-file are run.")
    parser.add_argument("--participants-file", default="participants_mvpa.tsv",
                        help="TSV filename (relative to --base-dir) listing subject IDs "
                             "(default: participants_mvpa.tsv)")
    parser.add_argument("--base-dir",
                        default="/home/ubuntu/data/learning-habits",
                        help="Root data dir containing spm_format/")
    parser.add_argument("--bids-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/fmriprep-24.0.1-noSDC",
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--output-dir",
                        default="/home/ubuntu/data/learning-habits/bids_dataset"
                                "/derivatives/glmsingle",
                        help="Output root directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rerun even if outputs already exist")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subject list and log destination
    if args.subject:
        subjects = [args.subject]
        log_dir  = output_dir / f"sub-{args.subject}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"glmsingle_sub-{args.subject}.log"
    else:
        subjects = load_participant_list(str(base_dir), file_name=args.participants_file)
        log_file = output_dir / "glmsingle_batch.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    logging.info(f"Processing {len(subjects)} subject(s) "
                 f"[source: {args.participants_file if not args.subject else '--subject flag'}]")

    for subject in subjects:
        run_subject(
            subject    = subject,
            base_dir   = base_dir,
            bids_dir   = Path(args.bids_dir),
            output_dir = output_dir,
            overwrite  = args.overwrite,
        )


if __name__ == "__main__":
    main()
