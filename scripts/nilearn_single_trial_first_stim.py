import os
import sys
import time
import json
import pickle
import warnings
from datetime import datetime

from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list, create_dummy_regressors
# compute_parametric_modulator intentionally unused here to avoid leakage

# Dynamically set the number of workers based on available CPUs
max_workers = min(30, multiprocessing.cpu_count())

base_dir = '/home/ubuntu/data/learning-habits'
bids_dir = "/home/ubuntu/data/learning-habits/bids_dataset/derivatives/fmriprep-24.0.1"

sub_ids = load_participant_list(base_dir)

# -------------------------------
# UPDATED MODEL PARAMS (MVPA)
# -------------------------------
model_params = {
    'model_name': 'mvpa_firststim',
    'tr': 2.33384,
    'hrf_model': 'spm',
    'noise_model': 'ar1',
    # For MVPA keep smoothing very low or zero; override if needed
    'smoothing_fwhm': None,
    'motion_type': 'basic',
    'include_physio': True,
    'brain_mask': True,
    'fd_thresh': 0.5,
    'std_dvars_thresh': 2,
    'exclusion_threshold': 0.2,
    'scrub': 'dummies',        # keep as before
    'exclude_stimuli': False,   # keep your 1/8 exclusion if desired
    'duration': 'all',         # keep consistent with your design (often RT-based)
    'iti_included': False,

    # NEW: choose single-trial strategy
    # 'LSA' = one GLM per run with one column per trial (fast)
    # 'LSS' = one GLM per trial: target vs others (slower, often more robust)
    'beta_mode': 'LSA',

    # NEW: which phase to turn into single-trial betas
    'decoding_phase': 'first_stim_presentation',

    # QC toggles
    'save_design_png': True
}

def build_firststim_events(subject, run, model_params):
    """
    Builds an events dataframe restricted to the phase we want to decode,
    without inserting parametric modulators into the design.
    """
    exclude_stimuli = model_params['exclude_stimuli']
    decoding_phase = model_params['decoding_phase']

    # Use your helper to get a standardized events df with columns you rely on
    # (includes first_stim, first_stim_value_rl/ck columns, etc.)
    columns_event = {
        'first_stim_value_rl': 'first_stim_presentation',
        'first_stim_value_ck': 'first_stim_presentation',
        'first_stim': 'first_stim_presentation'
    }
    ev_all = getattr(subject, run).extend_events_df(columns_event)

    # Keep only the decoding phase rows
    ev = ev_all[ev_all['trial_type'] == decoding_phase].copy()

    # Optional: tag exclusions (e.g., stimulus IDs 1 and 8) while keeping them in the table
    # so that your mapping CSV still has their Qs if you want to drop later.
    if exclude_stimuli:
        ev['include_flag'] = ~ev['first_stim'].astype(int).isin([1, 8])
    else:
        ev['include_flag'] = True

    # Duration handling (your script used 'none' vs 'all')
    duration = model_params['duration']
    if duration == 'none':
        ev['duration'] = 0
    elif duration == 'all':
        # do nothing; your events already encode duration (e.g., RT if that's how you built them upstream)
        pass
    else:
        raise ValueError("Invalid duration type. Must be 'none' or 'all'")

    return ev


def model_run(subject, run, model_params):

    # Parameters
    model_name = model_params['model_name']
    tr = model_params['tr']
    hrf_model = model_params['hrf_model']
    noise_model = model_params['noise_model']
    smoothing_fwhm = model_params['smoothing_fwhm']
    motion_type = model_params['motion_type']
    include_physio = model_params['include_physio']
    fd_thresh = model_params['fd_thresh']
    std_dvars_thresh = model_params['std_dvars_thresh']
    exclusion_threshold = model_params['exclusion_threshold']
    scrub = model_params['scrub']
    brain_mask_flag = model_params['brain_mask']
    iti_included = model_params['iti_included']

    beta_mode = model_params.get('beta_mode', 'LSA').upper()
    decoding_phase = model_params.get('decoding_phase', 'first_stim_presentation')
    save_design_png = model_params.get('save_design_png', True)

    # Create output directory (encode mode)
    sub_id = subject.sub_id
    derivatives_dir = os.path.join(os.path.dirname(subject.bids_dir), 'nilearn')
    current_time = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(derivatives_dir, f"{model_name}_{beta_mode}_{current_time}")
    sub_output_dir = os.path.join(model_dir, sub_id, f"run-{run}")
    os.makedirs(sub_output_dir, exist_ok=True)

    # Load fMRI volume
    img_path = subject.img.get(run)
    fmri_img = load_img(img_path)
    n_volumes = fmri_img.shape[-1]

    # Load confounds (keep your fMRIPrep pipeline)
    confounds, sample_mask = subject.load_confounds(
        run, motion_type=motion_type,
        fd_thresh=fd_thresh, std_dvars_thresh=std_dvars_thresh,
        scrub=(0 if scrub == 'dummies' else scrub)
    )

    # Exclude runs with too many scrubbed volumes
    if sample_mask is not None and len(sample_mask) < (1 - exclusion_threshold) * n_volumes:
        with open(os.path.join(sub_output_dir, 'exclusion_flag.txt'), 'w') as f:
            f.write(f"Run {run} of {sub_id} excluded due to excessive scrubbing")
        print(f"Run {run} of {sub_id} excluded due to excessive scrubbing")
        return f"Run {run} of {sub_id} excluded due to excessive scrubbing"

    # Physio regressors
    if include_physio:
        physio_regressors = subject.load_physio_regressors(run)
        confounds = confounds.join(physio_regressors)

    # Scrub with dummies
    if scrub == 'dummies':
        dummies = create_dummy_regressors(sample_mask, len(confounds))
        confounds = pd.concat([confounds, dummies], axis=1)

    # Brain mask
    brain_mask = load_img(subject.brain_mask.get(run)) if brain_mask_flag else None

    # Build events for the phase of interest (no parametric modulators in design)
    events = build_firststim_events(subject, run, model_params)

    # Optionally drop ITI (your default is to exclude ITI anyway)
    if not iti_included:
        # events already filtered to first_stim_presentation; no ITI here
        pass

    # Frame times
    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr / 2., (n - .5) * tr, n)

    # Shared model kwargs
    base_model_kwargs = dict(
        t_r=tr,
        smoothing_fwhm=smoothing_fwhm,
        mask_img=brain_mask,
        hrf_model=hrf_model,
        noise_model=noise_model,
        drift_model=None,
        minimize_memory=True
    )

    # We?ll collect a mapping for decoding (per trial)
    mapping_rows = []

    # ---------------------
    # LSA: one GLM per run
    # ---------------------
    if beta_mode == 'LSA':
        # Give each trial a unique label so design has one column per trial
        trials = events.reset_index(drop=True).copy()
        trials['trial_label'] = [f"{decoding_phase}_t{ix+1:03d}" for ix in range(len(trials))]
        lsa_events = trials.rename(columns={'trial_type': 'old_trial_type'})
        lsa_events = lsa_events.rename(columns={'trial_label': 'trial_type'})

        warnings.filterwarnings("ignore", message=".*events with null duration.*")
        warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")

        design_matrix = make_first_level_design_matrix(
            frame_times=frametimes,
            events=lsa_events,
            hrf_model=hrf_model,
            drift_model=None,
            add_regs=confounds
        )

        model = FirstLevelModel(**base_model_kwargs).fit(
            fmri_img, design_matrices=design_matrix, sample_masks=sample_mask
        )

        # Save each trial's beta (1-hot contrast on its column)
        dm_cols = list(design_matrix.columns)
        trial_cols = [c for c in dm_cols if c.startswith(f"{decoding_phase}_t")]

        for i, col in enumerate(trial_cols, start=1):
            cvec = np.zeros(len(dm_cols), dtype=float)
            cvec[dm_cols.index(col)] = 1.0               # 1-hot for this trial
            beta_img = model.compute_contrast(cvec, output_type="effect_size")
            beta_path = os.path.join(sub_output_dir, f"{sub_id}_run-{run}_trial-{i:03d}_beta_LSA.nii.gz")
            beta_img.to_filename(beta_path)

            row_ev = trials.iloc[i-1]
            mapping_rows.append({
                'sub_id': sub_id,
                'run': run,
                'trial_index': i,
                'onset': float(row_ev['onset']),
                'duration': float(row_ev['duration']),
                'first_stim': int(row_ev.get('first_stim', np.nan)),
                'q_rl': float(row_ev.get('first_stim_value_rl', np.nan)),
                'q_ck': float(row_ev.get('first_stim_value_ck', np.nan)),
                'include_flag': bool(row_ev.get('include_flag', True)),
                'beta_path': beta_path,
                'mode': 'LSA'
            })

        # Optional QC
        if save_design_png:
            qc_design_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_design_matrix.png')
            plot_design_matrix(design_matrix, output_file=qc_design_path)

        # Persist design for audit
        design_matrix_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_design_matrix.csv')
        design_matrix.to_csv(design_matrix_path, index=False)

    # ----------------------
    # LSS: one GLM per trial
    # ----------------------
    elif beta_mode == 'LSS':
        trials = events.reset_index(drop=True).copy()

        for t in range(len(trials)):
            ev_t = trials.iloc[[t]].copy()
            ev_others = trials.drop(index=t).copy()
            ev_t['trial_type'] = 'target'
            ev_others['trial_type'] = 'others'
            events_t = pd.concat([ev_t, ev_others], ignore_index=True)

            model = FirstLevelModel(**base_model_kwargs).fit(
                fmri_img, events=events_t, confounds=confounds, sample_masks=sample_mask
            )

            dm_cols = model.design_matrices_[0].columns
            cvec = np.zeros(len(dm_cols), dtype=float)
            cvec[dm_cols.index(col)] = 1.0               # 1-hot for this trial
            beta_img = model.compute_contrast(cvec, output_type="effect_size")
            beta_path = os.path.join(sub_output_dir, f"{sub_id}_run-{run}_trial-{t+1:03d}_beta_LSS.nii.gz")
            beta_img.to_filename(beta_path)

            row_ev = trials.iloc[t]
            mapping_rows.append({
                'sub_id': sub_id,
                'run': run,
                'trial_index': t+1,
                'onset': float(row_ev['onset']),
                'duration': float(row_ev['duration']),
                'first_stim': int(row_ev.get('first_stim', np.nan)),
                'q_rl': float(row_ev.get('first_stim_value_rl', np.nan)),
                'q_ck': float(row_ev.get('first_stim_value_ck', np.nan)),
                'include_flag': bool(row_ev.get('include_flag', True)),
                'beta_path': beta_path,
                'mode': 'LSS'
            })
    else:
        raise ValueError("beta_mode must be 'LSA' or 'LSS'")

    # Save mapping CSV for decoding
    mapping_df = pd.DataFrame(mapping_rows)
    map_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_trial_beta_mapping.csv')
    mapping_df.to_csv(map_path, index=False)

    # Always save the phase-restricted events we used
    events_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_events_used.csv')
    events.to_csv(events_path, index=False)

    # Save analysis parameters
    params_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)

    return f"Run {run} of {sub_id}: {beta_mode} betas OK"


def process_subject(sub_id, model_params):
    print(f"Processing Subject {sub_id}...")
    try:
        subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
        messages = []
        for run in subject.runs:
            msg = model_run(subject, run, model_params)
            messages.append(msg)
        return f"Subject {sub_id} processed successfully\n" + "\n".join(messages)
    except Exception as e:
        return f"An error occurred for Subject {sub_id}: {e}"


if __name__ == "__main__":
    start_time = time.time()
    sub_ids = load_participant_list(base_dir)

    results = Parallel(n_jobs=max_workers)(
        delayed(process_subject)(sub, model_params) for sub in sub_ids
    )

    for result in results:
        print(result)

    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")