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
from nilearn.reporting import make_glm_report

sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list, create_dummy_regressors, collapse_events

# Dynamically set the number of workers based on available CPUs
max_workers = min(30, multiprocessing.cpu_count())

base_dir = '/home/ubuntu/data/learning-habits'
bids_dir = "/home/ubuntu/data/learning-habits/bids_dataset/derivatives/fmriprep-24.0.1"

sub_ids = load_participant_list(base_dir)

model_params = {
    'model_name': 'stim_id_collapsed',
    'tr': 2.33384,
    'hrf_model': 'spm',
    'noise_model': 'ar1',
    'smoothing_fwhm': 5,
    'motion_type': 'basic',
    'include_physio': True,
    'brain_mask': True,
    'fd_thresh': 0.5,
    'std_dvars_thresh': 2,
    'exclusion_threshold': 0.2,
    'scrub': 'dummies',
    'duration': 'all',
    'combine_events': True,
    'split_stim_by': 'stim_id',
    'iti_included': False
}

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
    brain_mask = model_params['brain_mask']
    duration = model_params['duration']
    combine_events = model_params['combine_events']
    split_stim_by = model_params['split_stim_by']
    iti_included = model_params['iti_included']

    # Create output directory
    sub_id = subject.sub_id
    derivatives_dir = os.path.join(os.path.dirname(subject.bids_dir), 'nilearn')
    current_time = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(derivatives_dir, f"{model_name}_{current_time}")
    sub_output_dir = os.path.join(model_dir,sub_id, f"run-{run}")
    if not os.path.exists(sub_output_dir):
        os.makedirs(sub_output_dir)

    # Load fMRI volume
    img_path = subject.img.get(run)
    fmri_img = load_img(img_path)
    n_volumes = fmri_img.shape[-1]

    # Load confounds
    confounds, sample_mask = subject.load_confounds(run, motion_type=motion_type,
                                                    fd_thresh=fd_thresh, std_dvars_thresh=std_dvars_thresh,
                                                    scrub=(0 if scrub == 'dummies' else scrub))
    
    # Exclude runs with too many scrubbed volumes
    if sample_mask is not None and len(sample_mask) < (1-exclusion_threshold)*n_volumes:
        exclusion_flag_path = os.path.join(sub_output_dir, 'exclusion_flag.txt')
        with open(exclusion_flag_path, 'w') as f:
            f.write(f"Run {run} of {sub_id} excluded due to excessive scrubbing")
        print(f"Run {run} of {sub_id} excluded due to excessive scrubbing")
        return f"Run {run} of {sub_id} excluded due to excessive scrubbing"

    # Load physio regressors
    if include_physio:
        physio_regressors = subject.load_physio_regressors(run)
        confounds = confounds.join(physio_regressors)

    # Create dummy regressors for outlier volumes
    if scrub == 'dummies':
        dummies = create_dummy_regressors(sample_mask, len(confounds))
        confounds = pd.concat([confounds, dummies], axis=1)

    # fmriprep's brain mask
    if brain_mask:
        brain_mask_path = subject.brain_mask.get(run)
        brain_mask = load_img(brain_mask_path)
    else:
        brain_mask = None

    # Load events
    if split_stim_by == 'stim_id':
        events = getattr(subject, run).extend_events_df(columns_event={'first_stim': 'first_stim_presentation'})

        # Add stim index to first stim presentation
        events['trial_type'] = events.apply(
            lambda row: f"{row['trial_type']}_{int(row['first_stim'])}" if row['trial_type'] == 'first_stim_presentation' else row['trial_type'],
            axis=1
        )
    elif split_stim_by == 'stim_cat':
        events = getattr(subject, run).extend_events_df(columns_event={'first_stim_cat': 'first_stim_presentation'})

        # Add stim category to first stim presentation
        events['trial_type'] = events.apply(
            lambda row: f"{row['trial_type']}_{row['first_stim_cat']}" if row['trial_type'] == 'first_stim_presentation' else row['trial_type'],
            axis=1
        )


    # Collapse events if specified
    if combine_events:
        events = collapse_events(events)

    # Handle the duration of events
    if duration == 'none':
        events['duration'] = 0
    elif duration == 'all':
        pass
    else:
        raise ValueError("Invalid duration type. Must be 'none' or 'all'")
    
    # Handle the ITI
    if not iti_included:
        events = events[events['trial_type'] != 'iti']
    else:
        pass

    # Compute frame timing    
    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr / 2., (n - .5) * tr, n)

    # Ignore warnings related to null duration events and unexpected columns in events data
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")
    
    # Create design matrix
    design_matrix = make_first_level_design_matrix(frame_times=frametimes,
                                        events=events,
                                        hrf_model=hrf_model,
                                        drift_model=None,
                                        add_regs=confounds)

    # Create the model
    model = FirstLevelModel(t_r=tr, 
                            smoothing_fwhm=smoothing_fwhm, 
                            mask_img=brain_mask,
                            hrf_model=hrf_model,
                            noise_model=noise_model,
                            drift_model=None)
    
    model = model.fit(fmri_img, design_matrices=design_matrix, sample_masks=sample_mask)

    # Save the events dataframe to csv
    events_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_events.csv')
    events.to_csv(events_path, index=False)
    #print(f"Events saved to {events_path}")

    # Save the fitted model using pickle
    model_filename = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_model.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    #print(f"GLM model saved to {model_filename}")

    # Save beta maps for each regressor (events only)
    for i, column in enumerate(events.trial_type.unique()):
        beta_map = model.compute_contrast(np.eye(len(design_matrix.columns))[i], output_type='effect_size')
        beta_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_betamap_{column}.nii.gz')
        beta_map.to_filename(beta_path)
        #print(f"Saved: {beta_path}")

    # Save the design matrix
    design_matrix_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_design_matrix.csv')
    design_matrix.to_csv(design_matrix_path, index=False)
    #print(f"Design matrix saved to {design_matrix_path}")

    # Save analysis parameters
    params_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    #print(f"Analysis parameters saved to {params_path}")

    # Suppress Tight layout warning
    warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

    # Save QC plot of design matrix
    qc_design_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_design_matrix.png')
    plot_design_matrix(design_matrix, output_file=qc_design_path)
    #print(f"Design matrix plot saved to {qc_design_path}")

    # Generate GLM report
    # report_path = os.path.join(sub_output_dir, f'{sub_id}_run-{run}_glm_report.html')
    # report = make_glm_report(model=model, contrasts={"response": "response"})
    # report.save_as_html(report_path)
    #print(f"GLM report saved to {report_path}")


def process_subject(sub_id, model_params):
    print(f"Processing Subject {sub_id}...")  
    try:
        subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
        for run in subject.runs:
            model_run(subject, run, model_params)
        return f"Subject {sub_id} processed successfully"
    except Exception as e:
        return f"An error occurred for Subject {sub_id}: {e}"

if __name__ == "__main__":
    start_time = time.time()
    sub_ids = load_participant_list(base_dir)

    # Parallel processing with joblib
    results = Parallel(n_jobs=max_workers)(
        delayed(process_subject)(sub, model_params) for sub in sub_ids
    )

    # Print results for each subject
    for result in results:
        print(result)

    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")