import sys
import os
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
from nilearn.plotting import plot_design_matrix
from nilearn.image import load_img
from nilearn.reporting import make_glm_report
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list
from utils.analysis import compute_parametric_modulator


base_dir = '/home/ubuntu/data/learning-habits'
bids_dir = "/home/ubuntu/data/learning-habits/bids_dataset/derivatives/fmriprep-24.0.1"

sub_ids = load_participant_list(base_dir)

model_params = {
    'model_name': 'rl_modulation',
    'tr': 2.33384,
    'hrf_model': 'spm',
    'noise_model': 'ar1',
    'smoothing_fwhm': 5,
    'high_pass': 0.01,
    'motion_type': 'basic',
    'include_physio': True,
    'brain_mask': False,
    'mask_samples': False,
    'demean_modulator': True,
}

run = 'test'

def model_run(subject, run, model_params):

    # Parameters
    model_name = model_params["model_name"]
    tr = model_params["tr"]
    hrf_model = model_params["hrf_model"]
    noise_model = model_params["noise_model"]
    smoothing_fwhm = model_params["smoothing_fwhm"]
    high_pass = model_params["high_pass"]
    motion_type = model_params["motion_type"]
    include_physio = model_params["include_physio"]
    brain_mask = model_params["brain_mask"]
    mask_samples = model_params["mask_samples"]
    demean_modulator = model_params["demean_modulator"]

    # Load confounds
    confounds, sample_mask = subject.load_confounds(run, motion_type=motion_type)
    if include_physio:
        physio_regressors = subject.load_physio_regressors(run)
        confounds = confounds.join(physio_regressors)

    if not mask_samples:
        sample_mask = None
    
    # Load fMRI volume
    img_path = subject.img.get(run)
    fmri_img = load_img(img_path)

    # Load events
    events = getattr(subject, run).events

    # This should always be None for now
    if brain_mask:
        brain_mask_path = subject.brain_mask.get(run)
        brain_mask = load_img(brain_mask_path)
    else:
        brain_mask = None

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
                                        high_pass=high_pass,
                                        add_regs=confounds)
    
    # Parametric modulation
    parametric_modulator_column = 'first_stim_value_rl'
    condition = 'first_stim_presentation'
    reg_value = compute_parametric_modulator(events, condition, parametric_modulator_column,
                                             frametimes, hrf_model, center=demean_modulator)
    design_matrix.insert(1, parametric_modulator_column, reg_value)

    # Create the model
    model = FirstLevelModel(t_r=tr, 
                            smoothing_fwhm=smoothing_fwhm, 
                            mask_img=brain_mask,
                            hrf_model=hrf_model,
                            noise_model=noise_model,
                            drift_model=None, 
                            high_pass=high_pass)
    
    model = model.fit(fmri_img, design_matrices=design_matrix, sample_masks=sample_mask)

    # Create output directory
    sub_id = subject.sub_id
    derivatives_dir = os.path.join(os.path.dirname(subject.bids_dir), 'nilearn')
    current_time = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(derivatives_dir, f"{model_name}_{current_time}")
    sub_output_dir = os.path.join(model_dir,sub_id, f"run-{run}")
    if not os.path.exists(sub_output_dir):
        os.makedirs(sub_output_dir)

    # Save the fitted model using pickle
    model_filename = os.path.join(sub_output_dir, f'sub-{sub_id}_run-{run}_model.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    #print(f"GLM model saved to {model_filename}")

    # Save beta maps for each regressor (events only)
    for i, column in enumerate(events.trial_type.unique()):
        beta_map = model.compute_contrast(np.eye(len(design_matrix.columns))[i], output_type='effect_size')
        beta_path = os.path.join(sub_output_dir, f'beta_{i:04d}_{column}.nii.gz')
        beta_map.to_filename(beta_path)
        #print(f"Saved: {beta_path}")

    # Save contrast map
    z_map = model.compute_contrast(contrast_def='response', output_type="effect_size")
    z_map_path = os.path.join(sub_output_dir, f'{subject.sub_id}_run-{run}_response_contrast.nii.gz')
    z_map.to_filename(z_map_path)
    #print(f"Contrast map saved to {z_map_path}")

    # Save the design matrix
    design_matrix_path = os.path.join(sub_output_dir, f'{subject.sub_id}_run-{run}_design_matrix.csv')
    design_matrix.to_csv(design_matrix_path, index=False)
    #print(f"Design matrix saved to {design_matrix_path}")

    # Save analysis parameters
    params_path = os.path.join(sub_output_dir, f'{subject.sub_id}_run-{run}_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    #print(f"Analysis parameters saved to {params_path}")

    # Save QC plot of design matrix
    qc_design_path = os.path.join(sub_output_dir, f'{subject.sub_id}_run-{run}_design_matrix.png')
    plot_design_matrix(design_matrix, output_file=qc_design_path)
    #print(f"Design matrix plot saved to {qc_design_path}")

    # Generate GLM report
    report_path = os.path.join(sub_output_dir, f'{subject.sub_id}_run-{run}_glm_report.html')
    report = make_glm_report(model=model, contrasts={"response": "response"})
    report.save_as_html(report_path)
    #print(f"GLM report saved to {report_path}")


def process_subject(sub_id, model_params):
    print(f"Processing Subject {sub_id}...")  
    
    #try:
    subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
        
    for run in subject.runs:
        #print(f"----------------- run {run}...")
        model_run(subject, run, model_params)

    return f"Subject {sub_id} processed successfully"

    #except Exception as e:
    #    return f"An error occurred for Subject {sub_id}: {e}"


for sub in sub_ids:
    print(process_subject(sub, model_params))
