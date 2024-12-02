import sys
import os
from nilearn.interfaces.fmriprep import load_confounds
sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list
from utils.analysis import run_model_rl, run_model_ck
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Set base directory and derivatives directory
base_dir = '/home/ubuntu/data/learning-habits'
bids_dir = '/home/ubuntu/data/learning-habits/bids_dataset/derivatives/fmriprep-24.0.1/'
derivatives_dir = os.path.join(base_dir, 'outputs/first_level/cloud-24.0.1')

# Create derivatives folder if it does not exist
if not os.path.exists(derivatives_dir):
    os.makedirs(derivatives_dir)

# Define parameters
parameters = {
    "tr": 2.33384,
    "hrf_model": 'spm',
    "noise_model": 'ar1',
    "smoothing_fwhm": 5,
    "high_pass": 0.01,
    "motion_type": 'basic',
    "mask_samples": True,
    "demean_modulators": True
}

# Write parameters to a JSON file in derivatives_dir
parameters_file = os.path.join(derivatives_dir, 'parameters.json')
with open(parameters_file, 'w') as f:
    json.dump(parameters, f, indent=4)

# Extract parameters for use in the script
tr = parameters["tr"]
hrf_model = parameters["hrf_model"]
noise_model = parameters["noise_model"]
smoothing_fwhm = parameters["smoothing_fwhm"]
high_pass = parameters["high_pass"]
motion_type = parameters["motion_type"]
mask_samples = parameters["mask_samples"]
demean_modulators = parameters["demean_modulators"]

def process_subject(sub_id, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, mask_samples):
    print(f"Processing Subject {sub_id}...")  
    try:
        subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)        
        for run in subject.runs:
            confounds, sample_mask = subject.load_confounds(run, motion_type=motion_type)
            if not mask_samples:
                sample_mask = None
            run_model_rl(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, demean_modulator=demean_modulators, plot_stat=False, plot_design=True)
            run_model_ck(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, demean_modulator=demean_modulators, plot_stat=False, plot_design=True)
    
        return f"Subject {sub_id} processed successfully"
    except Exception as e:
        return f"An error occurred for Subject {sub_id}: {e}"

# Entry point for the script
if __name__ == "__main__":
    start_time = time.time()
    subject_ids = load_participant_list(base_dir)

    # Set up parallel processing with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all subjects to be processed
        futures = {executor.submit(process_subject, sub_id, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, mask_samples): sub_id for sub_id in subject_ids}
        
        # Gather results as they are completed
        for future in as_completed(futures):
            print(future.result())
    
    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")