import sys
import os
sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list
import numpy as np
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
import warnings
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Set base directory and derivatives directory
base_dir = '/home/ubuntu/data/learning-habits'
bids_dir = '/home/ubuntu/data/learning-habits/bids_dataset/derivatives/fmriprep-23.2.1/'
derivatives_dir = os.path.join(base_dir, 'outputs/first_level/QC_cloud-23.2.1_brain_masked')

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
    "brain_mask": True,
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
brain_mask = parameters["brain_mask"]
demean_modulators = parameters["demean_modulators"]

def process_subject(sub_id, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, mask_samples, brain_mask):
    print(f"Processing Subject {sub_id}...")  
    model_label = "response_QC"
    try:
        subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)        
        for run in subject.runs:
            confounds, sample_mask = subject.load_confounds(run, motion_type=motion_type)
            if not mask_samples:
                sample_mask = None
            img_path = subject.img.get(run)
            fmri_img = image.load_img(img_path)
            if brain_mask:
                brain_mask_path = subject.brain_mask.get(run)
                brain_mask = image.load_img(brain_mask_path)
            else:
                brain_mask = None

            n = fmri_img.shape[-1]
            frametimes = np.linspace(tr/2., (n - .5)*tr, n) # from Gilles, checked with nilearn FirstLevelModel

            # Ignore warnings related to null duration events and unexpected columns in events data
            warnings.filterwarnings("ignore", message=".*events with null duration.*")
            warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")
            events = getattr(subject, run).events
            X1 = make_first_level_design_matrix(frame_times=frametimes,
                                                events=events,
                                                hrf_model=hrf_model,
                                                drift_model=None,
                                                high_pass=high_pass,
                                                add_regs=confounds)
            model = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=brain_mask)
            model = model.fit(fmri_img, design_matrices=X1, sample_masks=sample_mask)
            # Compute betamap and save it
            z_map = model.compute_contrast(
                contrast_def=f"{'response'}", output_type="effect_size"
            )
            z_map_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_b_map.nii.gz')
            z_map.to_filename(z_map_path)
            print(f"{model_label.capitalize()} betamap results saved to {z_map_path}")

                 
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
        futures = {executor.submit(process_subject, sub_id, tr, hrf_model, high_pass, smoothing_fwhm,
                                    derivatives_dir, mask_samples, brain_mask): sub_id for sub_id in subject_ids}
        
        # Gather results as they are completed
        for future in as_completed(futures):
            print(future.result())
    
    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")