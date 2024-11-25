import sys
import os
sys.path.append('/Users/hugofluhr/phd_local/repositories/RewardPairsTask_Analysis/')
from utils.data import Subject, load_participant_list
from utils.analysis import run_model_RSA

# Set base directory and derivatives directory
base_dir = '/Users/hugofluhr/data/LH_dev'
bids_dir = "/Users/hugofluhr/data/LH_dev/fmriprep-23.2.1"
derivatives_dir = os.path.join(base_dir, 'nilearn/first_level_RSA')

# Create derivatives folder if it does not exist
if not os.path.exists(derivatives_dir):
    os.makedirs(derivatives_dir)

# Define parameters
tr = 2.33384
hrf_model = 'spm'
noise_model = 'ar1'
smoothing_fwhm = 5
high_pass = 0.01
motion_type = 'basic'
mask_samples = True

# Entry point for the script
if __name__ == "__main__":
    subject_ids = load_participant_list(base_dir)

    for sub_id in subject_ids:
        print(f"Processing Subject {sub_id}...")
        try:
            subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
            for run in subject.runs:
                confounds, sample_mask = subject.load_confounds(run, motion_type=motion_type)
                if not mask_samples:
                    sample_mask = None
                run_model_RSA(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir,
                              model_label='model_RSA', plot_stat=True, plot_design=True)
        
        except Exception as e:
            print(f"An error occurred for Subject {sub_id}: {e}")