import sys
import os
import warnings
import numpy as np
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_stat_map
from nilearn.interfaces.fmriprep import load_confounds
sys.path.append('/Users/hugofluhr/phd_local/repositories/RewardPairsTask_Analysis/')
from utils.data import Subject
from utils.analysis import run_model_1, run_model_2

# Set base directory and derivatives directory
base_dir = '/Users/hugofluhr/data/LH_dev'
bids_dir = "/Users/hugofluhr/data/LH_dev/fmriprep-23.2.1"
derivatives_dir = os.path.join(base_dir, 'nilearn_first_level')

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

# Function to load subject data including confounds
def load_subject_data(sub_id):
    subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
    
    confounds_dict = {}
    for run in subject.runs:
        img_path = subject.img.get(run)
        confounds, sample_mask = load_confounds(
            img_path,
            strategy=('motion', 'high_pass', 'wm_csf', 'scrub'),
            motion=motion_type,
            scrub=0,
            fd_threshold=0.5,
            std_dvars_threshold=1.5
        )

        # Filter to keep only the first 5 cosine columns
        cosine_columns = [col for col in confounds.columns if col.startswith('cosine')]
        cosine_columns_to_keep = cosine_columns[:5]
        columns_to_keep = [col for col in confounds.columns if not col.startswith('cosine')] + cosine_columns_to_keep
        confounds_dict[run] = confounds[columns_to_keep]

    return subject, confounds_dict


# Entry point for the script
if __name__ == "__main__":
    sub_id = '01'  # Example subject ID list
    subject, confounds_dict = load_subject_data(sub_id)

    # Ignore warnings related to null duration events and unexpected columns in events data
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")
    for run in subject.runs:
        run_model_1(subject, run, confounds_dict[run], tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, plot_stat=False, plot_design=True)
        run_model_2(subject, run, confounds_dict[run], tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, plot_stat=False, plot_design=True)