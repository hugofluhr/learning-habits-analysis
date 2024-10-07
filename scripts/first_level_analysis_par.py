import sys
import os
from nilearn.interfaces.fmriprep import load_confounds
sys.path.append('/Users/hugofluhr/phd_local/repositories/RewardPairsTask_Analysis/')
from utils.data import Subject, load_participant_list
from utils.analysis import run_model_rl, run_model_ck

from concurrent.futures import ProcessPoolExecutor, as_completed

# Set base directory and derivatives directory
base_dir = '/Users/hugofluhr/data/LH_dev'
bids_dir = "/Users/hugofluhr/data/LH_dev/fmriprep-23.2.1"
derivatives_dir = os.path.join(base_dir, 'nilearn_first_level_no_baseline_masked')

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

# Function to load subject data including confounds
def load_subject_data(sub_id):
    subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
    
    confounds_dict = {}
    sample_mask_dict = {}
    for run in subject.runs:
        img_path = subject.img.get(run)
        confounds, sample_mask = load_confounds(
            img_path,
            strategy=('motion', 'high_pass', 'wm_csf', 'scrub'),
            motion=motion_type,
            scrub=0,
            fd_threshold=0.5,
            std_dvars_threshold=2.5
        )

        # Filter to keep only the first 5 cosine columns
        cosine_columns = [col for col in confounds.columns if col.startswith('cosine')]
        cosine_columns_to_keep = cosine_columns[:5]
        columns_to_keep = [col for col in confounds.columns if not col.startswith('cosine')] + cosine_columns_to_keep
        confounds_dict[run] = confounds[columns_to_keep]
        sample_mask_dict[run] = sample_mask

    return subject, confounds_dict, sample_mask_dict

def process_subject(sub_id, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, mask_samples):
    print(f"Processing Subject {sub_id}...")
    try:
        subject, confounds_dict, sample_mask = load_subject_data(sub_id)
        if not mask_samples:
            sample_mask = {run: None for run in subject.runs}
        
        for run in subject.runs:
            run_model_rl(subject, run, confounds_dict[run], sample_mask[run], tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, remove_baseline=True, plot_stat=False, plot_design=True)
            run_model_ck(subject, run, confounds_dict[run], sample_mask[run], tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, remove_baseline=True, plot_stat=False, plot_design=True)
        
        return f"Subject {sub_id} processed successfully"
    except Exception as e:
        return f"An error occurred for Subject {sub_id}: {e}"

# Entry point for the script
if __name__ == "__main__":
    subject_ids = load_participant_list(base_dir)

    # Set up parallel processing with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6) as executor:
        # Submit all subjects to be processed
        futures = {executor.submit(process_subject, sub_id, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, mask_samples): sub_id for sub_id in subject_ids}
        
        # Gather results as they are completed
        for future in as_completed(futures):
            print(future.result())