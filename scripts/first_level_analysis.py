
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
run = 'learning1'

# Function to load subject data
def load_subject_data(sub_id):
    subject = Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir)
    block = getattr(subject, run)
    events = block.extend_events_df()
    img_path = subject.img.get(run)
    fmri_img = image.load_img(img_path)

    # Load confounds
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
    confounds = confounds[columns_to_keep]

    return fmri_img, events, confounds

# Function to run first-level analysis for Model 1
def run_model_1(fmri_img, events, confounds):
    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr/2., (n - .5) * tr, n)

    X1 = make_first_level_design_matrix(
        frame_times=frametimes,
        events=events,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=high_pass,
        add_regs=confounds
    )

    # Fit the first-level model
    model1 = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=False)
    model1 = model1.fit(fmri_img, design_matrices=X1)

    # Compute the contrast
    z_map1 = model1.compute_contrast(
        contrast_def="first_stim_presentation - iti", output_type="z_score"
    )

    # Save the z-map to derivatives directory
    z_map1_path = os.path.join(derivatives_dir, 'model1_z_map.nii.gz')
    z_map1.to_filename(z_map1_path)
    print(f"Model 1 results saved to {z_map1_path}")

    # Plot and save the statistical map as an image
    plot_stat_map(
        z_map1,
        threshold=3.0,
        title="Model 1: Toy Contrast",
        output_file=os.path.join(derivatives_dir, 'model1_stat_map.png')
    )

# Function to run first-level analysis for Model 2
def run_model_2(fmri_img, events, confounds):
    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr/2., (n - .5) * tr, n)

    X2 = make_first_level_design_matrix(
        frame_times=frametimes,
        events=events,
        hrf_model=hrf_model,
        drift_model='cosine',
        high_pass=high_pass,
        add_regs=confounds
    )

    # Fit the second model
    model2 = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=False)
    model2 = model2.fit(fmri_img, design_matrices=X2)

    # Compute a different contrast for Model 2
    z_map2 = model2.compute_contrast(
        contrast_def="second_stim_presentation - iti", output_type="z_score"
    )

    # Save the z-map to derivatives directory
    z_map2_path = os.path.join(derivatives_dir, 'model2_z_map.nii.gz')
    z_map2.to_filename(z_map2_path)
    print(f"Model 2 results saved to {z_map2_path}")

    # Plot and save the statistical map as an image
    plot_stat_map(
        z_map2,
        threshold=3.0,
        title="Model 2: Different Contrast",
        output_file=os.path.join(derivatives_dir, 'model2_stat_map.png')
    )

# Entry point for the script
if __name__ == "__main__":
    sub_ids = ['12']  # Example subject ID list
    fmri_img, events, confounds = load_subject_data(sub_ids[0])
    
    # Run both models
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")

    run_model_1(fmri_img, events, confounds)
    run_model_2(fmri_img, events, confounds)
