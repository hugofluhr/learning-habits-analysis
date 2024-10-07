import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
sys.path.append('.')
from utils.data import Subject, load_participant_list
from utils.analysis import compute_parametric_modulator

# Ignore warnings related to null duration events and unexpected columns in events data
warnings.filterwarnings("ignore", message=".*events with null duration.*")
warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")

# Directories
base_dir = '/Users/hugofluhr/data/LH_dev'
bids_dir = "/Users/hugofluhr/data/LH_dev/fmriprep-23.2.1"
sub_ids = load_participant_list(base_dir)
subjects = [Subject(base_dir, sub_id, include_modeling=True, include_imaging=True, bids_dir=bids_dir) for sub_id in sub_ids]

# Define constants
tr = 2.33384
hrf_model = 'spm'
smoothing_fwhm = 5
high_pass = 0.01
motion_type = 'basic'

# Create an empty list to collect the results
results = []

# Loop through each subject and each run to calculate correlations
for subject in subjects[:5]:
    print(f"Processing Subject {subject.sub_id}...")
    for run in subject.runs:
        # Initialize a dictionary to store correlation coefficients for the current run
        run_results = {
            'subject': subject.sub_id,
            'run': run
        }

        # Load data for the current run
        block = getattr(subject, run)
        events = block.extend_events_df()
        fmri_img = image.load_img(subject.img.get(run))
        confounds, sample_mask = subject.load_confounds(run, motion_type=motion_type)
        n = fmri_img.shape[-1]
        frametimes = np.linspace(tr / 2., (n - 0.5) * tr, n)

        # Create the initial design matrix
        X1 = make_first_level_design_matrix(
            frame_times=frametimes,
            events=events,
            hrf_model=hrf_model,
            drift_model=None,
            high_pass=high_pass,
            add_regs=confounds
        )

        # Compute parametric modulators
        reg_rl_value = compute_parametric_modulator(events, 'first_stim_presentation', 'first_stim_value_rl', frametimes, hrf_model)
        reg_ck_value = compute_parametric_modulator(events, 'first_stim_presentation', 'first_stim_value_ck', frametimes, hrf_model)

        # Orthogonalize RL and CK regressors
        unmodulated = X1['first_stim_presentation'].values.reshape(-1, 1)
        rl_value_ortho = reg_rl_value - LinearRegression().fit(unmodulated, reg_rl_value).predict(unmodulated)
        ck_value_ortho = reg_ck_value - LinearRegression().fit(unmodulated, reg_ck_value).predict(unmodulated)

        # Compute correlations and add them to run_results
        run_results['corr_unmod_rl'] = np.corrcoef(unmodulated.flatten(), reg_rl_value.flatten())[0, 1]
        run_results['corr_rl_ck'] = np.corrcoef(reg_rl_value.flatten(), reg_ck_value.flatten())[0, 1]
        run_results['corr_unmod_rl_ortho'] = np.corrcoef(unmodulated.flatten(), rl_value_ortho.flatten())[0, 1]
        run_results['corr_rl_ortho_ck_ortho'] = np.corrcoef(rl_value_ortho.flatten(), ck_value_ortho.flatten())[0, 1]

        # Fit the initial model
        model1 = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
        model1 = model1.fit(fmri_img, design_matrices=X1, sample_masks=sample_mask)
        z_map1 = model1.compute_contrast(contrast_def="first_stim_presentation", output_type="z_score")
        
        # Complete model with both RL and CK regressors
        X_complete = X1.copy()
        X_complete.insert(1, 'first_stim_value_rl', reg_rl_value)
        X_complete.insert(2, 'first_stim_value_ck', reg_ck_value)

        model_complete = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
        model_complete = model_complete.fit(fmri_img, design_matrices=X_complete, sample_masks=sample_mask)

        # Orthogonalized model
        X_ortho = X_complete.copy()
        X_ortho['first_stim_value_rl'] = rl_value_ortho
        X_ortho['first_stim_value_ck'] = ck_value_ortho

        model_ortho = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
        model_ortho = model_ortho.fit(fmri_img, design_matrices=X_ortho, sample_masks=sample_mask)

        # RL only model
        X_rl = X1.copy()
        X_rl.insert(1, 'first_stim_value_rl', reg_rl_value)

        model_rl = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
        model_rl = model_rl.fit(fmri_img, design_matrices=X_rl, sample_masks=sample_mask)

        # CK only model
        X_ck = X1.copy()
        X_ck.insert(1, 'first_stim_value_ck', reg_ck_value)

        model_ck = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
        model_ck = model_ck.fit(fmri_img, design_matrices=X_ck, sample_masks=sample_mask)

        # Compute contrasts for complete model
        z_map_complete_pres = model_complete.compute_contrast(contrast_def="first_stim_presentation", output_type="z_score")
        z_map_complete_rl = model_complete.compute_contrast(contrast_def="first_stim_value_rl", output_type="z_score")
        z_map_complete_ck = model_complete.compute_contrast(contrast_def="first_stim_value_ck", output_type="z_score")

        # Compute correlations between complete model z-maps and add to run_results
        run_results['corr_complete_pres'] = np.corrcoef(z_map1.get_fdata().ravel(), z_map_complete_pres.get_fdata().ravel())[0, 1]
        run_results['corr_complete_rl_ck'] = np.corrcoef(z_map_complete_rl.get_fdata().ravel(), z_map_complete_ck.get_fdata().ravel())[0, 1]

        # Compute contrasts for Orthogonalized model
        z_map_ortho_pres = model_ortho.compute_contrast(contrast_def="first_stim_presentation", output_type="z_score")
        z_map_ortho_rl = model_ortho.compute_contrast(contrast_def="first_stim_value_rl", output_type="z_score")
        z_map_ortho_ck = model_ortho.compute_contrast(contrast_def="first_stim_value_ck", output_type="z_score")

        # Compute correlations between z-maps and add to run_results
        run_results['corr_ortho_pres'] = np.corrcoef(z_map1.get_fdata().ravel(), z_map_ortho_pres.get_fdata().ravel())[0, 1]
        run_results['corr_ortho_rl'] = np.corrcoef(z_map_complete_rl.get_fdata().ravel(), z_map_ortho_rl.get_fdata().ravel())[0, 1]
        run_results['corr_ortho_ck'] = np.corrcoef(z_map_complete_ck.get_fdata().ravel(), z_map_ortho_ck.get_fdata().ravel())[0, 1]
        run_results['corr_ortho_rl_ck'] = np.corrcoef(z_map_ortho_rl.get_fdata().ravel(), z_map_ortho_ck.get_fdata().ravel())[0, 1]

        # Compute contrasts for RL only model
        z_map_rl_pres = model_rl.compute_contrast(contrast_def="first_stim_presentation", output_type="z_score")
        z_map_rl_rl = model_rl.compute_contrast(contrast_def="first_stim_value_rl", output_type="z_score")
    
        # Compute correlations between z-maps and add to run_results
        run_results['corr_rl_pres'] = np.corrcoef(z_map1.get_fdata().ravel(), z_map_rl_pres.get_fdata().ravel())[0, 1]
        run_results['corr_rl_rl'] = np.corrcoef(z_map_complete_rl.get_fdata().ravel(), z_map_rl_rl.get_fdata().ravel())[0, 1]     

        # Compute contrasts for CK only model
        z_map_ck_pres = model_ck.compute_contrast(contrast_def="first_stim_presentation", output_type="z_score")
        z_map_ck_ck = model_ck.compute_contrast(contrast_def="first_stim_value_ck", output_type="z_score")

        # Compute correlations between z-maps and add to run_results
        run_results['corr_ck_pres'] = np.corrcoef(z_map1.get_fdata().ravel(), z_map_ck_pres.get_fdata().ravel())[0, 1]
        run_results['corr_ck_ck'] = np.corrcoef(z_map_complete_ck.get_fdata().ravel(), z_map_ck_ck.get_fdata().ravel())[0, 1]
        run_results['corr_rlOnly_ckOnly'] = np.corrcoef(z_map_rl_rl.get_fdata().ravel(), z_map_ck_ck.get_fdata().ravel())[0, 1]
        run_results['corr_rlOnly_ckOnly_pres'] = np.corrcoef(z_map_rl_pres.get_fdata().ravel(), z_map_ck_pres.get_fdata().ravel())[0, 1]

        # Append the current run's results to the results list
        results.append(run_results)

# Convert the results list to a DataFrame and save it as a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('/Users/hugofluhr/data/LH_dev/first_level/model_comparison.csv', index=False)