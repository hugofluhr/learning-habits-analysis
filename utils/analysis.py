import os
import warnings
import numpy as np
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_stat_map, plot_design_matrix
from nilearn.glm.first_level.hemodynamic_models import compute_regressor


def compute_parametric_modulator(events, condition, modulator, frametimes, hrf_model, center=True):
    """
    Compute the parametric modulator for a given condition and modulator
    
    Parameters
    ----------
    events : pd.DataFrame
        The events dataframe
    condition : str
        The condition for which to compute the parametric modulator
    modulator : str
        The modulator to use
    frametimes : array-like
        The frame times
    hrf_model : str
        The HRF model to use
    center : bool, optional
        Whether to center the modulator. Default is True.
    
    Returns
    -------
    regressor : array-like
        The modulated regressor
    """
    # adapted from nilearn's code

    condition_mask = events.trial_type == condition
    exp_condition = (
        events.onset[condition_mask].values,
        events.duration[condition_mask].values,
        events[modulator][condition_mask].values,
    )
    if center:
        exp_condition = (exp_condition[0], exp_condition[1], exp_condition[2] - exp_condition[2].mean())

    regressor, _ = compute_regressor(exp_condition=exp_condition,
                                     hrf_model=hrf_model,
                                     frame_times=frametimes
                                     )
    
    return regressor

# Function to run first-level analysis for a given model
def run_model(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir,
              model_label, parametric_modulator_column, demean_modulator=True, plot_stat=False, plot_design=False):
    """
    Run the first-level fMRI analysis model for a given subject and run.

    Parameters:
    subject (object): The subject object containing fMRI data and metadata.
    run (str): The specific run identifier within the subject's data.
    confounds (DataFrame): Confounding variables to include in the design matrix.
    sample_mask (array): Boolean mask to indicate which volumes to include in the analysis.
    tr (float): Repetition time of the fMRI acquisition.
    hrf_model (str): Hemodynamic response function model to use.
    high_pass (float): High-pass filter cutoff frequency in Hz.
    smoothing_fwhm (float): Full-width at half maximum for spatial smoothing in mm.
    derivatives_dir (str): Directory path to save the output z-map and statistical map.
    model_label (str): Label for the model, e.g., 'model1' or 'model2'.
    parametric_modulator_column (str): Column name for the parametric modulator to be added to the design matrix.
    demean_modulator (bool, optional): Whether to demean the parametric modulators. Default is True.
    plot_stat (bool, optional): Whether to plot and save the statistical map as an image. Default is False.
    plot_design (bool, optional): Whether to plot and save the design matrix as an image. Default is False.

    Returns:
    None

    Outputs:
    - Saves the z-map to the specified derivatives directory.
    - Optionally plots and saves the statistical map as an image if plot_stat is True.
    - Optionally plots and saves the design matrix as an image if plot_design is True.
    """
    # Load events and fMRI image for the run
    block = getattr(subject, run)
    events = block.extend_events_df()
    img_path = subject.img.get(run)
    fmri_img = image.load_img(img_path)

    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr / 2., (n - .5) * tr, n)

    # if demean_modulator is True, change the model label.
    if demean_modulator:
        model_label = f"{model_label}_demeaned_modulator"
    # if sample_mask is not None, change the model label.
    if sample_mask is not None:
        model_label = f"{model_label}_masked"

    # Ignore warnings related to null duration events and unexpected columns in events data
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")

    # Build the design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frametimes,
        events=events,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=high_pass,
        add_regs=confounds
    )

    # Add the parametric modulator for the first stimulus presentation
    condition = 'first_stim_presentation'
    reg_value = compute_parametric_modulator(events, condition, parametric_modulator_column,
                                             frametimes, hrf_model, center=demean_modulator)
    design_matrix.insert(1, parametric_modulator_column, reg_value)

    # Optionally plot and save the design matrix
    if plot_design:
        design_matrix_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_design_matrix.png')
        plot_design_matrix(design_matrix, output_file=design_matrix_path)
        print(f"Design matrix for {model_label} saved to {design_matrix_path}")

    # Fit the first-level model
    model = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
    model = model.fit(fmri_img, design_matrices=design_matrix, sample_masks=sample_mask)

    # Compute betamap and save it
    z_map = model.compute_contrast(
        contrast_def=f"{parametric_modulator_column}", output_type="z_score"
    )
    z_map_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_z_map.nii.gz')
    z_map.to_filename(z_map_path)
    print(f"{model_label.capitalize()} betamap results saved to {z_map_path}")

    # Compute constrast and save it
    z_map_contrast = model.compute_contrast(
        contrast_def=f"{parametric_modulator_column} - first_stim_presentation", output_type="z_score"
    )
    z_map_contrast_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_contrast_z_map.nii.gz')
    z_map_contrast.to_filename(z_map_contrast_path)
    print(f"{model_label.capitalize()} contrast results saved to {z_map_contrast_path}")

    # Optionally plot and save the statistical map
    if plot_stat:
        plot_stat_map(
            z_map,
            threshold=3.0,
            title=f"{model_label.capitalize()}: Subject {subject.sub_id}, Run {run} Contrast",
            output_file=os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_stat_map.png')
        )
    return model

# Wrapper functions for running specific models
def run_model_rl(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, demean_modulator=True, plot_stat=False, plot_design=True):
    """
    Wrapper to run Model 1 analysis.
    """
    _ = run_model(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir,
              model_label='model_rl', parametric_modulator_column='first_stim_value_rl', demean_modulator=demean_modulator,
              plot_stat=plot_stat, plot_design=plot_design)


def run_model_ck(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, demean_modulator=True, plot_stat=False, plot_design=True):
    """
    Wrapper to run Model 2 analysis.
    """
    _ = run_model(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir,
              model_label='model_ck', parametric_modulator_column='first_stim_value_ck', demean_modulator=demean_modulator,
              plot_stat=plot_stat, plot_design=plot_design)


def run_model_non_parametric(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir,
                            model_label, modulator, demean_modulator=True, plot_stat=False, plot_design=True):

    # LUT for stimulus rewards and frequencies
    stim_rewards = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5}
    stim_frequ = {0: 0, 1: 0, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1, 8: 0}
    # Load events and fMRI image for the run
    block = getattr(subject, run)
    columns_event = {'first_stim':'first_stim_presentation'}
    events = block.extend_events_df(columns_event)
    events['first_stim'] = events['first_stim'].astype(int)
    events['first_stim_reward'] = events['first_stim'].map(stim_rewards)
    events['first_stim_frequ'] = events['first_stim'].map(stim_frequ)

    img_path = subject.img.get(run)
    fmri_img = image.load_img(img_path)

    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr / 2., (n - .5) * tr, n)

    model_label = f"{model_label}_{modulator}"
    # if demean_modulator is True, change the model label.
    if demean_modulator:
        model_label = f"{model_label}_demeaned_modulator"
    # if sample_mask is not None, change the model label.
    if sample_mask is not None:
        model_label = f"{model_label}_masked"

    # Ignore warnings related to null duration events and unexpected columns in events data
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")

    # Build the design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frametimes,
        events=events,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=high_pass,
        add_regs=confounds
    )

    # Add the parametric modulator for the first stimulus presentation
    condition = 'first_stim_presentation'
    reg_value = compute_parametric_modulator(events, condition, f'first_stim_{modulator}',
                                             frametimes, hrf_model, center=demean_modulator)
    design_matrix.insert(1, f'first_stim_{modulator}', reg_value)

    # Optionally plot and save the design matrix
    if plot_design:
        design_matrix_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_design_matrix.png')
        plot_design_matrix(design_matrix, output_file=design_matrix_path)
        print(f"Design matrix for {model_label} saved to {design_matrix_path}")

    # Fit the first-level model
    model = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
    model = model.fit(fmri_img, design_matrices=design_matrix, sample_masks=sample_mask)

    # Compute betamap and save it
    z_map = model.compute_contrast(
        contrast_def=f'first_stim_{modulator}', output_type="z_score"
    )
    z_map_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_z_map.nii.gz')
    z_map.to_filename(z_map_path)
    print(f"{model_label.capitalize()} betamap results saved to {z_map_path}")

    # # Compute constrast and save it
    # z_map_contrast = model.compute_contrast(
    #     contrast_def=f"{parametric_modulator_column} - first_stim_presentation", output_type="z_score"
    # )
    # z_map_contrast_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_contrast_z_map.nii.gz')
    # z_map_contrast.to_filename(z_map_contrast_path)
    # print(f"{model_label.capitalize()} contrast results saved to {z_map_contrast_path}")

    # Optionally plot and save the statistical map
    if plot_stat:
        plot_stat_map(
            z_map,
            threshold=3.0,
            title=f"{model_label.capitalize()}: Subject {subject.sub_id}, Run {run} Contrast",
            output_file=os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_stat_map.png')
        )
    return model

# Model with no modulator, for RSA
# Function to run first-level analysis for a given model
def run_model_RSA(subject, run, confounds, sample_mask, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir,
              model_label, plot_stat=False, plot_design=False):
    """
    Run the first-level fMRI analysis model for a given subject and run.

    Parameters:
    subject (object): The subject object containing fMRI data and metadata.
    run (str): The specific run identifier within the subject's data.
    confounds (DataFrame): Confounding variables to include in the design matrix.
    sample_mask (array): Boolean mask to indicate which volumes to include in the analysis.
    tr (float): Repetition time of the fMRI acquisition.
    hrf_model (str): Hemodynamic response function model to use.
    high_pass (float): High-pass filter cutoff frequency in Hz.
    smoothing_fwhm (float): Full-width at half maximum for spatial smoothing in mm.
    derivatives_dir (str): Directory path to save the output z-map and statistical map.
    model_label (str): Label for the model, e.g., 'model1' or 'model2'.
    plot_stat (bool, optional): Whether to plot and save the statistical map as an image. Default is False.
    plot_design (bool, optional): Whether to plot and save the design matrix as an image. Default is False.

    Returns:
    None

    Outputs:
    - Saves the z-map to the specified derivatives directory.
    - Optionally plots and saves the statistical map as an image if plot_stat is True.
    - Optionally plots and saves the design matrix as an image if plot_design is True.
    """
    # maybe put this as function parameter
    bmap = 'first_stim_presentation'
    # Load events and fMRI image for the run
    block = getattr(subject, run)
    # here we need to add a column that has the ID of the stimuli at the time of first stim presentation
    events = block.extend_events_df(columns_event={'first_stim':'first_stim_presentation'})
    img_path = subject.img.get(run)
    fmri_img = image.load_img(img_path)

    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr / 2., (n - .5) * tr, n)

    # if sample_mask is not None, change the model label.
    if sample_mask is not None:
        model_label = f"{model_label}_masked"

    # Ignore warnings related to null duration events and unexpected columns in events data
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", message=".*following unexpected columns in events data.*")

    # Build the design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frametimes,
        events=events,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=high_pass,
        add_regs=confounds
    )

    # Optionally plot and save the design matrix
    if plot_design:
        design_matrix_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_design_matrix.png')
        plot_design_matrix(design_matrix, output_file=design_matrix_path)
        print(f"Design matrix for {model_label} saved to {design_matrix_path}")

    # Fit the first-level model
    model = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
    model = model.fit(fmri_img, design_matrices=design_matrix, sample_masks=sample_mask)

    # Compute betamap and save it
    z_map = model.compute_contrast(
        contrast_def=f"{bmap}", output_type="z_score"
    )
    z_map_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_z_map.nii.gz')
    z_map.to_filename(z_map_path)
    print(f"{model_label.capitalize()} betamap results saved to {z_map_path}")

    # # Compute constrast and save it
    # z_map_contrast = model.compute_contrast(
    #     contrast_def=f"{event} - first_stim_presentation", output_type="z_score"
    # )
    # z_map_contrast_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_contrast_z_map.nii.gz')
    # z_map_contrast.to_filename(z_map_contrast_path)
    # print(f"{model_label.capitalize()} contrast results saved to {z_map_contrast_path}")

    # Optionally plot and save the statistical map
    if plot_stat:
        plot_stat_map(
            z_map,
            threshold=3.0,
            title=f"{model_label.capitalize()}: Subject {subject.sub_id}, Run {run} Contrast",
            output_file=os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_{model_label}_stat_map.png')
        )
    return model