import os
import numpy as np
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_stat_map, plot_design_matrix
from nilearn.glm.first_level.hemodynamic_models import compute_regressor


def compute_parametric_modulator(events, condition, modulator, frametimes, hrf_model):
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
    regressor, _ = compute_regressor(exp_condition=exp_condition,
                                     hrf_model=hrf_model,
                                     frame_times=frametimes
                                     )
    
    return regressor

# Function to run first-level analysis for Model 1
def run_model_1(subject, run, confounds, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, plot_stat=False, plot_design=False):
    """
    Run the first-level fMRI analysis model for a given subject and run.

    Parameters:
    subject (object): The subject object containing fMRI data and metadata.
    run (str): The specific run identifier within the subject's data.
    confounds (DataFrame): Confounding variables to include in the design matrix.
    tr (float): Repetition time of the fMRI acquisition.
    hrf_model (str): Hemodynamic response function model to use.
    high_pass (float): High-pass filter cutoff frequency in Hz.
    smoothing_fwhm (float): Full-width at half maximum for spatial smoothing in mm.
    derivatives_dir (str): Directory path to save the output z-map and statistical map.
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

    # Build design matrix
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

    # Optionally plot and save the design matrix
    if plot_design:
        design_matrix_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_model1_design_matrix.png')
        plot_design_matrix(X1, output_file=design_matrix_path)
        print(f"Design matrix for Model 1 saved to {design_matrix_path}")

    # Fit the first-level model
    model1 = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=False)
    model1 = model1.fit(fmri_img, design_matrices=X1)

    # Compute the contrast
    z_map1 = model1.compute_contrast(
        contrast_def="first_stim_presentation - iti", output_type="z_score"
    )

    # Save the z-map to derivatives directory
    z_map1_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_model1_z_map.nii.gz')
    z_map1.to_filename(z_map1_path)
    print(f"Model 1 results saved to {z_map1_path}")

    # Plot and save the statistical map as an image if requested
    if plot_stat:
        plot_stat_map(
            z_map1,
            threshold=3.0,
            title=f"Model 1: Subject {subject.sub_id}, Run {run} Contrast",
            output_file=os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_model1_stat_map.png')
        )

# Function to run first-level analysis for Model 2
def run_model_2(subject, run, confounds, tr, hrf_model, high_pass, smoothing_fwhm, derivatives_dir, plot_stat=False, plot_design=False):
    """
    Run the first-level fMRI analysis model for a given subject and run.
    This model includes a parametric modulator for the first stimulus presentation.

    Parameters:
    subject (object): The subject object containing fMRI data and metadata.
    run (str): The specific run identifier within the subject's data.
    confounds (DataFrame): Confounding variables to include in the design matrix.
    tr (float): Repetition time of the fMRI acquisition.
    hrf_model (str): Hemodynamic response function model to use.
    high_pass (float): High-pass filter cutoff frequency in Hz.
    smoothing_fwhm (float): Full-width at half maximum for spatial smoothing in mm.
    derivatives_dir (str): Directory path to save the output z-map and statistical map.
    plot_stat (bool, optional): Whether to plot and save the statistical map as an image. Default is False.
    plot_design (bool, optional): Whether to plot and save the design matrix as an image. Default is False.

    Returns:
    None

    Outputs:
    - Saves the z-map to the specified derivatives directory.
    - Optionally plots and saves the statistical map as an image if plot_stat is True.
    - Optionally plots and saves the design matrix as an image if plot_design is True.
    """
    # Similar to model 1 but using different settings
    block = getattr(subject, run)
    events = block.extend_events_df()
    img_path = subject.img.get(run)
    fmri_img = image.load_img(img_path)

    n = fmri_img.shape[-1]
    frametimes = np.linspace(tr/2., (n - .5) * tr, n)

    X2 = make_first_level_design_matrix(
        frame_times=frametimes,
        events=events,
        hrf_model=hrf_model,
        drift_model=None,
        high_pass=high_pass,
        add_regs=confounds
    )
    # Add the parametric modulator for the first stimulus presentation
    condition = 'first_stim_presentation'
    reg_rl_value = compute_parametric_modulator(events, condition, 'first_stim_value_rl',
                                            frametimes, hrf_model)
    X2.insert(1, 'first_stim_value_rl', reg_rl_value)

    # Optionally plot and save the design matrix
    if plot_design:
        design_matrix_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_model2_design_matrix.png')
        plot_design_matrix(X2, output_file=design_matrix_path)
        print(f"Design matrix for Model 2 saved to {design_matrix_path}")

    model2 = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, minimize_memory=False)
    model2 = model2.fit(fmri_img, design_matrices=X2)

    z_map2 = model2.compute_contrast(
        contrast_def="second_stim_presentation - iti", output_type="z_score"
    )

    z_map2_path = os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_model2_z_map.nii.gz')
    z_map2.to_filename(z_map2_path)
    print(f"Model 2 results saved to {z_map2_path}")

    if plot_stat:
        plot_stat_map(
            z_map2,
            threshold=3.0,
            title=f"Model 2: Subject {subject.sub_id}, Run {run} Contrast",
            output_file=os.path.join(derivatives_dir, f'{subject.sub_id}_run-{run}_model2_stat_map.png')
        )