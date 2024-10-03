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