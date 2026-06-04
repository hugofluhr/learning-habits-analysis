# References

Reference code from collaborators, kept for context and inspiration rather than direct use in the pipeline.

## SPM LSS (Sarah)

Single-trial beta estimation using the Least Squares Separate (LSS) approach in SPM:

- `run_1L_trialwise_LSS.m` — main script running LSS estimation trial-by-trial
- `get_onsets_except.m` — helper to build onset vectors excluding the current trial
- `get_trial_info.m` — helper to extract trial information from the SPM structure

## GLMdenoise (Maike)

- `fit_glmDenoise_bothStim.py` — fits GLMdenoise to both stimuli using the GLMsingle toolbox

## GLMsingle pipeline (Rishabh)

Python pipeline for single-trial response estimation using GLMsingle:

- `estimate_single_trials.py` — core estimation logic
- `glmsingle_pipeline_withOutcomes.py` — full pipeline including outcome regressors
- `submit_glm.sh` — cluster submission script
