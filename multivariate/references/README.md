# References

Reference code from collaborators, kept for context and inspiration rather than direct use in the pipeline.

## SPM LSS (Sarah)

Single-trial beta estimation using the Least Squares Separate (LSS) approach in SPM:

- `run_1L_trialwise_LSS.m` — main script running LSS estimation trial-by-trial
- `get_onsets_except.m` — helper to build onset vectors excluding the current trial
- `get_trial_info.m` — helper to extract trial information from the SPM structure

One full SPM GLM is estimated per trial, so it's computationally expensive but gives clean isolation. The trial-of-interest always lands in session 1 of the design matrix so its betas are reliably `beta_0001` (cue) and `beta_0002` (outcome). All other trials are collapsed into per-condition nuisance regressors (same condition as TOI, plus each remaining condition separately). Classical HRF, no derivatives; AR(1) autocorrelation correction; 24 motion params + FD + motion outlier scrubbing.

## GLMdenoise (Maike)

- `fit_glmDenoise_bothStim.py` — fits GLMdenoise to both stimuli using the GLMsingle toolbox

Early-stage prototype from a different project (`numrisk`), with open questions and TODOs still in the code — less directly portable. Conditions labelled by numerical magnitude (e.g. `stimulus1_8`), so repeated presentations of the same value share a label and enable GLMsingle's internal cross-validation. Design matrix built with nilearn's FIR model then binarized to impulse functions — GLMsingle handles HRF estimation internally from the stimulus duration scalar.

## GLMsingle pipeline (Rishabh + Sarah)

- `glmsingle_pipeline_withOutcomes.py` — full pipeline including outcome regressors

Conditions labelled by stimulus identity (e.g. `animal_camel`) across all runs, so repeated presentations of the same stimulus enable cross-validation; outcome events are added as nuisance regressors (`outcome_reward` / `outcome_hidden`) so they don't contaminate cue betas. Onset assigned by floor division (`int(onset / tr)`) rather than rounding; stimulus duration hardcoded to 1.5s for both cues and outcomes (acknowledged limitation in comments). All four GLMsingle model types saved to disk (A: canonical HRF; B: optimized HRF; C: + denoising; D: + ridge regression). TR read per-subject from BIDS JSON sidecar; excluded subjects from motion QC spreadsheet. Saves betas, R², HRF index map, tSNR, and a trial-info CSV mapping each beta volume to its condition label.

## GLMsingle (Gilles)

- `estimate_single_trials.py` — core estimation logic (from the `numloss` project)
- `submit_glm.sh` — SLURM cluster submission script

From the `numloss` project — different task structure, so some design choices reflect that experiment. Conditions labelled by stimulus magnitude value (same logic as Maike); `sessionindicator` derived from session/run numbers in filenames, which affects how GLMdenoise pools noise estimates across runs. Only type D (optimized HRF + denoising + ridge) saved to disk (`wantfileoutputs = [0,0,0,1]`), keeping output small; optional 5mm FWHM smoothing before fitting.
