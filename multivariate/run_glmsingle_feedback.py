#!/usr/bin/env python3
"""
GLMsingle single-trial beta estimation, locked to REWARD FEEDBACK onset — one subject.

Feedback-locked counterpart of run_glmsingle.py (which locks to first-stimulus/cue
onset). Same GLMsingle configuration and output conventions; three substantive
differences, all forced by what the feedback event is:

  1. Learning runs only. The test run has no reward feedback — Block.block_type is
     set by the presence of `raw_block.time.rewards_onset` (utils/data.py), and
     t_points_feedback is populated for learning blocks only.
  2. Conditions are the 8 STIMULUS PAIRS, not the 8 stimulus identities. At feedback
     the screen shows the reward for *both* options, so the display content is fully
     determined by the pair (empirically: reward_chosen + reward_unchosen has 0.000
     within-pair variance). Pairs are also the better-conditioned choice for
     GLMsingle's cross-validation — all 8 present in every run for every subject,
     min 3 repeats/run, no thin cells; chosen-stimulus identity gives 5.3% of cells
     <3 repeats and sometimes only 7 conditions in a run.
  3. Non-response trials are dropped. They have no feedback event (t_points_feedback
     is exactly 0 for all 221/11904 such trials); their variance is left to
     GLMdenoise, same rationale run_glmsingle.py uses for other unmodeled events.

IMPORTANT — read before interpreting these betas
------------------------------------------------
Feedback onset is only **1.95 s after cue onset (SD 0.16 s)** and TR is 2.334 s, so
82.4% of feedback events land exactly 1 TR after their cue's TR and 17.5% land in the
*same* TR. The RT jitter available for deconvolution is ~0.16 s. These betas are
therefore a cue+feedback composite, close to a 1-TR shift of run_glmsingle.py's cue
betas — not a feedback-specific signal. Modeling cue and feedback jointly is not a
fix: the two regressor sets are near-collinear by construction.

Related: objective reward level is a deterministic function of stimulus identity in
this design (0.000 within-(subject x stimulus) variance, checked across the full
62-subject BBT), so "reward" targets built from it are identity relabelings — at
feedback as much as at cue.

Usage
-----
# Single subject (local default paths)
python multivariate/run_glmsingle_feedback.py --subject 01

# Cluster, single subject (this is what submit_glmsingle_feedback.sh runs)
python multivariate/run_glmsingle_feedback.py --subject 01 \
    --base-dir /home/hfluhr/data/learninghabits \
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_feedback

# All subjects from participants_mvpa.tsv (sequential) — drop --subject
python multivariate/run_glmsingle_feedback.py --base-dir ... --bids-dir ... --output-dir ...

Subject list
------------
The canonical MVPA analysis sample is read from participants_mvpa.tsv (relative to
--base-dir) via utils.data.load_participant_list. Override with --participants-file.

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    TYPEA_ONOFF.npy
    TYPEB_FITHRF.npy
    TYPEC_FITHRF_GLMDENOISE.npy
    TYPED_FITHRF_GLMDENOISE_RR.npy             <- type-D: best model (HRF + denoise + ridge)
    DESIGNINFO.npy
    RUNWISEFIR.npy
    figures/
    sub-<id>_glmSingle_betas_FEEDBACK.nii.gz       <- (x,y,z, n_trials) type-D betas
    sub-<id>_glmSingle_betas_FEEDBACK_info.csv     <- trial -> pair / choice / reward / run
    sub-<id>_glmSingle_betas_FEEDBACK_mean.nii.gz  <- per-voxel mean |beta| across trials
    sub-<id>_glmSingle_betas_FEEDBACK_std.nii.gz   <- per-voxel STD across trials
    sub-<id>_glmSingle_betas_FEEDBACK_snr.nii.gz   <- per-voxel mean/STD ("SNR")
    sub-<id>_glmSingle_qc_R2.nii.gz                <- type-D R2 (3D)
    sub-<id>_glmSingle_qc_R2run.nii.gz             <- type-D per-run R2 (4D, 2 vols: learning1/learning2)
    sub-<id>_glmSingle_qc_HRFindex.nii.gz          <- chosen HRF library index (3D)
    sub-<id>_glmSingle_qc_FRACvalue.nii.gz         <- ridge regularization fraction (3D)
    sub-<id>_glmSingle_qc_noisepool.nii.gz         <- GLMdenoise noise-pool mask, int8 (3D)
    sub-<id>_glmSingle_qc_R2_bytype.nii.gz         <- R2 for types A,B,C,D stacked (4D, 4 vols)
<output-dir>/logs/
    glmsingle_feedback_sub-<id>.log     <- NOT under sub-<id>/: GLMsingle rmtree's that
                                           dir at the start of fit(), see main()

The info CSV carries every deterministic per-trial quantity (pair, chosen/unchosen
stimulus and reward, RT, accuracy) so downstream analyses need no BBT join for them.
`run` + `trial` uniquely identify the trial in `Subject.<run>.trials`. Model-derived
targets (Q-values) still require the BBT; note that utils.data.load_target_from_bbt
cannot be used as-is — it reconstructs order from t_first_stim over all three runs and
asserts on a `stim_name` column this CSV does not have.

Design choices (differences from run_glmsingle.py flagged inline)
-----------------------------------------------------------------
- 8 stimulus-pair conditions at feedback onset; within-trial events excluded.
- extra_regressors not used: GLMdenoise handles noise, and passing explicit confounds
  would prevent it from learning the noise pool.
- sessionindicator = [1,1]: both learning runs pooled for GLMdenoise noise estimation.
- stimdur: mean(t_iti_onset - t_points_feedback) over the modeled trials (~1.47 s).
- Beta ordering is verified, not assumed: the expected condition sequence is rebuilt
  from the trial tables and asserted against DESIGNINFO's stimorder.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from glmsingle.glmsingle import GLM_single

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject, load_participant_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TR = 2.33384  # confirmed in matlab/first_lvl/glm2_chosen.m

# Learning runs only — the test run has no reward feedback.
RUNS = ['learning1', 'learning2']

N_PAIRS_EXPECTED = 8  # 8 fixed stimulus pairs per subject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_feedback_trials(trials, run_name):
    """Responded trials only, sorted by feedback onset, with derived choice columns.

    Non-response trials carry t_points_feedback == 0 exactly (checked across the full
    62-subject BBT: 221/221), so the two filters below are redundant by design — both
    are applied anyway so a violation surfaces here rather than as a silent onset at
    volume 0.
    """
    t = trials[trials['action'].notna() & (trials['t_points_feedback'] > 0)].copy()
    n_dropped = len(trials) - len(t)
    if n_dropped:
        logging.info(f"  {run_name}: dropped {n_dropped}/{len(trials)} trials "
                     f"without a feedback event (no response)")

    chose_left = t['action'] == 1.0
    t['chosen_stim']        = np.where(chose_left, t['left_stim'],      t['right_stim'])
    t['unchosen_stim']      = np.where(chose_left, t['right_stim'],     t['left_stim'])
    t['chosen_stim_name']   = np.where(chose_left, t['left_stim_name'], t['right_stim_name'])
    t['unchosen_stim_name'] = np.where(chose_left, t['right_stim_name'], t['left_stim_name'])
    t['reward_chosen']      = t['reward']  # == value of the chosen stimulus
    t['reward_unchosen']    = np.where(chose_left, t['right_value'],    t['left_value'])

    # Pair key is order-invariant: the display shows both options regardless of side.
    lo = t[['left_stim', 'right_stim']].min(axis=1).astype(int)
    hi = t[['left_stim', 'right_stim']].max(axis=1).astype(int)
    t['pair'] = lo.astype(str) + '-' + hi.astype(str)
    lo_name = np.where(t['left_stim'] <= t['right_stim'], t['left_stim_name'],  t['right_stim_name'])
    hi_name = np.where(t['left_stim'] <= t['right_stim'], t['right_stim_name'], t['left_stim_name'])
    t['pair_names'] = pd.Series(lo_name, index=t.index) + '+' + pd.Series(hi_name, index=t.index)

    return t.sort_values('t_points_feedback')


def build_pair_index(run_trials):
    """Map pair key -> design-matrix column, shared across runs.

    The pair set is subject-specific (unlike run_glmsingle.py's global STIM_NAMES),
    and the mapping *must* be identical across runs: GLMsingle's calcbadness matches
    conditions across runs by column index, so a per-run mapping would silently
    scramble the cross-validation.
    """
    pairs = sorted(set().union(*(set(t['pair']) for t in run_trials.values())))
    if len(pairs) != N_PAIRS_EXPECTED:
        logging.warning(f"expected {N_PAIRS_EXPECTED} stimulus pairs, found {len(pairs)}: {pairs}")
    for run, t in run_trials.items():
        missing = set(pairs) - set(t['pair'])
        if missing:
            logging.warning(f"  {run}: pairs absent from this run (no cross-run repeats "
                            f"for them): {sorted(missing)}")
    return {p: i for i, p in enumerate(pairs)}


def build_design_matrix(trials, n_volumes, run_name, pair2col):
    """Binary TR-indexed design matrix: (n_volumes, n_pairs). Floor-division onset
    assignment follows run_glmsingle.py / the reference pipeline convention.

    `scan_start` reproduces run_glmsingle.py's convention (subtract the run's first
    cue onset). Onsets are already scanner-trigger-relative, and the first cue lands
    at a median of 0.01 s post-trigger, so this is a no-op in practice — kept so the
    two pipelines share a time origin exactly.
    """
    dm = np.zeros((n_volumes, len(pair2col)))
    scan_start = trials['t_first_stim'].min()
    out_of_range = 0
    for _, row in trials.iterrows():
        tr_idx = int((row['t_points_feedback'] - scan_start) / TR)
        if 0 <= tr_idx < n_volumes:
            dm[tr_idx, pair2col[row['pair']]] = 1.0
        else:
            out_of_range += 1
    if out_of_range:
        logging.warning(f"{run_name}: {out_of_range} onsets outside [0, {n_volumes})")

    # GLMsingle cannot represent two events in one TR. Feedback onsets are ~10 s apart
    # (cue-to-cue ISI 10.14 s +/- 0.56), so collisions should never happen — but a
    # collision silently costs an event, so check rather than trust.
    n_marked, n_expected = int(dm.sum()), len(trials) - out_of_range
    if n_marked != n_expected:
        raise ValueError(
            f"{run_name}: {n_expected - n_marked} feedback event(s) collided in a TR bin "
            f"({n_marked} marked, {n_expected} expected) — betas would be misaligned."
        )
    logging.info(f"  {run_name}: DM {dm.shape}, {n_marked} events marked")
    return dm


def extract_betas(typed, subject_output, run_trials, pair2col):
    """Build the trial-info DataFrame and select the type-D feedback betas.

    `typed` is the already-loaded TYPED_FITHRF_GLMDENOISE_RR.npy dict — loaded once by
    run_subject() and reused here and in save_qc_maps() to avoid a second
    multi-hundred-MB unpickle.

    Unlike run_glmsingle.py (which recovers the run label by counting per-condition
    occurrences), the expected condition sequence is rebuilt directly from the trial
    tables and asserted against `stimorder`. GLMsingle emits one beta per event in
    chronological order, runs concatenated in RUNS order; the assertion turns that
    assumption into a checked precondition.
    """
    designinfo = np.load(subject_output / "DESIGNINFO.npy", allow_pickle=True).item()

    betas     = typed['betasmd']
    stimorder = np.array(designinfo['stimorder'])

    ordered = pd.concat(
        [run_trials[run].assign(run=run) for run in RUNS],
        ignore_index=False,
    ).reset_index()  # 'trial' index -> column

    expected = ordered['pair'].map(pair2col).to_numpy()
    if len(stimorder) != len(expected) or not np.array_equal(stimorder, expected):
        raise ValueError(
            f"beta ordering mismatch: DESIGNINFO stimorder has {len(stimorder)} entries, "
            f"trial tables give {len(expected)}"
            + ("" if len(stimorder) != len(expected)
               else f"; first divergence at index {int(np.argmax(stimorder != expected))}")
            + " — GLMsingle did not emit betas in the assumed chronological order."
        )

    trial_info = pd.DataFrame({
        'beta_vol_idx':       np.arange(len(expected), dtype=int),
        'run':                ordered['run'],
        'trial':              ordered['trial'].astype(int),
        'pair_col':           expected.astype(int),
        'pair':               ordered['pair'],
        'pair_names':         ordered['pair_names'],
        'chosen_stim':        ordered['chosen_stim'].astype(int),
        'chosen_stim_name':   ordered['chosen_stim_name'],
        'unchosen_stim':      ordered['unchosen_stim'].astype(int),
        'unchosen_stim_name': ordered['unchosen_stim_name'],
        'reward_chosen':      ordered['reward_chosen'].astype(float),
        'reward_unchosen':    ordered['reward_unchosen'].astype(float),
        'correct':            ordered['correct'].astype(float),
        'rt':                 ordered['rt'].astype(float),
        't_points_feedback':  ordered['t_points_feedback'].astype(float),
    })
    trial_info.index.name = 'trial_id'
    return betas, trial_info


# TYPEA/B/C source files + their R²-equivalent key, for the bonus by-type R² export.
# TYPED is excluded — its R2 is already in `typed` (passed in from run_subject()).
# Type A's key differs (`onoffR2`) because it's a single pooled on/off regressor.
BYTYPE_R2_SOURCES = [
    ('TYPEA_ONOFF.npy',             'onoffR2'),
    ('TYPEB_FITHRF.npy',            'R2'),
    ('TYPEC_FITHRF_GLMDENOISE.npy', 'R2'),
]


def save_qc_maps(typed, subject_output, subject, ref_img, betas):
    """Persist QC-diagnostic NIfTI volumes alongside the type-D betas.

    All maps are unmasked, full-FOV — same convention as betas_FEEDBACK.nii.gz.
    Reuses `typed` (no TYPED reload); TYPEA/B/C are each read once, only for the
    bonus by-type R² export.
    """
    def _save(arr, suffix, dtype=np.float32):
        image.new_img_like(ref_img, np.asarray(arr).astype(dtype)).to_filename(
            str(subject_output / f"sub-{subject}_glmSingle_qc_{suffix}.nii.gz")
        )

    _save(typed['R2'],        'R2')
    _save(typed['R2run'],     'R2run')          # (x,y,z,2): learning1, learning2
    _save(typed['HRFindex'],  'HRFindex')
    _save(typed['FRACvalue'], 'FRACvalue')
    _save(typed['noisepool'], 'noisepool', dtype=np.int8)

    # Per-voxel beta summary stats across trials (matches QC notebook Sec. 6)
    beta_mean = np.abs(betas).mean(axis=-1)
    beta_std  = betas.std(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        beta_snr = np.where(beta_std > 0, beta_mean / beta_std, 0.0).astype(np.float32)
    for arr, name in [(beta_mean, 'mean'), (beta_std, 'std'), (beta_snr, 'snr')]:
        image.new_img_like(ref_img, arr.astype(np.float32)).to_filename(
            str(subject_output / f"sub-{subject}_glmSingle_betas_FEEDBACK_{name}.nii.gz")
        )

    # Bonus: R² by model type (A, B, C, D order), 4D volume.
    r2_by_type = [None, None, None, typed['R2']]
    for i, (fname, key) in enumerate(BYTYPE_R2_SOURCES):
        d = np.load(subject_output / fname, allow_pickle=True).item()
        r2_by_type[i] = d[key]
        del d
    _save(np.stack(r2_by_type, axis=-1), 'R2_bytype')


# ---------------------------------------------------------------------------
# Main subject function
# ---------------------------------------------------------------------------

def run_subject(subject, base_dir, bids_dir, output_dir, overwrite=False):
    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / "TYPED_FITHRF_GLMDENOISE_RR.npy"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping "
                     f"(pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)
    figures_dir = subject_output / "figures"
    figures_dir.mkdir(exist_ok=True)

    # --- Load subject ---
    logging.info(f"Loading sub-{subject}")
    sub = Subject(
        base_dir=str(base_dir),
        subject_id=subject,
        include_imaging=True,
        bids_dir=str(bids_dir),
    )
    run_trials = {
        run: select_feedback_trials(getattr(sub, run).trials, run)
        for run in RUNS
    }
    pair2col = build_pair_index(run_trials)
    logging.info(f"{len(pair2col)} pair conditions: "
                 f"{ {p: i for p, i in sorted(pair2col.items(), key=lambda kv: kv[1])} }")

    # --- stimdur: mean feedback display duration ---
    all_trials = pd.concat(run_trials.values())
    stimdur = (all_trials['t_iti_onset'] - all_trials['t_points_feedback']).mean()
    logging.info(f"stimdur = {stimdur:.3f}s  ({len(all_trials)} feedback events)")

    # --- Load BOLD + design matrices ---
    fmri_data, design_matrices = [], []
    for run in RUNS:
        img = nib.load(sub.get_img_path(run))
        logging.info(f"  {run}: {img.shape}")
        img_tr = img.header.get_zooms()[-1]
        assert abs(img_tr - TR) < 1e-3, (
            f"sub-{subject} {run}: NIfTI header TR ({img_tr}) does not match "
            f"hardcoded TR ({TR}) — data may use a different acquisition "
            f"protocol than assumed."
        )
        fmri_data.append(img.get_fdata(dtype=np.float32))
        design_matrices.append(
            build_design_matrix(run_trials[run], img.shape[-1], run, pair2col)
        )

    # --- Fit GLMsingle ---
    opt = dict(
        sessionindicator = np.ones((1, len(RUNS)), dtype=int),
        wantlibrary      = 1,
        wantglmdenoise   = 1,
        wantfracridge    = 1,
        wantfileoutputs  = [1, 1, 1, 1],
    )
    logging.info("Fitting GLMsingle...")
    GLM_single(opt).fit(
        design_matrices, fmri_data, stimdur, TR,
        outputdir=str(subject_output),
        figuredir=str(figures_dir),
    )
    logging.info("Fitting complete.")

    # --- Extract and save type-D betas + QC diagnostic maps ---
    typed = np.load(subject_output / "TYPED_FITHRF_GLMDENOISE_RR.npy",
                    allow_pickle=True).item()
    betas, trial_info = extract_betas(typed, subject_output, run_trials, pair2col)

    ref_img = nib.load(sub.get_img_path(RUNS[0]))
    image.new_img_like(ref_img, betas).to_filename(
        str(subject_output / f"sub-{subject}_glmSingle_betas_FEEDBACK.nii.gz")
    )
    trial_info.to_csv(
        str(subject_output / f"sub-{subject}_glmSingle_betas_FEEDBACK_info.csv")
    )

    save_qc_maps(typed, subject_output, subject, ref_img, betas)
    del typed  # release before looping to the next subject in batch mode

    logging.info(f"sub-{subject}: done — betas shape {betas.shape}, "
                 f"{len(trial_info)} trials")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run GLMsingle single-trial estimation locked to reward-feedback "
                    "onset (learning runs only). If --subject is omitted, all subjects "
                    "in --participants-file are run."
    )
    parser.add_argument("--subject", default=None,
                        help="Subject ID without 'sub-' prefix, e.g. 01. "
                             "If omitted, all subjects in --participants-file are run.")
    parser.add_argument("--participants-file", default="participants_mvpa.tsv",
                        help="TSV filename (relative to --base-dir) listing subject IDs "
                             "(default: participants_mvpa.tsv)")
    parser.add_argument("--base-dir",
                        required=True,
                        help="Root data dir containing spm_format/")
    parser.add_argument("--bids-dir",
                        required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--output-dir",
                        required=True,
                        help="Output root directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rerun even if outputs already exist")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subject list and log destination
    # Logs live in <output-dir>/logs/, NOT in the per-subject dir: GLM_single.fit()
    # starts with `shutil.rmtree(outputdir)` on <output-dir>/sub-<id>, which unlinks
    # any file opened there beforehand. The FileHandler keeps writing to the unlinked
    # inode, so nothing errors — the log just silently doesn't exist afterwards.
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subjects = [args.subject]
        log_file = log_dir / f"glmsingle_feedback_sub-{args.subject}.log"
    else:
        subjects = load_participant_list(str(base_dir), file_name=args.participants_file)
        log_file = log_dir / "glmsingle_feedback_batch.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    logging.info(f"Processing {len(subjects)} subject(s) "
                 f"[source: {args.participants_file if not args.subject else '--subject flag'}]")

    for subject in subjects:
        run_subject(
            subject    = subject,
            base_dir   = base_dir,
            bids_dir   = Path(args.bids_dir),
            output_dir = output_dir,
            overwrite  = args.overwrite,
        )


if __name__ == "__main__":
    main()
