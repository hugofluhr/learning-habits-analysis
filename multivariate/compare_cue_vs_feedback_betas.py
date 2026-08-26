#!/usr/bin/env python3
"""
How redundant are the feedback-locked betas with the cue-locked betas? — one subject.

run_glmsingle_feedback.py predicts, from timing alone, that its betas are a cue+feedback
composite rather than a feedback-specific signal: feedback onset is only 1.95 s after cue
onset (SD 0.16 s) at TR = 2.334 s, so 82.4% of feedback events land exactly 1 TR after
their cue's TR and 17.5% in the same TR. This script measures whether that prediction
holds, before anything downstream is built on those betas.

A raw correlation between two beta sets is close to meaningless on its own — every beta
map shares a large common spatial structure, so even unrelated trials correlate highly.
Every measure here is therefore reported against two baselines:

  shuffled  — feedback trial t vs a random *different* cue trial in the same run.
              The floor: correlation from shared structure alone.
  adjacent  — cue trial k vs cue trial k+1 (same run). A reference for "two different
              trials of the same event type", i.e. what ordinary temporal proximity
              already buys you.

Read the output as:
  matched ~= shuffled  -> no trial-specific coupling beyond shared spatial structure
  matched >> adjacent  -> the feedback betas are largely redundant with the cue betas

Despite the name this compares any two GLMsingle beta sets; --cue-dir / --feedback-dir
and the *_info.csv schemas are the only cue/feedback-specific parts.

Usage
-----
python multivariate/compare_cue_vs_feedback_betas.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --cue-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --feedback-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle_feedback \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/cue_vs_feedback \\
    --roi-mask visualcortex /home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz

Which model type to compare (--beta-type)
-----------------------------------------
Defaults to D, the production betas every downstream analysis uses — but a type-D-only
comparison is confounded for this question. D's GLMdenoise PC count and fracridge
fraction are cross-validated over *condition repeats*, and the two models define
conditions differently (8 stimulus identities for cue, 8 stimulus pairs for feedback),
so any decorrelation mixes different event timing with different hyperparameter
selection. Type B (FITHRF only, no denoise, no ridge) isolates the timing effect.
Run `--beta-type B D` to separate them. Type A is unavailable: ONOFF pools every event
into a single beta, leaving nothing per-trial to correlate.

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/sub-<id>_cue_vs_feedback.csv          — one row per (beta_type, mask)
<output-dir>/sub-<id>/sub-<id>_cue_vs_feedback_r_type<T>.nii.gz — voxelwise matched r (--save-maps)
<output-dir>/logs/cue_vs_feedback_sub-<id>.log

--output-dir deliberately defaults outside the GLMsingle output trees: GLM_single.fit()
does `shutil.rmtree(outputdir)` on <glmsingle-dir>/sub-<id>, so results written there
would be destroyed by any later GLMsingle rerun (cf. commit ed8ec32).

Trial alignment
---------------
The two beta series have different lengths and share no key. Cue betas cover all 328
trials (learning1 96, learning2 96, test 136), chronological within run; their info CSV
has no trial number. Feedback betas cover only responded learning trials (~187) but
their info CSV does carry `run` and `trial`. Since every trial has a cue, position
within a run equals trial order:

    cue_idx(learning1, trial k) = k - 1
    cue_idx(learning2, trial k) = 96 + (k - 1)

That arithmetic is validated, not trusted — a misalignment would produce a low
correlation, which is exactly the result being tested for, so it must not be able to
pass silently. See match_trials().
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, math_img

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject

# Cue betas span all three runs in this order; feedback betas only the learning ones.
CUE_RUNS      = ['learning1', 'learning2', 'test']
EXPECTED_CUE_COUNTS = {'learning1': 96, 'learning2': 96, 'test': 136}

# Raw GLMsingle model outputs, for --beta-type. Type A is excluded: ONOFF pools every
# event into one beta per voxel, so there are no per-trial volumes to correlate.
BETA_FILES = {
    'B': 'TYPEB_FITHRF.npy',
    'C': 'TYPEC_FITHRF_GLMDENOISE.npy',
    'D': 'TYPED_FITHRF_GLMDENOISE_RR.npy',
}


def load_betas(subject_dir, subject, tag, beta_type, n_expected):
    """Load one model type's single-trial betas as (x, y, z, n_trials).

    Type D comes from the NIfTI the runners already export; B and C are only written
    as raw .npy, so they are unpickled here.
    """
    if beta_type == 'D':
        path = subject_dir / f"sub-{subject}_glmSingle_betas_{tag}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"sub-{subject}: missing input {path}")
        arr = nib.load(path).get_fdata(dtype=np.float32)
    else:
        path = subject_dir / BETA_FILES[beta_type]
        if not path.exists():
            raise FileNotFoundError(f"sub-{subject}: missing input {path}")
        arr = np.load(path, allow_pickle=True).item()['betasmd'].astype(np.float32)

    if arr.shape[-1] != n_expected:
        raise ValueError(
            f"sub-{subject}: type-{beta_type} {tag} betas have {arr.shape[-1]} volumes, "
            f"expected {n_expected} (one per trial)"
        )
    return arr


# ---------------------------------------------------------------------------
# Correlation helpers (NaN-free inputs assumed — see finite-voxel masking)
# ---------------------------------------------------------------------------

def voxelwise_temporal_r(A, B):
    """Per-voxel Pearson r between two (n_voxels, n_trials) arrays, across trials."""
    Ad = A - A.mean(axis=1, keepdims=True)
    Bd = B - B.mean(axis=1, keepdims=True)
    num = (Ad * Bd).sum(axis=1)
    den = np.sqrt((Ad ** 2).sum(axis=1) * (Bd ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def trialwise_spatial_r(A, B):
    """Per-trial Pearson r between two (n_voxels, n_trials) arrays, across voxels."""
    Ad = A - A.mean(axis=0, keepdims=True)
    Bd = B - B.mean(axis=0, keepdims=True)
    num = (Ad * Bd).sum(axis=0)
    den = np.sqrt((Ad ** 2).sum(axis=0) * (Bd ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def voxel_demean(X):
    """Remove each voxel's mean across trials, leaving trial-specific deviations.

    Without this, a trialwise spatial correlation is dominated by the mean response
    pattern that every trial shares, and reports ~1.0 regardless of trial coupling.
    """
    return X - X.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Trial alignment
# ---------------------------------------------------------------------------

def match_trials(cue_info, fb_info, sub, subject):
    """Map each feedback beta volume to its cue beta volume.

    Returns an int array `cue_idx` of the same length as fb_info.

    Three guards, because a silent misalignment looks exactly like the "low
    correlation" result this script exists to test for:
      1. cue run counts are the expected 96/96/136
      2. every matched pair's stimulus identity agrees between the cue info CSV and the
         behavioural table (the real checksum — same idea as the assertion in
         utils.data.load_target_from_bbt and run_glmsingle_feedback.extract_betas)
      3. the cue index stays inside its own run's block
    """
    counts = cue_info['run'].value_counts().to_dict()
    if counts != EXPECTED_CUE_COUNTS:
        raise ValueError(
            f"sub-{subject}: cue info run counts {counts} != expected "
            f"{EXPECTED_CUE_COUNTS} — positional trial alignment is not valid here."
        )

    # Offsets from the observed order rather than hardcoded, so the check above is the
    # single source of truth about the layout.
    offsets, running = {}, 0
    for run in CUE_RUNS:
        offsets[run] = running
        running += counts[run]

    cue_idx = np.empty(len(fb_info), dtype=int)
    for i, (_, row) in enumerate(fb_info.iterrows()):
        run, trial = row['run'], int(row['trial'])
        idx = offsets[run] + (trial - 1)
        if not (offsets[run] <= idx < offsets[run] + counts[run]):
            raise ValueError(
                f"sub-{subject}: {run} trial {trial} maps to cue index {idx}, "
                f"outside that run's block — trial numbering assumption violated."
            )
        cue_idx[i] = idx

    # Checksum: the stimulus shown must agree between the two sources.
    behav = {run: getattr(sub, run).trials for run in ('learning1', 'learning2')}
    mismatches = []
    for i, (_, row) in enumerate(fb_info.iterrows()):
        expected = behav[row['run']].loc[int(row['trial']), 'first_stim_name']
        got      = cue_info.iloc[cue_idx[i]]['stim_name']
        if expected != got:
            mismatches.append((row['run'], int(row['trial']), expected, got))
    if mismatches:
        raise ValueError(
            f"sub-{subject}: {len(mismatches)} matched pair(s) disagree on stimulus "
            f"identity — cue/feedback betas are misaligned. First: {mismatches[0]}"
        )

    logging.info(f"  aligned {len(cue_idx)} feedback trials to cue trials "
                 f"(stimulus-identity checksum passed)")
    return cue_idx


# ---------------------------------------------------------------------------
# Per-mask comparison
# ---------------------------------------------------------------------------

def compare_in_mask(cue_arr, fb_arr, mask_arr, runs, rng, n_shuffles):
    """All measures + baselines for one mask. Arrays are (x,y,z,n_trials), matched."""
    C = cue_arr[mask_arr]          # (n_vox, n_match)
    F = fb_arr[mask_arr]

    finite = np.isfinite(C).all(axis=1) & np.isfinite(F).all(axis=1)
    C, F = C[finite], F[finite]
    n_vox, n_trials = C.shape
    if n_vox == 0:
        return {'n_voxels': 0}

    Cd, Fd = voxel_demean(C), voxel_demean(F)

    vox_matched   = voxelwise_temporal_r(C, F)
    trial_matched = trialwise_spatial_r(Cd, Fd)
    trial_raw     = trialwise_spatial_r(C, F)   # inflated by the shared mean pattern

    # --- baseline 1: shuffled pairing, permuted within run ---
    vox_shuf, trial_shuf = [], []
    for _ in range(n_shuffles):
        perm = np.arange(n_trials)
        for r in np.unique(runs):
            m = np.where(runs == r)[0]
            if len(m) > 1:
                # derangement-ish: reroll until nothing maps to itself
                for _attempt in range(100):
                    p = rng.permutation(m)
                    if not (p == m).any():
                        break
                perm[m] = p
        vox_shuf.append(np.nanmedian(voxelwise_temporal_r(C, F[:, perm])))
        trial_shuf.append(np.nanmean(trialwise_spatial_r(Cd, Fd[:, perm])))

    # --- baseline 2: adjacent cue-cue pairs within run ---
    adj = np.where(runs[:-1] == runs[1:])[0]
    if len(adj) > 1:
        vox_adj   = np.nanmedian(voxelwise_temporal_r(C[:, adj], C[:, adj + 1]))
        trial_adj = np.nanmean(trialwise_spatial_r(Cd[:, adj], Cd[:, adj + 1]))
    else:
        vox_adj = trial_adj = np.nan

    return {
        'n_voxels': int(n_vox),
        'n_trials': int(n_trials),
        'vox_r_matched_median':  float(np.nanmedian(vox_matched)),
        'vox_r_matched_q25':     float(np.nanpercentile(vox_matched, 25)),
        'vox_r_matched_q75':     float(np.nanpercentile(vox_matched, 75)),
        'vox_r_shuffled_median': float(np.mean(vox_shuf)),
        'vox_r_adjacent_median': float(vox_adj),
        'trial_r_matched_mean':  float(np.nanmean(trial_matched)),
        'trial_r_matched_sd':    float(np.nanstd(trial_matched)),
        'trial_r_shuffled_mean': float(np.mean(trial_shuf)),
        'trial_r_adjacent_mean': float(trial_adj),
        'trial_r_matched_raw_mean': float(np.nanmean(trial_raw)),
        '_vox_matched_map':      vox_matched,
        '_finite':               finite,
    }


# ---------------------------------------------------------------------------
# Main subject function
# ---------------------------------------------------------------------------

def run_subject(subject, base_dir, bids_dir, cue_dir, feedback_dir, output_dir,
                roi_masks=None, beta_types=('D',), n_shuffles=10, seed=0,
                save_maps=False, overwrite=False):
    roi_masks = roi_masks or []
    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_cue_vs_feedback.csv"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)

    cue_sub_dir = cue_dir / f"sub-{subject}"
    fb_sub_dir  = feedback_dir / f"sub-{subject}"
    cue_info_path = cue_sub_dir / f"sub-{subject}_glmSingle_betas_CUES_info.csv"
    fb_info_path  = fb_sub_dir / f"sub-{subject}_glmSingle_betas_FEEDBACK_info.csv"
    for p in (cue_info_path, fb_info_path):
        if not p.exists():
            raise FileNotFoundError(f"sub-{subject}: missing input {p}")

    cue_info = pd.read_csv(cue_info_path)
    fb_info  = pd.read_csv(fb_info_path)

    sub = Subject(base_dir=str(base_dir), subject_id=subject,
                  include_imaging=True, bids_dir=str(bids_dir))

    cue_idx = match_trials(cue_info, fb_info, sub, subject)
    runs    = fb_info['run'].to_numpy()

    brain_mask_img = nib.load(sub.get_brain_mask('learning1'))
    masks = [('wholebrain', brain_mask_img)]
    for name, path in roi_masks:
        roi_func = resample_to_img(nib.load(str(path)), brain_mask_img, interpolation='nearest')
        masks.append((name, math_img('(v > 0) & (b > 0)', v=roi_func, b=brain_mask_img)))
    mask_arrays = [(name, img.get_fdata() > 0) for name, img in masks]

    rows = []
    for beta_type in beta_types:
        # Select matched cue volumes up front so both arrays are trial-aligned from here on.
        cue_arr = load_betas(cue_sub_dir, subject, 'CUES', beta_type, len(cue_info))[..., cue_idx]
        fb_arr  = load_betas(fb_sub_dir, subject, 'FEEDBACK', beta_type, len(fb_info))
        logging.info(f"sub-{subject}: type-{beta_type} — cue {cue_arr.shape}, feedback {fb_arr.shape}")

        # Reseed per type so the shuffled baseline is identical across types and the
        # B-vs-D difference can't be an artefact of different random pairings.
        rng = np.random.default_rng(seed)

        for mask_name, mask_arr in mask_arrays:
            res = compare_in_mask(cue_arr, fb_arr, mask_arr, runs, rng, n_shuffles)
            vox_map = res.pop('_vox_matched_map', None)
            finite  = res.pop('_finite', None)

            if res.get('n_voxels', 0) == 0:
                logging.warning(f"  type-{beta_type} {mask_name}: no finite voxels, skipping")
                continue

            logging.info(
                f"  type-{beta_type} {mask_name}: {res['n_voxels']:,} vox | "
                f"voxelwise r matched {res['vox_r_matched_median']:.3f} "
                f"(shuffled {res['vox_r_shuffled_median']:.3f}, "
                f"adjacent {res['vox_r_adjacent_median']:.3f}) | "
                f"trialwise r matched {res['trial_r_matched_mean']:.3f} "
                f"(shuffled {res['trial_r_shuffled_mean']:.3f}, "
                f"adjacent {res['trial_r_adjacent_mean']:.3f})"
            )
            # Direction sanity check — if matched isn't above the shuffled floor,
            # suspect the alignment before believing the science.
            if res['vox_r_matched_median'] <= res['vox_r_shuffled_median']:
                logging.warning(f"  type-{beta_type} {mask_name}: matched r <= shuffled r "
                                f"— check alignment")

            rows.append({'subject': subject, 'beta_type': beta_type,
                         'mask': mask_name, **res})

            if save_maps and mask_name == 'wholebrain':
                full = np.full(mask_arr.shape, np.nan, dtype=np.float32)
                keep = tuple(v[finite] for v in np.where(mask_arr))
                full[keep] = vox_map.astype(np.float32)
                nib.Nifti1Image(full, brain_mask_img.affine, brain_mask_img.header).to_filename(
                    str(subject_output /
                        f"sub-{subject}_cue_vs_feedback_r_type{beta_type}.nii.gz")
                )

        del cue_arr, fb_arr   # each pair is ~350 MB; don't hold two types at once

    pd.DataFrame(rows).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done — {len(rows)} row(s) written to {done_flag.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure how redundant feedback-locked betas are with cue-locked "
                    "betas, against shuffled-pair and adjacent-trial baselines."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", required=True,
                        help="Root data directory")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--cue-dir", required=True,
                        help="Cue-locked GLMsingle betas directory")
    parser.add_argument("--feedback-dir", required=True,
                        help="Feedback-locked GLMsingle betas directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--roi-mask", action="append", nargs=2, metavar=("NAME", "PATH"),
                        default=[],
                        help="ROI mask given as NAME PATH; repeatable. Whole-brain is "
                             "always included automatically.")
    parser.add_argument("--beta-type", nargs='+', choices=['B', 'C', 'D'], default=['D'],
                        help="GLMsingle model type(s) to compare; repeatable, e.g. "
                             "--beta-type B D (default: D). Type A is excluded — ONOFF "
                             "pools all events into one beta, leaving nothing per-trial. "
                             "Comparing B alongside D matters here: D's GLMdenoise/ridge "
                             "hyperparameters are cross-validated over condition repeats, "
                             "and the two models define conditions differently (8 stimulus "
                             "identities vs 8 stimulus pairs), so a type-D-only comparison "
                             "conflates different event timing with different hyperparameter "
                             "selection. Type B (FITHRF only) isolates the timing effect.")
    parser.add_argument("--n-shuffles", type=int, default=10,
                        help="Shuffled-pairing repetitions for the baseline (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the shuffles")
    parser.add_argument("--save-maps", action="store_true",
                        help="Also write the whole-brain voxelwise matched-r map")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    # Logs outside the per-subject dir purely for consistency with the GLMsingle
    # runners, where <output-dir>/sub-<id> is rmtree'd by GLM_single.fit().
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"cue_vs_feedback_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject      = args.subject,
        base_dir     = Path(args.base_dir),
        bids_dir     = Path(args.bids_dir),
        cue_dir      = Path(args.cue_dir),
        feedback_dir = Path(args.feedback_dir),
        output_dir   = output_dir,
        roi_masks    = args.roi_mask,
        beta_types   = args.beta_type,
        n_shuffles   = args.n_shuffles,
        seed         = args.seed,
        save_maps    = args.save_maps,
        overwrite    = args.overwrite,
    )


if __name__ == "__main__":
    main()
