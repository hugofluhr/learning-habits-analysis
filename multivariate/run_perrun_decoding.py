#!/usr/bin/env python3
"""
Stimulus category decoding — per-run vs combined-run comparison, one subject.

`run_decoding.py` pools trials from all 3 runs/sessions (`learning1`,
`learning2`, `test`) and cross-validates with `LeaveOneGroupOut` over runs
(train on 2 runs, test on the held-out run, ~328 trials total). This script
asks how much that matters: it reproduces that scheme unchanged as the
`combined_logo` baseline, and additionally decodes *within* each single run
in isolation (~96-136 trials/run, no cross-run generalization at all).

A single run has no natural "leave one X out" grouping to define CV folds
from, and far fewer trials than the pooled 328 (min ~18 trials/category in
the smallest run), so a single random k-fold split would be noisy. Instead
each within-run scheme repeats `StratifiedKFold` several times with a
different fold assignment each time and averages accuracy across repeats
(`cross_val_predict` doesn't accept `RepeatedStratifiedKFold` directly, since
each sample would land in more than one test fold — looping over single-repeat
`StratifiedKFold`s is the clean equivalent).

Same standardized decoder as `run_decoding.py`: type-D GLMsingle betas,
`NiftiMasker(standardize=True)`, `LinearSVC(max_iter=10000, dual='auto')`,
whole-brain + visual-cortex ROI masks (Harvard-Oxford atlas, `Subject`-based
brain mask for spatial reference — see `run_decoding.py`'s docstring for the
standardize=True rationale).

Usage
-----
python multivariate/run_perrun_decoding.py --subject 01 \\
    --base-dir /home/hfluhr/data/learninghabits \\
    --bids-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/fmriprep-24.0.1-noSDC \\
    --glmsingle-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/glmsingle \\
    --output-dir /home/hfluhr/shares-hare/ds-learning-habits/derivatives/perrun_decoding \\
    --visual-cortex-mask /home/hfluhr/shares-hare/ds-learning-habits/derivatives/decoding/visual_cortex_mask.nii.gz

Outputs (per subject)
---------------------
<output-dir>/sub-<id>/
    sub-<id>_perrun_decoding_accuracy.csv                     — mask, scheme, accuracy
    sub-<id>_perrun_decoding_confusion_<mask>_<scheme>.npy     — one per (mask, scheme), 8 total
    sub-<id>_perrun_decoding_labels.npy                        — category order (shared)
    perrun_decoding_sub-<id>.log

`scheme` is one of `combined_logo` (production LOGO over all 3 runs) or
`run-learning1` / `run-learning2` / `run-test` (repeated within-run k-fold).
Within-run confusion matrices are summed (not averaged) across repeats — the
absolute count scale doesn't matter since results notebooks row-normalize
before comparing/averaging across subjects.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, math_img
from nilearn.maskers import NiftiMasker
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import Subject

RUN_NAMES = ["learning1", "learning2", "test"]

# Within-run repeated k-fold: 5 folds is well under the smallest per-run,
# per-category trial count (~18), and 10 repeats (each with a different fold
# assignment) smooths out the noise of any single random split on these
# smaller samples. Base seed offsets each repeat's StratifiedKFold shuffle.
N_SPLITS_PERRUN = 5
N_REPEATS_PERRUN = 10
RANDOM_STATE = 42


def repeated_kfold_predict_accuracy(X, y, n_splits, n_repeats, base_seed, labels):
    """Average accuracy (and summed confusion matrix) over `n_repeats`
    independent StratifiedKFold splits, each fully cross-validated.

    Equivalent in spirit to RepeatedStratifiedKFold, but computed as an
    explicit loop of single-repeat cross_val_predict calls, since
    cross_val_predict requires the test folds to partition the samples
    exactly once (RepeatedStratifiedKFold violates that).
    """
    rep_accs = []
    cm_sum = np.zeros((len(labels), len(labels)), dtype=int)
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + rep)
        pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'), X, y, cv=cv)
        rep_accs.append(float((pred == y).mean()))
        cm_sum += confusion_matrix(y, pred, labels=labels)
    return float(np.mean(rep_accs)), rep_accs, cm_sum


def run_subject(subject, base_dir, bids_dir, glmsingle_dir, output_dir,
                visual_cortex_mask_path, overwrite=False):

    subject_output = output_dir / f"sub-{subject}"
    done_flag = subject_output / f"sub-{subject}_perrun_decoding_accuracy.csv"

    if done_flag.exists() and not overwrite:
        logging.info(f"sub-{subject}: outputs exist, skipping (pass --overwrite to rerun)")
        return

    subject_output.mkdir(parents=True, exist_ok=True)

    # --- Load betas and trial info ---
    betas_path = glmsingle_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES.nii.gz"
    info_path  = glmsingle_dir / f"sub-{subject}" / f"sub-{subject}_glmSingle_betas_CUES_info.csv"

    betas_img  = nib.load(betas_path)
    trial_info = pd.read_csv(info_path, index_col='trial_id')

    logging.info(f"sub-{subject}: betas {betas_img.shape}, {len(trial_info)} trials")

    y      = trial_info['stim_cat'].values
    groups = trial_info['run'].values
    cats   = sorted(set(y))

    for run_name in RUN_NAMES:
        n_run = int((groups == run_name).sum())
        logging.info(f"  {run_name}: {n_run} trials")

    # --- Brain mask (3D, used as spatial reference) ---
    sub = Subject(
        base_dir=str(base_dir),
        subject_id=subject,
        include_imaging=True,
        bids_dir=str(bids_dir),
    )
    brain_mask_img = nib.load(sub.brain_mask['learning1'])

    # --- Visual cortex mask: resample atlas mask to subject functional space ---
    vis_mask_mni  = nib.load(str(visual_cortex_mask_path))
    vis_mask_func = resample_to_img(vis_mask_mni, brain_mask_img, interpolation='nearest')
    vis_mask      = math_img('(v > 0) & (b > 0)', v=vis_mask_func, b=brain_mask_img)

    # --- Decode for each mask, reusing the extracted features across schemes ---
    logo    = LeaveOneGroupOut()
    results = []

    for mask_name, mask_img in [('wholebrain', brain_mask_img), ('visualcortex', vis_mask)]:
        # standardize=True: see run_decoding.py's docstring — whole-brain
        # features are ~65k voxels of widely varying signal scale, which
        # biases LinearSVC's L2 penalty toward high-magnitude voxels.
        masker = NiftiMasker(mask_img=mask_img, standardize=True).fit()
        X      = masker.transform(betas_img)
        logging.info(f"  {mask_name}: {X.shape[1]:,} voxels")

        # scheme 1: combined_logo — production between-run LOGO, unchanged
        y_pred = cross_val_predict(LinearSVC(max_iter=10000, dual='auto'), X, y, cv=logo, groups=groups)
        acc    = float((y_pred == y).mean())
        logging.info(f"  {mask_name} combined_logo: accuracy = {acc:.3f}  (chance = 0.25)")
        results.append({'mask': mask_name, 'scheme': 'combined_logo', 'accuracy': acc})
        cm = confusion_matrix(y, y_pred, labels=cats)
        np.save(subject_output / f"sub-{subject}_perrun_decoding_confusion_{mask_name}_combined_logo.npy", cm)

        # schemes 2-4: run-<name> — repeated within-run stratified k-fold
        for run_name in RUN_NAMES:
            run_sel = (groups == run_name)
            X_run, y_run = X[run_sel], y[run_sel]
            acc, rep_accs, cm_sum = repeated_kfold_predict_accuracy(
                X_run, y_run, N_SPLITS_PERRUN, N_REPEATS_PERRUN, RANDOM_STATE, cats)
            acc_sem = float(np.std(rep_accs, ddof=1) / np.sqrt(N_REPEATS_PERRUN))
            scheme = f'run-{run_name}'
            logging.info(f"  {mask_name} {scheme} (n={run_sel.sum()}): "
                        f"accuracy = {acc:.3f} +/- {acc_sem:.3f} "
                        f"(repeat-SEM over {N_REPEATS_PERRUN} reps)")
            results.append({'mask': mask_name, 'scheme': scheme, 'accuracy': acc})
            np.save(subject_output / f"sub-{subject}_perrun_decoding_confusion_{mask_name}_{scheme}.npy", cm_sum)

    np.save(subject_output / f"sub-{subject}_perrun_decoding_labels.npy", np.array(cats))
    pd.DataFrame(results).to_csv(done_flag, index=False)
    logging.info(f"sub-{subject}: done")


def main():
    parser = argparse.ArgumentParser(
        description="Run per-run vs combined-run stimulus category decoding for one subject.")
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-' prefix, e.g. 01")
    parser.add_argument("--base-dir", required=True,
                        help="Root data directory")
    parser.add_argument("--bids-dir", required=True,
                        help="fMRIPrep derivatives directory")
    parser.add_argument("--glmsingle-dir", required=True,
                        help="GLMsingle betas directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--visual-cortex-mask", required=True,
                        help="Path to pre-built visual cortex mask NIfTI "
                             "(from build_visual_cortex_mask.py)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir     = Path(args.output_dir)
    subject_output = output_dir / f"sub-{args.subject}"
    subject_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(subject_output / f"perrun_decoding_sub-{args.subject}.log"),
        ],
    )

    run_subject(
        subject              = args.subject,
        base_dir             = Path(args.base_dir),
        bids_dir             = Path(args.bids_dir),
        glmsingle_dir        = Path(args.glmsingle_dir),
        output_dir           = output_dir,
        visual_cortex_mask_path = Path(args.visual_cortex_mask),
        overwrite            = args.overwrite,
    )


if __name__ == "__main__":
    main()
