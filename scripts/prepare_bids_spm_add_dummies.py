import os
import gzip
import shutil
import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import Subject, create_dummy_regressors, load_participant_list
from scipy.io import savemat
import numpy as np

def prepare_bids_for_spm(base_dir, bids_dir, output_dir, participants_file='participants_mvpa.tsv', subjects=None):
    """
    Prepare BIDS data for SPM analysis.
    - only add a new confounds file with dummy regressors for censored volumes

    Parameters:
    - base_dir (str): Path to the project data root (participants list, behav_data, bbt.csv, ...).
    - bids_dir (str): Path to the BIDS directory.
    - output_dir (str): Path to the output directory for SPM-ready data.
    - participants_file (str): TSV file (under base_dir) listing subject IDs to process.
    - subjects (list[str] or None): explicit subject-id override (e.g. for a smoke test);
        if None, all subjects from participants_file are processed.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup log file with datestamp
    log_file = os.path.join(output_dir, f"prepare_bids_for_spm_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_f = open(log_file, 'a')
    def log(message):
        print(message)
        log_f.write(message + '\n')
        log_f.flush()

    # Loop through all subjects in the BIDS directory
    if subjects is None:
        subjects = load_participant_list(base_dir, file_name=participants_file)
    for subject in subjects:
        log(f"Processing {subject}...")
        output_subject_dir = os.path.join(output_dir, 'sub-' + subject, "func")
        os.makedirs(output_subject_dir, exist_ok=True)

        subject = Subject(base_dir, subject, include_modeling=False, include_imaging=True, bids_dir=bids_dir)

        for run in subject.runs:

            confounds, sample_mask = subject.load_confounds(run, include_cos=False,
            scrub=0, fd_thresh=0.5, std_dvars_thresh=None)
            dummies = create_dummy_regressors(sample_mask, len(confounds))
            physio_regressors = subject.load_physio_regressors(run)
            confounds = confounds.join(physio_regressors)
            confounds_with_dummies = confounds.join(dummies)

            # to get the filename
            bold_file = subject.img.get(run)

            # save regressors with dummies file
            output_regressors = os.path.join(output_subject_dir, os.path.basename(bold_file).replace(".nii.gz", "_motion_with_dummies.txt"))
            pd.DataFrame(confounds_with_dummies).to_csv(output_regressors, sep='\t', header=False, index=False)
            log(f"Saved motion regressors to {output_regressors}")


    log("Preparation complete.")
    log_f.close()
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Prepare BIDS data for SPM analysis.")
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Path to the project data root (participants list, behav_data, bbt.csv, ...).",
    )
    # bids dir: /mnt/data/learning-habits/bids_dataset
    parser.add_argument(
        "--bids-dir",
        type=str,
        required=True,
        help="Path to the BIDS directory.",
    )
    # output dir: /mnt/data/learning-habits/spm_format_20250603
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory for SPM-ready data.",
    )
    parser.add_argument(
        "--participants-file",
        type=str,
        default="participants_mvpa.tsv",
        help="TSV file (under base_dir) listing subject IDs to process.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Explicit subject-id override (e.g. for a smoke test), e.g. --subjects 01 02.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run preparation script
    prepare_bids_for_spm(
        args.base_dir,
        args.bids_dir,
        args.output_dir,
        participants_file=args.participants_file,
        subjects=args.subjects,
    )
