import os
import gzip
import shutil
import pandas as pd
import argparse
import sys
import os
sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, create_dummy_regressors, load_participant_list
from scipy.io import savemat
import numpy as np

# Set base directory and derivatives directory
base_dir = '/home/ubuntu/data/learning-habits'

def prepare_bids_for_spm(bids_dir, output_dir):
    """
    Prepare BIDS data for SPM analysis.
    - only add a new confounds file with dummy regressors for censored volumes
    
    Parameters:
    - bids_dir (str): Path to the BIDS directory.
    - output_dir (str): Path to the output directory for SPM-ready data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all subjects in the BIDS directory
    subjects = load_participant_list(base_dir)
    for subject in subjects:
        print(f"Processing {subject}...")
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
            print(f"Saved motion regressors to {output_regressors}")


    print("Preparation complete.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Prepare BIDS data for SPM analysis.")
    # bids dir: /mnt/data/learning-habits/bids_dataset
    parser.add_argument(
        "--bids-dir",
        type=str,
        required=True,
        help="Path to the BIDS directory.",
    )
    # output dir: /mnt/data/learning-habits/spm_format_20250603
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory for SPM-ready data.",
    )
    
    # Parse arguments
    args = parser.parse_args()
    bids_dir = args.bids_dir
    output_dir = args.output_dir

    # Run preparation script
    prepare_bids_for_spm(bids_dir, output_dir)