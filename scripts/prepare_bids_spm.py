import os
import gzip
import shutil
import pandas as pd
import argparse
import sys
import os
sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list
from scipy.io import savemat

# Set base directory and derivatives directory
base_dir = '/home/ubuntu/data/learning-habits'
#bids_dir = '/home/ubuntu/data/learning-habits/bids_dataset/derivatives/fmriprep-23.2.1/'


def prepare_bids_for_spm(bids_dir, output_dir):
    """
    Prepare BIDS data for SPM analysis.
    - Unzips necessary images
    - Converts regressors to SPM-readable .txt files
    
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
        output_subject_dir = os.path.join(output_dir, subject, "func")
        os.makedirs(output_subject_dir, exist_ok=True)

        subject = Subject(base_dir, subject, include_modeling=False, include_imaging=True, bids_dir=bids_dir)        
        for run in subject.runs:
            confounds, sample_mask = subject.load_confounds(run, motion_type='basic')

            bold_file = subject.img.get(run)
            mask_file = subject.brain_mask.get(run)
            events = getattr(subject, run).events

            # save BOLD file
            bold_output = os.path.join(output_subject_dir, os.path.basename(bold_file).replace(".gz", ""))
            print(f"Unzipping {bold_file} -> {bold_output}")
            with gzip.open(bold_file, 'rb') as f_in:
                with open(bold_output, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # save mask file
            mask_output = os.path.join(output_subject_dir, os.path.basename(mask_file).replace(".gz", ""))
            print(f"Unzipping {mask_file} -> {mask_output}")
            with gzip.open(mask_file, 'rb') as f_in:
                with open(mask_output, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # save regressors file
            output_regressors = os.path.join(output_subject_dir, os.path.basename(bold_file).replace(".nii.gz", "_motion.txt"))
            pd.DataFrame(confounds).to_csv(output_regressors, sep='\t', header=False, index=False)
            print(f"Saved motion regressors to {output_regressors}")

            # save events file
            for trial_type, group in events.groupby("trial_type"):
                onsets = group["onset"].tolist()
                durations = group["duration"].tolist()
                # Save to .mat file
                mat_data = {
                    "onsets": onsets,
                    "durations": durations,
                }
                output_file = os.path.join(output_subject_dir, f"{trial_type}_events.mat")
                savemat(output_file, mat_data)
                print(f"Saved: {output_file}")

    print("Preparation complete.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Prepare BIDS data for SPM analysis.")
    parser.add_argument(
        "--bids-dir",
        type=str,
        required=True,
        help="Path to the BIDS directory.",
    )
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