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
import numpy as np

# Set base directory and derivatives directory
base_dir = '/home/ubuntu/data/learning-habits'

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
        output_subject_dir = os.path.join(output_dir, 'sub-' + subject, "func")
        os.makedirs(output_subject_dir, exist_ok=True)

        subject = Subject(base_dir, subject, include_modeling=False, include_imaging=True, bids_dir=bids_dir)
                
        for run in subject.runs:
            confounds, _ = subject.load_confounds(run, include_cos=False)
            physio_regressors = subject.load_physio_regressors(run)
            confounds = confounds.join(physio_regressors)
            
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
            # Convert DataFrame columns to NumPy arrays (which can be saved as MATLAB cell arrays)
            # Group by 'trial_type' to aggregate the events for each condition
            grouped_events = events.groupby('trial_type')

            # Prepare lists for names, onsets, and durations
            names = []
            onsets = []
            durations = []

            # Iterate through each group and collect the onsets and durations for each condition
            for trial_type, group in grouped_events:
                names.append(trial_type)  # Add the trial type (condition name)
                onsets.append(group['onset'].tolist())  # Add the onsets as a list
                durations.append(group['duration'].tolist())  # Add the durations as a list

            # Convert lists to numpy arrays (which will be saved as MATLAB cell arrays)
            names_cell = np.array(names, dtype=object)
            onsets_cell = np.empty(len(onsets), dtype=object)
            onsets_cell[:] = onsets
            durations_cell = np.empty(len(durations), dtype=object)
            durations_cell[:] = durations

            output_events = os.path.join(output_subject_dir, os.path.basename(bold_file).replace(".nii.gz", "_events.mat")) 
            savemat(output_events, {"names": names_cell, "onsets": onsets_cell, "durations": durations_cell})
            print(f"Saved events to {output_events}")

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