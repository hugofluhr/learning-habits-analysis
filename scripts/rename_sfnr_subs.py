
sfnr_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/sfnr'
import os
import csv

# Path to the lookup table
lut_path = '/Users/hugofluhr/phd_local/data/LearningHabits/subjects_lookup_table.csv'

# Build mapping from original ID to BIDS subject ID
id_map = {}
with open(lut_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        orig = row[0].replace('SNS_MRI_LH_', '').replace('_1', '').replace('_01', '').replace('_1_1', '').replace('_1_1_1', '').replace('_', '').lower()
        bids = row[1].replace('SNS_MRI_LH_', '')
        id_map[orig.lower()] = bids

# Rename subject directories and .png files in one go
for entry in os.listdir(sfnr_dir):
    entry_path = os.path.join(sfnr_dir, entry)
    if os.path.isdir(entry_path):
        key = entry.replace('_', '').lower()
        if key in id_map:
            bids_name = id_map[key]
            new_subject_path = os.path.join(sfnr_dir, bids_name)
            if not os.path.exists(new_subject_path):
                print(f'Renaming {entry} -> {bids_name}')
                os.rename(entry_path, new_subject_path)
            else:
                print(f'SKIP: {bids_name} already exists, not renaming {entry}')
                new_subject_path = entry_path
            # Now rename .png files inside all run subfolders
            for run_entry in os.listdir(new_subject_path):
                run_path = os.path.join(new_subject_path, run_entry)
                if os.path.isdir(run_path):
                    for fname in os.listdir(run_path):
                        if fname.lower().endswith('.png'):
                            old_png = os.path.join(run_path, fname)
                            # Use BIDS subject and run for new filename
                            new_png = os.path.join(run_path, f'{bids_name}_{run_entry}.png')
                            if not os.path.exists(new_png):
                                print(f'  Renaming {fname} -> {bids_name}_{run_entry}.png')
                                os.rename(old_png, new_png)
                            else:
                                print(f'  SKIP: {bids_name}_{run_entry}.png already exists, not renaming {fname}')
        else:
            print(f'No mapping for {entry}, skipping.')