import argparse
import pandas as pd
import datetime
import re
import os
import pprint
import glob

# Lookup table for runs
runs_lookup = {
    1: 'learning',
    2: 'learning',
    3: 'test'
}

mri_file_patterns = [
    '*_t1w*.nii',
    '.*_run_(?P<run>\d+)s.*$',
    '*run*scanphyslog*.log',
    '*bo_fieldmap_v01_ec1*_typ0.nii',
    '*bo_fieldmap_v01_ec2*_typ0.nii',
    '*bo_fieldmap_v01_ec1*_typ3.nii',
    '*bo_fieldmap_v01_ec2*_typ3.nii',
    '*fieldmap*scanphyslog*.log'
]

def search_patterns(subj_directory, patterns):
    files = os.listdir(subj_directory)
    matching_files = []
    for pattern in patterns:
        matching_files.extend(glob.glob(os.path.join(subj_directory, pattern)))
    return matching_files

def check_data(subject_id, subject_dir, mri_file_patterns):

    incomplete_data = {}

    # Reward pairs data
    mri_files = search_patterns(subject_dir, mri_file_patterns)
    if not all(len(files) == 1 for files in mri_files.values()):
        if subject_id not in incomplete_data:
            incomplete_data[subject_id] = {}
        incomplete_data[subject_id] = [pattern for pattern, files in mri_files.items() if len(files) != 1]

    return incomplete_data

def insensitive_glob(pattern):
    '''
    from https://stackoverflow.com/questions/8151300/ignore-case-in-glob-on-linux
    '''
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))

def find_subject_directory(root_directory, subject_id, acquisition_date):
    _, month, day = acquisition_date.split('-')
    search_pattern = os.path.join(root_directory, month,day,f'SNS_MRI_LH_{subject_id.upper()}_*')
    #matching_folders = glob.glob(search_pattern)
    matching_folders = insensitive_glob(search_pattern)
    if len(matching_folders) > 1:
        print(f"Warning: Found multiple folders for subject {subject_id} on {acquisition_date}: {matching_folders}")
    if matching_folders:
        return matching_folders[0]  # Assuming only one matching folder
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check data for subjects.')
    parser.add_argument('-d', '--rootdir', type=str, help='Path to the raw data directory',
                        default='/Volumes/g_econ_rawdata$/rawdata/2024')
    parser.add_argument('--participants_info_path', type=str, help='Path to the participants info file',
                         default='/Volumes/g_econ_department$/projects/2024/nebe_fluhr_timokhov_tobler_learning_habits/data/documentation/Participant_Documentation.xlsx')
    parser.add_argument('-s', '--subjects', type=str, help='Subject ID(s) to check data for. Use "all" to check all subjects.', default='all')
    args = parser.parse_args()

    # Get the list of acquired subjects
    subj_info = pd.read_excel(args.participants_info_path, usecols=['ID', 'Date MRI'])
    subj_info['ID'] = subj_info['ID'].str.lower().str.replace(' ', '')
    acquired = subj_info[subj_info["Date MRI"] <= datetime.date.today().strftime("%Y-%m-%d")]
    acquired = acquired[acquired['ID'].str.len() == 6]
    acquired['Date MRI'] = acquired['Date MRI'].astype(str)
    acquired = acquired.set_index('ID').to_dict()['Date MRI']

    if args.subjects == 'all':
        subjects2check = acquired
    elif ',' in args.subjects:
        subjects2check = dict((k, acquired['k']) for k in args.subjects.split(','))
    else:   
        subjects2check = dict((args.subjects, acquired[args.subjects]))

    print(f"Checking data for the following subjects: {subjects2check.keys()}")

    # get the MRI data directories
    mri_directories = {sub:find_subject_directory(args.rootdir, sub, subjects2check[sub]) for sub in subjects2check}
    #pprint.pprint(mri_directories)

    complete_participants = []
    incomplete_data_dict = {}

    for sub_id, sub_dir in mri_directories.items():
        if sub_dir is None:
            incomplete_data_dict[sub_id] = 'No MRI data directory found'
            continue
        incomplete_data = check_data(sub_id, sub_dir, mri_file_patterns)
        if not incomplete_data:
            complete_participants.append(sub_id)
        else:
            incomplete_data_dict.update(incomplete_data)

    # for sub in subjects2check:
    #     # Check that the subject ID is in the format of 1 letter followed by 2 digits followed by 1 letter followed by 2 digits
    #     assert re.match(r'^[a-zA-Z]\d{2}[a-zA-Z]\d[a-zA-Z]$', sub), f"Subject ID {sub} is not in the correct format."

    #     # Checking data for each subject
    #     # print(f"Checking data for subject {sub}")
    #     incomplete_data = check_data(sub, args.exp_data_dir, mri_file_patterns)
    #     if not incomplete_data:
    #         complete_participants.append(sub)
    #     else:
    #         incomplete_data_dict.update(incomplete_data)

    print(f"\nComplete data: {len(complete_participants)} participants have complete data.")
    print(complete_participants)

    print(f"\nMissing data: {len(incomplete_data_dict)} participants have incomplete data.")
    pprint.pprint(incomplete_data_dict, indent=4)
