import argparse
import pandas as pd
import datetime
import re
import os
import pprint

file_patterns = {'rewardpairs': [
                            r'habit_{}_1_2024-.{{11}}.mat',
                            r'habit_{}_2_2024-.{{11}}.mat',
                            r'habit_{}_2_2024-.{{11}}_learning_1.mat',
                            r'habit_{}_2_2024-.{{11}}_learning_2.mat',
                            r'habit_{}_3_2024-.{{11}}.mat'
                            ],
                'gonogo':      [r'results_{}.mat'],
                'eyetracking': [r'{}.edf']
            }
# mr_file_patters = {'T1': [r'{}_T1.nii.gz'],
                   
#                     }

def search_patterns(subject_id, directory, patterns):
    '''
    Check if files exist in the specified directory for a given subject ID and patterns.

    Parameters:
    - subject_id (str): The ID of the subject.
    - directory (str): The directory to search for files.
    - patterns (list): A list of file name patterns to search for.

    Returns:
    - found_files (dict): A dictionary where the keys are the patterns and the values are lists of found file names.

    Example:
    >>> subject_id = 'aaaaaa'
    >>> directory = '/path/to/directory'
    >>> patterns = ['data_{}.csv', 'results_{}.txt']
    >>> search_patterns(subject_id, directory, patterns)
    {'data_{}.csv': ['data_001.csv'], 'results_{}.txt': []}
    '''
    files = os.listdir(directory)
    found_files = {pattern: [] for pattern in patterns}
    
    for pattern in patterns:
        pattern_with_id = pattern.format(re.escape(subject_id))
        matches = [file for file in files if re.match(pattern_with_id, file, re.IGNORECASE)]
        if matches:
            found_files[pattern] = matches

    return found_files

def check_data(subject_id, exp_data_dir, file_patterns):
    """
    Check the completeness of different types of data for a given subject.

    Args:
        subject_id (str): The ID of the subject.
        exp_data_dir (str): The directory where the experiment data is stored.
        file_patterns (dict): A dictionary containing file patterns for different types of data.

    Returns:
        incomplete_data (dict): A dictionary containing the types of data that are incomplete for the subject.

    Example:
    >>> subject_id = 'aaaaaa'
    >>> exp_data_dir = '/path/to/experiment/data'
    >>> file_patterns = {'rewardpairs': [r'habit_{}_1_2024-.{{11}}.mat'], 'gonogo': [r'results_{}.mat']}
    >>> check_data(subject_id, exp_data_dir, file_patterns)
    {'001': {'rewardpairs': ['habit_001_1_2024-01-01.mat'], 'gonogo': []}}
    """
    incomplete_data = {}

    # Reward pairs data
    reward_pairs_dir = os.path.join(exp_data_dir, 'data')
    reward_pairs_files = search_patterns(subject_id, reward_pairs_dir, file_patterns['rewardpairs'])
    if not all(len(files) == 1 for files in reward_pairs_files.values()):
        if subject_id not in incomplete_data:
            incomplete_data[subject_id] = {}
        incomplete_data[subject_id]['rewardpairs'] = [pattern for pattern, files in reward_pairs_files.items() if len(files) != 1]

    # GoNoGo data
    gonogo_dir = os.path.join(exp_data_dir, 'data_gonogo')
    gonogo_files = search_patterns(subject_id, gonogo_dir, file_patterns['gonogo'])
    if not len(gonogo_files['results_{}.mat']) == 1:
        if subject_id not in incomplete_data:
            incomplete_data[subject_id] = {}
        incomplete_data[subject_id]['gonogo'] = f'results_{subject_id}.mat'

    # Eye tracking data
    eyetracking_dir = os.path.join(exp_data_dir, 'data_eyetracker')
    eyetracking_files = search_patterns(subject_id, eyetracking_dir, file_patterns['eyetracking'])
    if not len(eyetracking_files['{}.edf']) == 1:
        if subject_id not in incomplete_data:
            incomplete_data[subject_id] = {}
        incomplete_data[subject_id]['eyetracking'] = f'{subject_id}.edf'

    return incomplete_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check data for subjects.')
    parser.add_argument('-d', '--exp_data_dir', type=str, help='Path to the experiment data directory',
                        default='/Volumes/g_econ_department$/projects/2024/nebe_fluhr_timokhov_tobler_learning_habits/data/exp_data')
    parser.add_argument('--participants_info_path', type=str, help='Path to the participants info file',
                         default='/Volumes/g_econ_department$/projects/2024/nebe_fluhr_timokhov_tobler_learning_habits/data/documentation/Participant_Documentation.xlsx')
    args = parser.parse_args()

    subj_info = pd.read_excel(args.participants_info_path)
    acquired = subj_info[subj_info["Date MRI"] <= datetime.date.today().strftime("%Y-%m-%d")]
    acquired_subj = acquired['ID'].str.lower().values

    complete_participants = []
    incomplete_data_dict = {}

    for sub in acquired_subj:
        # Check that the subject ID is a string of length 6
        assert isinstance(sub, str) and len(sub) == 6, f"Subject ID {sub} is not a string of length 6."

        # Checking data for each subject
        # print(f"Checking data for subject {sub}")
        incomplete_data = check_data(sub, args.exp_data_dir, file_patterns)
        if not incomplete_data:
            complete_participants.append(sub)
        else:
            incomplete_data_dict[sub] = incomplete_data

    print(f"\nComplete data participants: {len(complete_participants)}")
    print(complete_participants)

    print("\nMissing data:")
    pprint.pprint(incomplete_data_dict, indent=4)
