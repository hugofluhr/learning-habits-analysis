import os
import glob
import scipy.io
import numpy as np
import warnings
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds

def load_subject_lut(base_dir):
    """
    Load the subject lookup table (LUT) from the base directory.

    Parameters
    ----------
    base_dir : str
        The base directory containing the subject lookup table.

    Returns
    -------
    dict
        A DataFrame containing the subject lookup table.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(base_dir, 'subjects_lookup_table.csv'), header=None)
    
    # Function to clean the IDs by stripping the prefix and handling suffixes
    def clean_id(id_value):
        # Remove the 'SNS_MRI_LH_' prefix
        cleaned_id = id_value.replace('SNS_MRI_LH_', '')
        # Extract the first 6 characters of the cleaned ID
        cleaned_id = cleaned_id[:6]
        return cleaned_id
    
    # Create an inverted lookup dictionary with cleaned values as keys and keys as values
    lookup_dict = {
        value.replace('SNS_MRI_LH_', ''): clean_id(key)
        for key, value in zip(df.iloc[:, 0], df.iloc[:, 1])
    }
    
    return lookup_dict

def load_participant_list(base_dir, file_name='participants_sne2024.tsv'):
    """
    Load the list of subjects from the base directory.

    Parameters
    ----------
    base_dir : str
        The base directory containing the subject list.

    Returns
    -------
    list
        A list of subject IDs.
    """
    with open(os.path.join(base_dir, file_name), 'r') as file:
        # Read all lines, strip newline characters, and return as a list
        ids = [line.strip() for line in file.readlines()]
    return ids

class Block:
    """
    A class to represent a block of trials in an experimental session.

    Parameters
    ----------
    raw_block : custom data structure
        A data structure containing the raw data for the block, including timing, stimuli, and trial sequences as loaded from matlab.

    Attributes
    ----------
    raw_block : custom data structure
        Stores the raw block data.
    iti_seq : array-like
        Inter-Trial Interval sequence for the block.
    isi_seq : array-like
        Inter-Stimulus Interval sequence for the block.
    scanner_trigger : float
        Time when the scanner trigger occurred. Used as time reference for the block.
    start_time : float
        Block start time.
    end_time : float
        Block end time.
    total_length : float
        Total duration of the block.
    n_trials : int
        Number of trials in the block.
    trials : pd.DataFrame
        DataFrame containing detailed information about each trial in the block.
    """
    def __init__(self, raw_block):
        """
        Initialize a Block object by loading trial data and correcting time references.

        Parameters
        ----------
        raw_block : custom data structure
            The raw data for this block, containing sequences and timing information for each trial.
        """
        self.raw_block = raw_block

        # determine if the block is a learning block or a test block
        self.block_type = 'learning' if hasattr(raw_block.time, 'rewards_onset') else 'test'

        # ITI and ISI sequences used in the block
        self.iti_seq = raw_block.iti_seq
        self.isi_seq = raw_block.isi_seq

        # Time references
        self.scanner_trigger = raw_block.time.scanner_trigger
        self.start_time = raw_block.time.start_time
        self.end_time = raw_block.time.end_time
        self.total_length = raw_block.time.length

        # Number of trials in the block
        self.n_trials = self.iti_seq.shape[0]

        # Load trial data into a DataFrame
        self.trials = self._load_trials(raw_block)

        # Correct the time references relative to the scanner trigger
        self._correct_time_ref()

        # Add the events DataFrame
        self.create_event_df()

    def _load_trials(self, raw_block):
        """
        Load trial data into a DataFrame from the raw block data.

        Parameters
        ----------
        raw_block : custom data structure
            Raw block data containing trial sequences, actions, and timing.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about each trial, including stimuli, actions, rewards, and timings.
        """
        # Initialize an empty DataFrame with trial-related columns
        trials = pd.DataFrame(columns=['left_stim', 'right_stim', 'left_value', 'right_value', 'shift',
                                       'action', 'rt', 'chosen_stim', 'reward', 'correct',
                                       't_first_stim', 't_second_stim', 't_action', 't_purple_frame', 't_points_feedback', 't_iti_onset', 't_trial_end'],
                              index=range(1, self.n_trials+1))
        # name the index to explicitly show it's the trial number
        trials.index.name = 'trial'

        # Populate the DataFrame with trial sequence and action data
        trials.iloc[:, :5] = raw_block.seq1
        trials['action'] = raw_block.a
        trials['rt'] = raw_block.rt1
        trials['chosen_stim'] = raw_block.chosen

        # Calculate reward and correctness of each trial
        trials['reward'] = self._compute_reward(trials)
        trials['correct'] = self._compute_correct(trials)

        # Add timing information
        trials['t_first_stim'] = raw_block.time.first_stim_onset
        trials['t_second_stim'] = raw_block.time.onset

        # Handle missing response times, warning the user if data is missing
        n_missing = self.n_trials - len(raw_block.time.response)
        if n_missing > 0:
            warnings.warn(f"Last {n_missing} trial(s) of block had no response, filling with 0")
        trials['t_action'] = np.append(raw_block.time.response, np.full(n_missing, 0))
        trials['t_purple_frame'] = np.append(raw_block.time.purple_frame_onset, np.full(n_missing, 0))
        if self.block_type == 'learning':
            trials['t_points_feedback'] = np.append(raw_block.time.rewards_onset, np.full(n_missing, 0))
        trials['t_iti_onset'] = raw_block.time.iti_onset
        trials['t_trial_end'] = np.append(raw_block.time.first_stim_onset[1:], self.end_time-self.scanner_trigger)

        # Specify the data types for each column
        trials = trials.astype({
            'left_stim': 'int32',
            'right_stim': 'int32',
            'left_value': 'int32',
            'right_value': 'int32',
            'shift': 'int32',
            'action': 'float64',
            'rt': 'float64',
            'chosen_stim': 'float64',
            'reward': 'float64',
            'correct': 'float64',
            't_first_stim': 'float64',
            't_second_stim': 'float64',
            't_action': 'float64',
            't_purple_frame': 'float64',
            't_points_feedback': 'float64',
            't_iti_onset': 'float64',
            't_trial_end': 'float64'
        })

        return trials

    def _compute_reward(self, trials):
        """
        Calculate the reward for each trial based on the chosen action.

        Parameters
        ----------
        trials : pd.DataFrame
            DataFrame containing trial data, including actions and stimuli values.

        Returns
        -------
        np.ndarray
            An array of rewards for each trial, based on the chosen stimulus.
        """
        return np.where(
            trials['action'].isna(),  # If no action was taken, reward is NaN
            np.nan,
            np.where(trials['action'] == 1.0, trials['left_value'], trials['right_value'])  # Reward based on chosen stimulus
        )

    def _compute_correct(self, trials):
        """
        Determine whether the chosen stimulus was the correct one, based on its value.

        Parameters
        ----------
        trials : pd.DataFrame
            DataFrame containing trial data, including actions and stimuli values.

        Returns
        -------
        np.ndarray
            An array indicating whether each trial was correct (1) or incorrect (0).
            NaN values indicate non response OR same value for both stimuli.
        """
        return np.where(
            trials['action'].isna(),  # If no action was taken, correctness is NaN
            np.nan,
            np.where(
                trials['left_value'] == trials['right_value'],  # If left and right values are equal, correctness is NaN
                np.nan,
                np.where(
                    # Correct if the higher-value stimulus was chosen
                    ((trials['action'] == 1.0) & (trials['left_value'] > trials['right_value'])) | 
                    ((trials['action'] == 2.0) & (trials['right_value'] > trials['left_value'])),
                    1,
                    0
                )
            )
        )

    def _correct_time_ref(self, time_ref='scanner_trigger'):
        """
        Correct the time references of the trials to be relative to the scanner trigger.

        Only applies the correction if the resulting time is positive.

        Parameters
        ----------
        time_ref : str, optional
            The reference time to use for the correction (default is 'scanner_trigger').
        """
        time_ref = getattr(self, time_ref)
        time_columns = ['t_first_stim', 't_second_stim', 't_action', 't_purple_frame', 't_points_feedback', 't_iti_onset','t_trial_end']

        # Adjust the time columns based on the reference time
        for time_col in time_columns:
            condition = (self.trials[time_col] - time_ref) > 0  # Only correct positive times
            self.trials.loc[condition, time_col] -= time_ref
        
    def add_modeling_data(self, modeling_data):
        """
        Add modeling data to an extended trials dataframe.

        Parameters
        ----------
        modeling_data : pd.DataFrame
            DataFrame containing modeling data to add to the trials.
        """
        # making copy to avoid side effects
        modeling_df = modeling_data.copy()

        # define a valid mask for rows where 'action' is not NaN and 'rt' >= 0.05
        valid_mask = (~self.trials['action'].isna()) & (self.trials['rt'] >= 0.05)

        # updated sanity checks with valid_mask
        assert self.trials.loc[valid_mask, 'action'].astype(float).equals(modeling_df.loc[valid_mask, 'action'].astype(float)), 'actions do not match'
        assert self.trials.loc[valid_mask, 'left_stim'].astype('int32').equals(modeling_df.loc[valid_mask, 'stim1'].astype('int32')), 'left stim do not match'
        assert self.trials.loc[valid_mask, 'right_stim'].astype('int32').equals(modeling_df.loc[valid_mask, 'stim2'].astype('int32')), 'right stim do not match'
        assert (abs(self.trials.loc[valid_mask, 'rt'] - modeling_df.loc[valid_mask, 'rt1']) < 1e-3).all(), 'response times do not match'
        assert self.trials.loc[valid_mask, 'reward'].astype(float).equals(modeling_df.loc[valid_mask, 'reward_chosen'].astype(float)), 'rewards do not match'
        assert self.trials.loc[valid_mask, 'correct'].astype(float).equals(modeling_df.loc[valid_mask, 'corr_choice'].astype(float)), 'correct choices do not match'

        # drop columns that are redundant
        columns2drop = ['action', 'stim1', 'stim2', 'rt1', 'corr_choice']
        modeling_df.drop(columns2drop, axis=1, inplace=True)
        
        # perform the merge
        merged = pd.merge(self.trials, modeling_df, on='trial', how='left')

        # fill the missing values for non response trials
        columns2fill = ['flag_therapy1', 'flag_accuracy', 'flag_include', 'alpha_rl20',
                        'beta_rl20', 'alpha_ck20', 'beta_ck20', 'choice_prob_left',
                        'choice_prob_right', 'action_prob', 'stim1_value_rl', 'stim2_value_rl',
                        'stim3_value_rl', 'stim4_value_rl', 'stim5_value_rl', 'stim6_value_rl',
                        'stim7_value_rl', 'stim8_value_rl', 'stim1_value_ck', 'stim2_value_ck',
                        'stim3_value_ck', 'stim4_value_ck', 'stim5_value_ck', 'stim6_value_ck',
                        'stim7_value_ck', 'stim8_value_ck']
        for col in columns2fill:
            merged.loc[self.trials['action'].isna(), col] = merged[col].ffill()
        
        # Backward fill if the first trial(s) are non-response
        for col in columns2fill:
            merged[col] = merged[col].bfill()

        # TODO: fill missing values for diff_val, choice_prob, action_prob and value_diff
        # TODO: decide how to handle trials with rt < 0.05

        # Add columns with info about which stimulus was presented first and second
        merged['first_stim'] = np.where(merged['shift'] == 0, merged['right_stim'], merged['left_stim'])
        merged['second_stim'] = np.where(merged['shift'] == 0, merged['left_stim'], merged['right_stim'])
        
        # CK and RL values for those stimuli
        merged['first_stim_value_rl'] = merged.apply(lambda row: row[f"stim{row['first_stim']}_value_rl"], axis=1)
        merged['second_stim_value_rl'] = merged.apply(lambda row: row[f"stim{row['second_stim']}_value_rl"], axis=1)
        merged['first_stim_value_ck'] = merged.apply(lambda row: row[f"stim{row['first_stim']}_value_ck"], axis=1)
        merged['second_stim_value_ck'] = merged.apply(lambda row: row[f"stim{row['second_stim']}_value_ck"], axis=1)

        self.extended_trials = merged

    def create_event_df(self):
        """Create an event DataFrame from the trials DataFrame.
            The DataFrame containing trial information.
            The event DataFrame containing information about each event.
        Notes
        -----
        This function takes a trials DataFrame and creates an event DataFrame based on the trial information. The event DataFrame contains information about each event, such as the onset time, duration, and trial type.
        For each trial in the trials DataFrame, the function creates events for the first stimulus presentation, second stimulus presentation (if applicable), response (if applicable), purple frame presentation, points feedback presentation, non-response feedback (if applicable), and inter-trial interval (iti).
        The event DataFrame is then converted to a pandas DataFrame and returned.
        Examples
        --------
        >>> trials_df = pd.DataFrame(...)
        >>> events_df = create_event_df(trials_df)

        Parameters
        ----------
        trials : pd.DataFrame

        Returns
        -------pd.DataFrame
        """     
        events = []  # Define the events list inside the function

        for t, row in self.trials.iterrows():
            # Create event for first_stim presentation
            events.append({
                'onset': row['t_first_stim'],
                'duration': row['t_second_stim'] - row['t_first_stim'],
                'trial_type': 'first_stim_presentation', 'trial': t
            })

            if row['action'] in [1., 2.]:
                # response trial
                # Create event for second_stim presentation
                events.append({
                    'onset': row['t_second_stim'],
                    'duration': row['t_action'] - row['t_second_stim'],
                    'trial_type': 'second_stim_presentation', 'trial': t
                })

                # Create event for response
                events.append({
                    'onset': row['t_action'],
                    'duration': 0,
                    'trial_type': 'response', 'trial': t
                })

                # Create event for purple frame presentation
                events.append({
                    'onset': row['t_purple_frame'],
                    'duration': row['t_iti_onset'] - row['t_purple_frame'],
                    'trial_type': 'purple_frame', 'trial': t
                })

                # Create event for points feedback presentation, only for learning trials
                if self.block_type == 'learning':
                    events.append({
                        'onset': row['t_points_feedback'],
                        'duration': row['t_iti_onset'] - row['t_points_feedback'],
                        'trial_type': 'points_feedback', 'trial': t
                    })

            elif pd.isna(row['action']):
                # non response trial
                # Create event for second_stim presentation
                events.append({
                    'onset': row['t_second_stim'],
                    'duration': 1,
                    'trial_type': 'second_stim_presentation', 'trial': t
                })

                # Create event for non-response feedback
                events.append({
                    'onset': row['t_second_stim'] + 1,
                    'duration': row['t_iti_onset'] - row['t_second_stim'] - 1,
                    'trial_type': 'non_response_feedback', 'trial': t
                })

            # Create event for iti
            events.append({
                'onset': row['t_iti_onset'],
                'duration': row['t_trial_end'] - row['t_iti_onset'],
                'trial_type': 'iti', 'trial': t
            })

        # Convert to DataFrame and return it
        events_df = pd.DataFrame(events)
        events_df['trial'] = events_df['trial'].astype(int)
        
        self.events = events_df
    
    def _add_column_to_events_df(self, event, column_name):
        """
        Add a new column to the events DataFrame based on the trials DataFrame.

        Parameters
        ----------
        event : str
            The event type to add the column to.
        column_name : str
            The name of the column to add to the events DataFrame
        
        Returns
        -------
        pd.DataFrame
            An events DataFrame with the new column added.
        """
        ext_events_df = self.events.copy()
        ext_events_df[column_name] = 0.
        ext_events_df.loc[ext_events_df['trial_type']==event, column_name] = self.extended_trials[column_name].values
        return ext_events_df
    
    def extend_events_df(self, columns_event='default'):
        """
        Extend the events DataFrame with additional columns from the trials DataFrame.

        Parameters
        ----------
        columns : Dict or str
            A list of column names and event type to add to the events DataFrame.
        
        Returns
        -------
        pd.DataFrame
            An extended events DataFrame with the additional columns.
        """
        if columns_event == 'default':
            columns_event = {'first_stim_value_rl':'first_stim_presentation',
                             'first_stim_value_ck':'first_stim_presentation'}

        ext_events_df = self.events.copy()            
        for col, event in columns_event.items():
            ext_events_df[col] = 0.
            ext_events_df.loc[ext_events_df['trial_type']==event, col] = self.extended_trials[col].values
        
        return ext_events_df

class Subject:
    """
    A class to represent a subject and handle loading of various experimental data.

    Parameters
    ----------
    subject_data : str or dict
        If a string is provided, it is assumed to be a file path to a .mat file containing the subject data.
        If a dictionary is provided, it is assumed to already contain the subject's data in a pre-loaded format.

    Attributes
    ----------
    stimuli : dict
        Contains stimulus assignment and stimulus values for the subject.
    metadata : dict
        Contains metadata such as date, subject ID, eyetracking file, and file name.
    learning1 : Block
        Contains a Block object representing the subject's first learning block.
    learning2 : Block
        Contains a Block object representing the subject's second learning block.
    test : Block
        Contains a Block object representing the subject's test phase data.
    """
    def __init__(self, base_dir, subject_id, include_imaging=False, include_modeling=False, bids_dir=None):
        """
        Initializes the Subject class by loading the necessary data.

        If a string is passed, assumes it is a path to a .mat file and loads the data from the file.
        Otherwise, assumes the data is already provided in a dictionary format.
        """
        # Load the subject metadata
        self.base_dir = base_dir
        self.sub_id = f"sub-{subject_id}" if len(subject_id) == 2 else subject_id
        self.legacy_id = load_subject_lut(base_dir).get(self.sub_id, None)
        self.rp_files = self._get_rp_files()
        self.runs = ['learning1', 'learning2', 'test']

        # Load the Reward Pairing Task data 
        self._load_scanner_behav_data()

        # Can skip loading imaging data if not needed
        if include_imaging:
            if bids_dir is None:
                raise ValueError("BIDS directory must be provided to load imaging data.")
            self.bids_dir = bids_dir
        # Preload most common fmriprep files for easy access
            self._preload_fmriprep_files()

        # Load modeling data if needed
        if include_modeling:
            self.add_modeling_data()
    
    def _get_rp_files(self):
        """
        Search for .mat files in the given directory that contain the specified ID (case insensitive).

        Parameters
        ----------
        behav_dir : str
            The directory to search in.

        Returns
        -------
        list
            A list of matching .mat filenames that contain the ID.
        """
        # Define the directory to search in
        behav_dir = os.path.join(self.base_dir, 'behav_data', 'RP_data')
        
        # Convert id_to_search to lowercase for case-insensitive matching
        id_to_search_lower = self.legacy_id.lower()

        # List to store matching filenames
        matching_files = []

        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(behav_dir):
            for file in files:
                # Check if the file is a .mat file and if the ID is part of the filename (case insensitive)
                if file.endswith('.mat') and id_to_search_lower in file.lower():
                    matching_files.append(os.path.join(root, file))  # Append full path to the matching file
        matching_files.sort()  # Sort the filenames for consistency
        return matching_files
    
    def _get_scanner_rp_file(self):
        """
        Find the file that matches the format *_(subj_id)_2_*.mat but does NOT contain '_learning' from the given list.

        Parameters
        -------
        self

        Returns
        -------
        str or None
            The first matching file that does NOT contain '_learning', or None if no match is found.
        """
        # Define the pattern we're looking for
        pattern = f"_{self.legacy_id}_2_".lower()
        
        # Iterate over the file list
        for file_path in self.rp_files:
            # Get the file name
            file_name = os.path.basename(file_path).lower()
            
            # Check if the pattern is in the file name and it does not contain '_learning'
            if pattern in file_name and '_learning' not in file_name:
                return file_path
        
        # Return None if no file matches the pattern
        return None
    
    def _get_rp_modeling_files(self, modeling_dir='modeling_data'):
        """
        Find the modeling files for the subject based on the legacy ID.
        
        Parameters
        -------
        self
        modeling_dir : str
            The directory containing the modeling files.

        Returns
        -------
        dict
            A dictionary containing the modeling files for the subject.
        """
        modeling_dir = os.path.join(self.base_dir, modeling_dir)
        self.modeling_dir = modeling_dir
        modeling_files = {}
        for file in os.listdir(modeling_dir):
            if file.startswith(self.legacy_id.lower()) and file.endswith('.csv'):
                if '_learning' in file:
                    modeling_files['learning'] = file
                elif '_test' in file:
                    modeling_files['test'] = file
        return modeling_files

    def _load_scanner_behav_data(self):
        """
        Load the scanner behavioral data from the .mat file.

        Returns
        -------
        dict
            A dictionary containing the scanner behavioral data.
        """
        # Check that the correct .mat file exists
        scanner_rp_file = self._get_scanner_rp_file()
        assert scanner_rp_file is not None, f"No scanner RP file found for subject {self.sub_id}."
        # Load the .mat file
        scanner_rp_data = scipy.io.loadmat(scanner_rp_file, squeeze_me=True, struct_as_record=False)

        self.stimuli = self._load_stimuli_data(scanner_rp_data)
        self.metadata = self._load_metadata(scanner_rp_data)
        self.learning1, self.learning2 = self._load_learning_phase(scanner_rp_data)
        self.test = self._load_test_phase(scanner_rp_data)

        # make all trials accessible in a single DataFrame
        self.trials = self._concatenate_trials()


    def _load_block(self, block_data):
        """
        Loads a single block from the subject data.

        Parameters
        ----------
        block_data : dict or custom data structure
            Data structure containing the block information to be loaded.

        Returns
        -------
        Block
            Returns a Block object containing the block data.
        """
        return Block(block_data)

    def _load_stimuli_data(self, subject_data):
        """
        Loads and returns the stimuli data from the subject data.

        Parameters
        ----------
        subject_data : dict
            Dictionary containing the subject's data, which includes stimuli information.

        Returns
        -------
        dict
            A dictionary with the stimuli assignment and corresponding values.
        """
        # Extract stimuli assignment and predefined values
        assignment = subject_data['phase2_1'].stimuli_assignment
        # To Do: find a better way to set this (always the same across subjects)
        values = np.array([1, 2, 2, 3, 3, 4, 4, 5])
        return {'stim_assignment': assignment, 'stim_values': values}

    def _load_metadata(self, subject_data):
        """
        Loads and returns metadata information from the subject data.

        Parameters
        ----------
        subject_data : dict
            Dictionary containing the subject's metadata information.

        Returns
        -------
        dict
            A dictionary containing date, subject ID, eyetracking file, and file name.
        """
        # Extract and return metadata, including anonymized subject information
        date = subject_data['setup'].date
        eyetracking_file = subject_data['eyetracker_log_filename']
        file_name = subject_data['save_path']
        return {'date': date, 'eyetracking_file': eyetracking_file, 'file_name': file_name}
    
    def _load_learning_phase(self, subject_data):
        """
        Loads the learning phase data, which contains multiple blocks.

        Parameters
        ----------
        subject_data : dict
            Dictionary containing the subject's data, including the learning phase information.

        Returns
        -------
        list of Block
            A list of Block objects representing each block in the learning phase.
        """
        # Load each block in the learning phase based on the number of learning blocks
        return [self._load_block(subject_data['phase2_1'].blocks[i]) 
                for i in range(subject_data['setup'].learning_n_blocks)]
    
    def _load_test_phase(self, subject_data):
        """
        Loads the test phase data, which contains a single block.

        Parameters
        ----------
        subject_data : dict
            Dictionary containing the subject's test phase data.

        Returns
        -------
        Block
            A Block object representing the test phase data.
        """
        # Load the test phase, assumed to be a single block
        return self._load_block(subject_data['phase2_2'])
    
    def _concatenate_trials(self):
        """
        Concatenates all trials from the subject's learning phase and test phase into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing all trials, with an additional 'block' column.
        """
        all_trials = []

        # Iterate over each block
        # Add a 'block' column to the trials DataFrame indicating the block number
        for run in self.runs:
            block_trials = getattr(self, run).trials.copy()
            block_trials['block'] = f"{run}"
            all_trials.append(block_trials)

        # handling column type to avoid warning due to NaN values
        for df in all_trials:
            for col in df.columns:
                if df[col].isna().all():
                    df[col] = df[col].astype('float')

        # Concatenate all trials into a single DataFrame
        concatenated_trials = pd.concat(all_trials, ignore_index=True)

        # put the block column first
        concatenated_trials = concatenated_trials[['block'] + [col for col in concatenated_trials.columns if col != 'block']]

        return concatenated_trials
    
    def get_event_df(self, block):
        """
        Create an event DataFrame from the subject's trials DataFrame.

        Parameters
        ----------
        block : str, optional
            The block to create events for.
            Can be one of: 'learning1', 'learning2', 'test'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing event information for the specified block.
        """
        return getattr(self, block).create_event_df()
    
    @classmethod
    def get_or_create_layout(cls, bids_dir):
        # Check if the layout is already created
        if cls.bids_layout is None:
            print("Creating BIDSLayout...")
            if bids_dir is None:
                raise ValueError("BIDS directory must be provided to create BIDSLayout.")
            else:
                cls.bids_dir = bids_dir
                if 'fmriprep' in bids_dir:
                    cls.bids_layout = BIDSLayout(bids_dir, is_derivative=True)
                else:
                    cls.bids_layout = BIDSLayout(bids_dir, derivatives=True)
        else:
            print("Using existing BIDSLayout")
    
    def get_base_dir (self):
        """
        Get the base directory for the subject's fMRI data.

        Returns
        -------
        str
            The base directory for the subject's fmriprep func files
        """
        assert self.bids_dir is not None and os.path.isdir(self.bids_dir), "BIDS directory does not exist."
        if 'fmriprep' in self.bids_dir:
            base_dir = os.path.join(self.bids_dir, self.sub_id, 'ses-1', 'func')
        else:
            fmriprep_dir = glob.glob(os.path.join(self.bids_dir, 'derivatives', 'fmriprep*'))[0]
            if not fmriprep_dir:
                raise FileNotFoundError("No fmriprep directory found in derivatives.")
            base_dir  = os.path.join(fmriprep_dir, self.sub_id, 'ses-1', 'func')
        return base_dir

    def get_img_path (self, run):
        """
        Get the path to the fMRI image file for the specified run.

        Parameters
        ----------
        run : str, ['learning1', 'learning2', 'test']
            The run for which to get the fMRI image file.

        Returns
        -------
        str
            The path to the fMRI image file for the specified run.
        """
        base_dir = self.get_base_dir()
        f_task = 'learning' if 'learning' in run else 'test'
        f_run = self.runs.index(run) + 1 # neat way to get the run number
        img_file = os.path.join(base_dir, f"{self.sub_id}_ses-1_task-{f_task}_run-{f_run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"File {img_file} not found.")
        return img_file
    
    def load_confounds(self, run, motion_type='basic'):
        """ Load the confounds file for the specified run.

        Parameters
        ----------
        run : str
            The run for which to load the confounds file.
        """
        img_path = self.img.get(run)
        confounds, sample_mask = load_confounds(
            img_path,
            strategy=('motion', 'high_pass', 'wm_csf', 'scrub'),
            motion=motion_type,
            scrub=0,
            fd_threshold=0.5,
            std_dvars_threshold=2.5
        )

        # Filter to keep only the first 5 cosine columns
        cosine_columns = [col for col in confounds.columns if col.startswith('cosine')]
        cosine_columns_to_keep = cosine_columns[:5]
        columns_to_keep = [col for col in confounds.columns if not col.startswith('cosine')] + cosine_columns_to_keep
        confounds = confounds[columns_to_keep]

        return confounds, sample_mask

    def _preload_fmriprep_files(self):
        """
        Make most common fmriprep files available as attributes.
        """
        # Initialize self.img as a dictionary
        self.img = {}

        # Assign the fMRIprep files using keyword arguments
        for run in self.runs:
            self.img[run] = self.get_img_path(run)

    def _load_modeling_data(self, modeling_dir = 'modeling_data'):
        """
        Load the modeling data for the subject.
        """

        modeling_files = self._get_rp_modeling_files(modeling_dir)
        modeling_data = {}
        for key, file in modeling_files.items():
            df = pd.read_csv(os.path.join(self.modeling_dir, file))
            df.drop(['ID','session'], axis=1, inplace=True)
            df.set_index('trial', inplace=True) 
            if key == 'learning':
                df['run'] = df['run'].apply(lambda x: x.split(' ')[1])
                modeling_data['learning1'] = df[df['run'] == '1']
                modeling_data['learning2'] = df[df['run'] == '2']
            elif key == 'test':
                modeling_data['test'] = df
        self.modeling_data = modeling_data

    def _combine_modeling_data(self):
        """
        Combine the modeling data with the trials data for each block in 
        an extended trials DataFrame.
        """
        for key, block in self.modeling_data.items():
            getattr(self,key).add_modeling_data(block)

    def add_modeling_data(self, modeling_dir = 'modeling_data'):
        """
        Add modeling data to the subject's trials DataFrame.
        
        Parameters
        ----------
        modeling_dir : str
            The directory containing the modeling data.
        """
        self._load_modeling_data(modeling_dir)
        self._combine_modeling_data()