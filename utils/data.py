import os
import scipy.io
import numpy as np
import warnings
import pandas as pd
from bids import BIDSLayout

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
        self.sequence_end_time = raw_block.time.sequence_end_time

        # Number of trials in the block
        self.n_trials = self.iti_seq.shape[0]

        # Load trial data into a DataFrame
        self.trials = self._load_trials(raw_block)

        # Correct the time references relative to the scanner trigger
        self._correct_time_ref()

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
                              index=range(self.n_trials))

        # Populate the DataFrame with trial sequence and action data
        trials.iloc[:, :5] = raw_block.seq1
        trials['action'] = raw_block.a
        trials['rt'] = raw_block.rt1
        trials['chosen_stim'] = raw_block.chosen

        # Calculate reward and correctness of each trial
        trials['reward'] = self._calculate_reward(trials)
        trials['correct'] = self._calculate_correct(trials)

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

        return trials

    def _calculate_reward(self, trials):
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

    def _calculate_correct(self, trials):
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
        """
        return np.where(
            trials['action'].isna(),  # If no action was taken, correctness is NaN
            np.nan,
            np.where(
                # Correct if the higher-value stimulus was chosen
                ((trials['action'] == 1.0) & (trials['left_value'] > trials['right_value'])) | 
                ((trials['action'] == 2.0) & (trials['right_value'] > trials['left_value'])),
                1,
                0
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

        for _, row in self.trials.iterrows():
            # Create event for first_stim presentation
            events.append({
                'onset': row['t_first_stim'],
                'duration': row['t_second_stim'] - row['t_first_stim'],
                'trial_type': 'first_stim_presentation'
            })

            if row['action'] in [1., 2.]:
                # response trial
                # Create event for second_stim presentation
                events.append({
                    'onset': row['t_second_stim'],
                    'duration': row['t_action'] - row['t_second_stim'],
                    'trial_type': 'second_stim_presentation'
                })

                # Create event for response
                events.append({
                    'onset': row['t_action'],
                    'duration': 0,
                    'trial_type': 'response'
                })

                # Create event for purple frame presentation
                events.append({
                    'onset': row['t_purple_frame'],
                    'duration': row['t_iti_onset'] - row['t_purple_frame'],
                    'trial_type': 'purple_frame'
                })

                # Create event for points feedback presentation, only for learning trials
                if self.block_type == 'learning':
                    events.append({
                        'onset': row['t_points_feedback'],
                        'duration': row['t_iti_onset'] - row['t_points_feedback'],
                        'trial_type': 'points_feedback'
                    })

            elif pd.isna(row['action']):
                # non response trial
                # Create event for second_stim presentation
                events.append({
                    'onset': row['t_second_stim'],
                    'duration': 1,
                    'trial_type': 'second_stim_presentation'
                })

                # Create event for non-response feedback
                events.append({
                    'onset': row['t_second_stim'] + 1,
                    'duration': row['t_iti_onset'] - row['t_second_stim'] - 1,
                    'trial_type': 'non_response_feedback'
                })

            # Create event for iti
            events.append({
                'onset': row['t_iti_onset'],
                'duration': row['t_trial_end'] - row['t_iti_onset'],
                'trial_type': 'iti'
            })

        # Convert to DataFrame and return it
        events_df = pd.DataFrame(events)
        
        return events_df


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
    learning_phase : list of Block
        Contains a list of Block objects representing the subject's learning phase data.
    test_phase : Block
        Contains a Block object representing the subject's test phase data.
    """
    bids_layout = None # Class attribute to store the BIDSL layout

    def __init__(self, base_dir, subject_id, skip_imaging=False):
        """
        Initializes the Subject class by loading the necessary data.

        If a string is passed, assumes it is a path to a .mat file and loads the data from the file.
        Otherwise, assumes the data is already provided in a dictionary format.
        """
        # Load the subject metadata
        self.base_dir = base_dir
        self.sub_id = subject_id
        self.legacy_id = load_subject_lut(base_dir).get(subject_id, None)
        self.rp_files = self._get_rp_files()

        # Load the Reward Pairing Task data 
        self._load_scanner_behav_data()

        # Can skip loading imaging data if not needed
        if not skip_imaging:
        # Load the BIDSLayout if it hasn't been created yet
            bids_dir = next((d for d in os.listdir(base_dir) if d.startswith('bids')), None)
            assert bids_dir is not None, "No directory starting with 'bids' found in base_dir"
            self.bids_dir = os.path.join(base_dir, bids_dir)
            self.get_or_create_layout(self.bids_dir)
        # Preload most common fmriprep files for easy access
            self._preload_fmriprep_files()
    
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
        self.learning_phase = self._load_learning_phase(scanner_rp_data)
        self.test_phase = self._load_test_phase(scanner_rp_data)

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

        # Iterate over each block in the learning phase
        for i, block in enumerate(self.learning_phase):
            # Add a 'block' column to the trials DataFrame indicating the block number
            block_trials = block.trials.copy()
            block_trials['block'] = f"learning_{i+1}"  # Block numbering starts from 1
            all_trials.append(block_trials)
        
        # Include the test phase trials
        test_trials = self.test_phase.trials.copy()
        test_trials['block'] = 'test'  # Use a string identifier for the test phase
        all_trials.append(test_trials)

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
        if block == 'test':
            events = self.test_phase.create_event_df()
        elif block in ['learning1', 'learning2']:
            block_num = int(block[-1]) - 1
            events = self.learning_phase[block_num].create_event_df()
        return events
    
    @classmethod
    def get_or_create_layout(cls, bids_dir):
        # Check if the layout is already created
        if cls.bids_layout is None:
            print("Creating BIDSLayout...")
            cls.bids_layout = BIDSLayout(bids_dir, derivatives=True)
        else:
            print("Using existing BIDSLayout")

    def get_bids_files(self, **kwargs):
        """
        Wrapper for the BIDSLayout.get() method that always includes the subject ID.
        Additional filters can be passed as keyword arguments (kwargs).
        """
        # Always include the subject ID in the query
        kwargs['subject'] = self.sub_id[-2:]  # Assuming sub_id refers to the subject identifier
        
        # Use layout.get() with the updated filters
        return self.bids_layout.get(**kwargs)

    def _preload_fmriprep_files(self):
        """
        Make most common fmriprep files available as attributes.
        """
        # Initialize self.img as a dictionary
        self.img = {}

        # Assign the fMRIprep files using keyword arguments
        # 'run':1,'suffix': 'bold', 'desc': 'preproc', 'extension': 'nii.gz'}
        self.img['learning1'] = self.get_bids_files(suffix='bold', run=1, desc= 'preproc', extension='nii.gz')[0]
        self.img['learning2'] = self.get_bids_files(suffix='bold', run=2, desc= 'preproc', extension='nii.gz')[0]
        self.img['test'] = self.get_bids_files(suffix='bold', run=3, desc= 'preproc', extension='nii.gz')[0]