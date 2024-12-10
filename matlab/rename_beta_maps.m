% Base directory for SPM results
base_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_format/outputs/spm_results';  % Adjust to your base directory

% Get list of subjects
subject_dirs = dir(fullfile(base_dir, 'sub-*'));
subject_dirs = subject_dirs([subject_dirs.isdir]); % Only keep directories
subjects = {subject_dirs.name};

% Loop over subjects
for i = 1:length(subjects)
    sub_id = subjects{i};
    sub_dir = fullfile(base_dir, sub_id);
    
    % Get list of runs for the current subject
    run_dirs = dir(fullfile(sub_dir, 'run-*'));
    run_dirs = run_dirs([run_dirs.isdir]); % Only keep directories
    runs = {run_dirs.name};
    
    % Loop over runs
    for j = 1:length(runs)
        run_id = runs{j};
        run_dir = fullfile(sub_dir, run_id);
        spm_path = fullfile(run_dir, 'SPM.mat');
        
        % Check if SPM.mat exists
        if ~isfile(spm_path)
            warning('SPM.mat not found for %s %s. Skipping...', sub_id, run_id);
            continue;
        end
        
        % Load SPM.mat
        load(spm_path, 'SPM');
        
        % Get regressor names
        regressor_names = SPM.xX.name;
        
        % Find beta files in the run directory
        beta_files = dir(fullfile(run_dir, 'beta_*.nii'));
        
        if isempty(beta_files)
            warning('No beta files found for %s %s. Skipping...', sub_id, run_id);
            continue;
        end
        
        % Loop through beta files and rename
        for k = 1:length(beta_files)
            % Original beta file name
            original_name = fullfile(run_dir, beta_files(k).name);
            
            % Get corresponding regressor name
            if k > length(regressor_names)
                warning('No regressor name for beta %d in %s %s. Skipping...', k, sub_id, run_id);
                continue;
            end
            regressor_name = regressor_names{k};
            
            % Sanitize regressor name for file system
            sanitized_name = regexprep(regressor_name, '[^a-zA-Z0-9]', '_');
            
            % Create new file name
            new_name = fullfile(run_dir, ['beta_' sanitized_name '.nii']);
            
            % Rename the file
            try
                movefile(original_name, new_name);
                fprintf('Renamed %s to %s\n', beta_files(k).name, ['beta_' sanitized_name '.nii']);
            catch ME
                warning('Could not rename %s: %s', beta_files(k).name, ME.message);
            end
        end
    end
end

disp('Beta map renaming completed!');