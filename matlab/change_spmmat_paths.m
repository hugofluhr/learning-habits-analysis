%% Batch-fix SPM paths after moving first-level results from VM to local machine
% This script:
% 1) loops over multiple subject first-level folders
% 2) updates paths inside SPM.mat
%
% Tested logic for standard SPM workflows.

clear; clc;

%% -------- USER SETTINGS --------

% Old root path as stored in SPM.mat on the VM
oldRoot = '/home/ubuntu/data/learning-habits/spm_format_noSDC';

% New local root path
newRoot = '/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/spm_format_noSDC';

% Name of first level model
first_level_name = 'glm2_chosen_all_runs_scrubbed_2025-12-11-11-22';
% Folder containing all subject first-level folders
firstLevelRoot = fullfile(newRoot, 'outputs', first_level_name);
% Check if first level root exists
if ~exist(firstLevelRoot, 'dir')
    error('First level root folder not found: %s', firstLevelRoot);
end

% Subject IDs / folder names
subjects = { ...
    'sub-01', ...
    'sub-02', ...
    'sub-03'  ...
    };

% If true, make a backup copy of SPM.mat before editing
makeBackup = true;

%% -------- START --------

fprintf('Starting path update...\n\n');

for i = 1:numel(subjects)
    subj = subjects{i};
    subjDir = fullfile(firstLevelRoot, subj);
    % Check if subject directory exists, stop if not
    if ~exist(subjDir, 'dir')
        error('Subject directory not found: %s', subjDir);
    end

    fprintf('--- %s ---\n', subj);
    fprintf('Folder: %s\n', subjDir);

    spmFile = fullfile(subjDir, 'SPM.mat');

    if ~exist(spmFile, 'file')
        warning('SPM.mat not found for %s. Skipping.\n', subj);
        continue;
    end

    try
        % Backup
        if makeBackup
            backupFile = fullfile(subjDir, 'SPM_backup_before_pathfix.mat');
            copyfile(spmFile, backupFile);
            fprintf('Backup created: %s\n', backupFile);
        end

        % First try SPM's path replacement utility
        spm_changepath(spmFile, oldRoot, newRoot);
        fprintf('spm_changepath applied.\n');

        % Reload and force-update SPM.swd in case needed
        S = load(spmFile);
        if isfield(S, 'SPM')
            SPM = S.SPM;
            SPM.swd = subjDir;
            save(spmFile, 'SPM', '-v7');
            fprintf('Updated SPM.swd to: %s\n', subjDir);
        else
            warning('No SPM variable found in %s\n', spmFile);
        end

        % Quick existence check on first image path if available
        S = load(spmFile);
        if isfield(S, 'SPM') && isfield(S.SPM, 'xY') && isfield(S.SPM.xY, 'P') && ~isempty(S.SPM.xY.P)
            firstImg = strtrim(S.SPM.xY.P(1,:));

            % Remove volume index ",n" for file existence check
            commaPos = find(firstImg == ',', 1, 'last');
            if ~isempty(commaPos)
                firstImgFile = firstImg(1:commaPos-1);
            else
                firstImgFile = firstImg;
            end

            if exist(firstImgFile, 'file')
                fprintf('First functional image found.\n');
            else
                warning('First functional image NOT found: %s\n', firstImgFile);
            end
        end


        fprintf('Done for %s.\n\n', subj);

    catch ME
        warning('Error processing %s: %s\n', subj, ME.message);
    end
end

fprintf('All subjects processed.\n');