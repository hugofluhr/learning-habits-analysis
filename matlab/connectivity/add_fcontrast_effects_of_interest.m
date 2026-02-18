clear;

%% ===========================
%% Paths
%% ===========================
%spmpath     = '/home/ubuntu/repos/spm12';
%analysis_dir= '/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs';
%bbt_path    = '/home/ubuntu/data/learning-habits/bbt.csv';
spmpath = '/Users/hugofluhr/code/spm12';
analysis_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_first_lvl_full';
bbt_path = '/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bbt.csv';
addpath(spmpath);

%% ===========================
%% Select first-level model to use
%% ===========================
model_name = 'glm2_all_runs_scrubbed_2025-12-11-12-44';
first_lvl_dir = fullfile(analysis_dir, model_name);
if ~exist(first_lvl_dir, 'dir')
    error('No directories found matching %s', fullfile(analysis_dir, model_name));
end
disp(['Using first-level output_dir: ' first_lvl_dir]);

%% ===========================
%% Logging
%% ===========================
log_path = fullfile(first_lvl_dir, 'add_fcontrast_matlab_log.txt');
diary(log_path);
diary on;

%% ===========================
%% Load behavioral data + subjects
%% ===========================
bbt = readtable(bbt_path);
subjects = unique(bbt.sub_id);

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

%% ===========================
%% Contrast settings
%% ===========================
contrast_name   = 'Effects of interest';
delete_existing = 0;  % keep existing contrasts

% Nuisance patterns to exclude from "effects of interest"
% - multi_reg columns often become "Sn(1) R1" etc
% - constants are "Sn(1) constant" or include "constant"
% - extra safety: exclude anything containing "motion" or "dummy"
exclude_patterns = { ...
    'Sn\(\d+\)\s*R\d+', ...    % e.g., "Sn(1) R1"
    '^\s*R\d+\s*$', ...        % e.g., "R1"
    'constant', ...            % session constant
    };

%% ===========================
%% Loop through subjects
%% ===========================
for s = 1:length(subjects)
    sub_id = subjects{s};

    % Same skip rule as your GLM script
    if strcmp(sub_id,'sub-04') || strcmp(sub_id, 'sub-45')
        continue; % Skip these subjects, alpha_H is 0
    end

    disp(['Adding F-contrast for subject: ' sub_id]);

    sub_output_dir = fullfile(first_lvl_dir, sub_id);
    SPM_path = fullfile(sub_output_dir, 'SPM.mat');

    if ~exist(SPM_path, 'file')
        warning(['SPM.mat not found for ' sub_id ': ' SPM_path]);
        continue;
    end

    % Load SPM
    S = load(SPM_path);
    if ~isfield(S, 'SPM')
        warning(['No SPM struct inside: ' SPM_path]);
        continue;
    end
    SPM = S.SPM;

    % If contrast already exists, skip
    has_contrast = false;
    if isfield(SPM, 'xCon') && ~isempty(SPM.xCon)
        for c = 1:numel(SPM.xCon)
            if strcmp(SPM.xCon(c).name, contrast_name)
                has_contrast = true;
                break;
            end
        end
    end
    if has_contrast
        disp(['  Contrast already exists: ' contrast_name ' (skipping)']);
        continue;
    end

    colnames = SPM.xX.name(:);
    nCols = numel(colnames);

    % Decide which columns are nuisance
    is_excluded = false(nCols,1);
    for i = 1:nCols
        this = colnames{i};
        for k = 1:numel(exclude_patterns)
            if ~isempty(regexp(lower(this), lower(exclude_patterns{k}), 'once'))
                is_excluded(i) = true;
                break;
            end
        end
    end

    task_cols = find(~is_excluded);

    if isempty(task_cols)
        warning(['  No task regressors found after exclusions for ' sub_id '. Check patterns.']);
        continue;
    end
    %% Check estimability and drop non-estimable task columns
    % It seems to mostly be nresp_screen (at least for glm2)
    KXs = SPM.xX.xKXs;
    
    estimable = false(numel(task_cols),1);
    for i = 1:numel(task_cols)
        c = zeros(nCols,1);
        c(task_cols(i)) = 1;  % unit contrast for that beta
        estimable(i) = spm_SpUtil('isCon', KXs, c);
    end
    
    bad_cols  = task_cols(~estimable);
    good_cols = task_cols(estimable);
    
    if ~isempty(bad_cols)
        disp(['  Dropping ' num2str(numel(bad_cols)) ' NON-estimable task columns:']);
        disp(colnames(bad_cols));
    end
    
    if isempty(good_cols)
        warning(['  No estimable task columns left for ' sub_id '. Skipping subject.']);
        continue;
    end
    
    task_cols = good_cols;

    % F-contrast weights: identity over task columns
    W = zeros(numel(task_cols), nCols);
    for i = 1:numel(task_cols)
        W(i, task_cols(i)) = 1;
    end

    % Batch: add F-contrast
    matlabbatch = [];
    matlabbatch{1}.spm.stats.con.spmmat = {SPM_path};
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.name = contrast_name;
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.weights = W;
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.delete = delete_existing;

    spm_jobman('run', matlabbatch);

    disp(['  Added "' contrast_name '" with ' num2str(numel(task_cols)) '/' num2str(nCols) ' columns included.']);

    % Quick printout (helps confirm nuisance detection)
    disp('  Example EXCLUDED columns (first 10):');
    excl = colnames(is_excluded);
    disp(excl(1:min(10,numel(excl))));

    disp('  Example INCLUDED columns (first 10):');
    incl = colnames(~is_excluded);
    disp(incl(1:min(10,numel(incl))));
end

diary off;
disp('Done.');