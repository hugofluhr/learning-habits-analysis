clear;

%% ===========================
%% Paths
%% ===========================
% spmpath      = '/home/ubuntu/repos/spm12';
% analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs';

spmpath      = '/Users/hugofluhr/code/spm12';
analysis_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/spm_format_noSDC/outputs';

addpath(spmpath);

%% ===========================
%% FIRST-LEVEL model to use
%% ===========================
model_name = 'glm2_chosen_all_runs_scrubbed_2025-12-11-11-22';
first_lvl_dir = fullfile(analysis_dir, model_name);

if ~exist(first_lvl_dir, 'dir')
    error('First-level model folder not found: %s', first_lvl_dir);
end
disp(['Using base first-level dir: ' first_lvl_dir]);

%% ===========================
%% Logging
%% ===========================
ts = datestr(now,'yyyy-mm-dd-HH-MM-SS');
log_path = fullfile(first_lvl_dir, ['add_pppi_omnibus_fcontrast_' ts '.txt']);
diary(log_path);
diary on;

spm('Defaults','fMRI');
spm_jobman('initcfg');

%% ===========================
%% Subjects
%% ===========================
d = dir(fullfile(first_lvl_dir, 'sub-*'));
subjects = {d([d.isdir]).name};

%% ===========================
%% Settings
%% ===========================
contrast_name = 'Omnibus F-test for PPI Analyses';

% Set to true if you want to overwrite an existing omnibus contrast with the
% same name. Otherwise existing contrasts are left untouched.
overwrite_existing = true;

%% ===========================
%% Run
%% ===========================
for i = 1:numel(subjects)
    sub_id = subjects{i};
    disp('==================================================');
    disp(['Processing subject: ' sub_id]);

    if strcmp(sub_id,'sub-04') || strcmp(sub_id,'sub-45')
        disp(['Skipping subject (known issues): ' sub_id]);
        continue;
    end

    sub_base_dir = fullfile(first_lvl_dir, sub_id);
    spm_mat = fullfile(sub_base_dir, 'SPM.mat');

    if ~exist(spm_mat, 'file')
        warning('SPM.mat not found for subject %s: %s. Skipping.', sub_id, spm_mat);
        continue;
    end

    try
        load(spm_mat, 'SPM');

        if ~isfield(SPM, 'xX') || ~isfield(SPM.xX, 'X') || ~isfield(SPM.xX, 'name')
            warning('SPM structure incomplete for subject %s. Skipping.', sub_id);
            continue;
        end

        X = SPM.xX.X;
        reg_names = SPM.xX.name(:);
        n_cols = size(X, 2);

        if numel(reg_names) ~= n_cols
            warning('Mismatch between SPM.xX.X columns and SPM.xX.name for %s. Skipping.', sub_id);
            continue;
        end

        % --------------------------------------------------------------
        % Find any existing omnibus contrast
        % --------------------------------------------------------------
        existing_idx = [];
        if isfield(SPM, 'xCon') && ~isempty(SPM.xCon)
            existing_names = arrayfun(@(x) x.name, SPM.xCon, 'UniformOutput', false);
            existing_idx = find(strcmp(existing_names, contrast_name));
        end

        if ~isempty(existing_idx) && ~overwrite_existing
            disp(['Contrast already exists for ' sub_id ' -> leaving unchanged.']);
            continue;
        end

        % --------------------------------------------------------------
        % Infer effects-of-interest columns from SPM.xX.name
        %
        % Keep task-like regressors:
        %   Sn(k) something*bf(m)
        %
        % Exclude nuisance-like regressors:
        %   constants, motion, nuisance regressors, etc.
        %
        % Also exclude columns that are all zeros.
        % --------------------------------------------------------------
        include_cols = false(1, n_cols);
        include_labels = {};

        for j = 1:n_cols
            this_name = reg_names{j};

            if is_effect_of_interest_column(this_name)
                this_col = X(:, j);

                % Skip all-zero columns
                if all(abs(this_col) < eps)
                    disp(['Skipping all-zero column: ' this_name]);
                    continue;
                end

                include_cols(j) = true;
                include_labels{end+1,1} = this_name;
            end
        end

        kept_idx = find(include_cols);

        if isempty(kept_idx)
            warning('No valid effects-of-interest columns found for subject %s. Skipping.', sub_id);
            continue;
        end

        disp(['Keeping ' num2str(numel(kept_idx)) ' columns in omnibus F-contrast:']);

        % --------------------------------------------------------------
        % Build F-contrast:
        % one row per retained column
        % --------------------------------------------------------------
        c = zeros(numel(kept_idx), n_cols);
        for k = 1:numel(kept_idx)
            c(k, kept_idx(k)) = 1;
        end

        % --------------------------------------------------------------
        % Remove existing omnibus contrast if overwriting
        % --------------------------------------------------------------
        if ~isempty(existing_idx) && overwrite_existing
            keep_mask = true(1, numel(SPM.xCon));
            keep_mask(existing_idx) = false;
            SPM.xCon = SPM.xCon(keep_mask);
            disp(['Removed existing omnibus contrast for ' sub_id]);
        end

        % --------------------------------------------------------------
        % Add and estimate new omnibus contrast
        % --------------------------------------------------------------
        xCon = spm_FcUtil('Set', ...
            contrast_name, ...
            'F', ...
            'c', ...
            c', ...
            SPM.xX.xKXs);

        init = 0;
        if isfield(SPM, 'xCon') && ~isempty(SPM.xCon)
            init = length(SPM.xCon);
            SPM.xCon(init + 1) = xCon;
        else
            SPM.xCon = xCon;
        end

        SPM = spm_contrasts(SPM, init + 1);
        save(spm_mat, 'SPM');

        disp(['Added contrast #' num2str(init + 1) ' for subject ' sub_id]);
        disp(['Contrast name: ' contrast_name]);

    catch ME
        warning('Error for subject %s: %s', sub_id, ME.message);
    end
end

diary off;

%% ===========================
%% Helper
%% ===========================
function tf = is_effect_of_interest_column(reg_name)
    % Start pessimistic
    tf = false;

    % Must be a session-specific regressor
    if isempty(regexp(reg_name, '^Sn\(\d+\)\s+', 'once'))
        return;
    end

    % Must look like a basis-function-expanded task regressor
    % Examples:
    %   Sn(1) first_stim*bf(1)
    %   Sn(2) second_stimxHval_chosen^1*bf(1)
    %   Sn(1) cond*bf(2)
    if isempty(regexp(reg_name, '\*bf\(\d+\)$', 'once'))
        return;
    end

    % Exclude common nuisance/session regressors
    nuisance_patterns = {
        'Sn\(\d+\)\s*R\d+', ...    % e.g., "Sn(1) R1"
        '^\s*R\d+\s*$', ...        % e.g., "R1"
        'constant', ...            % session constant, e.g., "Sn(1) constant"
    };

    for ii = 1:numel(nuisance_patterns)
        if ~isempty(regexpi(reg_name, nuisance_patterns{ii}, 'once'))
            return;
        end
    end

    tf = true;
end