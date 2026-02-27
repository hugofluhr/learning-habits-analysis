clear;

%% ===========================
%% Paths
%% ===========================
spmpath     = '/home/ubuntu/repos/spm12';
analysis_dir= '/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs';
bbt_path    = '/home/ubuntu/data/learning-habits/bbt.csv';
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
log_path = fullfile(first_lvl_dir, 'count_nonestimable_regressors.txt');
diary(log_path);
diary on;

%% ===========================
%% Load behavioral data + subjects
%% ===========================
bbt = readtable(bbt_path);
subjects = unique(bbt.sub_id);

spm('Defaults', 'fMRI');

% Results will include a semicolon-separated list of non-estimable columns
results = {};

for s = 1:length(subjects)
    sub_id = subjects{s};

    if strcmp(sub_id,'sub-04') || strcmp(sub_id,'sub-45')
        continue;
    end

    sub_output_dir = fullfile(first_lvl_dir, sub_id);
    SPM_path = fullfile(sub_output_dir, 'SPM.mat');

    if ~exist(SPM_path, 'file')
        warning(['Missing SPM.mat for ' sub_id]);
        continue;
    end

    load(SPM_path);

    colnames = SPM.xX.name(:);
    nCols = numel(colnames);

    %% Identify nuisance columns
    is_excluded = false(nCols,1);
    for i = 1:nCols
        name = colnames{i};
        if ~isempty(regexp(name,'Sn\(\d+\)\s+R\d+','once')) || ...
           ~isempty(regexp(name,'Sn\(\d+\)\s+constant','once'))
            is_excluded(i) = true;
        end
    end

    task_cols = find(~is_excluded);

    %% Check estimability (unit contrasts for each task column)
    KXs = SPM.xX.xKXs;

    estimable = false(numel(task_cols),1);
    for i = 1:numel(task_cols)
        c = zeros(nCols,1);
        c(task_cols(i)) = 1;
        estimable(i) = spm_SpUtil('isCon', KXs, c);
    end

    n_task = numel(task_cols);
    n_bad  = sum(~estimable);

    bad_cols = task_cols(~estimable);
    bad_names = colnames(bad_cols);

    % Store as a single string for CSV readability
    if isempty(bad_names)
        bad_list = '';
    else
        % Join with '; ' and ensure it's a char row
        bad_list = strjoin(bad_names, '; ');
    end

    fprintf('%s: %d / %d task columns NOT estimable\n', sub_id, n_bad, n_task);
    if n_bad > 0
        fprintf('  Non-estimable examples (up to 5):\n');
        disp(bad_names(1:min(5,numel(bad_names))));
    end

    results(end+1,:) = {sub_id, n_task, n_bad, bad_list};
end

%% Save summary
results_table = cell2table(results, ...
    'VariableNames', {'sub_id','n_task_cols','n_nonestimable','nonestimable_columns'});

writetable(results_table, fullfile(first_lvl_dir, 'nonestimable_task_columns.csv'));

diary off;
disp('Saved summary table.');