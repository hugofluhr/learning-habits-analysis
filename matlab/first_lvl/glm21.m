% Model description:
%   First level GLM with two learning runs (combined, run-01 and run-02)
%   and one test run (run-03).
%   Each run has the same set of conditions:
%       - first_stim (with Qval and Hval parametric modulators)
%       - second_stim (with Qval and Hval parametric modulators)
%       - response + feedback are collapsed into one regressor
%   Contrasts are defined for each run separately, as well as combined
%   across learning runs (run-01 + run-02).
%   Additional contrasts for sum and difference of Qval and Hval across
%   learning runs and test run.
clear;

% Paths
spmpath = '/home/ubuntu/repos/spm12';
data_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603';
analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603';
bbt_path = '/home/ubuntu/data/learning-habits/bbt.csv';
addpath(spmpath);

current_date = char(datetime('now', 'Format', 'yyyy-MM-dd-hh-mm'));
output_dir = fullfile(analysis_dir, 'outputs', ['glm21_' current_date]);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Start logging
log_path = fullfile(output_dir, 'matlab_log.txt');
diary(log_path);
diary on;

% Load behavioral data
bbt = readtable(bbt_path);
subjects = unique(bbt.sub_id);
block_names = {'learning1', 'learning2', 'test'};

% Parameters
TR = 2.33384; % Repetition time
high_pass_cutoff = 128; % High-pass filter in seconds

% set up contrasts
connames = {
    'first_stim', 'first_stimxQval', 'first_stimxHval', ...
    'second_stim', 'second_stimxQval', 'second_stimxHval', ...
    'response_feedback'
    };

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Loop through subjects
for s = 1:length(subjects)
    sub_id = subjects{s};
    if strcmp(sub_id,'sub-04') || strcmp(sub_id, 'sub-45')
        continue; % Skip these subjects, alpha_H is 0
    end
    disp(['Processing subject: ', sub_id]);
    func_dir = fullfile(data_dir, sub_id, 'func');
    
    % Identify unique runs based on BOLD file names
    bold_files = spm_select('FPList', func_dir, '^sub-.*_desc-preproc_bold.nii$');
    run_ids = unique(regexp(cellstr(bold_files), 'run-\d+', 'match', 'once'));
    
    %% ===========================
    %% A) LEARNING MODEL (run01+run02)  ->  outputs/<sub>/learning
    %% ===========================
    learn_output_dir = fullfile(output_dir, sub_id, 'learning');
    if ~exist(learn_output_dir, 'dir'); mkdir(learn_output_dir); end

    % Initialize batch
    clear matlabbatch
    matlabbatch{1}.spm.stats.fmri_spec.dir = {learn_output_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
    % TO DO: check if these are correct, these are default values
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
        
    for r = 1:2 % Only learning runs
        run_id = run_ids{r};
        % Select BOLD files for the current run
        currBOLD = spm_select('FPList', func_dir, ['^smoothed_.*' run_id '_.*_bold.nii$']);
        brain_mask = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_desc-brain_mask.nii$']);
        confounds_file = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_motion.txt$']);
        
        % Sanity check for missing files
        if isempty(currBOLD)
            warning(['No BOLD files found for ', run_id, ' in subject ', sub_id]);
            continue;
        end
        if isempty(brain_mask)
            warning(['Brain mask not found for ', run_id, ' in subject ', sub_id]);
            continue;
        end
        if isempty(confounds_file)
            warning(['Confounds not found for ', run_id, ' in subject ', sub_id]);
            continue;
        end
        
        % Get block data and filter trials
        block_data = bbt(strcmp(bbt.sub_id, sub_id) & strcmp(bbt.block, block_names{r}), :);
        % Ignore non response trials
        block_data = block_data(~isnan(block_data.action), :);

        % General stuff
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).scans = cellstr(currBOLD);
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi_reg = {confounds_file};
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).hpf = high_pass_cutoff;

        % First stimulus
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).name = 'first_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).onset = block_data.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).duration = block_data.t_second_stim - block_data.t_first_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod(1).name = 'Qval';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod(1).param = zscore(block_data.first_stim_value_rl);
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod(1).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod(2).name = 'Hval';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod(2).param = zscore(block_data.first_stim_value_ck);
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod(2).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).orth = 0;

        % Second stimulus
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).name = 'second_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).onset = block_data.t_second_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).duration = block_data.t_action - block_data.t_second_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod(1).name = 'Qval';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod(1).param = zscore(block_data.second_stim_value_rl);
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod(1).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod(2).name = 'Hval';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod(2).param = zscore(block_data.second_stim_value_ck);
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod(2).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).orth = 0;

        % Response + feedback are collapsed
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).name = 'response_feedback';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).onset = block_data.t_action;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).duration = block_data.t_iti_onset - block_data.t_action; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).orth = 0;
    end

    % Other specifications, from Jae-Chang's script
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mask = cellstr(brain_mask);
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    
    %% Model estimation
    matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    
    save(fullfile(learn_output_dir, 'batch_spec_est.mat'), 'matlabbatch');
    spm_jobman('run', matlabbatch);
    
    % Learning Contrasts (combined) 
    SPM_path = fullfile(learn_output_dir, 'SPM.mat');
    load(SPM_path);

    matlabbatch_con = [];
    matlabbatch_con{1}.spm.stats.con.spmmat = {fullfile(learn_output_dir, 'SPM.mat')};
    matlabbatch_con{1}.spm.stats.con.delete = 1;

    % Indices for session 1 columns in the design matrix
    sess1_cols = SPM.Sess(1).col;  % column indices belonging to session 1
    sess1_names = SPM.xX.name(sess1_cols);

    for cc = 1:length(connames)
        % Build a single-session weight vector since I'm using 'repl'
        w = zeros(1, numel(sess1_cols));
        
        idx = find(contains(sess1_names, connames{cc}), 1);
        if isempty(idx)
            warning('No regressor found for %s in %s', connames{cc}, run_id);
            continue
        end
        w(idx) = 1;
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.name = connames{cc};
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.weights = w;
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.sessrep = 'repl';
    end

    % Add custom extra contrasts
    extra_contrasts = {
        'Qval_sum',  [0 1 0 0 1 0];
        'Hval_sum',  [0 0 1 0 0 1];
        'Qval_diff', [0 1 0 0 -1 0];
        'Hval_diff', [0 0 1 0 0 -1]
    };
    % Append each extra contrast
    for ec = 1:size(extra_contrasts, 1)
        matlabbatch_con{1}.spm.stats.con.consess{end+1}.tcon.name    = extra_contrasts{ec, 1};
        matlabbatch_con{1}.spm.stats.con.consess{end}.tcon.weights   = extra_contrasts{ec, 2};
        matlabbatch_con{1}.spm.stats.con.consess{end}.tcon.sessrep   = 'none';
    end
    
    save(fullfile(learn_output_dir, 'batch_contrasts_.mat'), 'matlabbatch_con');
    spm_jobman('run', matlabbatch_con);

    disp(['Model for learning in ', sub_id, ' complete.']);


    %% ===========================
    %% B) TEST MODEL (run03 only)  ->  outputs/<sub>/test
    %% ===========================
    test_output_dir = fullfile(output_dir, sub_id, 'test');
    if ~exist(test_output_dir, 'dir'); mkdir(test_output_dir); end

    % Initialize batch
    clear matlabbatch
    matlabbatch{1}.spm.stats.fmri_spec.dir = {test_output_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
    % TO DO: check if these are correct, these are default values
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

    r = 3; % Only test run
    run_id = run_ids{r};
    % Select BOLD files for the current run
    currBOLD = spm_select('FPList', func_dir, ['^smoothed_.*' run_id '_.*_bold.nii$']);
    brain_mask = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_desc-brain_mask.nii$']);
    confounds_file = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_motion.txt$']);
    
    % Sanity check for missing files
    if isempty(currBOLD)
        warning(['No BOLD files found for ', run_id, ' in subject ', sub_id]);
        continue;
    end
    if isempty(brain_mask)
        warning(['Brain mask not found for ', run_id, ' in subject ', sub_id]);
        continue;
    end
    if isempty(confounds_file)
        warning(['Confounds not found for ', run_id, ' in subject ', sub_id]);
        continue;
    end
    
    % Get block data and filter trials
    block_data = bbt(strcmp(bbt.sub_id, sub_id) & strcmp(bbt.block, block_names{r}), :);
    % Ignore non response trials
    block_data = block_data(~isnan(block_data.action), :);

    % General stuff
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(currBOLD);
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {confounds_file};
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = high_pass_cutoff;

    % First stimulus
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'first_stim';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = block_data.t_first_stim;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = block_data.t_second_stim - block_data.t_first_stim; % check duration
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(1).name = 'Qval';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(1).param = zscore(block_data.first_stim_value_rl);
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(1).poly = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(2).name = 'Hval';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(2).param = zscore(block_data.first_stim_value_ck);
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(2).poly = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 0;

    % Second stimulus
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'second_stim';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = block_data.t_second_stim;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = block_data.t_action - block_data.t_second_stim; % check duration
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).name = 'Qval';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).param = zscore(block_data.second_stim_value_rl);
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).poly = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).name = 'Hval';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).param = zscore(block_data.second_stim_value_ck);
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).poly = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 0;

    % Response
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name = 'response_feedback';
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset = block_data.t_action;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = block_data.t_iti_onset - block_data.t_action;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).orth = 0;

    % Other specifications, from Jae-Chang's script
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mask = cellstr(brain_mask);
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    
    %% Model estimation
    matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    
    save(fullfile(test_output_dir, 'batch_spec_test.mat'), 'matlabbatch');
    spm_jobman('run', matlabbatch);
    
    % Test Contrasts
    SPM_path = fullfile(test_output_dir, 'SPM.mat');
    load(SPM_path);

    matlabbatch_con = [];
    matlabbatch_con{1}.spm.stats.con.spmmat = {fullfile(test_output_dir, 'SPM.mat')};
    
    for cc = 1:length(connames)
        idx = find(contains(SPM.xX.name, connames{cc}), 1);
        if isempty(idx)
            warning('No regressor found for %s in %s', connames{cc}, run_id);
            continue
        end
        weights = zeros(1, length(SPM.xX.name));
        weights(idx) = 1;
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.name = connames{cc};
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.weights = weights;
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.sessrep = 'none';
    end
    matlabbatch_con{1}.spm.stats.con.delete = 1;

    % Add custom extra contrasts
    extra_contrasts = {
        'Qval_sum',  [0 1 0 0 1 0];
        'Hval_sum',  [0 0 1 0 0 1];
        'Qval_diff', [0 1 0 0 -1 0];
        'Hval_diff', [0 0 1 0 0 -1]
    };
    % Append each extra contrast
    for ec = 1:size(extra_contrasts, 1)
        matlabbatch_con{1}.spm.stats.con.consess{end+1}.tcon.name    = extra_contrasts{ec, 1};
        matlabbatch_con{1}.spm.stats.con.consess{end}.tcon.weights   = extra_contrasts{ec, 2};
        matlabbatch_con{1}.spm.stats.con.consess{end}.tcon.sessrep   = 'none';
    end
    
    save(fullfile(test_output_dir, 'batch_contrasts_.mat'), 'matlabbatch_con');
    spm_jobman('run', matlabbatch_con);

    disp(['Model for test in ', sub_id, ' complete.']);
end

diary off;