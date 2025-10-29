clear;

% Paths
spmpath = '/home/ubuntu/repos/spm12';
data_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603';
analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603';
bbt_path = '/home/ubuntu/data/learning-habits/bbt.csv';
addpath(spmpath);

current_date = char(datetime('now', 'Format', 'yyyy-MM-dd-hh-mm'));
output_dir = fullfile(analysis_dir, 'outputs', ['glm2b_exclude_' current_date]);
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
    'response', 'feedback'
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
    
    % Loop through each run separately
    for r = 1:length(run_ids)
        run_id = run_ids{r};
        disp(['Processing run: ', run_id]);
        
        % Define output directory for this run
        run_output_dir = fullfile(output_dir, sub_id, run_id);
        if ~exist(run_output_dir, 'dir')
            mkdir(run_output_dir);
        end
        
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
        % Ignore trials with highest/lowest stimuli
        % Create a mask for excluding specific trials
        trial_mask = (block_data.first_stim ~= 1) & (block_data.first_stim ~= 8) & (block_data.second_stim ~= 1) & (block_data.second_stim ~= 8);
        block_incl = block_data(trial_mask,:);
        block_excl = block_data(~trial_mask,:);
        
        %% -----------------
        % 1) SPEC + EST BATCH
        %% -----------------
        clear matlabbatch
        matlabbatch{1}.spm.stats.fmri_spec.dir = {run_output_dir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
        % TO DO: check if these are correct, these are default values
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
        
        % General stuff
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(currBOLD);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {confounds_file};
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = high_pass_cutoff;
        
        % First stimulus - included 
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'first_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = block_incl.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = block_incl.t_second_stim - block_incl.t_first_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;   
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(1).name = 'Qval';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(1).param = zscore(block_incl.first_stim_value_rl);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(1).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(2).name = 'Hval';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(2).param = zscore(block_incl.first_stim_value_ck);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod(2).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 0;
        
        % First stimulus - excluded
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).name = 'first_stim_excl';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).onset = block_excl.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).duration = block_excl.t_second_stim - block_excl.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).tmod = 0;   
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).orth = 0;

        % Second stimulus
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'second_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = block_incl.t_second_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = block_incl.t_action - block_incl.t_second_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).name = 'Qval';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).param = zscore(block_incl.second_stim_value_rl);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).name = 'Hval';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).param = zscore(block_incl.second_stim_value_ck);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 0;

        % Second stimulus - excluded
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).name = 'second_stim_excl';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).onset = block_excl.t_second_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).duration = block_excl.t_action - block_excl.t_second_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).orth = 0;
        
        % Response
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name = 'response';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset = block_data.t_action;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = 0; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod = 0;   
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).orth = 0;
        
        % Feedback
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).name = 'feedback';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).onset = block_data.t_purple_frame;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).duration = block_data.t_iti_onset - block_data.t_purple_frame; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).tmod = 0;   
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).orth = 0;
               
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
        
        save(fullfile(run_output_dir, ['batch_spec_est_', run_id, '.mat']), 'matlabbatch');
        spm_jobman('run', matlabbatch);
        
        %% -----------------
        % 2) CONTRAST BATCH
        %% -----------------
        SPM_path = fullfile(run_output_dir, 'SPM.mat');
        load(SPM_path, 'SPM');
        
        matlabbatch_con = [];
        matlabbatch_con{1}.spm.stats.con.spmmat = {SPM_path};
        
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

        matlabbatch_con{1}.spm.stats.con.delete = 1;
        
        save(fullfile(run_output_dir, ['batch_contrasts_', run_id, '.mat']), 'matlabbatch_con');
        spm_jobman('run', matlabbatch_con);
        
        %% Save regressor names
        reg_names = SPM.xX.name;
        save(fullfile(run_output_dir, 'regressor_names.mat'), 'reg_names');
        
        disp(['Model for ', run_id, ' in ', sub_id, ' complete.']);
    end
end

diary off;