clear;

% Paths
spmpath = '/home/ubuntu/repos/spm12';
data_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603';
analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603';
bbt_path = '/home/ubuntu/data/learning-habits/bbt.csv';
addpath(spmpath);

current_date = char(datetime('now', 'Format', 'yyyy-MM-dd-hh-mm'));
output_dir = fullfile(analysis_dir, 'outputs', ['glm2_' current_date]);
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
connames = {'first_stim', 'second_stim', 'second_stimxQval_diff', 'second_stimxHval_diff', ...
'response', 'feedback'};

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
    disp(func_dir);
    
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

        
        % Model specification for this run
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
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = block_data.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = block_data.t_second_stim - block_data.t_first_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;   
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 0;

        % Second stimulus
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'second_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = block_data.t_second_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = block_data.t_action - block_data.t_second_stim; % check duration
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).name = 'Qval_diff';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).param = zscore(block_data.first_stim_value_rl-block_data.second_stim_value_rl);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(1).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).name = 'Hval_diff';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).param = zscore(block_data.first_stim_value_ck-block_data.second_stim_value_ck);
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod(2).poly = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 0;
        
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
        
        % Other specifications
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
        % Use confounds from fmriprep
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = cellstr(confounds_file);
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = high_pass_cutoff;
        
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
        
        %% Contrast definition
        matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        
        for cc = 1:length(connames)
            matlabbatch{3}.spm.stats.con.consess{cc}.tcon.name = connames{cc};
            matlabbatch{3}.spm.stats.con.consess{cc}.tcon.weights = [zeros(1,cc-1) 1];
            matlabbatch{3}.spm.stats.con.consess{cc}.tcon.sessrep = 'none';
        end
        
        matlabbatch{3}.spm.stats.con.delete = 1;
        
        % Save and run batch
        save(fullfile(run_output_dir, ['batch_job_', run_id, '.mat']), 'matlabbatch');
        spm_jobman('run', matlabbatch);
        
        %% Saving regressor names for traceability
        % Load SPM.mat to extract regressor names
        SPM_path = fullfile(run_output_dir, 'SPM.mat');
        load(SPM_path, 'SPM');
        
        % Save regressor names
        reg_names = SPM.xX.name;
        save(fullfile(run_output_dir, 'regressor_names.mat'), 'reg_names');
        
        disp(['Model for ', run_id, ' in ', sub_id, ' complete.']);
    end
end

diary off;