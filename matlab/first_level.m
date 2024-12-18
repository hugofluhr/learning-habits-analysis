addpath('/home/ubuntu/repos/spm12');

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Directory paths
base_dir = '/home/ubuntu/data/learning-habits/spm_format';
derivatives_dir = fullfile(base_dir, 'outputs', 'spm_results'); 
if ~exist(derivatives_dir, 'dir')
    mkdir(derivatives_dir);
end

% Parameters
TR = 2.33384; % Repetition time
smoothing_fwhm = 5; % Smoothing kernel
high_pass_cutoff = 128; % High-pass filter in seconds

% Subjects to process
sub_dirs = dir(fullfile(base_dir, 'sub-*'));
subjects = {sub_dirs([sub_dirs.isdir]).name}; % Get all sub- directory names

% Loop through subjects
for i = 1:length(subjects)
    sub_id = subjects{i};
    disp(['Processing subject: ', sub_id]);
    func_dir = fullfile(base_dir, sub_id, 'func');
    disp(func_dir);
    
    % Identify unique runs based on BOLD file names
    bold_files = spm_select('FPList', func_dir, '^sub-.*_desc-preproc_bold.nii$');
    run_ids = unique(regexp(cellstr(bold_files), 'run-\d+', 'match', 'once'));
    
    % Step 1: Smoothing
    smoothed_files = {};
    for j = 1:length(run_ids)
        run_id = run_ids{j};
        disp(['Smoothing run: ', run_id]);
        % Select BOLD files for current run
        run_bold_files = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_desc-preproc_bold.nii$']);
        if isempty(run_bold_files)
            error(['No BOLD files found for run: ', run_id, ' in subject: ', sub_id]);
        end
        
        % Smoothing
        matlabbatch = [];
        matlabbatch{1}.spm.spatial.smooth.data = cellstr(run_bold_files);
        matlabbatch{1}.spm.spatial.smooth.fwhm = [smoothing_fwhm smoothing_fwhm smoothing_fwhm];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = 'smoothed_';
        
        % Save and run smoothing batch
        spm_jobman('run', matlabbatch);
        
        % Collect smoothed files for model specification
        smoothed_files{j} = spm_select('FPList', func_dir, ['^smoothed_.*' run_id '_.*_bold.nii$']);
    end
    disp('Smoothing complete');
    % Step 2: Model specification and estimation
    for j = 1:length(run_ids)
        run_id = run_ids{j};
        output_dir = fullfile(derivatives_dir, sub_id, run_id);
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % Select associated files for the current run
        motion_file = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_motion.txt$']);
        events_file = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_events.mat$']);
        
        if isempty(smoothed_files{j}) || isempty(motion_file) || isempty(events_file)
            error(['Missing required files for run: ', run_id, ' in subject: ', sub_id]);
        end
        
        % Initialize batch
        matlabbatch = [];
        
        % Model specification
        matlabbatch{1}.spm.stats.fmri_spec.dir = {output_dir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(smoothed_files{j});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {events_file};
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {motion_file};
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = high_pass_cutoff;
        
        % Model estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};
        
        % Contrast definition
        matlabbatch{3}.spm.stats.con.spmmat = {fullfile(output_dir, 'SPM.mat')};
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Response';
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1];
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        
        % Save and run batch
        save(fullfile(output_dir, ['batch_job_', run_id, '.mat']), 'matlabbatch');
        spm_jobman('run', matlabbatch);
    end
end