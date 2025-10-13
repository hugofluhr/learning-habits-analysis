addpath('/home/ubuntu/repos/spm12');

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Directory paths
base_dir = '/home/ubuntu/data/learning-habits/spm_format_noSDC';

% Parameters
smoothing_fwhm = 5; % Smoothing kernel

% Subjects to process
sub_dirs = dir(fullfile(base_dir, 'sub-*'));
subjects = {sub_dirs([sub_dirs.isdir]).name}; % Get all sub- directory names

% Loop through subjects
for i = 1:length(subjects)
    sub_id = subjects{i};
    disp(['Processing subject: ', sub_id]);

    func_dir = fullfile(base_dir, sub_id,'func');
    disp(func_dir);
    
    % Identify unique runs based on BOLD file names
    bold_files = spm_select('FPList', func_dir, '^sub-.*_desc-preproc_bold.nii$');
    run_ids = unique(regexp(cellstr(bold_files), 'run-\d+', 'match', 'once'));
    
    % Check for BOLD files
    if isempty(bold_files)
        warning(['No BOLD files found for ' sub_id]);
        continue;
    end

    % Step 1: Smoothing
    for j = 1:length(run_ids)
        run_id = run_ids{j};
        disp(['Smoothing run: ', run_id]);
        % Select BOLD files for current run
        run_bold_files = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_desc-preproc_bold.nii$']);
        if isempty(run_bold_files)
            warning(['    No BOLD file found for run: ' run_id]);
            continue;
        end
        
        % Smoothing
        matlabbatch = [];
        matlabbatch{1}.spm.spatial.smooth.data = cellstr(run_bold_files);
        matlabbatch{1}.spm.spatial.smooth.fwhm = [smoothing_fwhm smoothing_fwhm smoothing_fwhm];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = ['smoothed_' num2str(smoothing_fwhm) 'mm_'];
        
        % Save and run smoothing batch
        spm_jobman('run', matlabbatch);
    end
end