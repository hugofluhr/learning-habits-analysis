%addpath('/home/ubuntu/repos/spm12');

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Directory paths
base_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bids_dataset';
fmriprep_dir = dir(fullfile(base_dir, 'derivatives', 'fmriprep*'));
if isempty(fmriprep_dir)
    error('No directory starting with "fmriprep" found in base_dir.');
end
fmriprep_dir = fullfile(base_dir, 'derivatives', fmriprep_dir(1).name);
spm_dir = fullfile(base_dir, 'derivatives', 'spm'); 
if ~exist(spm_dir, 'dir')
    mkdir(spm_dir);
end

% Parameters
smoothing_fwhm = 5; % Smoothing kernel

% Subjects to process
sub_dirs = dir(fullfile(base_dir, 'sub-*'));
subjects = {sub_dirs([sub_dirs.isdir]).name}; % Get all sub- directory names

% Loop through subjects
for i = 1:length(subjects)
    sub_id = subjects{i};
    disp(['Processing subject: ', sub_id]);
    func_dir = fullfile(fmriprep_dir, sub_id, 'ses-1','func');
    disp(func_dir);
    
    % Identify unique runs based on BOLD file names
    bold_files = spm_select('FPList', func_dir, '^sub-.*_desc-preproc_bold.nii$');
    run_ids = unique(regexp(cellstr(bold_files), 'run-\d+', 'match', 'once'));
    
    % Step 1: Smoothing
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
        matlabbatch{1}.spm.spatial.smooth.prefix = ['smoothed_' num2str(smoothing_fwhm) 'mm_'];
        
        % Save and run smoothing batch
        spm_jobman('run', matlabbatch);
    end
    % Move smoothed files
    smoothed_files = dir(fullfile(source_dir, 'smoothed_*.nii'));
    for i = 1:length(smoothed_files)
        movefile(fullfile(source_dir, smoothed_files(i).name), target_dir);
    end
end