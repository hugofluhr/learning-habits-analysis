%% Get files
% Set base directories
base_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_format/outputs';
derivatives_dir = fullfile(base_dir, 'spm_results');

% Find all subject directories
subject_dirs = dir(fullfile(derivatives_dir, 'sub-*'));
subjects = {subject_dirs([subject_dirs.isdir]).name};

% Initialize file list
beta_files = {};

% Loop through subjects
for i = 1:length(subjects)
    sub_id = subjects{i};
    beta_dir = fullfile(derivatives_dir, sub_id, 'run-1');
    
    % Locate beta_response map (e.g., beta_0001.nii)
    beta_map = spm_select('FPList', beta_dir, '^beta_Sn_1__response.*\.nii$');
    
    if isempty(beta_map)
        warning(['No beta map found for Subject: ', sub_id, ', Run: run-1']);
    else
        beta_files{end+1, 1} = beta_map; % Collect the beta map file
    end
end

% Set output directory for second-level results
second_level_dir = fullfile(derivatives_dir, 'second_level', 'beta_response_run-1');
if ~exist(second_level_dir, 'dir')
    mkdir(second_level_dir);
end

%% Analysis
% Create batch for second-level analysis
matlabbatch = [];

% Model specification
matlabbatch{1}.spm.stats.factorial_design.dir = {second_level_dir};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = beta_files'; % List of beta maps
matlabbatch{1}.spm.stats.factorial_design.cov = struct([]); % No covariates for now
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1; % No threshold masking
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1; % Implicit mask
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''}; % No explicit mask
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1; % Ignore global scaling
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1; % No grand mean scaling
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1; % Normalization method

% Model estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(second_level_dir, 'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0; % Do not save residuals
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1; % Classical estimation

% Contrast definition: One-sample t-test
matlabbatch{3}.spm.stats.con.spmmat = {fullfile(second_level_dir, 'SPM.mat')};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Mean Response';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1; % Weight for the mean
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none'; % No session replication

% Save and run batch
save(fullfile(second_level_dir, 'second_level_job.mat'), 'matlabbatch');
spm_jobman('run', matlabbatch);