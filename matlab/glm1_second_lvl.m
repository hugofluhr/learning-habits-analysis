% Initialize SPM
spm('Defaults','fMRI'); spm_jobman('initcfg');

% Define root paths
data_root = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_outputs/glm1_2025-06-05-05-15';
base_output_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_outputs/glm1_2025-06-05-05-15/second-lvl';

% Define your contrast names
connames = {'first_stim', 'first_stimxQval', 'first_stimxHval', ...
            'second_stim', 'response', 'feedback'};

% Loop over contrast numbers and names
for c = 1:numel(connames)
    contrast_num = sprintf('%04d', c);  % zero-padded contrast number
    contrast_name = connames{c};
    fprintf('\n===== Processing contrast %s (%s) =====\n', contrast_num, contrast_name);

    % Find matching contrast files
    all_con_files = spm_select('FPListRec', data_root, ['con_' contrast_num '.nii$']);
    all_con_files = cellstr(all_con_files);

    % Filter for run-3 only
    con_files = all_con_files(contains(all_con_files, '/run-3/'));

    if isempty(con_files)
        warning('No contrast files found for contrast %s. Skipping.\n', contrast_num);
        continue;
    end

    % Define output directory using the contrast name
    output_dir = fullfile(base_output_dir, contrast_name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % === STEP 1: Design Specification ===
    clear matlabbatch
    matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = con_files;
    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    spm_jobman('run', matlabbatch);

    % === STEP 2: Model Estimation ===
    clear matlabbatch
    spm_mat_path = fullfile(output_dir, 'SPM.mat');
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {spm_mat_path};
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

    spm_jobman('run', matlabbatch);

    % === STEP 3: Define Contrast ===
    clear matlabbatch
    matlabbatch{1}.spm.stats.con.spmmat = {spm_mat_path};
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = contrast_name;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.delete = 0;

    spm_jobman('run', matlabbatch);
end