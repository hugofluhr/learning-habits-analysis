clear;

% Define paths
spmpath = '/home/ubuntu/repos/spm12';
first_lvl_dir = '/home/ubuntu/data/learning-habits/spm_format_20250603/outputs/glm2_2025-06-05-08-56';
base_output_dir = fullfile(first_lvl_dir, 'second-lvl');
addpath(spmpath);

% Define contrast names
connames = {'first_stim', 'second_stim', 'second_stimxQval_diff', 'second_stimxHval_diff', ...
'response', 'feedback'};

% Initialize SPM
spm('Defaults','fMRI'); spm_jobman('initcfg');

% Loop over runs
for r = 1:3
    run_name = ['run-' num2str(r)];
    fprintf('\n===== Processing %s =====\n', run_name);
    % Loop over contrasts
    for c = 1:numel(connames)
        contrast_num = sprintf('%04d', c);  % zero-padded contrast number
        contrast_name = connames{c};
        fprintf('\n===== Processing contrast %s (%s) =====\n', contrast_num, contrast_name);

        % Find matching contrast files for this run
        all_con_files = spm_select('FPListRec', first_lvl_dir, ['con_' contrast_num '.nii$']);
        all_con_files = cellstr(all_con_files);
        con_files = all_con_files(contains(all_con_files, ['/' run_name '/']));

        if isempty(con_files)
            warning('No contrast files found for contrast %s, %s. Skipping.\n', contrast_num, run_name);
            continue;
        end

        % Define output directory using run and contrast name
        output_dir = fullfile(base_output_dir, run_name, contrast_name);
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
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = [contrast_name '_' run_name];
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.delete = 0;

        spm_jobman('run', matlabbatch);
    end
end