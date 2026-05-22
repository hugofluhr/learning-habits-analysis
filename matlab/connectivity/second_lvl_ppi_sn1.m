% second_lvl_ppi_sn1.m
%
% Runs a group-level one-sample t-test on Session 1 (learning1) PPI
% contrast images. No averaging needed — uses con_XXXX.nii directly.
%
% Requires add_session_contrasts_ppi.m to have been run first.
%
% Outputs:
%   <gppi_root>/second-lvl/session-01/
%     contrast-XX_<name>/
%       SPM.mat            second-level model
%       con_*.nii / spmT_*.nii
%       subjects_included.txt

%% ===========================
%% User settings
%% ===========================
spmpath = '/home/ubuntu/repos/spm12';
if ~exist('gppi_root', 'var') || isempty(gppi_root)
    gppi_root = '/mnt/data/learning-habits/spm_format/outputs/PPI/gppi_putamen_Hvalchosen_deconv_2026-03-18-07-39-25';
end

% Base contrast names as stored in SPM.xCon (without the " - Session N" suffix).
% These must match the names used in add_session_contrasts_ppi.m.
contrast_basenames = {
    'PPI_second_stimxHval_chosen^1'
    % 'PPI_first_stim'
    % 'PPI_second_stim'
    % 'PPI_response'
    % 'PPI_purple_frame'
    % 'PPI_points_feedback'
};

excluded_subjects = {
    'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'
};

%% ===========================
%% Setup
%% ===========================
addpath(spmpath);
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');

base_output_dir = fullfile(gppi_root, 'second-lvl', 'session-01');
if ~exist(base_output_dir, 'dir'), mkdir(base_output_dir); end

log_path = fullfile(base_output_dir, 'second_level_sn1_log.txt');
diary(log_path);
diary on;

fprintf('gPPI root: %s\n', gppi_root);
fprintf('Output:    %s\n', base_output_dir);

%% ===========================
%% Find subjects
%% ===========================
all_entries = dir(gppi_root);
sub_dirs    = all_entries([all_entries.isdir] & startsWith({all_entries.name}, 'sub-'));
sub_names   = {sub_dirs.name};
sub_dirs    = sub_dirs(~ismember(sub_names, excluded_subjects));

fprintf('Found %d subject folders after exclusions.\n', numel(sub_dirs));

%% ===========================
%% Loop over contrasts
%% ===========================
for c = 1:numel(contrast_basenames)
    base_name   = contrast_basenames{c};
    base_name_s = sanitize(base_name);
    label_sn1   = sprintf('%s - Session 1', base_name);

    fprintf('\n===== Contrast %d/%d: %s =====\n', c, numel(contrast_basenames), base_name);

    contrast_out = fullfile(base_output_dir, sprintf('contrast-%02d_%s', c, base_name_s));
    if ~exist(contrast_out, 'dir'), mkdir(contrast_out); end

    con_files = {};
    used_subs = {};

    %% -- Collect Session 1 contrast image per subject --
    for s = 1:numel(sub_dirs)
        sub_id   = sub_dirs(s).name;
        ppi_dir  = fullfile(gppi_root, sub_id, 'PPI_putamen');
        spm_path = fullfile(ppi_dir, 'SPM.mat');

        if ~isfile(spm_path)
            fprintf('[SKIP] No PPI_putamen/SPM.mat for %s\n', sub_id);
            continue;
        end

        load(spm_path, 'SPM');
        con_names = string({SPM.xCon.name});

        idx_sn1 = find(con_names == label_sn1, 1);

        if isempty(idx_sn1)
            fprintf('[SKIP] %s: contrast "%s" not found in SPM.xCon\n', sub_id, label_sn1);
            continue;
        end

        con_sn1 = fullfile(ppi_dir, sprintf('con_%04d.nii', idx_sn1));

        if ~isfile(con_sn1)
            fprintf('[SKIP] %s: contrast image missing on disk\n', sub_id);
            continue;
        end

        con_files{end+1, 1} = con_sn1; %#ok<AGROW>
        used_subs{end+1, 1} = sub_id;  %#ok<AGROW>
    end

    fprintf('Session 1 images ready for %d subjects.\n', numel(con_files));

    if isempty(con_files)
        warning('No subjects with Session 1 image for %s. Skipping.', base_name);
        continue;
    end

    % Subject manifest
    fid = fopen(fullfile(contrast_out, 'subjects_included.txt'), 'w');
    for ii = 1:numel(used_subs), fprintf(fid, '%s\n', used_subs{ii}); end
    fclose(fid);

    %% -- Second-level one-sample t-test --

    % Step 1: Design specification
    clear matlabbatch
    matlabbatch{1}.spm.stats.factorial_design.dir              = {contrast_out};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans     = con_files;
    matlabbatch{1}.spm.stats.factorial_design.cov              = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im       = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em       = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit   = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm  = 1;
    spm_jobman('run', matlabbatch);

    % Step 2: Model estimation
    clear matlabbatch
    spm_mat_path = fullfile(contrast_out, 'SPM.mat');
    matlabbatch{1}.spm.stats.fmri_est.spmmat           = {spm_mat_path};
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    spm_jobman('run', matlabbatch);

    % Step 3: Group contrasts (positive and negative effects)
    clear matlabbatch
    matlabbatch{1}.spm.stats.con.spmmat                         = {spm_mat_path};
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name           = [base_name ' positive'];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights        = 1;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep        = 'none';
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name           = [base_name ' negative'];
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights        = -1;
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep        = 'none';
    matlabbatch{1}.spm.stats.con.delete                         = 0;
    spm_jobman('run', matlabbatch);

    fprintf('[DONE] %s\n', base_name);
end

diary off;
fprintf('\nAll done.\n');
