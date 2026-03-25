clear;

%% ===========================
%% User settings
%% ===========================
% Edit these paths for your machine
spmpath      = '/home/ubuntu/repos/spm12';
gppi_root = '/mnt/data/learning-habits/spm_format/outputs/PPI/gppi_putamen_Hvalchosen_deconv_2026-03-18-07-39-25';

% Enter the exact PPPI contrast names you want to run here.
% These should match the filenames after the "con_PPI_" prefix and before
% the "_sub-XX.nii" suffix.
contrast_names_to_run = {
    'second_stimxHval_chosen^1'
    };
%con_PPI_second_stimxHval_chosen^1_sub-01
% Optional: exclude subjects here
excluded_subjects = {
    'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31' ... % motion exclusions from your second-level script
    };

%% ===========================
%% Setup
%% ===========================
addpath(spmpath);
spm('Defaults','fMRI');
spm_jobman('initcfg');

if ~exist(gppi_root, 'dir')
    error('gPPI root directory not found: %s', gppi_root);
end

base_output_dir = fullfile(gppi_root, 'second-lvl');
if ~exist(base_output_dir, 'dir')
    mkdir(base_output_dir);
end

log_path = fullfile(base_output_dir, 'second_level_log.txt');
diary(log_path);
diary on;

fprintf('Using gPPI root: %s\n', gppi_root);
fprintf('Second-level output dir: %s\n', base_output_dir);

%% ===========================
%% Find subject folders
%% ===========================
sub_dirs = dir(fullfile(gppi_root, 'sub-*'));
sub_dirs = sub_dirs([sub_dirs.isdir]);

if isempty(sub_dirs)
    error('No subject folders found in %s', gppi_root);
end

sub_names = {sub_dirs.name};
keep_sub = ~ismember(sub_names, excluded_subjects);
sub_dirs = sub_dirs(keep_sub);

fprintf('Found %d subject folders after exclusions.\n', numel(sub_dirs));

if isempty(sub_dirs)
    error('No subject folders remain after exclusions.');
end

%% ===========================
%% Loop over requested PPPI contrast names
%% ===========================
included_subs_all = {};
manifest = {};

sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-zA-Z0-9._-]', '');

for c = 1:numel(contrast_names_to_run)
    contrast_name = contrast_names_to_run{c};
    contrast_name_sanitized = sanitize(contrast_name);

    fprintf('\n===== Processing requested contrast %d/%d: %s =====\n', ...
        c, numel(contrast_names_to_run), contrast_name);

    con_files = {};
    used_subs = {};

    % Gather the named PPPI contrast from each subject directory
    for s = 1:numel(sub_dirs)
        sub_id = sub_dirs(s).name;
        sub_dir = fullfile(gppi_root, sub_id, 'PPI_putamen');

        % Expected PPPI naming pattern:
        % con_PPI_<contrast_name>_<sub_id>.nii
        con_filename = sprintf('con_PPI_%s_%s.nii', contrast_name, sub_id);
        con_file = fullfile(sub_dir, con_filename);

        if exist(con_file, 'file')
            con_files{end+1,1} = con_file;
            used_subs{end+1,1} = sub_id;
        else
            fprintf('Missing contrast file for %s: %s\n', sub_id, con_filename);
        end
    end

    fprintf('Found %d subject contrast files for %s\n', numel(con_files), contrast_name);

    if isempty(con_files)
        warning('No files found for requested contrast %s. Skipping.', contrast_name);
        continue;
    end

    included_subs_all = unique([included_subs_all; used_subs]);

    % Save contrast-specific subject manifest
    contrast_manifest = fullfile(base_output_dir, sprintf('subjects_%02d_%s.txt', c, contrast_name_sanitized));
    fid = fopen(contrast_manifest, 'w');
    for ii = 1:numel(used_subs)
        fprintf(fid, '%s\n', used_subs{ii});
    end
    fclose(fid);

    manifest{end+1,1} = sprintf('%02d\t%s\t%d', c, contrast_name, numel(used_subs));

    % Create output directory for this contrast
    output_dir = fullfile(base_output_dir, sprintf('contrast-%02d_%s', c, contrast_name_sanitized));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% STEP 1: Design specification (one-sample t-test)
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

    %% STEP 2: Model estimation
    clear matlabbatch
    spm_mat_path = fullfile(output_dir, 'SPM.mat');
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {spm_mat_path};
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    spm_jobman('run', matlabbatch);

    %% STEP 3: Define group-level contrasts
    clear matlabbatch
    matlabbatch{1}.spm.stats.con.spmmat = {spm_mat_path};

    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = [contrast_name ' positive'];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = [contrast_name ' negative'];
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = -1;
    matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

    matlabbatch{1}.spm.stats.con.delete = 0;
    spm_jobman('run', matlabbatch);
end

%% ===========================
%% Write manifests
%% ===========================
summary_path = fullfile(base_output_dir, 'contrast_summary.txt');
fid = fopen(summary_path, 'w');
fprintf(fid, 'contrast_order\tcontrast_name\tn_subjects\n');
for ii = 1:numel(manifest)
    fprintf(fid, '%s\n', manifest{ii});
end
fclose(fid);

subjects_manifest = fullfile(base_output_dir, 'subjects_included_any_contrast.txt');
fid = fopen(subjects_manifest, 'w');
included_subs_all = unique(included_subs_all);
for ii = 1:numel(included_subs_all)
    fprintf(fid, '%s\n', included_subs_all{ii});
end
fclose(fid);

fprintf('\nDone. Wrote summary to %s\n', summary_path);
fprintf('Wrote included-subject manifest to %s\n', subjects_manifest);

diary off;
