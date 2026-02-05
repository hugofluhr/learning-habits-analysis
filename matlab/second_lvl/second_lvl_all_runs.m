
clear;

% List of subjects to exclude due to excessive motion (edit as needed)
excluded_subjects = {
    'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'};

% Define paths
spmpath = '/home/ubuntu/repos/spm12';
first_lvl_dir = '/home/ubuntu/data/learning-habits/spm_outputs_noSDC/glm2_all_runs_sustained_2026-02-05-03-03';
base_output_dir = fullfile(first_lvl_dir, 'second-lvl');
addpath(spmpath);

% Initialize SPM
spm('Defaults','fMRI'); spm_jobman('initcfg');

% Read contrast order file (all runs)
order_file = fullfile(first_lvl_dir, 'contrast_list_order_allruns.txt');
if ~exist(order_file, 'file')
    error('Contrast order file not found: %s', order_file);
end
fid = fopen(order_file, 'r');
data = textscan(fid, '%d\t%s');
fclose(fid);
contrast_indices = data{1};
contrast_names = data{2};
fprintf('Found %d contrasts in all-runs export\n', numel(contrast_indices));

included_subs = {};

for c = 1:numel(contrast_names)
    contrast_idx = contrast_indices(c);
    contrast_name = contrast_names{c};
    fprintf('\n===== Processing contrast %02d (%s) =====\n', contrast_idx, contrast_name);

    % Create sanitized contrast name for file matching
    sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');
    contrast_name_sanitized = sanitize(contrast_name);

    % Find exported contrast files in the contrast-specific subdirectory
    contrast_subdir = fullfile(first_lvl_dir, sprintf('contrast-%02d_%s', contrast_idx, contrast_name_sanitized));
    if exist(contrast_subdir, 'dir')
        all_con_files = spm_select('FPList', contrast_subdir, '.*_con\.nii$');
        all_con_files = cellstr(all_con_files);
        con_files = all_con_files(~cellfun(@isempty, all_con_files));
    else
        % Fallback: look for files directly in the root directory (if perContrastDirs=false was used)
        search_pattern = sprintf('.*_desc-%s_con.nii$', contrast_name_sanitized);
        all_con_files = spm_select('FPList', first_lvl_dir, search_pattern);
        all_con_files = cellstr(all_con_files);
        con_files = all_con_files(~cellfun(@isempty, all_con_files));
    end

    if isempty(con_files)
        warning('No aliased contrast files found for contrast %02d (%s). Searching pattern: %s', ...
                contrast_idx, contrast_name, contrast_subdir);
        continue;
    end

    fprintf('Found %d contrast files for %s\n', numel(con_files), contrast_name);

    % === Filter out excluded subjects ===
    keep_mask = true(size(con_files));
    for e = 1:numel(excluded_subjects)
        excl_pattern = excluded_subjects{e};
        match_idx = contains(con_files, excl_pattern);
        if any(match_idx)
            fprintf('Excluding %d files matching %s\n', sum(match_idx), excl_pattern);
            keep_mask(match_idx) = false;
        end
    end
    con_files = con_files(keep_mask);

    fprintf('Remaining subjects after exclusion: %d\n', numel(con_files));
    if isempty(con_files)
        warning('No subjects left after exclusion for contrast %s. Skipping.', contrast_name);
        continue;
    end

    % Define output directory using contrast name
    output_dir = fullfile(base_output_dir, contrast_name_sanitized);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % collect subject IDs from the available contrast files
    for iCon = 1:numel(con_files)
        cf = con_files{iCon};
        tk = regexp(cf, 'sub-([A-Za-z0-9]+)', 'tokens', 'once');
        if ~isempty(tk)
            included_subs{end+1} = tk{1};
        end
    end
    included_subs = unique(included_subs);

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

% After processing all contrasts, write a manifest of included subjects
manifest_path = fullfile(base_output_dir, 'subjects_included.txt');
fid = fopen(manifest_path, 'w');
for ii = 1:numel(included_subs)
    fprintf(fid, 'sub-%s\n', included_subs{ii});
end
fclose(fid);
fprintf('Wrote subjects manifest to %s\n', manifest_path);