clear;

% List of subjects to exclude due to excessive motion
excluded_subjects = {
    'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'};

% Define paths
%spmpath  = '/home/ubuntu/repos/spm12';
export_root = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_outputs_noSDC/session_split/glm2_chosen_all_runs_scrubbed_2025-12-11-11-22';  % <-- SET: same export root used in average_sn2_sn3_contrasts.m
                   %         (contains allruns/, session-01/, session-02/, session-03/, session-02-03/)
%addpath(spmpath);

% Initialize SPM
spm('Defaults','fMRI'); spm_jobman('initcfg');

sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');

subdir_name = 'session-02-03';
subdir_path = fullfile(export_root, subdir_name);

if ~exist(subdir_path, 'dir')
    error('Averaged contrast directory not found: %s\nRun average_sn2_sn3_contrasts.m first.', subdir_path);
end

% Auto-detect contrast subdirectories (contrast-XX_<name>)
contrast_dirs = dir(fullfile(subdir_path, 'contrast-*'));
contrast_dirs = contrast_dirs([contrast_dirs.isdir]);

if isempty(contrast_dirs)
    error('No contrast subdirectories found in %s', subdir_path);
end

fprintf('Found %d contrast directories in %s\n', numel(contrast_dirs), subdir_name);

base_output_dir = fullfile(export_root, 'second-lvl', subdir_name);
if ~exist(base_output_dir, 'dir'), mkdir(base_output_dir); end

included_subs = {};

for c = 1:numel(contrast_dirs)
    con_dir_name = contrast_dirs(c).name;

    % Parse contrast index and name from directory name (contrast-XX_<name>)
    tok = regexp(con_dir_name, '^contrast-(\d+)_(.+)$', 'tokens', 'once');
    if isempty(tok)
        warning('Could not parse contrast directory name: %s — skipping.', con_dir_name);
        continue;
    end
    contrast_idx  = str2double(tok{1});
    contrast_name = tok{2};

    fprintf('\n  --- Contrast %02d (%s) ---\n', contrast_idx, contrast_name);

    contrast_subdir = fullfile(subdir_path, con_dir_name);
    all_con_files   = spm_select('FPList', contrast_subdir, '.*_con\.nii$');
    all_con_files   = cellstr(all_con_files);
    con_files       = all_con_files(~cellfun(@isempty, all_con_files));

    if isempty(con_files)
        warning('No contrast files found in %s — skipping.', contrast_subdir);
        continue;
    end
    fprintf('  Found %d files.\n', numel(con_files));

    % Filter excluded subjects
    keep_mask = true(size(con_files));
    for e = 1:numel(excluded_subjects)
        match_idx = contains(con_files, excluded_subjects{e});
        if any(match_idx)
            fprintf('  Excluding %d file(s) matching %s\n', sum(match_idx), excluded_subjects{e});
            keep_mask(match_idx) = false;
        end
    end
    con_files = con_files(keep_mask);

    fprintf('  Remaining after exclusion: %d\n', numel(con_files));
    if isempty(con_files)
        warning('No subjects left after exclusion for %s. Skipping.', contrast_name);
        continue;
    end

    % Collect included subject IDs
    for iCon = 1:numel(con_files)
        tk = regexp(con_files{iCon}, 'sub-([A-Za-z0-9]+)', 'tokens', 'once');
        if ~isempty(tk), included_subs{end+1} = tk{1}; end %#ok<AGROW>
    end
    included_subs = unique(included_subs);

    output_dir = fullfile(base_output_dir, contrast_name);
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end

    % === STEP 1: Design Specification ===
    clear matlabbatch
    matlabbatch{1}.spm.stats.factorial_design.dir            = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans   = con_files;
    matlabbatch{1}.spm.stats.factorial_design.cov            = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im     = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em     = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
    spm_jobman('run', matlabbatch);

    % === STEP 2: Model Estimation ===
    clear matlabbatch
    spm_mat_path = fullfile(output_dir, 'SPM.mat');
    matlabbatch{1}.spm.stats.fmri_est.spmmat             = {spm_mat_path};
    matlabbatch{1}.spm.stats.fmri_est.method.Classical   = 1;
    spm_jobman('run', matlabbatch);

    % === STEP 3: Define Contrast ===
    clear matlabbatch
    matlabbatch{1}.spm.stats.con.spmmat                  = {spm_mat_path};
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name    = contrast_name;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.delete                  = 0;
    spm_jobman('run', matlabbatch);
end

% Write subject manifest
manifest_path = fullfile(base_output_dir, 'subjects_included.txt');
fid = fopen(manifest_path, 'w');
for ii = 1:numel(included_subs)
    fprintf(fid, 'sub-%s\n', included_subs{ii});
end
fclose(fid);
fprintf('\n[%s] Done. Subjects manifest: %s\n', subdir_name, manifest_path);
fprintf('All done.\n');
