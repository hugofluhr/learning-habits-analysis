
clear;

% List of subjects to exclude due to excessive motion (edit as needed)
excluded_subjects = {
    'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'};

% Define paths
spmpath = '/home/ubuntu/repos/spm12';
export_root = '';  % <-- SET: root export directory produced by export_first_lvl_contrasts_with_sessions
                   %         (contains allruns/, session-01/, session-02/, session-03/)
addpath(spmpath);

% Initialize SPM
spm('Defaults','fMRI'); spm_jobman('initcfg');

% Auto-detect layout:
%   New layout  — export_root/allruns/ exists → loop over allruns + session subdirs
%   Flat layout — no allruns/ subdir → treat export_root itself as the allruns directory
if exist(fullfile(export_root, 'allruns'), 'dir')
    subdirs         = {'allruns', 'session-01', 'session-02', 'session-03'};
    order_filenames = {
        'contrast_list_order_allruns.txt',
        'contrast_list_order_session-01.txt',
        'contrast_list_order_session-02.txt',
        'contrast_list_order_session-03.txt'
    };
    fprintf('Detected new layout (allruns/ subdir present).\n');
else
    subdirs         = {''};   % empty string → use export_root directly
    order_filenames = {'contrast_list_order_allruns.txt'};
    fprintf('Detected flat layout (no allruns/ subdir — treating export_root as allruns).\n');
end

sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');

for sd = 1:numel(subdirs)
    subdir_name = subdirs{sd};
    if isempty(subdir_name)
        subdir_path = export_root;
    else
        subdir_path = fullfile(export_root, subdir_name);
    end

    if ~exist(subdir_path, 'dir')
        fprintf('[SKIP] %s not found - skipping.\n', subdir_path);
        continue;
    end

    order_file = fullfile(subdir_path, order_filenames{sd});
    if ~exist(order_file, 'file')
        fprintf('[SKIP] No contrast order file for %s: %s\n', subdir_name, order_file);
        continue;
    end

    fid = fopen(order_file, 'r');
    data = textscan(fid, '%d\t%s');
    fclose(fid);
    contrast_indices = data{1};
    contrast_names   = data{2};
    display_name = subdir_name;
    if isempty(display_name), display_name = 'allruns (flat)'; end
    fprintf('\n===== %s: found %d contrasts =====\n', display_name, numel(contrast_indices));

    if isempty(subdir_name)
        base_output_dir = fullfile(export_root, 'second-lvl');
    else
        base_output_dir = fullfile(export_root, 'second-lvl', subdir_name);
    end
    if ~exist(base_output_dir, 'dir'), mkdir(base_output_dir); end

    included_subs = {};

    for c = 1:numel(contrast_names)
        contrast_idx            = contrast_indices(c);
        contrast_name           = contrast_names{c};
        contrast_name_sanitized = sanitize(contrast_name);

        fprintf('\n  --- Contrast %02d (%s) ---\n', contrast_idx, contrast_name);

        % Find exported contrast files
        contrast_subdir = fullfile(subdir_path, ...
            sprintf('contrast-%02d_%s', contrast_idx, contrast_name_sanitized));

        if exist(contrast_subdir, 'dir')
            all_con_files = spm_select('FPList', contrast_subdir, '.*_con\.nii$');
        else
            % Fallback: flat layout (perContrastDirs=false)
            search_pattern = sprintf('.*_desc-%s_con.nii$', contrast_name_sanitized);
            all_con_files  = spm_select('FPList', subdir_path, search_pattern);
        end
        all_con_files = cellstr(all_con_files);
        con_files     = all_con_files(~cellfun(@isempty, all_con_files));

        if isempty(con_files)
            warning('No contrast files found for %s in %s.', contrast_name, subdir_name);
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
            warning('No subjects left after exclusion for %s (%s). Skipping.', contrast_name, subdir_name);
            continue;
        end

        % Collect included subject IDs
        for iCon = 1:numel(con_files)
            tk = regexp(con_files{iCon}, 'sub-([A-Za-z0-9]+)', 'tokens', 'once');
            if ~isempty(tk), included_subs{end+1} = tk{1}; end %#ok<AGROW>
        end
        included_subs = unique(included_subs);

        output_dir = fullfile(base_output_dir, contrast_name_sanitized);
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
        matlabbatch{1}.spm.stats.fmri_est.spmmat  = {spm_mat_path};
        matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
        spm_jobman('run', matlabbatch);

        % === STEP 3: Define Contrast ===
        clear matlabbatch
        matlabbatch{1}.spm.stats.con.spmmat                        = {spm_mat_path};
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.name          = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights       = 1;
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep       = 'none';
        matlabbatch{1}.spm.stats.con.delete                        = 0;
        spm_jobman('run', matlabbatch);
    end

    % Write subject manifest for this subdirectory
    manifest_path = fullfile(base_output_dir, 'subjects_included.txt');
    fid = fopen(manifest_path, 'w');
    for ii = 1:numel(included_subs)
        fprintf(fid, 'sub-%s\n', included_subs{ii});
    end
    fclose(fid);
    fprintf('[%s] Done. Subjects manifest: %s\n', subdir_name, manifest_path);
end

fprintf('\nAll done.\n');
