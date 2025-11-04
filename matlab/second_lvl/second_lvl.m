clear;

% List of subjects to exclude due to excessive motion
excluded_learning = {
    'sub-44', 'sub-48', 'sub-68'};
excluded_test = {
    'sub-17', 'sub-31', 'sub-48', 'sub-68'};

% Define paths
spmpath = '/home/ubuntu/repos/spm12';
first_lvl_dir = '/home/ubuntu/data/learning-habits/spm_outputs/glm2_mf_2025-10-29-11-27';
base_output_dir = fullfile(first_lvl_dir, 'second-lvl');
addpath(spmpath);

% Initialize SPM
spm('Defaults','fMRI'); spm_jobman('initcfg');

% Read contrast order files for each phase
phases = {'learning', 'test'};
contrast_info = struct();

for p = 1:numel(phases)
    phase = phases{p};
    phase_dir = fullfile(first_lvl_dir, phase);
    order_file = fullfile(phase_dir, sprintf('contrast_list_order_phase-%s.txt', phase));
    
    if ~exist(order_file, 'file')
        warning('Contrast order file not found for phase %s: %s', phase, order_file);
        continue;
    end
    
    % Read contrast order
    fid = fopen(order_file, 'r');
    data = textscan(fid, '%d\t%s');
    fclose(fid);
    
    contrast_info.(phase).indices = data{1};
    contrast_info.(phase).names = data{2};
    
    fprintf('Found %d contrasts for phase %s\n', numel(data{1}), phase);
end

% Loop over phases and contrasts
for p = 1:numel(phases)
    phase = phases{p};
    
    if ~isfield(contrast_info, phase)
        warning('No contrast information for phase %s, skipping.', phase);
        continue;
    end
    
    fprintf('\n===== Processing phase %s =====\n', phase);
    
    % Loop over contrasts for this phase
    for c = 1:numel(contrast_info.(phase).names)
        contrast_idx = contrast_info.(phase).indices(c);
        contrast_name = contrast_info.(phase).names{c};
        
        fprintf('\n===== Processing contrast %02d (%s) for phase %s =====\n', contrast_idx, contrast_name, phase);
        
        % Create sanitized contrast name for file matching
        sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');
        contrast_name_sanitized = sanitize(contrast_name);
        
        % Find exported contrast files for this phase
        % When using perContrastDirs=true (default), look in contrast-specific subdirectories
        contrast_subdir = fullfile(first_lvl_dir, phase, sprintf('contrast-%02d_%s', contrast_idx, contrast_name_sanitized));
        
        if exist(contrast_subdir, 'dir')
            % Look for exported contrast files in the contrast-specific subdirectory
            all_con_files = spm_select('FPList', contrast_subdir, '.*_con\.nii$');
            all_con_files = cellstr(all_con_files);
            con_files = all_con_files(~cellfun(@isempty, all_con_files));
        else
            % Fallback: look for files directly in the phase directory (if perContrastDirs=false was used)
            phase_dir = fullfile(first_lvl_dir, phase);
            search_pattern = sprintf('.*_desc-%s_con.nii$', contrast_name_sanitized);
            all_con_files = spm_select('FPList', phase_dir, search_pattern);
            all_con_files = cellstr(all_con_files);
            con_files = all_con_files(~cellfun(@isempty, all_con_files));
        end
        
        if isempty(con_files)
            warning('No aliased contrast files found for contrast %02d (%s) in phase %s. Searching pattern: %s', ...
                    contrast_idx, contrast_name, phase, search_pattern);
            continue;
        end
        
        fprintf('Found %d contrast files for %s in phase %s\n', numel(con_files), contrast_name, phase);

        % === Filter out excluded subjects ===
        switch phase
            case 'learning'
                excl_list = excluded_learning;
            case 'test'
                excl_list = excluded_test;
        end

        keep_mask = true(size(con_files));
        for e = 1:numel(excl_list)
            excl_pattern = excl_list{e};
            match_idx = contains(con_files, excl_pattern);
            if any(match_idx)
                fprintf('Excluding %d files matching %s from phase %s\n', sum(match_idx), excl_pattern, phase);
                keep_mask(match_idx) = false;
            end
        end

        con_files = con_files(keep_mask);

        fprintf('Remaining subjects after exclusion: %d\n', numel(con_files));
        if isempty(con_files)
            warning('No subjects left after exclusion for contrast %s (%s phase). Skipping.', contrast_name, phase);
            continue;
        end
        
        % Define output directory using phase and contrast name
        output_dir = fullfile(base_output_dir, phase, contrast_name_sanitized);
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
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = sprintf('%s_%s', contrast_name, phase);
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.delete = 0;

        spm_jobman('run', matlabbatch);
    end
end